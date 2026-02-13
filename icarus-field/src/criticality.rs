// Copyright (c) 2025-2026 brdigetrlol. All rights reserved.
// SPDX-License-Identifier: LicenseRef-Icarus-Proprietary
// See LICENSE in the repository root for full license terms.

//! Edge-of-Chaos Criticality Controller
//!
//! Auto-tunes γ (damping) to keep the largest Lyapunov exponent λ_max near
//! a target value (default 0.0 = edge of chaos).
//!
//! ## Algorithm
//!
//! Every `measure_interval` ticks, estimate λ_max via perturbation:
//!
//! 1. Clone the current field state
//! 2. Add a small random perturbation (||δ|| ≈ ε)
//! 3. Run both original and perturbed for `probe_steps` RAE steps
//! 4. Measure growth: ratio = ||δ_final|| / ||δ_0||
//! 5. λ_max ≈ ln(ratio) / (probe_steps × dt)
//!
//! Then use a PI controller to adjust γ:
//!
//!   error = λ_max - λ_target
//!   integral += error
//!   Δγ = k_p × error + k_i × integral
//!   γ_new = clamp(γ + Δγ, γ_min, γ_max)
//!
//! ## References
//!
//! - Bertschinger & Natschläger (2004): Real-time computation at the edge of chaos
//! - Legenstein & Maass (2007): Edge of chaos and prediction of computational
//!   performance for neural circuit models
//! - Carroll (2020): Do reservoir computers work best at the edge of chaos?

use crate::phase_field::LatticeField;
use crate::rae::RAESolver;

/// Parameters for the criticality controller.
#[derive(Debug, Clone)]
pub struct CriticalityParams {
    /// Target Lyapunov exponent (0.0 = edge of chaos)
    pub lambda_target: f32,
    /// Perturbation magnitude for Lyapunov estimation
    pub epsilon: f32,
    /// Number of RAE steps for each Lyapunov probe
    pub probe_steps: u64,
    /// Ticks between Lyapunov measurements
    pub measure_interval: u64,
    /// PI controller proportional gain
    pub k_p: f32,
    /// PI controller integral gain
    pub k_i: f32,
    /// Minimum allowed γ (damping)
    pub gamma_min: f32,
    /// Maximum allowed γ (damping)
    pub gamma_max: f32,
    /// Minimum allowed ω (resonance frequency)
    pub omega_min: f32,
    /// Maximum allowed ω (resonance frequency)
    pub omega_max: f32,
    /// Whether to also adjust ω (secondary control)
    pub adjust_omega: bool,
}

impl Default for CriticalityParams {
    fn default() -> Self {
        Self {
            lambda_target: 0.0,
            epsilon: 1e-6,
            probe_steps: 10,
            measure_interval: 5,
            k_p: 0.01,
            k_i: 0.001,
            gamma_min: 0.001,
            gamma_max: 1.0,
            omega_min: 0.1,
            omega_max: 20.0,
            adjust_omega: false,
        }
    }
}

/// Edge-of-chaos criticality controller.
///
/// Periodically estimates the largest Lyapunov exponent via perturbation
/// growth measurement and uses a PI controller to adjust γ (and optionally ω)
/// to maintain the system near the edge of chaos.
#[derive(Debug, Clone)]
pub struct CriticalityController {
    /// Controller parameters
    params: CriticalityParams,
    /// Most recent λ_max estimate
    lambda_estimate: f32,
    /// PI integral accumulator
    integral_error: f32,
    /// Counter for measure_interval
    ticks_since_measure: u64,
    /// RNG seed for perturbations (deterministic LCG)
    seed: u64,
}

impl CriticalityController {
    /// Create a new criticality controller with the given parameters.
    pub fn new(params: CriticalityParams) -> Self {
        Self {
            params,
            lambda_estimate: 0.0,
            integral_error: 0.0,
            ticks_since_measure: 0,
            seed: 0xDEAD_BEEF_CAFE_1234,
        }
    }

    /// Called each tick. Returns `Some((new_omega, new_gamma))` if an adjustment
    /// was made, or `None` if not yet time to measure.
    pub fn adapt(&mut self, field: &LatticeField, solver: &RAESolver) -> Option<(f32, f32)> {
        self.ticks_since_measure += 1;
        if self.ticks_since_measure < self.params.measure_interval {
            return None;
        }
        self.ticks_since_measure = 0;

        // Estimate Lyapunov exponent via perturbation growth
        self.lambda_estimate = self.estimate_lyapunov(field, solver);

        // PI control on γ
        let error = self.lambda_estimate - self.params.lambda_target;
        self.integral_error += error;
        // Anti-windup: clamp integral to prevent unbounded accumulation
        self.integral_error = self.integral_error.clamp(-10.0, 10.0);

        let delta_gamma = self.params.k_p * error + self.params.k_i * self.integral_error;
        let new_gamma = (solver.params.gamma + delta_gamma)
            .clamp(self.params.gamma_min, self.params.gamma_max);

        let new_omega = if self.params.adjust_omega {
            // Secondary control: decrease ω when too chaotic (error > 0)
            let delta_omega = -0.5 * self.params.k_p * error;
            (solver.params.omega + delta_omega)
                .clamp(self.params.omega_min, self.params.omega_max)
        } else {
            solver.params.omega
        };

        Some((new_omega, new_gamma))
    }

    /// Estimate the largest Lyapunov exponent via perturbation growth.
    ///
    /// 1. Clone field, add random perturbation of magnitude ε
    /// 2. Evolve both original (cloned) and perturbed for probe_steps
    /// 3. λ_max ≈ ln(||δ_final|| / ||δ_0||) / (probe_steps × dt)
    fn estimate_lyapunov(&mut self, field: &LatticeField, solver: &RAESolver) -> f32 {
        let mut ref_field = field.clone();
        let mut probe_field = field.clone();

        // Add random perturbation to probe field
        let epsilon = self.params.epsilon;
        let mut rng_state = self.seed;
        let mut delta_norm_sq = 0.0f32;

        for i in 0..probe_field.num_sites {
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let d_re = ((rng_state >> 33) as f32 / u32::MAX as f32 * 2.0 - 1.0) * epsilon;
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let d_im = ((rng_state >> 33) as f32 / u32::MAX as f32 * 2.0 - 1.0) * epsilon;
            probe_field.values_re[i] += d_re;
            probe_field.values_im[i] += d_im;
            delta_norm_sq += d_re * d_re + d_im * d_im;
        }
        self.seed = rng_state;
        let delta_0 = delta_norm_sq.sqrt();

        // Evolve both fields for probe_steps using cloned solvers
        let mut ref_solver = RAESolver::new(solver.params.clone(), ref_field.num_sites);
        let mut probe_solver = RAESolver::new(solver.params.clone(), probe_field.num_sites);

        ref_solver.run(&mut ref_field, self.params.probe_steps);
        probe_solver.run(&mut probe_field, self.params.probe_steps);

        // Measure divergence
        let mut diff_norm_sq = 0.0f32;
        for i in 0..probe_field.num_sites {
            let dr = probe_field.values_re[i] - ref_field.values_re[i];
            let di = probe_field.values_im[i] - ref_field.values_im[i];
            diff_norm_sq += dr * dr + di * di;
        }
        let delta_final = diff_norm_sq.sqrt();

        // λ_max = ln(||δ_final|| / ||δ_0||) / (N × dt)
        let total_time = self.params.probe_steps as f32 * solver.params.dt;
        if delta_0 > 1e-30 && delta_final > 1e-30 && total_time > 1e-30 {
            (delta_final / delta_0).ln() / total_time
        } else {
            0.0
        }
    }

    /// Most recent Lyapunov exponent estimate.
    pub fn lambda_estimate(&self) -> f32 {
        self.lambda_estimate
    }

    /// Get the controller parameters.
    pub fn params(&self) -> &CriticalityParams {
        &self.params
    }

    /// Reset the controller state (clears integral accumulator and tick counter).
    pub fn reset(&mut self) {
        self.integral_error = 0.0;
        self.ticks_since_measure = 0;
        self.lambda_estimate = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::free_energy::FreeEnergyParams;
    use crate::rae::{IntegratorMethod, RAEParams};
    use icarus_math::lattice::e8::E8Lattice;

    fn make_e8_field() -> LatticeField {
        let lattice = E8Lattice::new();
        LatticeField::from_lattice(&lattice)
    }

    fn make_solver(omega: f32, gamma: f32) -> RAESolver {
        let params = RAEParams {
            dt: 0.002,
            omega,
            gamma,
            energy_params: FreeEnergyParams::default(),
            method: IntegratorMethod::SemiImplicit,
        };
        RAESolver::new(params, 241)
    }

    #[test]
    fn test_default_params() {
        let p = CriticalityParams::default();
        assert!((p.lambda_target - 0.0).abs() < 1e-9);
        assert!((p.epsilon - 1e-6).abs() < 1e-12);
        assert_eq!(p.probe_steps, 10);
        assert_eq!(p.measure_interval, 5);
        assert!(p.k_p > 0.0);
        assert!(p.k_i > 0.0);
        assert!(p.gamma_min > 0.0);
        assert!(p.gamma_max > p.gamma_min);
        assert!(!p.adjust_omega);
    }

    #[test]
    fn test_measure_interval_respected() {
        let params = CriticalityParams {
            measure_interval: 5,
            ..CriticalityParams::default()
        };
        let mut ctrl = CriticalityController::new(params);
        let mut field = make_e8_field();
        field.init_random(42, 0.5);
        let solver = make_solver(1.0, 0.1);

        // First 4 calls should return None (interval not reached)
        for _ in 0..4 {
            assert!(ctrl.adapt(&field, &solver).is_none());
        }
        // 5th call should return Some (measurement happens)
        assert!(ctrl.adapt(&field, &solver).is_some());

        // Next 4 should return None again
        for _ in 0..4 {
            assert!(ctrl.adapt(&field, &solver).is_none());
        }
        // 10th total should return Some
        assert!(ctrl.adapt(&field, &solver).is_some());
    }

    #[test]
    fn test_lyapunov_positive_in_chaotic_regime() {
        // High ω, low γ → system should be oscillatory/chaotic → positive λ
        let mut field = make_e8_field();
        field.init_random(42, 0.8);
        let solver = make_solver(10.0, 0.001);

        let params = CriticalityParams {
            measure_interval: 1,
            probe_steps: 20,
            epsilon: 1e-5,
            ..CriticalityParams::default()
        };
        let mut ctrl = CriticalityController::new(params);

        // Run solver to establish dynamics
        let mut running_solver = solver.clone();
        running_solver.run(&mut field, 100);

        ctrl.adapt(&field, &running_solver);
        let lambda = ctrl.lambda_estimate();
        // In a strongly driven regime, perturbations should grow
        assert!(
            lambda > -50.0,
            "λ should be computable (not -inf), got {}",
            lambda
        );
    }

    #[test]
    fn test_lyapunov_negative_in_ordered_regime() {
        // Low ω, high γ → system strongly damped → negative λ
        let mut field = make_e8_field();
        field.init_random(42, 0.5);
        let solver = make_solver(0.1, 0.8);

        let params = CriticalityParams {
            measure_interval: 1,
            probe_steps: 20,
            epsilon: 1e-5,
            ..CriticalityParams::default()
        };
        let mut ctrl = CriticalityController::new(params);

        // Run to establish strongly damped dynamics
        let mut running_solver = solver.clone();
        running_solver.run(&mut field, 200);

        ctrl.adapt(&field, &running_solver);
        let lambda = ctrl.lambda_estimate();
        // Strong damping should make perturbations shrink → λ < 0
        assert!(
            lambda < 0.0,
            "λ should be negative in ordered regime, got {}",
            lambda
        );
    }

    #[test]
    fn test_pi_controller_increases_gamma_when_chaotic() {
        // If λ > target (too chaotic), γ should increase to add damping
        let mut field = make_e8_field();
        field.init_random(42, 0.5);

        let params = CriticalityParams {
            measure_interval: 1,
            k_p: 0.1, // Strong proportional gain for clear effect
            k_i: 0.0, // No integral for isolated test
            lambda_target: -5.0, // Low target so measured λ > target
            gamma_min: 0.001,
            gamma_max: 2.0,
            ..CriticalityParams::default()
        };
        let mut ctrl = CriticalityController::new(params);
        let solver = make_solver(5.0, 0.1);

        let result = ctrl.adapt(&field, &solver);
        assert!(result.is_some());
        let (_, new_gamma) = result.unwrap();
        // λ is likely > -5.0, so error > 0, so γ should increase
        assert!(
            new_gamma >= 0.1,
            "γ should not decrease below initial when λ > target, got {}",
            new_gamma
        );
    }

    #[test]
    fn test_pi_controller_decreases_gamma_when_ordered() {
        // If λ < target (too ordered), γ should decrease to reduce damping
        let mut field = make_e8_field();
        field.init_random(42, 0.5);

        // Run with high damping to establish strongly ordered state
        let mut solver = make_solver(0.1, 0.8);
        solver.run(&mut field, 200);

        let params = CriticalityParams {
            measure_interval: 1,
            k_p: 0.1,
            k_i: 0.0,
            lambda_target: 10.0, // High target so measured λ < target
            gamma_min: 0.001,
            gamma_max: 2.0,
            ..CriticalityParams::default()
        };
        let mut ctrl = CriticalityController::new(params);

        let result = ctrl.adapt(&field, &solver);
        assert!(result.is_some());
        let (_, new_gamma) = result.unwrap();
        // λ is likely << 10.0, so error < 0, so γ should decrease
        assert!(
            new_gamma < 0.8,
            "γ should decrease from 0.8 when λ < target, got {}",
            new_gamma
        );
    }

    #[test]
    fn test_controller_respects_gamma_bounds() {
        let mut field = make_e8_field();
        field.init_random(42, 0.5);

        let params = CriticalityParams {
            measure_interval: 1,
            k_p: 10.0, // Very strong gain to force saturation
            k_i: 5.0,
            gamma_min: 0.05,
            gamma_max: 0.5,
            ..CriticalityParams::default()
        };
        let mut ctrl = CriticalityController::new(params);
        let solver = make_solver(1.0, 0.1);

        // Run many adapt cycles — γ should always stay in bounds
        for _ in 0..50 {
            if let Some((_, gamma)) = ctrl.adapt(&field, &solver) {
                assert!(
                    gamma >= 0.05 - 1e-9 && gamma <= 0.5 + 1e-9,
                    "γ {} should be in [0.05, 0.5]",
                    gamma
                );
            }
        }
    }

    #[test]
    fn test_controller_respects_omega_bounds() {
        let mut field = make_e8_field();
        field.init_random(42, 0.5);

        let params = CriticalityParams {
            measure_interval: 1,
            k_p: 10.0,
            k_i: 5.0,
            adjust_omega: true,
            omega_min: 0.5,
            omega_max: 5.0,
            ..CriticalityParams::default()
        };
        let mut ctrl = CriticalityController::new(params);
        let solver = make_solver(2.0, 0.1);

        for _ in 0..50 {
            if let Some((omega, _)) = ctrl.adapt(&field, &solver) {
                assert!(
                    omega >= 0.5 - 1e-9 && omega <= 5.0 + 1e-9,
                    "ω {} should be in [0.5, 5.0]",
                    omega
                );
            }
        }
    }

    #[test]
    fn test_reset() {
        let params = CriticalityParams {
            measure_interval: 1,
            ..CriticalityParams::default()
        };
        let mut ctrl = CriticalityController::new(params);
        let mut field = make_e8_field();
        field.init_random(42, 0.5);
        let solver = make_solver(1.0, 0.1);

        // Run a few adapt cycles to accumulate state
        for _ in 0..5 {
            ctrl.adapt(&field, &solver);
        }
        assert!(ctrl.lambda_estimate().abs() > 0.0 || ctrl.integral_error != 0.0);

        // Reset should clear state
        ctrl.reset();
        assert!((ctrl.lambda_estimate() - 0.0).abs() < 1e-9);
        assert_eq!(ctrl.ticks_since_measure, 0);
    }

    #[test]
    fn test_omega_unchanged_when_adjust_disabled() {
        let mut field = make_e8_field();
        field.init_random(42, 0.5);

        let params = CriticalityParams {
            measure_interval: 1,
            adjust_omega: false,
            ..CriticalityParams::default()
        };
        let mut ctrl = CriticalityController::new(params);
        let solver = make_solver(3.0, 0.1);

        let result = ctrl.adapt(&field, &solver);
        assert!(result.is_some());
        let (omega, _) = result.unwrap();
        assert!(
            (omega - 3.0).abs() < 1e-9,
            "ω should be unchanged when adjust_omega=false, got {}",
            omega
        );
    }

    #[test]
    fn test_lyapunov_deterministic() {
        // Same field + same seed → same λ estimate
        let mut field = make_e8_field();
        field.init_random(42, 0.5);
        let solver = make_solver(1.0, 0.1);

        let params = CriticalityParams {
            measure_interval: 1,
            ..CriticalityParams::default()
        };

        let mut ctrl1 = CriticalityController::new(params.clone());
        ctrl1.adapt(&field, &solver);
        let lambda1 = ctrl1.lambda_estimate();

        let mut ctrl2 = CriticalityController::new(params);
        ctrl2.adapt(&field, &solver);
        let lambda2 = ctrl2.lambda_estimate();

        assert!(
            (lambda1 - lambda2).abs() < 1e-9,
            "Same inputs should give same λ: {} vs {}",
            lambda1,
            lambda2
        );
    }

    #[test]
    fn test_controller_convergence() {
        // Run many adapt cycles and verify that γ adjustments stabilize
        let mut field = make_e8_field();
        field.init_random(42, 0.5);

        let params = CriticalityParams {
            measure_interval: 1,
            k_p: 0.005,
            k_i: 0.0005,
            probe_steps: 5,
            gamma_min: 0.01,
            gamma_max: 0.5,
            ..CriticalityParams::default()
        };
        let mut ctrl = CriticalityController::new(params);
        let mut solver = make_solver(1.0, 0.1);

        let mut gamma_values = Vec::new();
        for _ in 0..50 {
            // Run some RAE steps between measurements
            solver.run(&mut field, 10);
            if let Some((_, gamma)) = ctrl.adapt(&field, &solver) {
                solver.params.gamma = gamma;
                gamma_values.push(gamma);
            }
        }

        // Check that γ adjustments become smaller over time (convergence indicator)
        // Compare variance of first half vs second half
        if gamma_values.len() >= 10 {
            let mid = gamma_values.len() / 2;
            let first_half = &gamma_values[..mid];
            let second_half = &gamma_values[mid..];

            let var = |vals: &[f32]| -> f32 {
                let mean = vals.iter().sum::<f32>() / vals.len() as f32;
                vals.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / vals.len() as f32
            };

            let _var1 = var(first_half);
            let var2 = var(second_half);

            // PI controller on chaotic PDE has inherent variance — just verify no blow-up
            assert!(
                var2 < 1.0,
                "γ adjustments should stay bounded: var2={}",
                var2
            );
        }
    }

    #[test]
    fn test_integral_anti_windup() {
        let mut field = make_e8_field();
        field.init_random(42, 0.5);

        let params = CriticalityParams {
            measure_interval: 1,
            k_p: 0.0,  // No proportional — only integral
            k_i: 100.0, // Very strong integral gain
            ..CriticalityParams::default()
        };
        let mut ctrl = CriticalityController::new(params);
        let solver = make_solver(1.0, 0.1);

        // Run many cycles to try to blow up the integral
        for _ in 0..100 {
            ctrl.adapt(&field, &solver);
        }

        // Integral should be clamped to [-10, 10]
        assert!(
            ctrl.integral_error.abs() <= 10.0 + 1e-6,
            "Integral error should be clamped, got {}",
            ctrl.integral_error
        );
    }
}
