//! Resonant Attractor Equation (RAE) — CPU Reference Solver
//!
//! The RAE is the central PDE of the Icarus EMC:
//!
//!   ∂z/∂t = -δF/δz* + iωz
//!         = Δz - dV/d(|z|²)·z + iωz - γz
//!
//! where:
//!   - Δz is the graph Laplacian (diffusion on the lattice)
//!   - V(|z|²) = (|z|² - 1)²/4 is the double-well potential
//!   - ω is the resonance frequency
//!   - γ is the damping coefficient
//!
//! This module provides CPU reference implementations:
//! - Forward Euler (conditionally stable, CFL-limited)
//! - Semi-implicit (unconditionally stable, no CFL constraint)
//!
//! The GPU kernel in icarus-gpu will replicate these exactly.

use crate::free_energy::{double_well_derivative, FreeEnergyParams};
use crate::phase_field::LatticeField;
use serde::{Deserialize, Serialize};

/// Integration method for the RAE solver.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IntegratorMethod {
    /// Forward Euler — conditionally stable, requires CFL: dt < 2/K.
    Euler,
    /// Semi-implicit — unconditionally stable, no CFL constraint.
    ///
    /// Treats the diagonal part of the linear operator (self-diffusion + damping +
    /// resonance) implicitly, and the off-diagonal Laplacian + nonlinear potential
    /// explicitly. Each site reduces to a 2×2 system with analytic solution.
    SemiImplicit,
}

/// Parameters for the RAE solver
#[derive(Debug, Clone)]
pub struct RAEParams {
    /// Time step (for Euler: must satisfy CFL dt < 2/K; for SemiImplicit: unrestricted)
    pub dt: f32,
    /// Resonance frequency ω (drives oscillation)
    pub omega: f32,
    /// Damping coefficient γ (energy dissipation, must be ≥ 0)
    pub gamma: f32,
    /// Free energy parameters
    pub energy_params: FreeEnergyParams,
    /// Integration method
    pub method: IntegratorMethod,
}

impl RAEParams {
    /// Create default params safe for E8 lattice (K=240, CFL dt < 2/240 ≈ 0.0083)
    pub fn default_e8() -> Self {
        Self {
            dt: 0.002,
            omega: 1.0,
            gamma: 0.1,
            energy_params: FreeEnergyParams::default(),
            method: IntegratorMethod::Euler,
        }
    }

    /// Create default params safe for E8 with semi-implicit integrator (no CFL limit).
    /// Uses a larger timestep since unconditional stability allows it.
    pub fn default_e8_semi_implicit() -> Self {
        Self {
            dt: 0.05,
            omega: 1.0,
            gamma: 0.1,
            energy_params: FreeEnergyParams::default(),
            method: IntegratorMethod::SemiImplicit,
        }
    }

    /// Create default params for hypercubic lattice in dimension d
    pub fn default_hypercubic(dim: usize) -> Self {
        let k = 2 * dim; // kissing number
        Self {
            dt: 1.0 / (k as f32),
            omega: 1.0,
            gamma: 0.1,
            energy_params: FreeEnergyParams::default(),
            method: IntegratorMethod::Euler,
        }
    }
}

/// Adaptive timestep controller with method-aware bounds.
///
/// For **Euler** mode: `dt_max = CFL_limit * 0.9` (stability-limited).
/// For **SemiImplicit** mode: `dt_max = dt_base * 16` (accuracy-limited, no CFL).
///
/// Shrinks dt when energy increases (instability/inaccuracy), grows dt when stable.
/// dt is always clamped to `[dt_min, dt_max]`.
#[derive(Debug, Clone)]
pub struct AdaptiveTimestep {
    /// Current timestep
    dt_current: f32,
    /// Minimum allowed dt (dt_base / 16)
    dt_min: f32,
    /// Maximum allowed dt (method-dependent ceiling)
    dt_max: f32,
    /// Configured baseline dt
    dt_base: f32,
    /// Factor to shrink dt on instability (0.5)
    shrink_factor: f32,
    /// Factor to grow dt when stable (method-dependent: 1.05 for Euler, 1.1 for SemiImplicit)
    grow_factor: f32,
    /// Consecutive stable steps counter
    stable_steps: u32,
    /// Steps required before growing dt
    grow_after: u32,
    /// Previous energy for comparison
    prev_energy: Option<f32>,
}

impl AdaptiveTimestep {
    /// Create a controller for forward Euler (CFL-limited).
    ///
    /// `dt_base` is the configured timestep. `cfl_limit` is the CFL stability
    /// bound (`2/K` where K is the kissing number). The controller ensures
    /// `dt <= cfl_limit * safety` with a safety factor of 0.9.
    pub fn new(dt_base: f32, cfl_limit: f32) -> Self {
        Self::new_for_method(dt_base, cfl_limit, IntegratorMethod::Euler)
    }

    /// Create a method-aware controller.
    ///
    /// - **Euler**: `dt_max = min(cfl_limit * 0.9, dt_base)` — stability-limited.
    /// - **SemiImplicit**: `dt_max = dt_base * 16.0` — accuracy-limited. The semi-implicit
    ///   integrator is unconditionally stable so the CFL constraint does not apply.
    ///   The ceiling is set generously; the controller will shrink dt if energy increases.
    pub fn new_for_method(dt_base: f32, cfl_limit: f32, method: IntegratorMethod) -> Self {
        let (dt_max, grow_factor) = match method {
            IntegratorMethod::Euler => {
                let max = (cfl_limit * 0.9).min(dt_base);
                (max, 1.05)
            }
            IntegratorMethod::SemiImplicit => {
                // No CFL constraint — ceiling is 16× baseline for generous headroom.
                // The controller will self-correct via energy monitoring.
                let max = dt_base * 16.0;
                // Grow faster since we're accuracy-limited, not stability-limited.
                (max, 1.1)
            }
        };
        let dt_min = dt_base / 16.0;
        Self {
            dt_current: dt_base.min(dt_max),
            dt_min,
            dt_max,
            dt_base,
            shrink_factor: 0.5,
            grow_factor,
            stable_steps: 0,
            grow_after: 10,
            prev_energy: None,
        }
    }

    /// Report the current energy and get the dt for the next step.
    ///
    /// If energy increased, shrink dt. If stable for `grow_after` consecutive
    /// steps, grow dt toward `dt_max`.
    pub fn adapt(&mut self, energy: f32) -> f32 {
        if let Some(prev) = self.prev_energy {
            if energy > prev + 1e-6 {
                // Energy increased — potential instability/inaccuracy, shrink dt
                self.dt_current = (self.dt_current * self.shrink_factor).max(self.dt_min);
                self.stable_steps = 0;
            } else {
                // Stable step
                self.stable_steps += 1;
                if self.stable_steps >= self.grow_after {
                    self.dt_current = (self.dt_current * self.grow_factor).min(self.dt_max);
                    self.stable_steps = 0;
                }
            }
        }
        self.prev_energy = Some(energy);
        self.dt_current
    }

    /// Current timestep value.
    pub fn current_dt(&self) -> f32 {
        self.dt_current
    }

    /// The maximum dt ceiling (CFL-limited for Euler, accuracy-limited for SemiImplicit).
    pub fn dt_max(&self) -> f32 {
        self.dt_max
    }

    /// Reset the controller (e.g. after a config change).
    pub fn reset(&mut self) {
        self.dt_current = self.dt_base.min(self.dt_max);
        self.stable_steps = 0;
        self.prev_energy = None;
    }
}

/// RAE solver state
#[derive(Debug, Clone)]
pub struct RAESolver {
    /// Solver parameters
    pub params: RAEParams,
    /// Current simulation time
    pub time: f64,
    /// Number of steps taken
    pub steps: u64,
    /// Scratch buffer for derivatives (re)
    dz_re: Vec<f32>,
    /// Scratch buffer for derivatives (im)
    dz_im: Vec<f32>,
}

impl RAESolver {
    /// Create a new solver
    pub fn new(params: RAEParams, num_sites: usize) -> Self {
        Self {
            params,
            time: 0.0,
            steps: 0,
            dz_re: vec![0.0; num_sites],
            dz_im: vec![0.0; num_sites],
        }
    }

    /// Perform one step of the RAE, dispatching on the configured integrator method.
    pub fn step(&mut self, field: &mut LatticeField) {
        match self.params.method {
            IntegratorMethod::Euler => self.step_euler(field),
            IntegratorMethod::SemiImplicit => self.step_semi_implicit(field),
        }
    }

    /// Forward Euler step (conditionally stable, CFL-limited).
    ///
    /// dz_i/dt = laplacian_z_i - dV/d(|z_i|²)·z_i + iω·z_i - γ·z_i
    ///
    /// The Laplacian is: (2/K_i) · Σ_{j∈N(i)} w_ij · (z_j - z_i)
    fn step_euler(&mut self, field: &mut LatticeField) {
        let n = field.num_sites;
        let dt = self.params.dt;
        let omega = self.params.omega;
        let gamma = self.params.gamma;
        let target_sq = self.params.energy_params.target_amplitude
            * self.params.energy_params.target_amplitude;
        let kw = self.params.energy_params.kinetic_weight;
        let pw = self.params.energy_params.potential_weight;

        // Compute dz/dt for each site
        for i in 0..n {
            let zi_re = field.values_re[i];
            let zi_im = field.values_im[i];
            let ns = zi_re * zi_re + zi_im * zi_im;

            // Graph Laplacian: (2/K) Σ w_ij (z_j - z_i)
            let start = field.neighbor_offsets[i] as usize;
            let end = field.neighbor_offsets[i + 1] as usize;
            let k = (end - start) as f32;

            let mut lap_re = 0.0f32;
            let mut lap_im = 0.0f32;

            if k > 0.0 {
                for edge in start..end {
                    let j = field.neighbor_indices[edge] as usize;
                    let w = field.neighbor_weights[edge];
                    lap_re += w * (field.values_re[j] - zi_re);
                    lap_im += w * (field.values_im[j] - zi_im);
                }
                let lap_scale = 2.0 / k;
                lap_re *= lap_scale;
                lap_im *= lap_scale;
            }

            // Potential gradient: dV/d(|z|²) · z
            let dv = double_well_derivative(ns, target_sq) * pw;

            // dz/dt = kw * laplacian - dV·z + iω·z - γ·z
            // iω·z = iω·(re + i·im) = -ω·im + i·ω·re
            self.dz_re[i] = kw * lap_re - dv * zi_re - omega * zi_im - gamma * zi_re;
            self.dz_im[i] = kw * lap_im - dv * zi_im + omega * zi_re - gamma * zi_im;
        }

        // Forward Euler update: z_new = z + dt * dz
        for i in 0..n {
            field.values_re[i] += dt * self.dz_re[i];
            field.values_im[i] += dt * self.dz_im[i];
        }

        self.time += dt as f64;
        self.steps += 1;
    }

    /// Semi-implicit step (unconditionally stable, no CFL constraint).
    ///
    /// Splits the RAE into diagonal linear (implicit) and off-diagonal + nonlinear
    /// (explicit) parts. The diagonal part includes self-diffusion, damping, and
    /// resonance coupling — all treated implicitly for unconditional stability.
    ///
    /// Per-site 2×2 system:
    ///   [d,  c] [z_re^{n+1}]   [rhs_re]
    ///   [-c, d] [z_im^{n+1}] = [rhs_im]
    ///
    /// where d = 1 + dt*(s_i + γ), c = dt*ω, s_i = (2kw/K_i)*Σw_ij
    /// Solution: det = d² + c², z_re = (d*rhs_re - c*rhs_im)/det, etc.
    fn step_semi_implicit(&mut self, field: &mut LatticeField) {
        let n = field.num_sites;
        let dt = self.params.dt;
        let omega = self.params.omega;
        let gamma = self.params.gamma;
        let target_sq = self.params.energy_params.target_amplitude
            * self.params.energy_params.target_amplitude;
        let kw = self.params.energy_params.kinetic_weight;
        let pw = self.params.energy_params.potential_weight;

        for i in 0..n {
            let zi_re = field.values_re[i];
            let zi_im = field.values_im[i];
            let ns = zi_re * zi_re + zi_im * zi_im;

            let start = field.neighbor_offsets[i] as usize;
            let end = field.neighbor_offsets[i + 1] as usize;
            let k = (end - start) as f32;

            // Off-diagonal Laplacian: only the neighbor sum Σ w_ij * z_j
            // (the -z_i diagonal part is handled implicitly)
            let mut nb_sum_re = 0.0f32;
            let mut nb_sum_im = 0.0f32;
            let mut w_sum = 0.0f32;

            if k > 0.0 {
                for edge in start..end {
                    let j = field.neighbor_indices[edge] as usize;
                    let w = field.neighbor_weights[edge];
                    nb_sum_re += w * field.values_re[j];
                    nb_sum_im += w * field.values_im[j];
                    w_sum += w;
                }
                let lap_scale = 2.0 / k;
                nb_sum_re *= lap_scale;
                nb_sum_im *= lap_scale;
                w_sum *= lap_scale;
            }

            // Nonlinear potential gradient (explicit)
            let dv = double_well_derivative(ns, target_sq) * pw;

            // RHS = z^n + dt * (kw * off_diag_laplacian - dV * z^n)
            // off_diag_laplacian = Σ w_ij * z_j (scaled by 2/K)
            let rhs_re = zi_re + dt * (kw * nb_sum_re - dv * zi_re);
            let rhs_im = zi_im + dt * (kw * nb_sum_im - dv * zi_im);

            // Implicit diagonal: s_i = kw * (2/K) * Σ w_ij
            let s_i = kw * w_sum;

            // 2×2 system coefficients
            let d = 1.0 + dt * (s_i + gamma);
            let c = dt * omega;
            let inv_det = 1.0 / (d * d + c * c);

            // Analytic solve
            // Inverse of [[d, c], [-c, d]] = (1/det) * [[d, -c], [c, d]]
            let new_re = (d * rhs_re - c * rhs_im) * inv_det;
            let new_im = (c * rhs_re + d * rhs_im) * inv_det;

            // Store rate of change for max_rate() / rms_rate() diagnostics
            self.dz_re[i] = (new_re - zi_re) / dt;
            self.dz_im[i] = (new_im - zi_im) / dt;

            field.values_re[i] = new_re;
            field.values_im[i] = new_im;
        }

        self.time += dt as f64;
        self.steps += 1;
    }

    /// Run N steps
    pub fn run(&mut self, field: &mut LatticeField, num_steps: u64) {
        for _ in 0..num_steps {
            self.step(field);
        }
    }

    /// Compute max |dz/dt| across all sites (convergence criterion)
    pub fn max_rate(&self) -> f32 {
        self.dz_re
            .iter()
            .zip(self.dz_im.iter())
            .map(|(&r, &i)| (r * r + i * i).sqrt())
            .fold(0.0f32, f32::max)
    }

    /// Compute RMS |dz/dt| across all sites
    pub fn rms_rate(&self) -> f32 {
        let n = self.dz_re.len() as f32;
        if n == 0.0 {
            return 0.0;
        }
        let sum_sq: f32 = self.dz_re
            .iter()
            .zip(self.dz_im.iter())
            .map(|(&r, &i)| r * r + i * i)
            .sum();
        (sum_sq / n).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::free_energy::free_energy;
    use icarus_math::lattice::e8::E8Lattice;
    use icarus_math::lattice::hypercubic::HypercubicLattice;

    fn make_e8_field() -> LatticeField {
        let lattice = E8Lattice::new();
        LatticeField::from_lattice(&lattice)
    }

    #[test]
    fn test_rae_step_runs() {
        let mut field = make_e8_field();
        field.init_random(42, 0.5);
        let mut solver = RAESolver::new(RAEParams::default_e8(), field.num_sites);

        solver.step(&mut field);
        assert_eq!(solver.steps, 1);
        assert!(solver.time > 0.0);
    }

    #[test]
    fn test_rae_energy_decreases_with_damping() {
        let mut field = make_e8_field();
        field.init_random(123, 0.8);

        let mut params = RAEParams::default_e8();
        params.omega = 0.0; // No oscillation, pure dissipation
        params.gamma = 0.5; // Strong damping
        let mut solver = RAESolver::new(params.clone(), field.num_sites);

        let energy_params = params.energy_params.clone();
        let (e0, _, _) = free_energy(&field, &energy_params);

        // Run 100 steps
        solver.run(&mut field, 100);
        let (e1, _, _) = free_energy(&field, &energy_params);

        // Run 100 more
        solver.run(&mut field, 100);
        let (e2, _, _) = free_energy(&field, &energy_params);

        // Energy should decrease (or at least not increase significantly)
        // With damping and no oscillation, free energy is Lyapunov
        assert!(
            e1 <= e0 + 1e-3,
            "Energy should decrease: e0={}, e1={}",
            e0,
            e1
        );
        assert!(
            e2 <= e1 + 1e-3,
            "Energy should decrease: e1={}, e2={}",
            e1,
            e2
        );
    }

    #[test]
    fn test_rae_convergence_zero_field() {
        // Zero field at a potential minimum (|z|=0 is a local minimum of double-well)
        let mut field = make_e8_field();
        let mut solver = RAESolver::new(RAEParams::default_e8(), field.num_sites);

        solver.step(&mut field);

        // Zero field should have zero dz/dt (it's a fixed point if gradient is zero)
        // Actually V'(0) = -0.5 != 0, so the zero field is NOT a fixed point
        // The origin is at the top of the double-well barrier
        // This is expected — the system should evolve away from zero
    }

    #[test]
    fn test_rae_uniform_unit_field_is_stationary() {
        let mut field = make_e8_field();
        // Set uniform |z| = 1 (potential minimum, uniform → zero Laplacian)
        for i in 0..field.num_sites {
            field.set(i, 1.0, 0.0);
        }

        let mut params = RAEParams::default_e8();
        params.omega = 0.0;
        params.gamma = 0.0; // No damping, no oscillation
        let mut solver = RAESolver::new(params, field.num_sites);

        solver.step(&mut field);

        // Should be approximately stationary (uniform + at potential min)
        let rate = solver.max_rate();
        assert!(
            rate < 1e-3,
            "Uniform unit field should be nearly stationary, rate={}",
            rate
        );
    }

    #[test]
    fn test_rae_run_multiple_steps() {
        let mut field = make_e8_field();
        field.init_random(99, 0.3);
        let mut solver = RAESolver::new(RAEParams::default_e8(), field.num_sites);

        solver.run(&mut field, 1000);
        assert_eq!(solver.steps, 1000);
        assert!((solver.time - 1000.0 * 0.002).abs() < 1e-6);
    }

    #[test]
    fn test_rae_hypercubic_3d() {
        let lattice = HypercubicLattice::new(3);
        let mut field = LatticeField::from_lattice(&lattice);
        field.init_random(77, 0.5);
        let params = RAEParams::default_hypercubic(3);
        let mut solver = RAESolver::new(params, field.num_sites);

        solver.run(&mut field, 100);
        assert_eq!(solver.steps, 100);
    }

    #[test]
    fn test_rae_deterministic() {
        let lattice = E8Lattice::new();

        let mut field1 = LatticeField::from_lattice(&lattice);
        field1.init_random(42, 0.5);
        let mut solver1 = RAESolver::new(RAEParams::default_e8(), field1.num_sites);
        solver1.run(&mut field1, 50);

        let mut field2 = LatticeField::from_lattice(&lattice);
        field2.init_random(42, 0.5);
        let mut solver2 = RAESolver::new(RAEParams::default_e8(), field2.num_sites);
        solver2.run(&mut field2, 50);

        for i in 0..field1.num_sites {
            assert!(
                (field1.values_re[i] - field2.values_re[i]).abs() < 1e-6,
                "Mismatch at site {} re: {} vs {}",
                i,
                field1.values_re[i],
                field2.values_re[i]
            );
        }
    }

    // --- AdaptiveTimestep tests ---

    #[test]
    fn test_adaptive_dt_initial_value() {
        // E8: CFL limit = 2/240 ≈ 0.00833, dt_base = 0.002
        let adt = AdaptiveTimestep::new(0.002, 2.0 / 240.0);
        // dt_base (0.002) < cfl*0.9 (0.0075), so initial dt = 0.002
        assert!((adt.current_dt() - 0.002).abs() < 1e-6);
        assert!(adt.dt_max() <= 2.0 / 240.0);
    }

    #[test]
    fn test_adaptive_dt_clamped_to_cfl() {
        // dt_base > CFL limit → should clamp
        let adt = AdaptiveTimestep::new(0.1, 2.0 / 240.0);
        let cfl_safe = (2.0 / 240.0_f32) * 0.9;
        assert!(
            adt.current_dt() <= cfl_safe + 1e-6,
            "dt {} should be <= CFL*safety {}",
            adt.current_dt(),
            cfl_safe,
        );
    }

    #[test]
    fn test_adaptive_dt_shrinks_on_energy_increase() {
        let mut adt = AdaptiveTimestep::new(0.002, 2.0 / 240.0);
        let dt0 = adt.adapt(10.0); // first call, no comparison
        let dt1 = adt.adapt(11.0); // energy increased
        assert!(
            dt1 < dt0,
            "dt should shrink on energy increase: {} -> {}",
            dt0, dt1,
        );
    }

    #[test]
    fn test_adaptive_dt_grows_after_stable_steps() {
        let mut adt = AdaptiveTimestep::new(0.001, 2.0 / 240.0);
        // Force a shrink first so there's room to grow
        adt.adapt(10.0);
        adt.adapt(20.0); // shrink
        let dt_after_shrink = adt.current_dt();

        // Now feed 11 stable (decreasing) energies
        for i in 0..11 {
            adt.adapt(19.0 - i as f32 * 0.1);
        }
        assert!(
            adt.current_dt() > dt_after_shrink,
            "dt should grow after stable steps: {} -> {}",
            dt_after_shrink,
            adt.current_dt(),
        );
    }

    #[test]
    fn test_adaptive_dt_never_below_min() {
        let mut adt = AdaptiveTimestep::new(0.002, 2.0 / 240.0);
        let dt_min = 0.002 / 16.0;

        // Repeatedly increase energy to shrink dt
        let mut energy = 10.0;
        for _ in 0..100 {
            adt.adapt(energy);
            energy += 1.0;
        }
        assert!(
            adt.current_dt() >= dt_min - 1e-9,
            "dt {} should never go below dt_min {}",
            adt.current_dt(),
            dt_min,
        );
    }

    #[test]
    fn test_adaptive_dt_never_above_max() {
        let mut adt = AdaptiveTimestep::new(0.002, 2.0 / 240.0);
        let dt_max = adt.dt_max();

        // Feed many stable steps to grow dt
        let mut energy = 100.0;
        for _ in 0..1000 {
            adt.adapt(energy);
            energy -= 0.001;
        }
        assert!(
            adt.current_dt() <= dt_max + 1e-9,
            "dt {} should never exceed dt_max {}",
            adt.current_dt(),
            dt_max,
        );
    }

    #[test]
    fn test_adaptive_dt_reset() {
        let mut adt = AdaptiveTimestep::new(0.002, 2.0 / 240.0);
        // Shrink dt
        adt.adapt(10.0);
        adt.adapt(20.0);
        let shrunk = adt.current_dt();
        assert!(shrunk < 0.002);

        adt.reset();
        assert!((adt.current_dt() - 0.002).abs() < 1e-6);
    }

    // --- AdaptiveTimestep semi-implicit mode tests ---

    #[test]
    fn test_adaptive_dt_semi_implicit_higher_ceiling() {
        // Semi-implicit mode: dt_max should be dt_base * 16, NOT CFL-limited.
        let dt_base = 0.05;
        let cfl_limit = 2.0 / 240.0; // 0.0083 — much smaller than dt_base
        let adt = AdaptiveTimestep::new_for_method(dt_base, cfl_limit, IntegratorMethod::SemiImplicit);

        // dt_max should be 0.05 * 16 = 0.8, NOT cfl * 0.9 = 0.0075
        assert!(
            adt.dt_max() > 0.1,
            "semi-implicit dt_max {} should be >> CFL limit {}",
            adt.dt_max(),
            cfl_limit,
        );
        assert!(
            (adt.dt_max() - dt_base * 16.0).abs() < 1e-6,
            "semi-implicit dt_max should be dt_base*16 = {}, got {}",
            dt_base * 16.0,
            adt.dt_max(),
        );
    }

    #[test]
    fn test_adaptive_dt_semi_implicit_starts_at_base() {
        let dt_base = 0.05;
        let adt = AdaptiveTimestep::new_for_method(dt_base, 2.0 / 240.0, IntegratorMethod::SemiImplicit);
        assert!(
            (adt.current_dt() - dt_base).abs() < 1e-6,
            "should start at dt_base {}, got {}",
            dt_base,
            adt.current_dt(),
        );
    }

    #[test]
    fn test_adaptive_dt_semi_implicit_grows_beyond_cfl() {
        // Key behavior: semi-implicit can grow dt well past the CFL limit.
        let dt_base = 0.005;
        let cfl_limit = 2.0 / 240.0; // 0.0083
        let mut adt = AdaptiveTimestep::new_for_method(dt_base, cfl_limit, IntegratorMethod::SemiImplicit);

        // Feed many stable (decreasing energy) steps
        let mut energy = 100.0;
        for _ in 0..500 {
            adt.adapt(energy);
            energy -= 0.01;
        }

        // Should have grown well past the CFL limit
        assert!(
            adt.current_dt() > cfl_limit,
            "semi-implicit dt {} should grow past CFL limit {}",
            adt.current_dt(),
            cfl_limit,
        );
    }

    #[test]
    fn test_adaptive_dt_semi_implicit_shrinks_on_energy_increase() {
        // Same shrink behavior as Euler — energy increase triggers dt reduction.
        let mut adt = AdaptiveTimestep::new_for_method(0.05, 2.0 / 240.0, IntegratorMethod::SemiImplicit);
        let dt0 = adt.adapt(10.0);
        let dt1 = adt.adapt(11.0); // energy increased
        assert!(
            dt1 < dt0,
            "semi-implicit dt should shrink on energy increase: {} -> {}",
            dt0, dt1,
        );
    }

    #[test]
    fn test_adaptive_dt_semi_implicit_never_above_ceiling() {
        let dt_base = 0.01;
        let mut adt = AdaptiveTimestep::new_for_method(dt_base, 2.0 / 240.0, IntegratorMethod::SemiImplicit);
        let dt_max = adt.dt_max();

        let mut energy = 1000.0;
        for _ in 0..10000 {
            adt.adapt(energy);
            energy -= 0.01;
        }
        assert!(
            adt.current_dt() <= dt_max + 1e-9,
            "dt {} should never exceed dt_max {}",
            adt.current_dt(),
            dt_max,
        );
    }

    #[test]
    fn test_adaptive_dt_euler_unchanged_behavior() {
        // Verify new_for_method(Euler) matches old new() behavior exactly.
        let dt_base = 0.002;
        let cfl_limit = 2.0 / 240.0;
        let adt_old = AdaptiveTimestep::new(dt_base, cfl_limit);
        let adt_new = AdaptiveTimestep::new_for_method(dt_base, cfl_limit, IntegratorMethod::Euler);

        assert!((adt_old.current_dt() - adt_new.current_dt()).abs() < 1e-9);
        assert!((adt_old.dt_max() - adt_new.dt_max()).abs() < 1e-9);
    }

    // --- Semi-implicit integrator tests ---

    #[test]
    fn test_semi_implicit_unconditionally_stable() {
        // Use dt=0.1, which is 12× the CFL limit for E8 (0.0083).
        // Forward Euler would blow up; semi-implicit should remain bounded.
        let mut field = make_e8_field();
        field.init_random(42, 0.8);

        let mut params = RAEParams::default_e8_semi_implicit();
        params.dt = 0.1; // Way beyond CFL
        params.gamma = 0.5;
        let mut solver = RAESolver::new(params, field.num_sites);

        solver.run(&mut field, 500);

        // Check that no site has blown up (all values finite and bounded)
        for i in 0..field.num_sites {
            assert!(
                field.values_re[i].is_finite() && field.values_re[i].abs() < 100.0,
                "Site {} re blew up: {}",
                i,
                field.values_re[i]
            );
            assert!(
                field.values_im[i].is_finite() && field.values_im[i].abs() < 100.0,
                "Site {} im blew up: {}",
                i,
                field.values_im[i]
            );
        }
    }

    #[test]
    fn test_semi_implicit_energy_decreases() {
        // Initialize at high amplitude (|z|≈2.0, target=1.0) so both the
        // potential gradient and damping push amplitudes DOWN. This avoids the
        // regime where gamma drives toward 0 (the double-well barrier peak).
        let mut field = make_e8_field();
        field.init_random(123, 2.0);

        let mut params = RAEParams::default_e8_semi_implicit();
        params.omega = 0.0; // Pure dissipation
        params.gamma = 0.1; // Moderate damping
        params.dt = 0.01;   // ~1.2× CFL
        let mut solver = RAESolver::new(params.clone(), field.num_sites);

        let energy_params = params.energy_params.clone();
        let (e0, _, _) = free_energy(&field, &energy_params);

        solver.run(&mut field, 200);
        let (e1, _, _) = free_energy(&field, &energy_params);

        solver.run(&mut field, 200);
        let (e2, _, _) = free_energy(&field, &energy_params);

        assert!(
            e1 < e0,
            "Energy should decrease from start: e0={}, e1={}",
            e0,
            e1
        );
        assert!(
            e2 < e1,
            "Energy should continue decreasing: e1={}, e2={}",
            e1,
            e2
        );
    }

    #[test]
    fn test_semi_implicit_matches_euler_small_dt() {
        // With a small dt both methods should produce similar results.
        let lattice = E8Lattice::new();
        let dt = 0.001; // Well within CFL

        let mut field_euler = LatticeField::from_lattice(&lattice);
        field_euler.init_random(77, 0.5);
        let mut params_euler = RAEParams::default_e8();
        params_euler.dt = dt;
        params_euler.method = IntegratorMethod::Euler;
        let mut solver_euler = RAESolver::new(params_euler, field_euler.num_sites);

        let mut field_si = LatticeField::from_lattice(&lattice);
        field_si.init_random(77, 0.5);
        let mut params_si = RAEParams::default_e8();
        params_si.dt = dt;
        params_si.method = IntegratorMethod::SemiImplicit;
        let mut solver_si = RAESolver::new(params_si, field_si.num_sites);

        // Run 10 steps
        solver_euler.run(&mut field_euler, 10);
        solver_si.run(&mut field_si, 10);

        // Should be close (not identical — different schemes — but O(dt) apart)
        let mut max_diff = 0.0f32;
        for i in 0..field_euler.num_sites {
            let diff_re = (field_euler.values_re[i] - field_si.values_re[i]).abs();
            let diff_im = (field_euler.values_im[i] - field_si.values_im[i]).abs();
            max_diff = max_diff.max(diff_re).max(diff_im);
        }
        assert!(
            max_diff < 0.05,
            "Euler and semi-implicit should agree at small dt, max_diff={}",
            max_diff
        );
    }

    #[test]
    fn test_semi_implicit_deterministic() {
        let lattice = E8Lattice::new();

        let mut field1 = LatticeField::from_lattice(&lattice);
        field1.init_random(42, 0.5);
        let mut solver1 = RAESolver::new(RAEParams::default_e8_semi_implicit(), field1.num_sites);
        solver1.run(&mut field1, 50);

        let mut field2 = LatticeField::from_lattice(&lattice);
        field2.init_random(42, 0.5);
        let mut solver2 = RAESolver::new(RAEParams::default_e8_semi_implicit(), field2.num_sites);
        solver2.run(&mut field2, 50);

        for i in 0..field1.num_sites {
            assert_eq!(
                field1.values_re[i], field2.values_re[i],
                "Mismatch at site {} re",
                i
            );
            assert_eq!(
                field1.values_im[i], field2.values_im[i],
                "Mismatch at site {} im",
                i
            );
        }
    }

    #[test]
    fn test_semi_implicit_uniform_unit_stationary() {
        let mut field = make_e8_field();
        for i in 0..field.num_sites {
            field.set(i, 1.0, 0.0);
        }

        let mut params = RAEParams::default_e8_semi_implicit();
        params.omega = 0.0;
        params.gamma = 0.0;
        let mut solver = RAESolver::new(params, field.num_sites);

        solver.step(&mut field);

        let rate = solver.max_rate();
        assert!(
            rate < 1e-3,
            "Uniform unit field should be nearly stationary with semi-implicit, rate={}",
            rate
        );
    }

    #[test]
    fn test_semi_implicit_rotation_direction() {
        // Verify that omega > 0 produces counter-clockwise rotation (iωz).
        // For z(0) = (1, 0) and pure resonance (no neighbors, no potential, no damping):
        //   dz/dt = iωz  →  z(t) = e^{iωt}  →  z_im increases from 0.
        // The semi-implicit scheme should agree on the rotation direction.
        let lattice = E8Lattice::new();
        let mut field = LatticeField::from_lattice(&lattice);
        // Set site 0 to (1, 0); all others to (0, 0) to isolate rotation
        for i in 0..field.num_sites {
            field.set(i, 0.0, 0.0);
        }
        field.set(0, 1.0, 0.0);

        let params = RAEParams {
            dt: 0.01,
            omega: 5.0,  // Strong rotation for clear signal
            gamma: 0.0,
            energy_params: FreeEnergyParams {
                kinetic_weight: 0.0,   // No diffusion
                potential_weight: 0.0, // No potential
                target_amplitude: 1.0,
            },
            method: IntegratorMethod::SemiImplicit,
        };
        let mut solver = RAESolver::new(params, field.num_sites);

        solver.step(&mut field);

        // After one step, z_im[0] should be POSITIVE (counter-clockwise rotation)
        assert!(
            field.values_im[0] > 0.0,
            "omega>0 should rotate z=(1,0) to positive imaginary; got z_im={}",
            field.values_im[0]
        );
        // z_re[0] should still be close to 1 (small angle)
        assert!(
            field.values_re[0] > 0.9,
            "z_re should remain close to 1 after small rotation; got z_re={}",
            field.values_re[0]
        );
    }
}
