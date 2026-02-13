// Copyright (c) 2025-2026 brdigetrlol. All rights reserved.
// SPDX-License-Identifier: LicenseRef-Icarus-Proprietary
// See LICENSE in the repository root for full license terms.

//! Free Energy Functional for the Icarus EMC
//!
//! The free energy F[z, g] is the Lyapunov functional for the RAE dynamics.
//! When damping γ > 0, dF/dt ≤ 0 (monotone decrease).
//!
//! F = F_kinetic + F_potential
//!   = ½ Σ_i Σ_{j∈N(i)} w_ij |z_j - z_i|² + Σ_i V(|z_i|)
//!
//! The double-well potential: V(r) = (r² - 1)² / 4
//! - Minima at |z| = 0 and |z| = 1
//! - Barrier at |z| = 1/√2

use crate::phase_field::LatticeField;

/// Parameters for the free energy functional
#[derive(Debug, Clone)]
pub struct FreeEnergyParams {
    /// Coefficient for kinetic (gradient) term
    pub kinetic_weight: f32,
    /// Coefficient for potential term
    pub potential_weight: f32,
    /// Target amplitude for the double-well (default: 1.0)
    pub target_amplitude: f32,
}

impl Default for FreeEnergyParams {
    fn default() -> Self {
        Self {
            kinetic_weight: 0.5,
            potential_weight: 1.0,
            target_amplitude: 1.0,
        }
    }
}

/// Double-well potential: V(r²) = (r² - a²)² / 4
/// where r² = |z|² and a = target_amplitude
#[inline]
pub fn double_well_potential(norm_sq: f32, target_sq: f32) -> f32 {
    let x = norm_sq - target_sq;
    x * x * 0.25
}

/// Derivative of double-well potential with respect to |z|²:
/// dV/d(|z|²) = (|z|² - a²) / 2
#[inline]
pub fn double_well_derivative(norm_sq: f32, target_sq: f32) -> f32 {
    (norm_sq - target_sq) * 0.5
}

/// Compute the free energy of a lattice field.
///
/// Returns (F_total, F_kinetic, F_potential) for monitoring.
pub fn free_energy(field: &LatticeField, params: &FreeEnergyParams) -> (f32, f32, f32) {
    let target_sq = params.target_amplitude * params.target_amplitude;

    // Kinetic energy: ½ Σ_i Σ_{j∈N(i)} w_ij |z_j - z_i|²
    // Note: each edge (i,j) is counted twice (once from i, once from j)
    // so we divide by 2 at the end
    let mut kinetic = 0.0f32;
    for i in 0..field.num_sites {
        let zi_re = field.values_re[i];
        let zi_im = field.values_im[i];
        let start = field.neighbor_offsets[i] as usize;
        let end = field.neighbor_offsets[i + 1] as usize;

        for edge in start..end {
            let j = field.neighbor_indices[edge] as usize;
            let w = field.neighbor_weights[edge];
            let dre = field.values_re[j] - zi_re;
            let dim = field.values_im[j] - zi_im;
            kinetic += w * (dre * dre + dim * dim);
        }
    }
    // Each directed edge counted once → divide by 2 for undirected
    kinetic *= params.kinetic_weight * 0.5;

    // Potential energy: Σ_i V(|z_i|²)
    let mut potential = 0.0f32;
    for i in 0..field.num_sites {
        let ns = field.norm_sq(i);
        potential += double_well_potential(ns, target_sq);
    }
    potential *= params.potential_weight;

    (kinetic + potential, kinetic, potential)
}

/// Compute the gradient of the free energy with respect to z (conjugate).
///
/// δF/δz_i* = -laplacian_z_i + dV/d(|z_i|²) · z_i
///
/// Returns (grad_re, grad_im) arrays of length num_sites.
pub fn free_energy_gradient(field: &LatticeField, params: &FreeEnergyParams) -> (Vec<f32>, Vec<f32>) {
    let n = field.num_sites;
    let target_sq = params.target_amplitude * params.target_amplitude;

    let mut grad_re = vec![0.0f32; n];
    let mut grad_im = vec![0.0f32; n];

    for i in 0..n {
        let zi_re = field.values_re[i];
        let zi_im = field.values_im[i];
        let ns = zi_re * zi_re + zi_im * zi_im;

        // Potential gradient: dV/d(|z|²) · z
        let dv = double_well_derivative(ns, target_sq) * params.potential_weight;
        grad_re[i] += dv * zi_re;
        grad_im[i] += dv * zi_im;

        // Kinetic gradient (negative Laplacian): -Σ_j w_ij (z_j - z_i) / K
        let start = field.neighbor_offsets[i] as usize;
        let end = field.neighbor_offsets[i + 1] as usize;
        let k = (end - start) as f32; // degree (kissing number for this site)

        if k > 0.0 {
            let mut lap_re = 0.0f32;
            let mut lap_im = 0.0f32;
            for edge in start..end {
                let j = field.neighbor_indices[edge] as usize;
                let w = field.neighbor_weights[edge];
                lap_re += w * (field.values_re[j] - zi_re);
                lap_im += w * (field.values_im[j] - zi_im);
            }
            // Laplacian: (2/K) * Σ w_ij (z_j - z_i), gradient is negative Laplacian
            let scale = 2.0 * params.kinetic_weight / k;
            grad_re[i] -= scale * lap_re;
            grad_im[i] -= scale * lap_im;
        }
    }

    (grad_re, grad_im)
}

/// Compute the L2 norm of the gradient: ||δF/δz*||²
pub fn gradient_norm_sq(grad_re: &[f32], grad_im: &[f32]) -> f32 {
    grad_re
        .iter()
        .zip(grad_im.iter())
        .map(|(&r, &i)| r * r + i * i)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use icarus_math::lattice::e8::E8Lattice;

    fn make_test_field() -> LatticeField {
        let lattice = E8Lattice::new();
        LatticeField::from_lattice(&lattice)
    }

    #[test]
    fn test_double_well_potential() {
        // At |z|² = 1 (target), V = 0
        assert!(double_well_potential(1.0, 1.0).abs() < 1e-10);
        // At |z|² = 0, V = 0.25
        assert!((double_well_potential(0.0, 1.0) - 0.25).abs() < 1e-6);
        // At |z|² = 2, V = 0.25
        assert!((double_well_potential(2.0, 1.0) - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_double_well_derivative() {
        // At |z|² = 1 (target), dV/d|z|² = 0
        assert!(double_well_derivative(1.0, 1.0).abs() < 1e-10);
        // At |z|² = 0, dV/d|z|² = -0.5
        assert!((double_well_derivative(0.0, 1.0) - (-0.5)).abs() < 1e-6);
    }

    #[test]
    fn test_zero_field_energy() {
        let field = make_test_field();
        let params = FreeEnergyParams::default();
        let (total, kinetic, potential) = free_energy(&field, &params);

        // All z = 0: kinetic = 0, potential = 241 * (0 - 1)² / 4 = 241 * 0.25
        assert!(kinetic.abs() < 1e-6);
        assert!((potential - 241.0 * 0.25).abs() < 1e-3);
        assert!((total - potential).abs() < 1e-6);
    }

    #[test]
    fn test_uniform_field_energy() {
        let mut field = make_test_field();
        // Set all sites to |z| = 1 (at the potential minimum)
        for i in 0..field.num_sites {
            field.set(i, 1.0, 0.0);
        }
        let params = FreeEnergyParams::default();
        let (_total, kinetic, potential) = free_energy(&field, &params);

        // Uniform field → kinetic = 0 (all differences zero)
        assert!(kinetic.abs() < 1e-6);
        // |z|² = 1 → potential = 0
        assert!(potential.abs() < 1e-6);
    }

    #[test]
    fn test_gradient_at_minimum() {
        let mut field = make_test_field();
        // Uniform |z| = 1 field is a minimum
        for i in 0..field.num_sites {
            field.set(i, 1.0, 0.0);
        }
        let params = FreeEnergyParams::default();
        let (grad_re, grad_im) = free_energy_gradient(&field, &params);
        let gnorm = gradient_norm_sq(&grad_re, &grad_im);

        // Gradient should be near zero at the minimum
        assert!(gnorm < 1e-6, "Gradient norm {} should be near zero", gnorm);
    }

    #[test]
    fn test_gradient_nonzero_away_from_minimum() {
        let mut field = make_test_field();
        // Set nonuniform field
        for i in 0..field.num_sites {
            field.set(i, (i as f32) * 0.01, 0.0);
        }
        let params = FreeEnergyParams::default();
        let (grad_re, grad_im) = free_energy_gradient(&field, &params);
        let gnorm = gradient_norm_sq(&grad_re, &grad_im);

        assert!(gnorm > 1e-6, "Gradient should be nonzero away from minimum");
    }
}
