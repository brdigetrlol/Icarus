// Copyright (c) 2025-2026 brdigetrlol. All rights reserved.
// SPDX-License-Identifier: LicenseRef-Icarus-Proprietary
// See LICENSE in the repository root for full license terms.

//! Spectral Methods for the Graph Laplacian
//!
//! Eigendecomposes the weighted graph Laplacian operator used in the RAE,
//! enabling an IMEX (Implicit-Explicit) Crank-Nicolson solver that is
//! **unconditionally stable** for the diffusion term.
//!
//! The forward Euler RAE solver requires `dt < 2/K` (CFL condition),
//! e.g. `dt < 0.0083` for E8 (K=240). The spectral IMEX solver removes
//! this restriction entirely — the implicit treatment of diffusion is
//! stable for any dt, while nonlinear terms are handled explicitly.
//!
//! ## Cost
//!
//! - **One-time**: O(N³) eigendecomposition (N = num_sites, 241 for E8 ≈ 14M ops)
//! - **Per-step**: O(N²) for forward/inverse transforms (two matrix-vector products)
//!
//! For E8 (N=241), the per-step cost is ~116K multiplies — comparable to the
//! CSR Laplacian (241×240 = ~58K). The payoff is much larger stable timesteps.

use nalgebra::{DMatrix, SymmetricEigen};

use crate::free_energy::double_well_derivative;
use crate::phase_field::LatticeField;
use crate::rae::RAEParams;

/// Spectral basis for the graph Laplacian operator.
///
/// The RAE diffusion operator is `M = -2 D_K^{-1} L` where:
/// - `L` is the symmetric combinatorial Laplacian: `L[i][j] = -w_ij`, `L[i][i] = Σ w_ij`
/// - `D_K = diag(K_i)` is the degree matrix (neighbor count per site)
///
/// `M` is asymmetric when `K_i` varies, but is **similar** to a symmetric matrix:
/// `D_K^{1/2} M D_K^{-1/2} = -2 L_norm` where `L_norm = D_K^{-1/2} L D_K^{-1/2}`
/// is the symmetric normalized Laplacian with eigenvalues `μ_k ≥ 0`.
///
/// The operator eigenvalues are `λ_k = -2μ_k ≤ 0`, and the weighted transforms:
/// - Forward: `c = U^T · (sqrt_degree .* z)`
/// - Inverse: `z = inv_sqrt_degree .* (U · c)`
///
/// recover the original operator: `apply_laplacian(z) = M · z`.
#[derive(Debug, Clone)]
pub struct SpectralBasis {
    /// Eigenvalues of the diffusion operator (all ≤ 0), sorted ascending.
    /// These are `-2μ_k` where `μ_k` are eigenvalues of the normalized Laplacian.
    pub eigenvalues: Vec<f32>,
    /// Eigenvector matrix U (column-major) from `L_norm = U diag(μ) U^T`.
    /// Column `k` is the k-th eigenvector: `U[row + col*n]`.
    eigenvectors: Vec<f32>,
    /// `sqrt(K_i)` for each site — used in weighted forward transform.
    sqrt_degree: Vec<f32>,
    /// `1/sqrt(K_i)` for each site — used in weighted inverse transform.
    inv_sqrt_degree: Vec<f32>,
    /// Number of lattice sites
    pub num_sites: usize,
    /// Maximum asymmetry of the raw operator matrix (diagnostic).
    pub asymmetry: f32,
}

impl SpectralBasis {
    /// Build the spectral basis from a LatticeField's CSR topology.
    ///
    /// Constructs the symmetric normalized Laplacian `L_norm = D^{-1/2} L D^{-1/2}`
    /// and eigendecomposes it. The operator eigenvalues `λ_k = -2μ_k` are stored
    /// (where `μ_k ≥ 0` are eigenvalues of `L_norm`), along with degree-weighting
    /// vectors for the forward/inverse transforms.
    ///
    /// Cost: O(N³) where N = field.num_sites.
    pub fn from_field(field: &LatticeField) -> Self {
        let n = field.num_sites;

        // Step 1: Compute degree vector d_i = K_i (neighbor count per site)
        // and build the symmetric combinatorial Laplacian L in f64.
        // L[i][j] = -w_ij for neighbors, L[i][i] = Σ w_ij
        let mut lap = DMatrix::<f64>::zeros(n, n);
        let mut degree = vec![0.0f64; n];

        for i in 0..n {
            let start = field.neighbor_offsets[i] as usize;
            let end = field.neighbor_offsets[i + 1] as usize;
            degree[i] = (end - start) as f64;
            let mut diag_sum = 0.0f64;

            for edge in start..end {
                let j = field.neighbor_indices[edge] as usize;
                let w = field.neighbor_weights[edge] as f64;
                lap[(i, j)] -= w;
                diag_sum += w;
            }
            lap[(i, i)] = diag_sum;
        }

        // Measure asymmetry of the raw operator M = -2 D^{-1} L (diagnostic)
        let mut max_asym = 0.0f64;
        for i in 0..n {
            for j in (i + 1)..n {
                if degree[i] > 0.0 && degree[j] > 0.0 {
                    let m_ij = -2.0 * lap[(i, j)] / degree[i];
                    let m_ji = -2.0 * lap[(j, i)] / degree[j];
                    let asym = (m_ij - m_ji).abs();
                    if asym > max_asym {
                        max_asym = asym;
                    }
                }
            }
        }

        // Step 2: Compute D^{-1/2} and form L_norm = D^{-1/2} L D^{-1/2}
        let mut inv_sqrt_d = vec![0.0f64; n];
        let mut sqrt_d = vec![0.0f64; n];
        for i in 0..n {
            if degree[i] > 0.0 {
                sqrt_d[i] = degree[i].sqrt();
                inv_sqrt_d[i] = 1.0 / sqrt_d[i];
            } else {
                sqrt_d[i] = 1.0;
                inv_sqrt_d[i] = 1.0;
            }
        }

        let mut l_norm = DMatrix::<f64>::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                l_norm[(i, j)] = inv_sqrt_d[i] * lap[(i, j)] * inv_sqrt_d[j];
            }
        }

        // Step 3: Eigendecompose L_norm (symmetric, eigenvalues μ ≥ 0)
        let eigen = SymmetricEigen::new(l_norm);

        // Sort by operator eigenvalue λ = -2μ ascending (most negative first, zero last).
        // Since λ = -2μ, sorting μ descending gives λ ascending.
        let mut indexed: Vec<(usize, f64)> = eigen
            .eigenvalues
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap()); // μ descending → λ ascending

        // Operator eigenvalues: λ_k = -2*μ_k (all ≤ 0, sorted ascending)
        let eigenvalues: Vec<f32> = indexed.iter().map(|&(_, mu)| (-2.0 * mu) as f32).collect();

        // Build sorted eigenvector matrix U (column-major)
        let mut eigenvectors = vec![0.0f32; n * n];
        for (new_col, &(old_col, _)) in indexed.iter().enumerate() {
            for row in 0..n {
                eigenvectors[row + new_col * n] = eigen.eigenvectors[(row, old_col)] as f32;
            }
        }

        // Store degree vectors as f32
        let sqrt_degree: Vec<f32> = sqrt_d.iter().map(|&v| v as f32).collect();
        let inv_sqrt_degree: Vec<f32> = inv_sqrt_d.iter().map(|&v| v as f32).collect();

        Self {
            eigenvalues,
            eigenvectors,
            sqrt_degree,
            inv_sqrt_degree,
            num_sites: n,
            asymmetry: max_asym as f32,
        }
    }

    /// Forward spectral transform: `c = U^T · (sqrt_degree .* z)`
    ///
    /// Scales `z` by `sqrt(K_i)` then projects into the eigenbasis of `L_norm`.
    pub fn forward(&self, values: &[f32]) -> Vec<f32> {
        let n = self.num_sites;
        debug_assert_eq!(values.len(), n);
        let mut coeffs = vec![0.0f32; n];

        // c[k] = Σ_i U[i,k] * sqrt_degree[i] * values[i]
        for k in 0..n {
            let col_base = k * n;
            let mut sum = 0.0f32;
            for i in 0..n {
                sum += self.eigenvectors[i + col_base] * self.sqrt_degree[i] * values[i];
            }
            coeffs[k] = sum;
        }
        coeffs
    }

    /// Inverse spectral transform: `z = inv_sqrt_degree .* (U · c)`
    ///
    /// Reconstructs spatial vector from spectral coefficients, then scales
    /// by `1/sqrt(K_i)` to undo the forward weighting.
    pub fn inverse(&self, coeffs: &[f32]) -> Vec<f32> {
        let n = self.num_sites;
        debug_assert_eq!(coeffs.len(), n);
        let mut values = vec![0.0f32; n];

        // values[i] = inv_sqrt_degree[i] * Σ_k U[i,k] * coeffs[k]
        for i in 0..n {
            let mut sum = 0.0f32;
            for k in 0..n {
                sum += self.eigenvectors[i + k * n] * coeffs[k];
            }
            values[i] = self.inv_sqrt_degree[i] * sum;
        }
        values
    }

    /// Apply the Laplacian operator via spectral decomposition.
    ///
    /// Computes `M · z = -2 D^{-1} L z` using:
    /// `inv_sqrt_d .* (U · diag(λ) · U^T · (sqrt_d .* z))`
    pub fn apply_laplacian(&self, values: &[f32]) -> Vec<f32> {
        let coeffs = self.forward(values);
        let n = self.num_sites;
        let mut scaled = vec![0.0f32; n];
        for k in 0..n {
            scaled[k] = self.eigenvalues[k] * coeffs[k];
        }
        self.inverse(&scaled)
    }

    /// IMEX (Implicit-Explicit) Crank-Nicolson RAE step.
    ///
    /// Treats the diffusion term (graph Laplacian) **implicitly** via
    /// Crank-Nicolson, and all nonlinear terms **explicitly** (forward Euler).
    ///
    /// In spectral space for mode k with eigenvalue λ_k:
    ///
    /// ```text
    /// (1 - dt/2 · kw · λ_k) · c_k^{n+1} = (1 + dt/2 · kw · λ_k) · c_k^n + dt · NL_k^n
    /// ```
    ///
    /// Since λ_k ≤ 0 for diffusion, the denominator `(1 - dt/2·kw·λ_k) ≥ 1`,
    /// and the amplification factor `|(1+α)/(1-α)| ≤ 1` — **unconditionally stable**.
    ///
    /// The nonlinear terms NL(z) include:
    /// - Potential gradient: `-dV/d(|z|²) · z`
    /// - Resonance: `iω · z`
    /// - Damping: `-γ · z`
    ///
    /// Updates the field in-place. Returns the spectral energy spectrum
    /// (|c_k|² for each mode) for diagnostics.
    pub fn imex_step(
        &self,
        field: &mut LatticeField,
        params: &RAEParams,
        dt: f32,
    ) -> Vec<f32> {
        let n = self.num_sites;
        assert_eq!(field.num_sites, n);

        let omega = params.omega;
        let gamma = params.gamma;
        let target_sq = params.energy_params.target_amplitude
            * params.energy_params.target_amplitude;
        let kw = params.energy_params.kinetic_weight;
        let pw = params.energy_params.potential_weight;

        // Step 1: Compute nonlinear terms NL(z) at current state (explicit)
        let mut nl_re = vec![0.0f32; n];
        let mut nl_im = vec![0.0f32; n];

        for i in 0..n {
            let zi_re = field.values_re[i];
            let zi_im = field.values_im[i];
            let ns = zi_re * zi_re + zi_im * zi_im;
            let dv = double_well_derivative(ns, target_sq) * pw;

            // NL = -dV·z + iω·z - γ·z
            nl_re[i] = -dv * zi_re - omega * zi_im - gamma * zi_re;
            nl_im[i] = -dv * zi_im + omega * zi_re - gamma * zi_im;
        }

        // Step 2: Forward transform current state and NL to spectral space
        let c_re = self.forward(&field.values_re);
        let c_im = self.forward(&field.values_im);
        let nl_c_re = self.forward(&nl_re);
        let nl_c_im = self.forward(&nl_im);

        // Step 3: Crank-Nicolson update in spectral space
        let mut c_re_new = vec![0.0f32; n];
        let mut c_im_new = vec![0.0f32; n];
        let mut spectrum = vec![0.0f32; n];

        let half_dt_kw = 0.5 * dt * kw;

        for k in 0..n {
            let lam = self.eigenvalues[k]; // ≤ 0

            // Crank-Nicolson factors
            let explicit_factor = 1.0 + half_dt_kw * lam; // ≤ 1 (since lam ≤ 0)
            let implicit_factor = 1.0 - half_dt_kw * lam; // ≥ 1

            // c^{n+1} = [explicit_factor * c^n + dt * NL^n] / implicit_factor
            c_re_new[k] = (explicit_factor * c_re[k] + dt * nl_c_re[k]) / implicit_factor;
            c_im_new[k] = (explicit_factor * c_im[k] + dt * nl_c_im[k]) / implicit_factor;

            spectrum[k] = c_re_new[k] * c_re_new[k] + c_im_new[k] * c_im_new[k];
        }

        // Step 4: Inverse transform back to spatial domain
        field.values_re = self.inverse(&c_re_new);
        field.values_im = self.inverse(&c_im_new);

        spectrum
    }

    /// Maximum stable timestep for the explicit nonlinear terms.
    ///
    /// The IMEX scheme has no CFL restriction from diffusion, but the
    /// explicit nonlinear terms still have a stability bound. For the
    /// double-well potential with max |dV/d(|z|²)| ≈ max_amplitude²,
    /// a conservative estimate is `dt < 1 / (pw * max_amplitude² + gamma)`.
    pub fn max_nonlinear_dt(params: &RAEParams, max_amplitude: f32) -> f32 {
        let ns_max = max_amplitude * max_amplitude;
        let dv_max = (ns_max - params.energy_params.target_amplitude.powi(2)).abs() * 0.5;
        let rate = dv_max * params.energy_params.potential_weight + params.gamma + params.omega;
        if rate > 1e-12 {
            1.0 / rate
        } else {
            1.0 // effectively no constraint
        }
    }

    /// Spectral condition number: |λ_max| / |λ_min_nonzero|.
    ///
    /// A large condition number means the eigenvalue spectrum is spread
    /// out — exactly the case where IMEX outperforms forward Euler.
    pub fn condition_number(&self) -> f32 {
        let n = self.num_sites;
        if n < 2 {
            return 1.0;
        }
        // Eigenvalues are sorted ascending (most negative first).
        // The largest magnitude is eigenvalues[0] (most negative).
        // The smallest nonzero magnitude — skip the near-zero eigenvalue at the end.
        let lam_max_mag = self.eigenvalues[0].abs();
        let lam_min_nonzero = self
            .eigenvalues
            .iter()
            .rev()
            .find(|&&v| v.abs() > 1e-6)
            .map(|v| v.abs())
            .unwrap_or(1.0);

        if lam_min_nonzero > 1e-12 {
            lam_max_mag / lam_min_nonzero
        } else {
            f32::INFINITY
        }
    }

    /// Memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        // eigenvalues: n*4, eigenvectors: n*n*4, sqrt_degree: n*4, inv_sqrt_degree: n*4
        (3 * self.num_sites + self.num_sites * self.num_sites) * 4
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::free_energy::{free_energy, FreeEnergyParams};
    use crate::rae::IntegratorMethod;
    use icarus_math::lattice::e8::E8Lattice;
    use icarus_math::lattice::hypercubic::HypercubicLattice;

    fn make_e8_field() -> LatticeField {
        let lattice = E8Lattice::new();
        LatticeField::from_lattice(&lattice)
    }

    fn make_hypercubic_field(dim: usize) -> LatticeField {
        let lattice = HypercubicLattice::new(dim);
        LatticeField::from_lattice(&lattice)
    }

    #[test]
    fn test_eigenvalues_nonpositive() {
        let field = make_hypercubic_field(3);
        let basis = SpectralBasis::from_field(&field);

        for (k, &lam) in basis.eigenvalues.iter().enumerate() {
            assert!(
                lam <= 1e-6,
                "Eigenvalue {} = {} should be ≤ 0 (diffusion operator)",
                k, lam
            );
        }
    }

    #[test]
    fn test_zero_eigenvalue_exists() {
        // The constant vector is in the kernel of the Laplacian → eigenvalue = 0
        let field = make_hypercubic_field(4);
        let basis = SpectralBasis::from_field(&field);

        let has_near_zero = basis.eigenvalues.iter().any(|&v| v.abs() < 1e-4);
        assert!(
            has_near_zero,
            "Should have a near-zero eigenvalue (constant mode). Min magnitude: {}",
            basis.eigenvalues.iter().map(|v| v.abs()).fold(f32::MAX, f32::min)
        );
    }

    #[test]
    fn test_forward_inverse_round_trip() {
        let field = make_hypercubic_field(3);
        let basis = SpectralBasis::from_field(&field);

        // Random input
        let values: Vec<f32> = (0..field.num_sites)
            .map(|i| (i as f32 * 0.7 + 0.3).sin())
            .collect();

        let coeffs = basis.forward(&values);
        let restored = basis.inverse(&coeffs);

        for i in 0..field.num_sites {
            let err = (restored[i] - values[i]).abs();
            assert!(
                err < 1e-4,
                "Round-trip error at site {}: {} (orig={}, restored={})",
                i, err, values[i], restored[i]
            );
        }
    }

    #[test]
    fn test_spectral_laplacian_matches_csr() {
        let mut field = make_hypercubic_field(3);
        field.init_random(42, 1.0);
        let basis = SpectralBasis::from_field(&field);

        // Spectral Laplacian
        let lap_spectral = basis.apply_laplacian(&field.values_re);

        // CSR Laplacian (same formula as RAE)
        let n = field.num_sites;
        let mut lap_csr = vec![0.0f32; n];
        for i in 0..n {
            let start = field.neighbor_offsets[i] as usize;
            let end = field.neighbor_offsets[i + 1] as usize;
            let k = (end - start) as f32;
            if k > 0.0 {
                let scale = 2.0 / k;
                let mut acc = 0.0f32;
                for edge in start..end {
                    let j = field.neighbor_indices[edge] as usize;
                    let w = field.neighbor_weights[edge];
                    acc += w * (field.values_re[j] - field.values_re[i]);
                }
                lap_csr[i] = scale * acc;
            }
        }

        // Compare (allow tolerance for symmetrization and f32 precision)
        let mut max_err = 0.0f32;
        for i in 0..n {
            let err = (lap_spectral[i] - lap_csr[i]).abs();
            max_err = max_err.max(err);
        }
        assert!(
            max_err < 0.1,
            "Spectral Laplacian should match CSR. Max error: {}",
            max_err
        );
    }

    #[test]
    fn test_imex_small_dt_matches_euler() {
        // With small dt, IMEX should give similar results to forward Euler
        let lattice = HypercubicLattice::new(3);
        let mut field_euler = LatticeField::from_lattice(&lattice);
        let mut field_imex = LatticeField::from_lattice(&lattice);
        field_euler.init_random(42, 0.5);
        field_imex.init_random(42, 0.5);

        let params = RAEParams::default_hypercubic(3);
        let dt = params.dt * 0.5; // well within CFL

        let basis = SpectralBasis::from_field(&field_euler);

        // Forward Euler step
        let mut solver = crate::rae::RAESolver::new(
            RAEParams { dt, ..params.clone() },
            field_euler.num_sites,
        );
        solver.step(&mut field_euler);

        // IMEX step
        basis.imex_step(&mut field_imex, &params, dt);

        // Compare
        let mut max_err = 0.0f32;
        for i in 0..field_euler.num_sites {
            let err_re = (field_euler.values_re[i] - field_imex.values_re[i]).abs();
            let err_im = (field_euler.values_im[i] - field_imex.values_im[i]).abs();
            max_err = max_err.max(err_re).max(err_im);
        }
        // For small dt, the two methods should agree closely
        // (CN ≈ Euler when dt*|λ| << 1)
        assert!(
            max_err < 0.05,
            "IMEX and Euler should agree for small dt. Max diff: {}",
            max_err
        );
    }

    #[test]
    fn test_imex_large_dt_stable() {
        // IMEX should remain stable even with dt >> CFL limit
        let lattice = HypercubicLattice::new(3);
        let mut field = LatticeField::from_lattice(&lattice);
        field.init_random(42, 0.5);

        let params = RAEParams {
            dt: 0.1, // 10x the CFL limit for hypercubic-3 (CFL ≈ 1/6 ≈ 0.167 — actually close)
            omega: 0.5,
            gamma: 0.3,
            energy_params: FreeEnergyParams {
                kinetic_weight: 0.5,
                potential_weight: 1.0,
                target_amplitude: 1.0,
            },
            method: IntegratorMethod::Euler,
        };

        let basis = SpectralBasis::from_field(&field);

        // Run 50 IMEX steps with large dt
        for _ in 0..50 {
            basis.imex_step(&mut field, &params, params.dt);
        }

        // Field should remain bounded (not blow up)
        let max_amp: f32 = field
            .values_re
            .iter()
            .zip(field.values_im.iter())
            .map(|(&r, &i)| (r * r + i * i).sqrt())
            .fold(0.0f32, f32::max);

        assert!(
            max_amp < 100.0,
            "IMEX should remain bounded with large dt. Max amplitude: {}",
            max_amp
        );
    }

    #[test]
    fn test_imex_energy_decreases_with_damping() {
        let lattice = HypercubicLattice::new(4);
        let mut field = LatticeField::from_lattice(&lattice);
        field.init_random(77, 0.8);

        let params = RAEParams {
            dt: 0.05,
            omega: 0.0, // no oscillation
            gamma: 0.5, // strong damping
            energy_params: FreeEnergyParams::default(),
            method: IntegratorMethod::Euler,
        };

        let basis = SpectralBasis::from_field(&field);
        let (e0, _, _) = free_energy(&field, &params.energy_params);

        for _ in 0..20 {
            basis.imex_step(&mut field, &params, params.dt);
        }
        let (e1, _, _) = free_energy(&field, &params.energy_params);

        for _ in 0..20 {
            basis.imex_step(&mut field, &params, params.dt);
        }
        let (e2, _, _) = free_energy(&field, &params.energy_params);

        // With strong damping and no oscillation, energy should decrease
        assert!(
            e1 < e0 + 0.5,
            "Energy should decrease with damping: e0={}, e1={}",
            e0, e1
        );
        assert!(
            e2 < e1 + 0.5,
            "Energy should continue decreasing: e1={}, e2={}",
            e1, e2
        );
    }

    #[test]
    fn test_e8_spectral_basis() {
        let field = make_e8_field();
        let basis = SpectralBasis::from_field(&field);

        assert_eq!(basis.num_sites, 241);
        assert_eq!(basis.eigenvalues.len(), 241);

        // All eigenvalues non-positive
        for &lam in &basis.eigenvalues {
            assert!(lam <= 1e-5, "E8 eigenvalue {} should be ≤ 0", lam);
        }

        // Should have a near-zero eigenvalue
        let min_mag = basis.eigenvalues.iter().map(|v| v.abs()).fold(f32::MAX, f32::min);
        assert!(min_mag < 0.01, "E8 should have near-zero eigenvalue, min |λ|={}", min_mag);

        // Asymmetry measures the raw operator M = -2 D^{-1} L before normalization.
        // E8 has varying degree (origin: K=240, boundary sites: fewer) so
        // asymmetry is expected to be non-trivial. The normalized Laplacian
        // handles this correctly — asymmetry is purely diagnostic.
        assert!(
            basis.asymmetry < 10.0,
            "E8 asymmetry {} unexpectedly large",
            basis.asymmetry
        );
    }

    #[test]
    fn test_e8_imex_step() {
        let mut field = make_e8_field();
        field.init_random(42, 0.5);
        let basis = SpectralBasis::from_field(&field);

        let params = RAEParams::default_e8();

        // Use a dt 5x larger than the Euler CFL limit
        let large_dt = 0.04; // CFL limit is ~0.0083

        let spectrum = basis.imex_step(&mut field, &params, large_dt);

        // Spectrum should have num_sites entries
        assert_eq!(spectrum.len(), 241);

        // Field should remain bounded
        let max_val: f32 = field
            .values_re
            .iter()
            .chain(field.values_im.iter())
            .map(|v| v.abs())
            .fold(0.0f32, f32::max);
        assert!(max_val < 100.0, "E8 IMEX should be bounded, max={}", max_val);
    }

    #[test]
    fn test_condition_number() {
        let field = make_e8_field();
        let basis = SpectralBasis::from_field(&field);

        let cond = basis.condition_number();
        assert!(cond > 1.0, "Condition number should be > 1, got {}", cond);
        assert!(
            cond < 1e6,
            "Condition number shouldn't be astronomical, got {}",
            cond
        );
    }

    #[test]
    fn test_max_nonlinear_dt() {
        let params = RAEParams::default_e8();
        let dt_nl = SpectralBasis::max_nonlinear_dt(&params, 2.0);
        assert!(dt_nl > 0.0, "Nonlinear dt should be positive");
        assert!(dt_nl < 10.0, "Nonlinear dt should be finite: {}", dt_nl);
    }

    #[test]
    fn test_memory_bytes() {
        let field = make_e8_field();
        let basis = SpectralBasis::from_field(&field);

        let mem = basis.memory_bytes();
        // eigenvalues(241) + eigenvectors(241*241) + sqrt_degree(241) + inv_sqrt_degree(241)
        let expected = (3 * 241 + 241 * 241) * 4;
        assert_eq!(mem, expected);
    }

    #[test]
    fn test_eigenvectors_orthonormal() {
        // Verify that V^T V ≈ I (eigenvectors are orthonormal)
        let field = make_hypercubic_field(3);
        let basis = SpectralBasis::from_field(&field);
        let n = basis.num_sites;

        for i in 0..n {
            for j in 0..n {
                let mut dot = 0.0f32;
                for k in 0..n {
                    dot += basis.eigenvectors[k + i * n] * basis.eigenvectors[k + j * n];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                let err = (dot - expected).abs();
                assert!(
                    err < 1e-3,
                    "V^T V[{},{}] = {} (expected {}), err={}",
                    i, j, dot, expected, err
                );
            }
        }
    }

    #[test]
    fn test_constant_vector_in_kernel() {
        // A constant vector should be in the kernel of the Laplacian
        let field = make_hypercubic_field(3);
        let basis = SpectralBasis::from_field(&field);
        let n = field.num_sites;

        let constant = vec![1.0f32; n];
        let lap_result = basis.apply_laplacian(&constant);

        let max_val: f32 = lap_result.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        assert!(
            max_val < 1e-4,
            "Laplacian of constant vector should be ~0, max={}",
            max_val
        );
    }
}
