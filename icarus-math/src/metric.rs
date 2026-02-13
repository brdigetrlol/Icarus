//! Metric Tensor for Icarus EMC
//!
//! Each lattice site has a local metric tensor g_μν that defines the
//! geometry of information space. The metric is learned, not fixed.
//!
//! Storage: upper-triangle packed symmetric matrix (dim*(dim+1)/2 floats per site).
//! For E8 (dim=8): 36 floats/site. For Leech (dim=24): 300 floats/site.
//!
//! Key quantities derived from the metric:
//! - Christoffel symbols Γ^λ_μν (connection)
//! - Ricci tensor R_μν (curvature)
//! - Metric determinant (volume element)

use serde::{Deserialize, Serialize};

/// Packed symmetric metric tensor for a single site.
/// Stores the upper triangle in row-major order:
/// For dim=3: [g00, g01, g02, g11, g12, g22]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SiteMetric {
    /// Upper-triangle packed components
    pub components: Vec<f32>,
    /// Dimension of the underlying space
    pub dim: usize,
}

impl SiteMetric {
    /// Number of independent components for a symmetric dim×dim matrix
    pub fn packed_size(dim: usize) -> usize {
        dim * (dim + 1) / 2
    }

    /// Create an identity (flat) metric for dimension `dim`
    pub fn identity(dim: usize) -> Self {
        let n = Self::packed_size(dim);
        let mut components = vec![0.0f32; n];
        // Set diagonal entries to 1.0
        for i in 0..dim {
            let idx = Self::packed_index(dim, i, i);
            components[idx] = 1.0;
        }
        Self { components, dim }
    }

    /// Create a metric from a full symmetric matrix (flattened row-major)
    pub fn from_full(dim: usize, full: &[f32]) -> Self {
        assert_eq!(full.len(), dim * dim);
        let n = Self::packed_size(dim);
        let mut components = vec![0.0; n];
        for i in 0..dim {
            for j in i..dim {
                let idx = Self::packed_index(dim, i, j);
                components[idx] = full[i * dim + j];
            }
        }
        Self { components, dim }
    }

    /// Get the packed index for (i, j) where i <= j
    #[inline]
    pub fn packed_index(dim: usize, i: usize, j: usize) -> usize {
        let (r, c) = if i <= j { (i, j) } else { (j, i) };
        r * dim - r * (r + 1) / 2 + c
    }

    /// Get g_μν (symmetric, so g_ij = g_ji)
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f32 {
        self.components[Self::packed_index(self.dim, i, j)]
    }

    /// Set g_μν (sets both (i,j) and (j,i) via packed storage)
    #[inline]
    pub fn set(&mut self, i: usize, j: usize, value: f32) {
        let idx = Self::packed_index(self.dim, i, j);
        self.components[idx] = value;
    }

    /// Convert to full dim×dim matrix (row-major)
    pub fn to_full(&self) -> Vec<f32> {
        let mut full = vec![0.0; self.dim * self.dim];
        for i in 0..self.dim {
            for j in 0..self.dim {
                full[i * self.dim + j] = self.get(i, j);
            }
        }
        full
    }

    /// Compute the inverse metric g^μν via Gauss-Jordan elimination.
    /// Returns None if singular.
    pub fn inverse(&self) -> Option<Self> {
        let n = self.dim;
        let mut aug = vec![0.0f32; n * 2 * n];

        // Build augmented matrix [g | I]
        for i in 0..n {
            for j in 0..n {
                aug[i * 2 * n + j] = self.get(i, j);
            }
            aug[i * 2 * n + n + i] = 1.0;
        }

        // Gauss-Jordan
        for col in 0..n {
            // Find pivot
            let mut max_val = aug[col * 2 * n + col].abs();
            let mut max_row = col;
            for row in (col + 1)..n {
                let val = aug[row * 2 * n + col].abs();
                if val > max_val {
                    max_val = val;
                    max_row = row;
                }
            }

            if max_val < 1e-12 {
                return None; // Singular
            }

            // Swap rows
            if max_row != col {
                for k in 0..(2 * n) {
                    let tmp = aug[col * 2 * n + k];
                    aug[col * 2 * n + k] = aug[max_row * 2 * n + k];
                    aug[max_row * 2 * n + k] = tmp;
                }
            }

            // Scale pivot row
            let pivot = aug[col * 2 * n + col];
            for k in 0..(2 * n) {
                aug[col * 2 * n + k] /= pivot;
            }

            // Eliminate
            for row in 0..n {
                if row != col {
                    let factor = aug[row * 2 * n + col];
                    for k in 0..(2 * n) {
                        aug[row * 2 * n + k] -= factor * aug[col * 2 * n + k];
                    }
                }
            }
        }

        // Extract inverse
        let mut inv_full = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                inv_full[i * n + j] = aug[i * 2 * n + n + j];
            }
        }

        Some(Self::from_full(n, &inv_full))
    }

    /// Compute determinant (for volume element √|det g|).
    /// Uses LU decomposition for numerical stability.
    pub fn determinant(&self) -> f32 {
        let n = self.dim;
        let mut mat = self.to_full();
        let mut det = 1.0f32;

        for col in 0..n {
            // Partial pivoting
            let mut max_val = mat[col * n + col].abs();
            let mut max_row = col;
            for row in (col + 1)..n {
                let val = mat[row * n + col].abs();
                if val > max_val {
                    max_val = val;
                    max_row = row;
                }
            }

            if max_val < 1e-20 {
                return 0.0;
            }

            if max_row != col {
                for k in 0..n {
                    let tmp = mat[col * n + k];
                    mat[col * n + k] = mat[max_row * n + k];
                    mat[max_row * n + k] = tmp;
                }
                det = -det;
            }

            det *= mat[col * n + col];

            for row in (col + 1)..n {
                let factor = mat[row * n + col] / mat[col * n + col];
                for k in (col + 1)..n {
                    mat[row * n + k] -= factor * mat[col * n + k];
                }
            }
        }

        det
    }

    /// Compute Christoffel symbols of the second kind: Γ^λ_μν.
    ///
    /// Γ^λ_μν = (1/2) g^{λσ} (∂_μ g_{νσ} + ∂_ν g_{μσ} - ∂_σ g_{μν})
    ///
    /// On a lattice, derivatives are approximated by finite differences
    /// using the metric at neighboring sites. This function computes
    /// Christoffel symbols given the metric and its neighbors' metrics.
    ///
    /// `neighbor_metrics`: metrics at the K nearest neighbors
    /// `neighbor_displacements`: direction vectors e_ij to each neighbor (as f32 coords)
    ///
    /// Returns Christoffel symbols as [λ * dim * dim + μ * dim + ν] (dim³ components)
    pub fn christoffel(
        &self,
        neighbor_metrics: &[SiteMetric],
        neighbor_displacements: &[Vec<f32>],
    ) -> Vec<f32> {
        let n = self.dim;
        let num_christoffel = n * n * n;

        // Compute inverse metric
        let g_inv = match self.inverse() {
            Some(inv) => inv,
            None => return vec![0.0; num_christoffel],
        };

        // Estimate ∂_σ g_{μν} via weighted least-squares over neighbors
        // For simplicity on the first-shell neighborhood, use a direct finite-difference:
        // ∂_σ g_{μν} ≈ Σ_j w_j * (g_{μν}(j) - g_{μν}(center)) * e_σ(j) / |e_j|²
        let k = neighbor_metrics.len();
        let mut dg = vec![0.0f32; n * n * n]; // dg[sigma * n*n + mu * n + nu]

        for j in 0..k {
            let e = &neighbor_displacements[j];
            let e_norm_sq: f32 = e.iter().map(|&x| x * x).sum();
            if e_norm_sq < 1e-12 {
                continue;
            }
            let inv_norm_sq = 1.0 / e_norm_sq;

            for mu in 0..n {
                for nu in mu..n {
                    let delta_g = neighbor_metrics[j].get(mu, nu) - self.get(mu, nu);
                    for sigma in 0..n {
                        let contrib = delta_g * e[sigma] * inv_norm_sq;
                        dg[sigma * n * n + mu * n + nu] += contrib;
                        if mu != nu {
                            dg[sigma * n * n + nu * n + mu] += contrib;
                        }
                    }
                }
            }
        }

        // Normalize by number of neighbors
        if k > 0 {
            let inv_k = 1.0 / k as f32;
            for val in &mut dg {
                *val *= inv_k;
            }
        }

        // Γ^λ_μν = (1/2) Σ_σ g^{λσ} (dg[μ,νσ] + dg[ν,μσ] - dg[σ,μν])
        let mut christoffel = vec![0.0f32; num_christoffel];

        for lambda in 0..n {
            for mu in 0..n {
                for nu in 0..n {
                    let mut val = 0.0;
                    for sigma in 0..n {
                        let d_mu_g_nu_sigma = dg[mu * n * n + nu * n + sigma];
                        let d_nu_g_mu_sigma = dg[nu * n * n + mu * n + sigma];
                        let d_sigma_g_mu_nu = dg[sigma * n * n + mu * n + nu];
                        val += g_inv.get(lambda, sigma)
                            * (d_mu_g_nu_sigma + d_nu_g_mu_sigma - d_sigma_g_mu_nu);
                    }
                    christoffel[lambda * n * n + mu * n + nu] = 0.5 * val;
                }
            }
        }

        christoffel
    }

    /// Compute the Ricci tensor R_μν from Christoffel symbols.
    ///
    /// R_μν = ∂_λ Γ^λ_μν - ∂_ν Γ^λ_μλ + Γ^λ_λσ Γ^σ_μν - Γ^λ_νσ Γ^σ_μλ
    ///
    /// This simplified version computes the quadratic Christoffel terms only,
    /// which dominates in the strong-field regime relevant for EMC dynamics.
    /// The derivative terms require second-shell neighbors and are added in Phase 3.
    pub fn ricci_from_christoffel(&self, christoffel: &[f32]) -> Vec<f32> {
        let n = self.dim;
        let mut ricci = vec![0.0f32; n * n];

        for mu in 0..n {
            for nu in 0..n {
                let mut val = 0.0f32;
                for lambda in 0..n {
                    for sigma in 0..n {
                        // Γ^λ_λσ * Γ^σ_μν
                        val += christoffel[lambda * n * n + lambda * n + sigma]
                            * christoffel[sigma * n * n + mu * n + nu];
                        // - Γ^λ_νσ * Γ^σ_μλ
                        val -= christoffel[lambda * n * n + nu * n + sigma]
                            * christoffel[sigma * n * n + mu * n + lambda];
                    }
                }
                ricci[mu * n + nu] = val;
            }
        }

        ricci
    }

    /// Compute Ricci scalar: R = g^{μν} R_{μν}
    pub fn ricci_scalar(&self, ricci: &[f32]) -> f32 {
        let n = self.dim;
        let g_inv = match self.inverse() {
            Some(inv) => inv,
            None => return 0.0,
        };

        let mut scalar = 0.0f32;
        for mu in 0..n {
            for nu in 0..n {
                scalar += g_inv.get(mu, nu) * ricci[mu * n + nu];
            }
        }
        scalar
    }

    /// Pin eigenvalues to [eps, 1/eps] for numerical stability.
    /// This prevents metric collapse or explosion during learning.
    pub fn pin_eigenvalues(&mut self, eps: f32) {
        // For a symmetric matrix, we do a simple diagonal clamping
        // Full eigenvalue pinching requires eigendecomposition;
        // this is the fast approximation that works for nearly-diagonal metrics
        let inv_eps = 1.0 / eps;
        for i in 0..self.dim {
            let diag = self.get(i, i);
            let clamped = diag.max(eps).min(inv_eps);
            self.set(i, i, clamped);
        }
    }
}

/// Collection of metrics across all sites in a lattice layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricField {
    /// Packed metric components for all sites, contiguous
    /// Layout: [site0_comp0, site0_comp1, ..., site1_comp0, ...]
    pub data: Vec<f32>,
    /// Dimension of the lattice
    pub dim: usize,
    /// Number of sites
    pub num_sites: usize,
    /// Components per site
    pub components_per_site: usize,
}

impl MetricField {
    /// Create an identity metric field (all sites have flat metric)
    pub fn identity(dim: usize, num_sites: usize) -> Self {
        let cps = SiteMetric::packed_size(dim);
        let identity = SiteMetric::identity(dim);
        let mut data = Vec::with_capacity(num_sites * cps);
        for _ in 0..num_sites {
            data.extend_from_slice(&identity.components);
        }
        Self {
            data,
            dim,
            num_sites,
            components_per_site: cps,
        }
    }

    /// Get the metric at a specific site
    pub fn get_site(&self, site: usize) -> SiteMetric {
        let start = site * self.components_per_site;
        let end = start + self.components_per_site;
        SiteMetric {
            components: self.data[start..end].to_vec(),
            dim: self.dim,
        }
    }

    /// Set the metric at a specific site
    pub fn set_site(&mut self, site: usize, metric: &SiteMetric) {
        let start = site * self.components_per_site;
        self.data[start..start + self.components_per_site]
            .copy_from_slice(&metric.components);
    }

    /// Total memory in bytes
    pub fn memory_bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<f32>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_metric() {
        let g = SiteMetric::identity(8);
        for i in 0..8 {
            assert!((g.get(i, i) - 1.0).abs() < 1e-6);
            for j in (i + 1)..8 {
                assert!(g.get(i, j).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_packed_size() {
        assert_eq!(SiteMetric::packed_size(8), 36);
        assert_eq!(SiteMetric::packed_size(24), 300);
        assert_eq!(SiteMetric::packed_size(3), 6);
    }

    #[test]
    fn test_symmetry() {
        let mut g = SiteMetric::identity(3);
        g.set(0, 1, 0.5);
        assert!((g.get(0, 1) - 0.5).abs() < 1e-6);
        assert!((g.get(1, 0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_inverse_identity() {
        let g = SiteMetric::identity(3);
        let inv = g.inverse().unwrap();
        for i in 0..3 {
            assert!((inv.get(i, i) - 1.0).abs() < 1e-5);
            for j in (i + 1)..3 {
                assert!(inv.get(i, j).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn test_inverse_diagonal() {
        let mut g = SiteMetric::identity(3);
        g.set(0, 0, 2.0);
        g.set(1, 1, 4.0);
        g.set(2, 2, 0.5);
        let inv = g.inverse().unwrap();
        assert!((inv.get(0, 0) - 0.5).abs() < 1e-5);
        assert!((inv.get(1, 1) - 0.25).abs() < 1e-5);
        assert!((inv.get(2, 2) - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_determinant_identity() {
        let g = SiteMetric::identity(8);
        assert!((g.determinant() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_determinant_diagonal() {
        let mut g = SiteMetric::identity(3);
        g.set(0, 0, 2.0);
        g.set(1, 1, 3.0);
        g.set(2, 2, 4.0);
        assert!((g.determinant() - 24.0).abs() < 1e-3);
    }

    #[test]
    fn test_roundtrip_full() {
        let mut g = SiteMetric::identity(3);
        g.set(0, 1, 0.3);
        g.set(0, 2, 0.1);
        g.set(1, 2, 0.2);
        let full = g.to_full();
        let g2 = SiteMetric::from_full(3, &full);
        for i in 0..3 {
            for j in 0..3 {
                assert!((g.get(i, j) - g2.get(i, j)).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_pin_eigenvalues() {
        let mut g = SiteMetric::identity(3);
        g.set(0, 0, 0.001);
        g.set(2, 2, 1000.0);
        g.pin_eigenvalues(0.01);
        assert!(g.get(0, 0) >= 0.01);
        assert!(g.get(2, 2) <= 100.0);
    }

    #[test]
    fn test_metric_field_identity() {
        let field = MetricField::identity(8, 241);
        assert_eq!(field.num_sites, 241);
        assert_eq!(field.components_per_site, 36);
        assert_eq!(field.data.len(), 241 * 36);

        let site0 = field.get_site(0);
        assert!((site0.get(0, 0) - 1.0).abs() < 1e-6);
        assert!(site0.get(0, 1).abs() < 1e-6);
    }

    #[test]
    fn test_metric_field_set_get() {
        let mut field = MetricField::identity(3, 10);
        let mut custom = SiteMetric::identity(3);
        custom.set(0, 0, 2.5);
        field.set_site(5, &custom);
        let retrieved = field.get_site(5);
        assert!((retrieved.get(0, 0) - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_memory_bytes() {
        let field = MetricField::identity(8, 241);
        // 241 sites × 36 components × 4 bytes = 34,704 bytes
        assert_eq!(field.memory_bytes(), 241 * 36 * 4);
    }

    #[test]
    fn test_christoffel_flat_space() {
        let g = SiteMetric::identity(3);
        // All neighbors also have identity metric → Christoffel = 0
        let neighbor_metrics = vec![SiteMetric::identity(3); 6];
        let displacements = vec![
            vec![1.0, 0.0, 0.0],
            vec![-1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, -1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![0.0, 0.0, -1.0],
        ];
        let gamma = g.christoffel(&neighbor_metrics, &displacements);
        for val in &gamma {
            assert!(val.abs() < 1e-5, "Flat space Christoffel should be zero");
        }
    }
}
