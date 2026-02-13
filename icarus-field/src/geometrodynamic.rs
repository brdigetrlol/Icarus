//! Geometrodynamic Metric Learning for the Icarus EMC
//!
//! The metric tensor g_μν(x) at each lattice site evolves according to:
//!
//!   ∂g_μν/∂t = -α · δL/δg^μν + β · R_μν
//!
//! where:
//!   - α is the learning rate
//!   - L is the loss functional (e.g., free energy)
//!   - R_μν is the Ricci curvature tensor
//!   - β controls curvature-driven flow (normalized Ricci flow)
//!
//! The metric is kept positive-definite by eigenvalue pinning.

use crate::phase_field::LatticeField;
use icarus_math::metric::MetricField;

/// Parameters for geometrodynamic metric learning
#[derive(Debug, Clone)]
pub struct GeometrodynamicParams {
    /// Learning rate for loss-gradient descent
    pub alpha: f32,
    /// Ricci flow coefficient
    pub beta: f32,
    /// Minimum eigenvalue for positive-definiteness
    pub eigenvalue_floor: f32,
    /// Maximum eigenvalue to prevent explosion
    pub eigenvalue_ceiling: f32,
    /// Whether to enable Ricci flow regularization
    pub enable_ricci_flow: bool,
}

impl Default for GeometrodynamicParams {
    fn default() -> Self {
        Self {
            alpha: 0.001,
            beta: 0.01,
            eigenvalue_floor: 0.01,
            eigenvalue_ceiling: 100.0,
            enable_ricci_flow: true,
        }
    }
}

/// Geometrodynamic learner that evolves the metric tensor
#[derive(Debug, Clone)]
pub struct GeometrodynamicLearner {
    pub params: GeometrodynamicParams,
    /// Dimension of the metric tensor
    pub dim: usize,
    /// Gradient accumulator (packed symmetric, per site)
    grad_buffer: Vec<f32>,
}

impl GeometrodynamicLearner {
    /// Create a new learner
    pub fn new(params: GeometrodynamicParams, dim: usize, num_sites: usize) -> Self {
        let packed_size = dim * (dim + 1) / 2;
        Self {
            params,
            dim,
            grad_buffer: vec![0.0; num_sites * packed_size],
        }
    }

    /// Compute the metric gradient from the phase field.
    ///
    /// The gradient of the kinetic energy with respect to g^μν is:
    ///   δF_kin/δg^μν_i ≈ (1/K_i) Σ_j w_ij · e^μ_ij · e^ν_ij · |z_j - z_i|²
    ///
    /// This measures how much the metric at site i contributes to gradient energy.
    pub fn compute_gradient(
        &mut self,
        field: &LatticeField,
        _metric: &MetricField,
    ) {
        let packed_size = self.dim * (self.dim + 1) / 2;

        // Zero the gradient buffer
        for g in self.grad_buffer.iter_mut() {
            *g = 0.0;
        }

        for i in 0..field.num_sites {
            let zi_re = field.values_re[i];
            let zi_im = field.values_im[i];
            let start = field.neighbor_offsets[i] as usize;
            let end = field.neighbor_offsets[i + 1] as usize;
            let k = (end - start) as f32;
            if k == 0.0 {
                continue;
            }

            let grad_base = i * packed_size;

            for edge_idx in start..end {
                let j = field.neighbor_indices[edge_idx] as usize;

                // |z_j - z_i|²
                let dre = field.values_re[j] - zi_re;
                let dim = field.values_im[j] - zi_im;
                let diff_sq = dre * dre + dim * dim;

                // Displacement vector e_ij
                let disp_base = edge_idx * field.dim;
                let disp = &field.displacement_vectors[disp_base..disp_base + field.dim];

                // Accumulate outer product: e^μ_ij · e^ν_ij · |Δz|²
                for mu in 0..self.dim.min(field.dim) {
                    for nu in mu..self.dim.min(field.dim) {
                        let packed_idx = mu * (2 * self.dim - mu - 1) / 2 + nu;
                        if grad_base + packed_idx < self.grad_buffer.len() {
                            self.grad_buffer[grad_base + packed_idx] +=
                                disp[mu] * disp[nu] * diff_sq / k;
                        }
                    }
                }
            }
        }
    }

    /// Update the metric field using the computed gradient and optional Ricci flow.
    ///
    /// g_μν ← g_μν - α·grad + β·R_μν
    pub fn update_metric(
        &self,
        metric: &mut MetricField,
        field: &LatticeField,
    ) {
        let packed_size = self.dim * (self.dim + 1) / 2;
        let alpha = self.params.alpha;
        let beta = self.params.beta;

        for i in 0..field.num_sites {
            let grad_base = i * packed_size;

            // Update each component
            for p in 0..packed_size {
                let old = metric.data[i * packed_size + p];
                let grad = if grad_base + p < self.grad_buffer.len() {
                    self.grad_buffer[grad_base + p]
                } else {
                    0.0
                };

                // Simple gradient descent on the metric
                metric.data[i * packed_size + p] = old - alpha * grad;
            }

            // Optionally add Ricci flow: g += beta * R_μν
            // For the initial implementation, we use a simplified approach:
            // compute Ricci only when beta > 0 and Ricci flow is enabled
            if self.params.enable_ricci_flow && beta.abs() > 1e-12 {
                // For now, use a simplified regularization:
                // push the metric toward identity (Ricci flow on flat space)
                for mu in 0..self.dim {
                    let diag_idx = mu * (2 * self.dim - mu - 1) / 2 + mu;
                    if i * packed_size + diag_idx < metric.data.len() {
                        let current = metric.data[i * packed_size + diag_idx];
                        // Regularize toward identity: g_μμ → g_μμ + β*(1 - g_μμ)
                        metric.data[i * packed_size + diag_idx] = current + beta * (1.0 - current);
                    }
                }
            }

            // Pin eigenvalues for stability
            let mut site_metric = metric.get_site(i);
            site_metric.pin_eigenvalues(self.params.eigenvalue_floor);
            // Also cap eigenvalues
            for mu in 0..self.dim {
                let diag_idx = mu * (2 * self.dim - mu - 1) / 2 + mu;
                if site_metric.components[diag_idx] > self.params.eigenvalue_ceiling {
                    site_metric.components[diag_idx] = self.params.eigenvalue_ceiling;
                }
            }
            metric.set_site(i, &site_metric);
        }
    }

    /// Get the computed gradient buffer (call after `compute_gradient`).
    pub fn grad_buffer(&self) -> &[f32] {
        &self.grad_buffer
    }

    /// Perform one complete geometrodynamic step: compute gradient + update metric.
    pub fn step(&mut self, metric: &mut MetricField, field: &LatticeField) {
        self.compute_gradient(field, metric);
        self.update_metric(metric, field);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use icarus_math::lattice::Lattice;
    use icarus_math::lattice::e8::E8Lattice;

    fn make_test_setup() -> (LatticeField, MetricField) {
        let lattice = E8Lattice::new();
        let field = LatticeField::from_lattice(&lattice);
        let metric = MetricField::identity(lattice.dimension(), field.num_sites);
        (field, metric)
    }

    #[test]
    fn test_learner_creation() {
        let (field, _metric) = make_test_setup();
        let learner = GeometrodynamicLearner::new(
            GeometrodynamicParams::default(),
            8,
            field.num_sites,
        );
        assert_eq!(learner.dim, 8);
    }

    #[test]
    fn test_gradient_zero_field() {
        let (field, metric) = make_test_setup();
        let mut learner = GeometrodynamicLearner::new(
            GeometrodynamicParams::default(),
            8,
            field.num_sites,
        );

        learner.compute_gradient(&field, &metric);

        // Zero field → zero gradient (no difference between z_j and z_i)
        let total_grad: f32 = learner.grad_buffer.iter().map(|g| g.abs()).sum();
        assert!(total_grad < 1e-10, "Gradient should be zero for zero field");
    }

    #[test]
    fn test_metric_stays_positive_definite() {
        let (mut field, mut metric) = make_test_setup();
        field.init_random(42, 2.0); // Large amplitude to create gradients

        let mut learner = GeometrodynamicLearner::new(
            GeometrodynamicParams::default(),
            8,
            field.num_sites,
        );

        // Run several steps
        for _ in 0..10 {
            learner.step(&mut metric, &field);
        }

        // Check all diagonal elements are positive (positive-definite proxy)
        let dim = 8;
        let packed_size = dim * (dim + 1) / 2;
        for i in 0..field.num_sites {
            for mu in 0..dim {
                let diag_idx = mu * (2 * dim - mu - 1) / 2 + mu;
                let val = metric.data[i * packed_size + diag_idx];
                assert!(
                    val > 0.0,
                    "Metric diagonal [{},{}] at site {} is {}, should be positive",
                    mu,
                    mu,
                    i,
                    val
                );
            }
        }
    }

    #[test]
    fn test_identity_metric_stable_with_zero_field() {
        let (field, mut metric) = make_test_setup();
        let mut learner = GeometrodynamicLearner::new(
            GeometrodynamicParams::default(),
            8,
            field.num_sites,
        );

        let original_data = metric.data.clone();
        learner.step(&mut metric, &field);

        // With zero field and Ricci regularization toward identity,
        // the identity metric should be approximately stable
        let max_change: f32 = metric
            .data
            .iter()
            .zip(original_data.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_change < 0.1,
            "Identity metric should be stable with zero field, max change={}",
            max_change
        );
    }
}
