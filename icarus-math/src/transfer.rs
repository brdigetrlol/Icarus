//! Inter-Layer Transfer Operators for Icarus EMC
//!
//! Embedding (upward) and distillation (downward) operators that move
//! information between lattice layers in the multi-resolution hierarchy.
//!
//! Layer hierarchy (low→high dimensional):
//!   E8 (8D) ←→ Leech (24D) ←→ HCP (64D) ←→ Hypercubic (1024D)
//!
//! Embedding: low-D → high-D (inject structure into a richer space)
//! Distillation: high-D → low-D (compress patterns into essential features)

use crate::lattice::LatticeLayer;

/// Transfer operator between two lattice layers.
///
/// Stores a dense matrix W of shape (target_dim × source_dim) plus bias.
/// The matrix is learned alongside the metric tensor during geometrodynamic training.
#[derive(Debug, Clone)]
pub struct TransferOperator {
    /// Source layer
    pub source: LatticeLayer,
    /// Target layer
    pub target: LatticeLayer,
    /// Weight matrix (row-major, target_dim × source_dim)
    pub weights: Vec<f32>,
    /// Bias vector (target_dim)
    pub bias: Vec<f32>,
    /// Source dimensionality
    pub source_dim: usize,
    /// Target dimensionality
    pub target_dim: usize,
}

impl TransferOperator {
    /// Create a transfer operator initialized with a truncated identity mapping.
    /// For embedding (source_dim < target_dim): pads with zeros.
    /// For distillation (source_dim > target_dim): truncates.
    pub fn identity_init(source: LatticeLayer, target: LatticeLayer, source_dim: usize, target_dim: usize) -> Self {
        let mut weights = vec![0.0f32; target_dim * source_dim];
        let min_dim = source_dim.min(target_dim);
        for i in 0..min_dim {
            weights[i * source_dim + i] = 1.0;
        }
        Self {
            source,
            target,
            weights,
            bias: vec![0.0; target_dim],
            source_dim,
            target_dim,
        }
    }

    /// Create a random orthogonal-ish transfer operator (for initialization).
    /// Uses simple scaled random init: W_ij ~ N(0, 1/sqrt(source_dim)).
    pub fn random_init(
        source: LatticeLayer,
        target: LatticeLayer,
        source_dim: usize,
        target_dim: usize,
        seed: u64,
    ) -> Self {
        let n = target_dim * source_dim;
        let scale = 1.0 / (source_dim as f32).sqrt();
        let mut weights = Vec::with_capacity(n);

        // Simple LCG-based pseudo-random for deterministic init
        let mut state = seed;
        for _ in 0..n {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            // Convert to [-1, 1] range then scale
            let uniform = ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0;
            weights.push(uniform * scale);
        }

        Self {
            source,
            target,
            weights,
            bias: vec![0.0; target_dim],
            source_dim,
            target_dim,
        }
    }

    /// Apply the transfer: y = W * x + bias
    /// `source_re`, `source_im`: complex field at source sites (SoA layout)
    /// Returns (target_re, target_im) with target_dim elements per site
    pub fn apply(&self, source_re: &[f32], source_im: &[f32]) -> (Vec<f32>, Vec<f32>) {
        assert_eq!(source_re.len(), self.source_dim);
        assert_eq!(source_im.len(), self.source_dim);

        let mut target_re = vec![0.0f32; self.target_dim];
        let mut target_im = vec![0.0f32; self.target_dim];

        for i in 0..self.target_dim {
            let mut sum_re = self.bias[i];
            let mut sum_im = 0.0f32;
            for j in 0..self.source_dim {
                let w = self.weights[i * self.source_dim + j];
                sum_re += w * source_re[j];
                sum_im += w * source_im[j];
            }
            target_re[i] = sum_re;
            target_im[i] = sum_im;
        }

        (target_re, target_im)
    }

    /// Apply transpose (adjoint): y = W^T * x
    /// Used for gradient backpropagation and distillation
    pub fn apply_transpose(&self, target_re: &[f32], target_im: &[f32]) -> (Vec<f32>, Vec<f32>) {
        assert_eq!(target_re.len(), self.target_dim);
        assert_eq!(target_im.len(), self.target_dim);

        let mut source_re = vec![0.0f32; self.source_dim];
        let mut source_im = vec![0.0f32; self.source_dim];

        for j in 0..self.source_dim {
            let mut sum_re = 0.0f32;
            let mut sum_im = 0.0f32;
            for i in 0..self.target_dim {
                let w = self.weights[i * self.source_dim + j];
                sum_re += w * target_re[i];
                sum_im += w * target_im[i];
            }
            source_re[j] = sum_re;
            source_im[j] = sum_im;
        }

        (source_re, source_im)
    }

    /// Compute gradient of the weight matrix given source and target gradients.
    /// dW[i][j] = target_grad[i] * source[j] (outer product)
    pub fn weight_gradient(
        &self,
        source_re: &[f32],
        source_im: &[f32],
        target_grad_re: &[f32],
        target_grad_im: &[f32],
    ) -> Vec<f32> {
        let mut grad = vec![0.0f32; self.target_dim * self.source_dim];
        for i in 0..self.target_dim {
            for j in 0..self.source_dim {
                // Real part of complex outer product
                grad[i * self.source_dim + j] =
                    target_grad_re[i] * source_re[j] + target_grad_im[i] * source_im[j];
            }
        }
        grad
    }

    /// Update weights with gradient descent: W -= lr * dW
    pub fn update_weights(&mut self, gradient: &[f32], learning_rate: f32) {
        assert_eq!(gradient.len(), self.weights.len());
        for (w, g) in self.weights.iter_mut().zip(gradient.iter()) {
            *w -= learning_rate * g;
        }
    }

    /// Memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        (self.weights.len() + self.bias.len()) * std::mem::size_of::<f32>()
    }
}

/// The full set of inter-layer transfer operators
#[derive(Debug, Clone)]
pub struct InterLayerTransfers {
    /// E8 (8D) → Leech (24D) embedding
    pub e8_to_leech: TransferOperator,
    /// Leech (24D) → E8 (8D) distillation
    pub leech_to_e8: TransferOperator,
    /// Leech (24D) → HCP (64D) embedding
    pub leech_to_hcp: TransferOperator,
    /// HCP (64D) → Leech (24D) distillation
    pub hcp_to_leech: TransferOperator,
    /// HCP (64D) → Hypercubic (1024D) embedding
    pub hcp_to_hyper: TransferOperator,
    /// Hypercubic (1024D) → HCP (64D) distillation
    pub hyper_to_hcp: TransferOperator,
}

impl InterLayerTransfers {
    /// Create default identity-initialized transfers with standard dimensions.
    pub fn new() -> Self {
        Self::from_dims(&[
            (LatticeLayer::Analytical, 8),
            (LatticeLayer::Creative, 24),
            (LatticeLayer::Associative, 64),
            (LatticeLayer::Sensory, 1024),
        ])
    }

    /// Create transfers from actual layer dimensions.
    ///
    /// Accepts a slice of (LatticeLayer, dimension) pairs in hierarchy order.
    /// Builds forward (embedding) and reverse (distillation) operators for each
    /// adjacent pair. Missing pairs get identity-initialized zero-size operators.
    pub fn from_dims(layer_dims: &[(LatticeLayer, usize)]) -> Self {
        // Find dimensions for each layer type, defaulting to standard if absent
        let analytical_dim = layer_dims.iter()
            .find(|(l, _)| *l == LatticeLayer::Analytical)
            .map(|(_, d)| *d)
            .unwrap_or(8);
        let creative_dim = layer_dims.iter()
            .find(|(l, _)| *l == LatticeLayer::Creative)
            .map(|(_, d)| *d)
            .unwrap_or(24);
        let associative_dim = layer_dims.iter()
            .find(|(l, _)| *l == LatticeLayer::Associative)
            .map(|(_, d)| *d)
            .unwrap_or(64);
        let sensory_dim = layer_dims.iter()
            .find(|(l, _)| *l == LatticeLayer::Sensory)
            .map(|(_, d)| *d)
            .unwrap_or(1024);

        Self {
            e8_to_leech: TransferOperator::identity_init(
                LatticeLayer::Analytical, LatticeLayer::Creative,
                analytical_dim, creative_dim,
            ),
            leech_to_e8: TransferOperator::identity_init(
                LatticeLayer::Creative, LatticeLayer::Analytical,
                creative_dim, analytical_dim,
            ),
            leech_to_hcp: TransferOperator::identity_init(
                LatticeLayer::Creative, LatticeLayer::Associative,
                creative_dim, associative_dim,
            ),
            hcp_to_leech: TransferOperator::identity_init(
                LatticeLayer::Associative, LatticeLayer::Creative,
                associative_dim, creative_dim,
            ),
            hcp_to_hyper: TransferOperator::identity_init(
                LatticeLayer::Associative, LatticeLayer::Sensory,
                associative_dim, sensory_dim,
            ),
            hyper_to_hcp: TransferOperator::identity_init(
                LatticeLayer::Sensory, LatticeLayer::Associative,
                sensory_dim, associative_dim,
            ),
        }
    }

    /// Total memory in bytes for all transfer operators
    pub fn memory_bytes(&self) -> usize {
        self.e8_to_leech.memory_bytes()
            + self.leech_to_e8.memory_bytes()
            + self.leech_to_hcp.memory_bytes()
            + self.hcp_to_leech.memory_bytes()
            + self.hcp_to_hyper.memory_bytes()
            + self.hyper_to_hcp.memory_bytes()
    }
}

impl Default for InterLayerTransfers {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_embedding() {
        let op = TransferOperator::identity_init(
            LatticeLayer::Analytical, LatticeLayer::Creative, 8, 24,
        );
        let src_re = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let src_im = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let (dst_re, dst_im) = op.apply(&src_re, &src_im);

        assert_eq!(dst_re.len(), 24);
        assert_eq!(dst_im.len(), 24);
        // First 8 components should match source
        for i in 0..8 {
            assert!((dst_re[i] - src_re[i]).abs() < 1e-6);
            assert!((dst_im[i] - src_im[i]).abs() < 1e-6);
        }
        // Remaining 16 should be zero
        for i in 8..24 {
            assert!(dst_re[i].abs() < 1e-6);
            assert!(dst_im[i].abs() < 1e-6);
        }
    }

    #[test]
    fn test_identity_distillation() {
        let op = TransferOperator::identity_init(
            LatticeLayer::Creative, LatticeLayer::Analytical, 24, 8,
        );
        let mut src_re = vec![0.0; 24];
        let mut src_im = vec![0.0; 24];
        for i in 0..24 {
            src_re[i] = (i + 1) as f32;
            src_im[i] = (i + 1) as f32 * 0.1;
        }
        let (dst_re, dst_im) = op.apply(&src_re, &src_im);

        assert_eq!(dst_re.len(), 8);
        // Should take first 8 components
        for i in 0..8 {
            assert!((dst_re[i] - src_re[i]).abs() < 1e-6);
            assert!((dst_im[i] - src_im[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_transpose_adjoint() {
        let op = TransferOperator::identity_init(
            LatticeLayer::Analytical, LatticeLayer::Creative, 8, 24,
        );
        let target_re = vec![1.0; 24];
        let target_im = vec![0.0; 24];
        let (src_re, _src_im) = op.apply_transpose(&target_re, &target_im);

        assert_eq!(src_re.len(), 8);
        // Transpose of identity embedding should sum the first 8 target components
        for i in 0..8 {
            assert!((src_re[i] - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_weight_update() {
        let mut op = TransferOperator::identity_init(
            LatticeLayer::Analytical, LatticeLayer::Creative, 8, 24,
        );
        let initial_w = op.weights[0];
        let grad = vec![0.1; 24 * 8];
        op.update_weights(&grad, 0.01);
        assert!((op.weights[0] - (initial_w - 0.001)).abs() < 1e-6);
    }

    #[test]
    fn test_memory_bytes() {
        let transfers = InterLayerTransfers::new();
        let bytes = transfers.memory_bytes();
        // E8→Leech: (24*8 + 24)*4 = 864, Leech→E8: (8*24 + 8)*4 = 800, etc.
        assert!(bytes > 0);
    }

    #[test]
    fn test_random_init_deterministic() {
        let op1 = TransferOperator::random_init(
            LatticeLayer::Analytical, LatticeLayer::Creative, 8, 24, 42,
        );
        let op2 = TransferOperator::random_init(
            LatticeLayer::Analytical, LatticeLayer::Creative, 8, 24, 42,
        );
        for (a, b) in op1.weights.iter().zip(op2.weights.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }
}
