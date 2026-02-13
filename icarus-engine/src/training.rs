//! Training utilities for reservoir computing with the EMC.
//!
//! Provides ridge regression for closed-form linear readout training,
//! and a `ReservoirTrainer` that orchestrates the full pipeline:
//! warmup → encode → tick → collect → train.

use crate::encoding::InputEncoder;
use crate::readout::{FeatureMode, LinearReadout, StateCollector};
use crate::EmergentManifoldComputer;
use anyhow::{Context, Result};
use icarus_gpu::npu_client::NpuBridgeClient;

// ─── Ridge Regression ────────────────────────────────

/// Ridge regression: closed-form linear readout training.
///
/// Given state matrix `X` (N × D) and target matrix `Y` (N × K):
///
/// `W = (X^T·X + λI)^{-1} · X^T · Y`
///
/// Solved via Cholesky decomposition in f64 for numerical stability.
/// No backpropagation needed.
#[derive(Debug, Clone)]
pub struct RidgeRegression {
    /// Regularization strength (λ)
    pub lambda: f32,
}

impl RidgeRegression {
    pub fn new(lambda: f32) -> Self {
        Self { lambda }
    }

    /// Train a linear readout from collected states and targets.
    ///
    /// Returns `(weights_flat, bias)` where `weights_flat` is row-major `K × D`.
    ///
    /// States are centered (mean-subtracted) before regression.
    /// Bias is computed to account for centering.
    pub fn train(&self, states: &[Vec<f32>], targets: &[Vec<f32>]) -> (Vec<f32>, Vec<f32>) {
        assert!(!states.is_empty(), "Need at least one training sample");
        assert_eq!(
            states.len(),
            targets.len(),
            "States ({}) and targets ({}) must have equal length",
            states.len(),
            targets.len()
        );

        let n = states.len();
        let d = states[0].len();
        let k = targets[0].len();
        let n_f = n as f64;

        // Compute means for centering
        let mut state_mean = vec![0.0f64; d];
        let mut target_mean = vec![0.0f64; k];

        for s in states {
            for j in 0..d {
                state_mean[j] += s[j] as f64;
            }
        }
        for t in targets {
            for j in 0..k {
                target_mean[j] += t[j] as f64;
            }
        }
        for j in 0..d {
            state_mean[j] /= n_f;
        }
        for j in 0..k {
            target_mean[j] /= n_f;
        }

        // Compute per-dimension standard deviation for variance normalization.
        // Normalizing to unit variance prevents high-variance dimensions from
        // dominating the regression and dramatically improves generalization.
        let mut state_std = vec![0.0f64; d];
        for s in states {
            for j in 0..d {
                let diff = s[j] as f64 - state_mean[j];
                state_std[j] += diff * diff;
            }
        }
        for j in 0..d {
            state_std[j] = (state_std[j] / n_f).sqrt().max(1e-10);
        }

        // Compute X^T·X (D × D, symmetric) and X^T·Y (D × K)
        // States are centered AND variance-normalized: z = (x - μ) / σ
        let mut xtx = vec![0.0f64; d * d];
        let mut xty = vec![0.0f64; d * k];

        for i in 0..n {
            for a in 0..d {
                let xa = (states[i][a] as f64 - state_mean[a]) / state_std[a];
                // Upper triangle only for X^T·X (symmetric)
                for b in a..d {
                    let xb = (states[i][b] as f64 - state_mean[b]) / state_std[b];
                    xtx[a * d + b] += xa * xb;
                }
                for b in 0..k {
                    let yb = targets[i][b] as f64 - target_mean[b];
                    xty[a * k + b] += xa * yb;
                }
            }
        }

        // Symmetrize X^T·X
        for a in 0..d {
            for b in (a + 1)..d {
                xtx[b * d + a] = xtx[a * d + b];
            }
        }

        // Add regularization: X^T·X + λI
        let lambda_f64 = self.lambda as f64;
        for a in 0..d {
            xtx[a * d + a] += lambda_f64;
        }

        // Solve (X^T·X + λI) · W^T = X^T·Y via Cholesky
        let w_t = cholesky_solve(&xtx, &xty, d, k);

        // Transpose W^T (D × K) → W (K × D), converting to f32.
        // Un-scale by state_std to absorb normalization into the weights:
        //   W_eff[k,d] = W_normalized[k,d] / σ[d]
        // This makes the readout work on raw (unnormalized) states.
        let mut weights = vec![0.0f32; k * d];
        for a in 0..d {
            for b in 0..k {
                weights[b * d + a] = (w_t[a * k + b] / state_std[a]) as f32;
            }
        }

        // Bias: b = target_mean - W · state_mean
        // Uses un-scaled weights and original state_mean, so the formula is unchanged.
        let mut bias = vec![0.0f32; k];
        for b in 0..k {
            let mut dot = 0.0f64;
            for a in 0..d {
                dot += weights[b * d + a] as f64 * state_mean[a];
            }
            bias[b] = (target_mean[b] - dot) as f32;
        }

        (weights, bias)
    }

    /// Train and return a configured `LinearReadout`.
    pub fn train_readout(&self, collector: &StateCollector) -> LinearReadout {
        assert!(
            !collector.is_empty(),
            "StateCollector must have at least one sample"
        );

        let state_dim = collector.state_dim();
        let output_dim = collector.targets[0].len();

        let (weights, bias) = self.train(&collector.states, &collector.targets);

        LinearReadout::from_weights_with_mode(
            weights,
            bias,
            output_dim,
            state_dim,
            collector.feature_mode,
        )
    }

    /// Train using NPU-accelerated matmul for X^T·X and X^T·Y.
    ///
    /// The NPU performs the heavy O(N·D²) matrix multiplications in f32,
    /// then the Cholesky solve runs in f64 on CPU for numerical stability.
    /// Data is centered before NPU matmul to minimize f32 precision loss.
    pub fn train_npu(
        &self,
        states: &[Vec<f32>],
        targets: &[Vec<f32>],
        npu: &mut NpuBridgeClient,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        assert!(!states.is_empty(), "Need at least one training sample");
        assert_eq!(states.len(), targets.len());

        let n = states.len();
        let d = states[0].len();
        let k = targets[0].len();
        let n_f = n as f64;

        // Compute means for centering
        let mut state_mean = vec![0.0f64; d];
        let mut target_mean = vec![0.0f64; k];

        for s in states {
            for j in 0..d {
                state_mean[j] += s[j] as f64;
            }
        }
        for t in targets {
            for j in 0..k {
                target_mean[j] += t[j] as f64;
            }
        }
        for j in 0..d {
            state_mean[j] /= n_f;
        }
        for j in 0..k {
            target_mean[j] /= n_f;
        }

        // Compute per-dimension standard deviation for variance normalization
        let mut state_std = vec![0.0f64; d];
        for s in states {
            for j in 0..d {
                let diff = s[j] as f64 - state_mean[j];
                state_std[j] += diff * diff;
            }
        }
        for j in 0..d {
            state_std[j] = (state_std[j] / n_f).sqrt().max(1e-10);
        }

        // Build centered+normalized X (N×D) and centered Y (N×K) as contiguous f32 arrays
        let mut x_centered = vec![0.0f32; n * d];
        let mut y_centered = vec![0.0f32; n * k];

        for i in 0..n {
            for j in 0..d {
                x_centered[i * d + j] =
                    ((states[i][j] as f64 - state_mean[j]) / state_std[j]) as f32;
            }
            for j in 0..k {
                y_centered[i * k + j] = (targets[i][j] as f64 - target_mean[j]) as f32;
            }
        }

        // Transpose X to get X^T (D×N)
        let mut x_t = vec![0.0f32; d * n];
        for i in 0..n {
            for j in 0..d {
                x_t[j * n + i] = x_centered[i * d + j];
            }
        }

        // NPU matmul: X^T·X (D×N) × (N×D) → (D×D)
        let (xtx_f32, _dur1) = npu
            .matmul(d as u32, n as u32, d as u32, &x_t, &x_centered)
            .map_err(|e| anyhow::anyhow!("NPU matmul X^T·X failed: {}", e))?;

        // NPU matmul: X^T·Y (D×N) × (N×K) → (D×K)
        let (xty_f32, _dur2) = npu
            .matmul(d as u32, n as u32, k as u32, &x_t, &y_centered)
            .map_err(|e| anyhow::anyhow!("NPU matmul X^T·Y failed: {}", e))?;

        // Convert to f64 for Cholesky solve
        let mut xtx = vec![0.0f64; d * d];
        for i in 0..xtx_f32.len() {
            xtx[i] = xtx_f32[i] as f64;
        }
        let mut xty = vec![0.0f64; d * k];
        for i in 0..xty_f32.len() {
            xty[i] = xty_f32[i] as f64;
        }

        // Add regularization: X^T·X + λI
        let lambda_f64 = self.lambda as f64;
        for a in 0..d {
            xtx[a * d + a] += lambda_f64;
        }

        // Solve (X^T·X + λI) · W^T = X^T·Y via Cholesky
        let w_t = cholesky_solve(&xtx, &xty, d, k);

        // Transpose W^T (D × K) → W (K × D), un-scaling by state_std
        let mut weights = vec![0.0f32; k * d];
        for a in 0..d {
            for b in 0..k {
                weights[b * d + a] = (w_t[a * k + b] / state_std[a]) as f32;
            }
        }

        // Bias: b = target_mean - W · state_mean
        let mut bias = vec![0.0f32; k];
        for b in 0..k {
            let mut dot = 0.0f64;
            for a in 0..d {
                dot += weights[b * d + a] as f64 * state_mean[a];
            }
            bias[b] = (target_mean[b] - dot) as f32;
        }

        Ok((weights, bias))
    }

    /// Train a LinearReadout using NPU-accelerated matmul.
    pub fn train_readout_npu(
        &self,
        collector: &StateCollector,
        npu: &mut NpuBridgeClient,
    ) -> Result<LinearReadout> {
        assert!(!collector.is_empty());
        let state_dim = collector.state_dim();
        let output_dim = collector.targets[0].len();
        let (weights, bias) =
            self.train_npu(&collector.states, &collector.targets, npu)?;
        Ok(LinearReadout::from_weights_with_mode(weights, bias, output_dim, state_dim, collector.feature_mode))
    }

    /// Train with automatic lambda selection via k-fold cross-validation.
    ///
    /// Tests a logarithmic grid of lambdas and picks the one with lowest
    /// average NMSE across folds. Uses a fold-subtraction trick: pre-compute
    /// the full XTX/XTY, then subtract each test fold's contribution rather
    /// than recomputing from scratch.
    ///
    /// Returns `(weights, bias, selected_lambda)`.
    pub fn train_auto_lambda(
        states: &[Vec<f32>],
        targets: &[Vec<f32>],
        k_folds: usize,
    ) -> (Vec<f32>, Vec<f32>, f32) {
        assert!(!states.is_empty(), "Need at least one training sample");
        assert_eq!(states.len(), targets.len());
        let k_folds = k_folds.max(2);

        let n = states.len();
        let d = states[0].len();
        let k = targets[0].len();
        let n_f = n as f64;

        // ── 1. Pre-compute means and std (once for all data) ──
        let mut state_mean = vec![0.0f64; d];
        let mut target_mean = vec![0.0f64; k];
        for s in states {
            for j in 0..d {
                state_mean[j] += s[j] as f64;
            }
        }
        for t in targets {
            for j in 0..k {
                target_mean[j] += t[j] as f64;
            }
        }
        for j in 0..d {
            state_mean[j] /= n_f;
        }
        for j in 0..k {
            target_mean[j] /= n_f;
        }

        let mut state_std = vec![0.0f64; d];
        for s in states {
            for j in 0..d {
                let diff = s[j] as f64 - state_mean[j];
                state_std[j] += diff * diff;
            }
        }
        for j in 0..d {
            state_std[j] = (state_std[j] / n_f).sqrt().max(1e-10);
        }

        // ── 2. Pre-compute normalized X rows and centered Y rows ──
        let mut x_norm: Vec<Vec<f64>> = Vec::with_capacity(n);
        let mut y_cent: Vec<Vec<f64>> = Vec::with_capacity(n);
        for i in 0..n {
            let mut xr = vec![0.0f64; d];
            for j in 0..d {
                xr[j] = (states[i][j] as f64 - state_mean[j]) / state_std[j];
            }
            x_norm.push(xr);
            let mut yr = vec![0.0f64; k];
            for j in 0..k {
                yr[j] = targets[i][j] as f64 - target_mean[j];
            }
            y_cent.push(yr);
        }

        // ── 3. Pre-compute full XTX_all and XTY_all ──
        let mut xtx_all = vec![0.0f64; d * d];
        let mut xty_all = vec![0.0f64; d * k];
        for i in 0..n {
            let x = &x_norm[i];
            let y = &y_cent[i];
            for a in 0..d {
                for b in a..d {
                    xtx_all[a * d + b] += x[a] * x[b];
                }
                for b in 0..k {
                    xty_all[a * k + b] += x[a] * y[b];
                }
            }
        }
        // Symmetrize
        for a in 0..d {
            for b in (a + 1)..d {
                xtx_all[b * d + a] = xtx_all[a * d + b];
            }
        }

        // ── 4. Lambda grid ──
        let lambda_grid: Vec<f64> = vec![
            1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1, 1.0, 10.0,
            100.0, 1_000.0, 10_000.0, 100_000.0,
        ];

        // ── 5. K-fold cross-validation (contiguous folds) ──
        // Contiguous folds work well for reservoir computing because temporally
        // adjacent states are correlated. Contiguous splits force extrapolation,
        // giving a more honest estimate of generalization error.
        let fold_size = n / k_folds;
        let mut best_lambda = 1e-4f64;
        let mut best_cv_nmse = f64::MAX;

        for &lam in &lambda_grid {
            let mut total_nmse = 0.0f64;
            let mut valid_folds = 0usize;

            for fold in 0..k_folds {
                let test_start = fold * fold_size;
                let test_end = if fold == k_folds - 1 { n } else { test_start + fold_size };

                // Subtract test fold contribution from XTX_all and XTY_all
                let mut xtx_train = xtx_all.clone();
                let mut xty_train = xty_all.clone();
                for i in test_start..test_end {
                    let x = &x_norm[i];
                    let y = &y_cent[i];
                    for a in 0..d {
                        for b in 0..d {
                            xtx_train[a * d + b] -= x[a] * x[b];
                        }
                        for b in 0..k {
                            xty_train[a * k + b] -= x[a] * y[b];
                        }
                    }
                }

                // Add regularization
                for a in 0..d {
                    xtx_train[a * d + a] += lam;
                }

                // Solve for W^T on training fold
                let w_t = cholesky_solve(&xtx_train, &xty_train, d, k);

                // Evaluate on test fold (in normalized space)
                let n_test = test_end - test_start;
                if n_test == 0 {
                    continue;
                }
                let mut fold_se = 0.0f64;
                let mut fold_var = 0.0f64;
                // Compute test-fold target mean for NMSE denominator
                let mut test_target_mean = vec![0.0f64; k];
                for i in test_start..test_end {
                    for b in 0..k {
                        test_target_mean[b] += y_cent[i][b];
                    }
                }
                for b in 0..k {
                    test_target_mean[b] /= n_test as f64;
                }

                for i in test_start..test_end {
                    let x = &x_norm[i];
                    let y = &y_cent[i];
                    for b in 0..k {
                        let mut pred = 0.0f64;
                        for a in 0..d {
                            pred += w_t[a * k + b] * x[a];
                        }
                        let err = pred - y[b];
                        fold_se += err * err;
                        let dev = y[b] - test_target_mean[b];
                        fold_var += dev * dev;
                    }
                }

                let fold_nmse = if fold_var > 1e-12 {
                    fold_se / fold_var
                } else {
                    fold_se
                };
                total_nmse += fold_nmse;
                valid_folds += 1;
            }

            if valid_folds > 0 {
                let avg_nmse = total_nmse / valid_folds as f64;
                if avg_nmse < best_cv_nmse {
                    best_cv_nmse = avg_nmse;
                    best_lambda = lam;
                }
            }
        }

        // ── 6. Train final model with best lambda on all data ──
        let ridge = RidgeRegression::new(best_lambda as f32);
        let (weights, bias) = ridge.train(states, targets);

        (weights, bias, best_lambda as f32)
    }

    /// Train a configured `LinearReadout` with automatic lambda selection.
    pub fn train_readout_auto_lambda(
        collector: &StateCollector,
        k_folds: usize,
    ) -> (LinearReadout, f32) {
        assert!(
            !collector.is_empty(),
            "StateCollector must have at least one sample"
        );

        let state_dim = collector.state_dim();
        let output_dim = collector.targets[0].len();

        let (weights, bias, selected_lambda) =
            Self::train_auto_lambda(&collector.states, &collector.targets, k_folds);

        let readout = LinearReadout::from_weights_with_mode(
            weights,
            bias,
            output_dim,
            state_dim,
            collector.feature_mode,
        );

        (readout, selected_lambda)
    }
}

/// Solve A·X = B via Cholesky decomposition.
///
/// `A` is `d × d` symmetric positive definite (row-major).
/// `B` is `d × k` (row-major).
/// Returns `X` as `d × k` (row-major).
fn cholesky_solve(a: &[f64], b: &[f64], d: usize, k: usize) -> Vec<f64> {
    // Cholesky factorization: A = L · L^T
    let mut l = vec![0.0f64; d * d];

    for i in 0..d {
        for j in 0..=i {
            let mut sum = 0.0f64;
            for p in 0..j {
                sum += l[i * d + p] * l[j * d + p];
            }
            if i == j {
                let val = a[i * d + i] - sum;
                // Clamp to small positive for numerical stability
                l[i * d + j] = if val > 0.0 { val.sqrt() } else { 1e-10 };
            } else {
                l[i * d + j] = (a[i * d + j] - sum) / l[j * d + j];
            }
        }
    }

    // Forward substitution: L · y = B
    let mut y = vec![0.0f64; d * k];
    for i in 0..d {
        for col in 0..k {
            let mut sum = b[i * k + col];
            for j in 0..i {
                sum -= l[i * d + j] * y[j * k + col];
            }
            y[i * k + col] = sum / l[i * d + i];
        }
    }

    // Back substitution: L^T · x = y
    let mut x = vec![0.0f64; d * k];
    for i in (0..d).rev() {
        for col in 0..k {
            let mut sum = y[i * k + col];
            for j in (i + 1)..d {
                sum -= l[j * d + i] * x[j * k + col];
            }
            x[i * k + col] = sum / l[i * d + i];
        }
    }

    x
}

// ─── Online Ridge Regression (RLS) ─────────────────

/// Online ridge regression via Recursive Least Squares (RLS).
///
/// Uses Sherman-Morrison rank-1 updates for O(d²) per-sample cost,
/// compared to O(nd²) for batch ridge regression. Maintains an inverse
/// covariance matrix P = (X^T·X + λI)^{-1} and weight matrix W.
///
/// All internal computation uses f64 for numerical stability.
/// External API accepts and returns f32.
///
/// # Hybrid Strategy
///
/// Designed for hybrid use with batch ridge regression:
/// - Batch retrain periodically for stability (uses all collected data)
/// - Online RLS for immediate per-tick adaptation between retrains
/// - After batch retrain, call `warm_start()` to reset P and adopt new weights
pub struct OnlineRidgeRegression {
    /// Inverse covariance matrix P = (X^T·X + λI)^{-1}, d×d, row-major.
    p: Vec<f64>,
    /// Weight matrix W, k×d, row-major. W[j*d + i] = weight for output j, feature i.
    weights: Vec<f64>,
    /// Bias vector, k elements. Not updated online (from batch).
    bias: Vec<f64>,
    /// Input/feature dimension (d).
    dim: usize,
    /// Output dimension (k).
    output_dim: usize,
    /// Total number of rank-1 updates performed.
    num_updates: u64,
    /// Regularization parameter used for initialization.
    lambda: f64,
}

impl OnlineRidgeRegression {
    /// Create a new online ridge regression with zero weights.
    ///
    /// P is initialized to (1/λ)·I, representing the prior assumption
    /// that features are uncorrelated with variance 1/λ.
    pub fn new(dim: usize, output_dim: usize, lambda: f32) -> Self {
        let lam = (lambda as f64).max(1e-15);
        let inv_lambda = 1.0 / lam;

        let mut p = vec![0.0f64; dim * dim];
        for i in 0..dim {
            p[i * dim + i] = inv_lambda;
        }

        Self {
            p,
            weights: vec![0.0f64; output_dim * dim],
            bias: vec![0.0f64; output_dim],
            dim,
            output_dim,
            num_updates: 0,
            lambda: lam,
        }
    }

    /// Create from a trained LinearReadout (warm-start).
    ///
    /// Copies weights and bias from the readout, initializes P to (1/λ)·I.
    pub fn from_readout(readout: &LinearReadout, lambda: f32) -> Self {
        let mut rls = Self::new(readout.state_dim, readout.output_dim, lambda);
        for i in 0..rls.weights.len() {
            rls.weights[i] = readout.weights[i] as f64;
        }
        for i in 0..rls.bias.len() {
            rls.bias[i] = readout.bias[i] as f64;
        }
        rls
    }

    /// Warm-start from batch ridge regression weights.
    ///
    /// Copies weights and bias from a LinearReadout, resets P to (1/λ)·I.
    /// Reuses existing P allocation. Use after a batch retrain to continue
    /// online updates from the batch-optimized weights.
    pub fn warm_start(&mut self, readout: &LinearReadout, lambda: f32) {
        assert_eq!(readout.state_dim, self.dim);
        assert_eq!(readout.output_dim, self.output_dim);

        let lam = (lambda as f64).max(1e-15);
        let inv_lambda = 1.0 / lam;

        for i in 0..self.weights.len() {
            self.weights[i] = readout.weights[i] as f64;
        }
        for i in 0..self.bias.len() {
            self.bias[i] = readout.bias[i] as f64;
        }

        self.p.fill(0.0);
        for i in 0..self.dim {
            self.p[i * self.dim + i] = inv_lambda;
        }

        self.lambda = lam;
        self.num_updates = 0;
    }

    /// Perform a rank-1 Sherman-Morrison update with a new (features, target) pair.
    ///
    /// Update steps (all in f64):
    /// 1. g = P·x — gain direction
    /// 2. α = 1 / (1 + x^T·g) — normalization scalar
    /// 3. k = α·g — Kalman gain
    /// 4. error = y - W·x - bias — prediction error (pre-update)
    /// 5. W += error ⊗ k^T — weight update (rank-1 outer product)
    /// 6. P -= k · g^T — covariance update (rank-1 downdate)
    ///
    /// Cost: O(d² + k·d) where d = feature dim, k = output dim.
    pub fn update(&mut self, features: &[f32], target: &[f32]) {
        assert_eq!(features.len(), self.dim);
        assert_eq!(target.len(), self.output_dim);

        let d = self.dim;
        let k = self.output_dim;

        // Convert to f64
        let x: Vec<f64> = features.iter().map(|&v| v as f64).collect();
        let y: Vec<f64> = target.iter().map(|&v| v as f64).collect();

        // Step 1: g = P · x
        let mut g = vec![0.0f64; d];
        for i in 0..d {
            let mut sum = 0.0f64;
            let row_base = i * d;
            for j in 0..d {
                sum += self.p[row_base + j] * x[j];
            }
            g[i] = sum;
        }

        // Step 2: α = 1 / (1 + x^T · g)
        let mut xtg = 0.0f64;
        for i in 0..d {
            xtg += x[i] * g[i];
        }
        let alpha = 1.0 / (1.0 + xtg);

        // Step 3: k = α · g (Kalman gain)
        let mut kalman = vec![0.0f64; d];
        for i in 0..d {
            kalman[i] = alpha * g[i];
        }

        // Step 4: error = y - W·x - bias
        let mut error = vec![0.0f64; k];
        for j in 0..k {
            let mut pred = self.bias[j];
            let row_base = j * d;
            for i in 0..d {
                pred += self.weights[row_base + i] * x[i];
            }
            error[j] = y[j] - pred;
        }

        // Step 5: W += error ⊗ k^T
        for j in 0..k {
            let row_base = j * d;
            let err_j = error[j];
            for i in 0..d {
                self.weights[row_base + i] += err_j * kalman[i];
            }
        }

        // Step 6: P -= k · g^T (symmetric rank-1 downdate)
        for i in 0..d {
            let row_base = i * d;
            let ki = kalman[i];
            for j in 0..d {
                self.p[row_base + j] -= ki * g[j];
            }
        }

        self.num_updates += 1;

        // Enforce symmetry every 64 updates to prevent numerical drift
        if self.num_updates % 64 == 0 {
            self.enforce_symmetry();
        }
    }

    /// Predict output from features: y = W·x + bias.
    pub fn predict(&self, features: &[f32]) -> Vec<f32> {
        assert_eq!(features.len(), self.dim);

        let mut output = vec![0.0f32; self.output_dim];
        for j in 0..self.output_dim {
            let mut sum = self.bias[j];
            let row_base = j * self.dim;
            for i in 0..self.dim {
                sum += self.weights[row_base + i] * features[i] as f64;
            }
            output[j] = sum as f32;
        }
        output
    }

    /// Convert to a LinearReadout for use with the ensemble prediction pipeline.
    pub fn to_readout(&self, feature_mode: FeatureMode) -> LinearReadout {
        let weights_f32: Vec<f32> = self.weights.iter().map(|&w| w as f32).collect();
        let bias_f32: Vec<f32> = self.bias.iter().map(|&b| b as f32).collect();

        LinearReadout::from_weights_with_mode(
            weights_f32,
            bias_f32,
            self.output_dim,
            self.dim,
            feature_mode,
        )
    }

    /// Total number of online updates performed.
    pub fn num_updates(&self) -> u64 {
        self.num_updates
    }

    /// Feature dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Output dimension.
    pub fn output_dim(&self) -> usize {
        self.output_dim
    }

    /// Current regularization lambda.
    pub fn lambda(&self) -> f32 {
        self.lambda as f32
    }

    /// Enforce symmetry of P by averaging upper and lower triangles.
    fn enforce_symmetry(&mut self) {
        let d = self.dim;
        for i in 0..d {
            for j in (i + 1)..d {
                let avg = (self.p[i * d + j] + self.p[j * d + i]) * 0.5;
                self.p[i * d + j] = avg;
                self.p[j * d + i] = avg;
            }
        }
    }
}

// ─── Reservoir Trainer ───────────────────────────────

/// Reservoir training orchestrator.
///
/// Runs the EMC on input sequences, collects reservoir states, and trains
/// a linear readout via ridge regression.
///
/// Pipeline:
/// 1. Warmup: Run EMC for `warmup_ticks` to let transients decay
/// 2. Drive: For each input, encode → tick × `ticks_per_input` → collect state
/// 3. Train: Ridge regression on collected (state, target) pairs
#[derive(Debug, Clone)]
pub struct ReservoirTrainer {
    /// Ridge regression regularization strength
    pub lambda: f32,
    /// Number of warmup ticks before state collection
    pub warmup_ticks: u64,
    /// Number of EMC ticks per input sample
    pub ticks_per_input: u64,
    /// Feature extraction mode for state collection
    pub feature_mode: FeatureMode,
    /// Leaking rate for encoding: blends new input with existing state.
    /// 1.0 = full overwrite (default, backward compatible).
    /// Lower values preserve reservoir memory across input steps.
    pub leak_rate: f32,
}

impl Default for ReservoirTrainer {
    fn default() -> Self {
        Self {
            lambda: 1e-4,
            warmup_ticks: 10,
            ticks_per_input: 1,
            feature_mode: FeatureMode::Linear,
            leak_rate: 1.0,
        }
    }
}

impl ReservoirTrainer {
    pub fn new(lambda: f32, warmup_ticks: u64, ticks_per_input: u64) -> Self {
        Self {
            lambda,
            warmup_ticks,
            ticks_per_input,
            feature_mode: FeatureMode::Linear,
            leak_rate: 1.0,
        }
    }

    /// Create a trainer with nonlinear feature extraction.
    pub fn with_feature_mode(mut self, mode: FeatureMode) -> Self {
        self.feature_mode = mode;
        self
    }

    /// Set the leaking rate for encoding.
    pub fn with_leak_rate(mut self, rate: f32) -> Self {
        self.leak_rate = rate.clamp(0.0, 1.0);
        self
    }

    /// Drive the EMC with input data and collect reservoir states.
    ///
    /// - Runs warmup ticks first (using `inputs[0]` to seed the reservoir)
    /// - For each (input, target) pair: encode → tick → collect state
    /// - Returns a `StateCollector` ready for training
    pub fn collect_states(
        &self,
        emc: &mut EmergentManifoldComputer,
        encoder: &dyn InputEncoder,
        inputs: &[Vec<f32>],
        targets: &[Vec<f32>],
        layer_idx: usize,
    ) -> Result<StateCollector> {
        assert_eq!(
            inputs.len(),
            targets.len(),
            "Inputs and targets must have equal length"
        );

        let layer_count = emc.manifold.layers.len();
        if layer_idx >= layer_count {
            anyhow::bail!(
                "Layer index {} out of range (have {} layers)",
                layer_idx,
                layer_count
            );
        }

        // Warmup: encode first input and let transients decay
        if !inputs.is_empty() {
            encoder.encode(&inputs[0], &mut emc.manifold.layers[layer_idx].field);
            emc.run(self.warmup_ticks)
                .context("Warmup ticks failed")?;
        }

        // Drive and collect
        let mut collector = StateCollector::with_mode(self.feature_mode);
        let use_leaky = self.leak_rate < 1.0;
        for (input, target) in inputs.iter().zip(targets.iter()) {
            if use_leaky {
                encoder.encode_leaky(input, &mut emc.manifold.layers[layer_idx].field, self.leak_rate);
            } else {
                encoder.encode(input, &mut emc.manifold.layers[layer_idx].field);
            }
            emc.run(self.ticks_per_input)
                .context("Tick during collection failed")?;
            collector.collect(&emc.manifold.layers[layer_idx].field, target.clone());
        }

        Ok(collector)
    }

    /// Full training pipeline: collect states → ridge regression → linear readout.
    pub fn train(
        &self,
        emc: &mut EmergentManifoldComputer,
        encoder: &dyn InputEncoder,
        inputs: &[Vec<f32>],
        targets: &[Vec<f32>],
        layer_idx: usize,
    ) -> Result<LinearReadout> {
        let collector = self.collect_states(emc, encoder, inputs, targets, layer_idx)?;
        let ridge = RidgeRegression::new(self.lambda);
        Ok(ridge.train_readout(&collector))
    }

    /// Full training pipeline with automatic lambda selection via k-fold CV.
    ///
    /// Returns `(readout, selected_lambda)`.
    pub fn train_auto(
        &self,
        emc: &mut EmergentManifoldComputer,
        encoder: &dyn InputEncoder,
        inputs: &[Vec<f32>],
        targets: &[Vec<f32>],
        layer_idx: usize,
        k_folds: usize,
    ) -> Result<(LinearReadout, f32)> {
        let collector = self.collect_states(emc, encoder, inputs, targets, layer_idx)?;
        Ok(RidgeRegression::train_readout_auto_lambda(&collector, k_folds))
    }
}

// ─── Evaluation utilities ────────────────────────────

/// Compute Normalized Mean Squared Error (NMSE).
///
/// `NMSE = MSE(predicted, actual) / variance(actual)`
///
/// Lower is better. NMSE < 1 means the model beats predicting the mean.
pub fn nmse(predicted: &[f32], actual: &[f32]) -> f32 {
    assert_eq!(predicted.len(), actual.len());
    let n = predicted.len() as f32;

    let mean: f32 = actual.iter().sum::<f32>() / n;
    let variance: f32 = actual.iter().map(|&y| (y - mean) * (y - mean)).sum::<f32>() / n;

    if variance < 1e-12 {
        return 0.0;
    }

    let mse: f32 = predicted
        .iter()
        .zip(actual.iter())
        .map(|(&p, &a)| (p - a) * (p - a))
        .sum::<f32>()
        / n;

    mse / variance
}

/// Compute classification accuracy.
///
/// Compares argmax of each prediction row against the integer labels.
pub fn accuracy(predictions: &[Vec<f32>], labels: &[usize]) -> f32 {
    assert_eq!(predictions.len(), labels.len());
    let correct = predictions
        .iter()
        .zip(labels.iter())
        .filter(|(pred, &label)| {
            let argmax = pred
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            argmax == label
        })
        .count();

    correct as f32 / predictions.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ManifoldConfig;
    use crate::encoding::SpatialEncoder;

    // ─── Ridge Regression ───

    #[test]
    fn test_ridge_identity() {
        // Train on y = x (1D → 1D) with 4 samples
        // States are 1D (simplified), targets are same as states
        let states: Vec<Vec<f32>> = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
        let targets: Vec<Vec<f32>> = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]];

        let ridge = RidgeRegression::new(1e-6);
        let (weights, bias) = ridge.train(&states, &targets);

        // Should learn w ≈ 1.0, b ≈ 0.0
        assert!(
            (weights[0] - 1.0).abs() < 0.01,
            "Weight should be ~1.0, got {}",
            weights[0]
        );
        assert!(
            bias[0].abs() < 0.01,
            "Bias should be ~0.0, got {}",
            bias[0]
        );
    }

    #[test]
    fn test_ridge_linear_function() {
        // Train on y = 2x + 3
        let states: Vec<Vec<f32>> = (0..100).map(|i| vec![i as f32 * 0.1]).collect();
        let targets: Vec<Vec<f32>> = states
            .iter()
            .map(|s| vec![2.0 * s[0] + 3.0])
            .collect();

        let ridge = RidgeRegression::new(1e-6);
        let (weights, bias) = ridge.train(&states, &targets);

        assert!(
            (weights[0] - 2.0).abs() < 0.05,
            "Weight should be ~2.0, got {}",
            weights[0]
        );
        assert!(
            (bias[0] - 3.0).abs() < 0.05,
            "Bias should be ~3.0, got {}",
            bias[0]
        );
    }

    #[test]
    fn test_ridge_multi_output() {
        // Train on y0 = x0 + x1, y1 = x0 - x1
        let states: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![2.0, 1.0],
            vec![1.0, 2.0],
            vec![3.0, 0.5],
        ];
        let targets: Vec<Vec<f32>> = states
            .iter()
            .map(|s| vec![s[0] + s[1], s[0] - s[1]])
            .collect();

        let ridge = RidgeRegression::new(1e-6);
        let (weights, _bias) = ridge.train(&states, &targets);

        // weights should be approximately [[1, 1], [1, -1]]
        // weights[0*2 + 0] = W[0][0] ≈ 1, weights[0*2 + 1] = W[0][1] ≈ 1
        // weights[1*2 + 0] = W[1][0] ≈ 1, weights[1*2 + 1] = W[1][1] ≈ -1
        assert!(
            (weights[0] - 1.0).abs() < 0.1,
            "W[0][0] should be ~1.0, got {}",
            weights[0]
        );
        assert!(
            (weights[1] - 1.0).abs() < 0.1,
            "W[0][1] should be ~1.0, got {}",
            weights[1]
        );
        assert!(
            (weights[2] - 1.0).abs() < 0.1,
            "W[1][0] should be ~1.0, got {}",
            weights[2]
        );
        assert!(
            (weights[3] - (-1.0)).abs() < 0.1,
            "W[1][1] should be ~-1.0, got {}",
            weights[3]
        );
    }

    #[test]
    fn test_ridge_regularization() {
        // With very high lambda, weights should be small
        let states: Vec<Vec<f32>> = vec![vec![1.0], vec![2.0], vec![3.0]];
        let targets: Vec<Vec<f32>> = vec![vec![100.0], vec![200.0], vec![300.0]];

        let ridge_low = RidgeRegression::new(1e-6);
        let (w_low, _) = ridge_low.train(&states, &targets);

        let ridge_high = RidgeRegression::new(1e6);
        let (w_high, _) = ridge_high.train(&states, &targets);

        assert!(
            w_high[0].abs() < w_low[0].abs(),
            "High regularization should shrink weights: {} vs {}",
            w_high[0],
            w_low[0]
        );
    }

    #[test]
    fn test_ridge_train_readout() {
        let mut collector = StateCollector::new();
        // Simulate 3 samples with state_dim=4 (2 sites × 2 for re+im)
        collector.states.push(vec![1.0, 0.0, 0.5, 0.0]);
        collector.targets.push(vec![1.0]);
        collector.states.push(vec![0.0, 1.0, 0.0, 0.5]);
        collector.targets.push(vec![0.0]);
        collector.states.push(vec![0.5, 0.5, 0.25, 0.25]);
        collector.targets.push(vec![0.5]);

        let ridge = RidgeRegression::new(1e-4);
        let readout = ridge.train_readout(&collector);

        assert_eq!(readout.output_dim, 1);
        assert_eq!(readout.state_dim, 4);
    }

    // ─── Evaluation ───

    #[test]
    fn test_nmse_perfect() {
        let actual = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let predicted = actual.clone();
        assert!(nmse(&predicted, &actual) < 1e-6);
    }

    #[test]
    fn test_nmse_mean_predictor() {
        let actual = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mean = 3.0;
        let predicted = vec![mean; 5];
        let score = nmse(&predicted, &actual);
        assert!(
            (score - 1.0).abs() < 1e-5,
            "Mean predictor should have NMSE ≈ 1.0, got {}",
            score
        );
    }

    #[test]
    fn test_accuracy() {
        let predictions = vec![
            vec![0.9, 0.1], // class 0
            vec![0.2, 0.8], // class 1
            vec![0.6, 0.4], // class 0
        ];
        let labels = vec![0, 1, 1];

        let acc = accuracy(&predictions, &labels);
        assert!(
            (acc - 2.0 / 3.0).abs() < 1e-5,
            "Accuracy should be 2/3, got {}",
            acc
        );
    }

    // ─── Reservoir Trainer (integration) ───

    #[test]
    fn test_reservoir_trainer_collects_states() {
        let config = ManifoldConfig::e8_only();
        let mut emc = EmergentManifoldComputer::new_cpu(config);
        emc.init_random(42, 0.5);

        let encoder = SpatialEncoder::default();
        let trainer = ReservoirTrainer::new(1e-4, 5, 1);

        let inputs: Vec<Vec<f32>> = (0..10).map(|i| vec![i as f32 * 0.1; 8]).collect();
        let targets: Vec<Vec<f32>> = (0..10).map(|i| vec![i as f32 * 0.2]).collect();

        let collector = trainer
            .collect_states(&mut emc, &encoder, &inputs, &targets, 0)
            .unwrap();

        assert_eq!(collector.len(), 10);
        assert_eq!(collector.state_dim(), 2 * 241); // 241 sites × (re + im)
    }

    #[test]
    fn test_reservoir_trainer_full_pipeline() {
        let config = ManifoldConfig::e8_only();
        let mut emc = EmergentManifoldComputer::new_cpu(config);
        emc.init_random(42, 0.5);

        let encoder = SpatialEncoder::default();
        let trainer = ReservoirTrainer::new(1e-2, 5, 1);

        // Simple training data: 20 samples
        let inputs: Vec<Vec<f32>> = (0..20).map(|i| vec![(i as f32) * 0.05; 8]).collect();
        let targets: Vec<Vec<f32>> = (0..20).map(|i| vec![(i as f32) * 0.1]).collect();

        let readout = trainer
            .train(&mut emc, &encoder, &inputs, &targets, 0)
            .unwrap();

        assert_eq!(readout.output_dim, 1);
        assert_eq!(readout.state_dim, 2 * 241);
        // Weights should be non-trivial
        let w_norm: f32 = readout.weights.iter().map(|w| w * w).sum::<f32>().sqrt();
        assert!(w_norm > 1e-6, "Trained weights should be non-zero");
    }

    #[test]
    fn test_reservoir_trainer_layer_out_of_range() {
        let config = ManifoldConfig::e8_only();
        let mut emc = EmergentManifoldComputer::new_cpu(config);
        emc.init_random(42, 0.5);

        let encoder = SpatialEncoder::default();
        let trainer = ReservoirTrainer::default();

        let result = trainer.collect_states(&mut emc, &encoder, &[vec![1.0]], &[vec![1.0]], 5);
        assert!(result.is_err());
    }

    // ─── Auto-Lambda ───

    #[test]
    fn test_auto_lambda_learns_linear() {
        // Auto-lambda on a clean linear problem should produce good weights
        let states: Vec<Vec<f32>> = (0..100).map(|i| vec![i as f32 * 0.1]).collect();
        let targets: Vec<Vec<f32>> = states.iter().map(|s| vec![2.0 * s[0] + 3.0]).collect();

        let (weights, bias, lambda) = RidgeRegression::train_auto_lambda(&states, &targets, 5);

        assert!(
            (weights[0] - 2.0).abs() < 0.1,
            "Weight should be ~2.0, got {}",
            weights[0]
        );
        assert!(
            (bias[0] - 3.0).abs() < 0.1,
            "Bias should be ~3.0, got {}",
            bias[0]
        );
        assert!(
            lambda < 1.0,
            "Lambda should be small for clean linear data, got {}",
            lambda
        );
    }

    #[test]
    fn test_auto_lambda_picks_higher_for_overparameterized() {
        // Many features (50) but few samples (20) → should pick higher lambda
        // than a well-conditioned problem (1 feature, 100 samples)
        let n_features = 50;
        let n_samples = 20;

        // Generate overparameterized data: random-ish features, linear target on first feature
        let states: Vec<Vec<f32>> = (0..n_samples)
            .map(|i| {
                (0..n_features)
                    .map(|j| {
                        let v = ((i * 37 + j * 13 + 7) % 100) as f32 / 100.0;
                        v
                    })
                    .collect()
            })
            .collect();
        let targets: Vec<Vec<f32>> = states.iter().map(|s| vec![s[0] * 2.0 + 1.0]).collect();

        let (_w_over, _b_over, lambda_over) =
            RidgeRegression::train_auto_lambda(&states, &targets, 5);

        // Well-conditioned: 1 feature, 100 samples
        let states_good: Vec<Vec<f32>> = (0..100).map(|i| vec![i as f32 * 0.1]).collect();
        let targets_good: Vec<Vec<f32>> =
            states_good.iter().map(|s| vec![s[0] * 2.0 + 1.0]).collect();

        let (_w_good, _b_good, lambda_good) =
            RidgeRegression::train_auto_lambda(&states_good, &targets_good, 5);

        assert!(
            lambda_over >= lambda_good,
            "Overparameterized should need more regularization: {} vs {}",
            lambda_over,
            lambda_good
        );
    }

    #[test]
    fn test_auto_lambda_matches_manual_on_clean_data() {
        // On clean linear data, auto-lambda result should be close to manual with small lambda
        let states: Vec<Vec<f32>> = (0..50).map(|i| vec![i as f32 * 0.1, (i as f32 * 0.05).sin()]).collect();
        let targets: Vec<Vec<f32>> = states.iter().map(|s| vec![s[0] + s[1]]).collect();

        let (w_auto, b_auto, _lambda) =
            RidgeRegression::train_auto_lambda(&states, &targets, 5);
        let ridge = RidgeRegression::new(1e-4);
        let (w_manual, b_manual) = ridge.train(&states, &targets);

        // Both should produce reasonable predictions
        let mut auto_err = 0.0f32;
        let mut manual_err = 0.0f32;
        for (s, t) in states.iter().zip(targets.iter()) {
            let pred_auto = w_auto[0] * s[0] + w_auto[1] * s[1] + b_auto[0];
            let pred_manual = w_manual[0] * s[0] + w_manual[1] * s[1] + b_manual[0];
            auto_err += (pred_auto - t[0]).powi(2);
            manual_err += (pred_manual - t[0]).powi(2);
        }
        auto_err /= states.len() as f32;
        manual_err /= states.len() as f32;

        // Auto should be at least as good (or very close)
        assert!(
            auto_err < manual_err * 2.0 + 0.01,
            "Auto-lambda train error ({}) should be comparable to manual ({})",
            auto_err,
            manual_err
        );
    }

    #[test]
    fn test_auto_lambda_readout() {
        let mut collector = StateCollector::new();
        for i in 0..30 {
            let x = i as f32 / 30.0;
            collector.states.push(vec![x, x * x, (x * 3.0).sin(), 1.0 - x]);
            collector.targets.push(vec![x * 2.0]);
        }

        let (readout, lambda) = RidgeRegression::train_readout_auto_lambda(&collector, 5);

        assert_eq!(readout.output_dim, 1);
        assert_eq!(readout.state_dim, 4);
        assert!(lambda > 0.0, "Lambda should be positive, got {}", lambda);
    }

    // ─── Online RLS ───

    #[test]
    fn test_online_rls_identity() {
        // Learn y = x (1D → 1D) with online updates
        let mut rls = OnlineRidgeRegression::new(1, 1, 1e-6);

        for i in 0..100 {
            let x = (i as f32) * 0.1;
            rls.update(&[x], &[x]);
        }

        // Should converge to w ≈ 1.0, bias stays 0
        let pred = rls.predict(&[5.0]);
        assert!(
            (pred[0] - 5.0).abs() < 0.5,
            "Online RLS identity prediction should be ~5.0, got {}",
            pred[0]
        );
        assert_eq!(rls.num_updates(), 100);
    }

    #[test]
    fn test_online_rls_linear_function() {
        // Learn y = 2x + 3 (bias not updated online, so this tests weight learning)
        // Since bias is zero and not updated, the RLS will learn an approximation
        // through the weight alone on centered-ish data
        let mut rls = OnlineRidgeRegression::new(1, 1, 1e-6);

        for i in 0..200 {
            let x = (i as f32) * 0.05;
            let y = 2.0 * x + 3.0;
            rls.update(&[x], &[y]);
        }

        // Predict at x=5: true = 13.0
        // Without bias update, RLS absorbs the offset into the weight
        let pred = rls.predict(&[5.0]);
        // Should be reasonably close (offset gets baked into weight for this input range)
        assert!(
            pred[0] > 5.0,
            "Should predict a positive value for positive input, got {}",
            pred[0]
        );
    }

    #[test]
    fn test_online_rls_multi_output() {
        // Learn y0 = x0 + x1, y1 = x0 - x1
        let mut rls = OnlineRidgeRegression::new(2, 2, 1e-6);

        for i in 0..200 {
            let x0 = ((i * 7 + 3) % 100) as f32 / 100.0;
            let x1 = ((i * 13 + 11) % 100) as f32 / 100.0;
            rls.update(&[x0, x1], &[x0 + x1, x0 - x1]);
        }

        let pred = rls.predict(&[0.6, 0.4]);
        assert!(
            (pred[0] - 1.0).abs() < 0.2,
            "y0 = 0.6 + 0.4 = 1.0, got {}",
            pred[0]
        );
        assert!(
            (pred[1] - 0.2).abs() < 0.2,
            "y1 = 0.6 - 0.4 = 0.2, got {}",
            pred[1]
        );
    }

    #[test]
    fn test_online_rls_warm_start() {
        // Train batch, warm-start online RLS, verify predictions match
        let states: Vec<Vec<f32>> = (0..50)
            .map(|i| vec![i as f32 * 0.1, (i as f32 * 0.05).sin()])
            .collect();
        let targets: Vec<Vec<f32>> = states.iter().map(|s| vec![s[0] + s[1]]).collect();

        let ridge = RidgeRegression::new(1e-4);
        let (weights, bias) = ridge.train(&states, &targets);
        let readout = LinearReadout::from_weights(weights, bias, 1, 2);

        // Batch prediction
        let batch_pred = readout.predict_from_raw_state(&[3.0, 0.5]);

        // Warm-start online RLS from batch
        let rls = OnlineRidgeRegression::from_readout(&readout, 1e-4);
        let online_pred = rls.predict(&[3.0, 0.5]);

        assert!(
            (batch_pred[0] - online_pred[0]).abs() < 1e-4,
            "Warm-started online RLS should match batch: {} vs {}",
            batch_pred[0],
            online_pred[0]
        );
    }

    #[test]
    fn test_online_rls_warm_start_then_update() {
        // Warm-start from batch, then continue online updates
        let states: Vec<Vec<f32>> = (0..30).map(|i| vec![i as f32 * 0.1]).collect();
        let targets: Vec<Vec<f32>> = states.iter().map(|s| vec![s[0] * 2.0]).collect();

        let ridge = RidgeRegression::new(1e-4);
        let (weights, bias) = ridge.train(&states, &targets);
        let readout = LinearReadout::from_weights(weights, bias, 1, 1);

        let mut rls = OnlineRidgeRegression::from_readout(&readout, 1e-4);

        // Continue with more data from the same distribution
        for i in 30..60 {
            let x = i as f32 * 0.1;
            rls.update(&[x], &[x * 2.0]);
        }

        // Should still predict well
        let pred = rls.predict(&[5.0]);
        assert!(
            (pred[0] - 10.0).abs() < 1.0,
            "After warm-start + updates, should predict ~10.0, got {}",
            pred[0]
        );
        assert_eq!(rls.num_updates(), 30);
    }

    #[test]
    fn test_online_rls_to_readout() {
        let mut rls = OnlineRidgeRegression::new(4, 2, 1e-4);

        // Feed some data
        for i in 0..20 {
            let x = vec![i as f32 * 0.1, 0.0, 0.0, 0.0];
            let y = vec![i as f32 * 0.2, i as f32 * -0.1];
            rls.update(&x, &y);
        }

        let readout = rls.to_readout(FeatureMode::Linear);
        assert_eq!(readout.output_dim, 2);
        assert_eq!(readout.state_dim, 4);
        assert_eq!(readout.feature_mode, FeatureMode::Linear);

        // Readout prediction should match RLS prediction
        let rls_pred = rls.predict(&[1.0, 0.0, 0.0, 0.0]);
        let readout_pred = readout.predict_from_raw_state(&[1.0, 0.0, 0.0, 0.0]);
        for j in 0..2 {
            assert!(
                (rls_pred[j] - readout_pred[j]).abs() < 1e-4,
                "Readout pred[{}] should match RLS: {} vs {}",
                j,
                rls_pred[j],
                readout_pred[j]
            );
        }
    }

    #[test]
    fn test_online_rls_symmetry_enforcement() {
        // Run enough updates to trigger symmetry enforcement (every 64)
        let mut rls = OnlineRidgeRegression::new(3, 1, 1e-4);

        for i in 0..128 {
            let x0 = ((i * 7 + 3) % 50) as f32 / 50.0;
            let x1 = ((i * 13 + 1) % 50) as f32 / 50.0;
            let x2 = ((i * 19 + 7) % 50) as f32 / 50.0;
            rls.update(&[x0, x1, x2], &[x0 + x1 - x2]);
        }

        assert_eq!(rls.num_updates(), 128);

        // P should be symmetric (enforced at updates 64 and 128)
        let d = rls.dim();
        for i in 0..d {
            for j in (i + 1)..d {
                let diff = (rls.p[i * d + j] - rls.p[j * d + i]).abs();
                assert!(
                    diff < 1e-10,
                    "P[{},{}] and P[{},{}] should be symmetric: diff = {}",
                    i, j, j, i, diff
                );
            }
        }
    }

    #[test]
    fn test_online_rls_higher_dim() {
        // Test with higher-dimensional features (like E8 linear: 482 dims)
        // Use a smaller proxy to keep test fast: 50 dims
        let d = 50;
        let k = 3;
        let mut rls = OnlineRidgeRegression::new(d, k, 1e-4);

        for i in 0..200 {
            let mut x = vec![0.0f32; d];
            for j in 0..d {
                x[j] = ((i * (j + 1) * 7 + 13) % 1000) as f32 / 1000.0;
            }
            // Target: linear combinations of first 3 features
            let y = vec![x[0] + x[1], x[1] - x[2], x[0] + x[2]];
            rls.update(&x, &y);
        }

        // Should produce non-trivial predictions
        let test_x: Vec<f32> = (0..d).map(|j| (j as f32) / d as f32).collect();
        let pred = rls.predict(&test_x);
        assert_eq!(pred.len(), k);

        // Predictions should be finite
        for j in 0..k {
            assert!(pred[j].is_finite(), "Prediction[{}] should be finite: {}", j, pred[j]);
        }
    }
}
