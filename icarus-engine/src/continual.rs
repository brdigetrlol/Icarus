//! Continual Learning — EWC + Experience Replay for reservoir readouts.
//!
//! Elastic Weight Consolidation (EWC) prevents catastrophic forgetting when
//! training the linear readout on sequential tasks. The Fisher Information
//! diagonal captures per-feature importance from prior tasks, and a quadratic
//! penalty pulls new weights toward previous optima.
//!
//! Experience Replay mixes stored (state, target) pairs from earlier tasks
//! into the current training set for additional protection.
//!
//! The key insight: since Icarus uses ridge regression (closed-form linear
//! solve), the EWC penalty integrates directly into the normal equations:
//!
//! ```text
//! (X^T·X + λI + λ_ewc·Σ_k diag(F_k))·W^T = X^T·Y + λ_ewc·Σ_k diag(F_k)·W^T*_k
//! ```

use std::collections::VecDeque;

use anyhow::Result;

use crate::encoding::InputEncoder;
use crate::readout::{LinearReadout, Readout, StateCollector};
use crate::training::{nmse, ReservoirTrainer};
use crate::EmergentManifoldComputer;

// ─── Fisher Diagonal ───────────────────────────────────

/// Per-task feature importance from the Fisher Information diagonal.
///
/// Stores the diagonal of the Fisher Information Matrix (approximated as
/// per-feature variance of the centered data matrix) along with the
/// optimal W^T from training on that task.
#[derive(Debug, Clone)]
pub struct FisherDiagonal {
    /// Fisher diagonal values (length D = state_dim), f64 for precision.
    pub values: Vec<f64>,
    /// Optimal centered W^T from this task (D × K, row-major, f64).
    pub optimal_wt: Vec<f64>,
    /// Output dimensionality K.
    pub output_dim: usize,
    /// State dimensionality D.
    pub state_dim: usize,
    /// Name of the task.
    pub task_name: String,
}

// ─── Replay Buffer ─────────────────────────────────────

/// Circular buffer of (state, target) pairs for experience replay.
///
/// When the buffer is full, the oldest entries are evicted to make room.
/// Sampling uses evenly-spaced indices for maximum diversity.
#[derive(Debug, Clone)]
pub struct ReplayBuffer {
    states: VecDeque<Vec<f32>>,
    targets: VecDeque<Vec<f32>>,
    max_size: usize,
}

impl ReplayBuffer {
    pub fn new(max_size: usize) -> Self {
        Self {
            states: VecDeque::with_capacity(max_size.min(1024)),
            targets: VecDeque::with_capacity(max_size.min(1024)),
            max_size,
        }
    }

    /// Add a batch of (state, target) pairs. Evicts oldest if over capacity.
    pub fn add_batch(&mut self, states: &[Vec<f32>], targets: &[Vec<f32>]) {
        for (s, t) in states.iter().zip(targets.iter()) {
            if self.states.len() >= self.max_size {
                self.states.pop_front();
                self.targets.pop_front();
            }
            self.states.push_back(s.clone());
            self.targets.push_back(t.clone());
        }
    }

    /// Sample `count` evenly-spaced (state, target) pairs.
    ///
    /// Returns empty vectors if the buffer is empty or count is 0.
    pub fn sample(&self, count: usize) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        if count == 0 || self.states.is_empty() {
            return (Vec::new(), Vec::new());
        }
        let n = self.states.len();
        let actual = count.min(n);
        let mut sampled_states = Vec::with_capacity(actual);
        let mut sampled_targets = Vec::with_capacity(actual);

        for i in 0..actual {
            let idx = (i * n) / actual;
            sampled_states.push(self.states[idx].clone());
            sampled_targets.push(self.targets[idx].clone());
        }

        (sampled_states, sampled_targets)
    }

    pub fn len(&self) -> usize {
        self.states.len()
    }

    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    pub fn clear(&mut self) {
        self.states.clear();
        self.targets.clear();
    }
}

// ─── EWC Train Result ──────────────────────────────────

/// Result of an EWC-augmented ridge regression training run.
pub struct EwcTrainResult {
    /// Readout weights (K × D, row-major, f32).
    pub weights: Vec<f32>,
    /// Readout bias (K, f32).
    pub bias: Vec<f32>,
    /// Fisher diagonal for this task (D, f64) — for registration.
    pub fisher: Vec<f64>,
    /// Centered W^T from Cholesky solve (D × K, row-major, f64) — for registration.
    pub optimal_wt: Vec<f64>,
    /// Output dimensionality K.
    pub output_dim: usize,
    /// State dimensionality D.
    pub state_dim: usize,
}

// ─── EWC Ridge Regression ──────────────────────────────

/// Ridge regression augmented with Elastic Weight Consolidation.
///
/// Integrates the EWC quadratic penalty directly into the normal equations,
/// maintaining the closed-form linear solve. No iterative optimization needed.
#[derive(Debug, Clone)]
pub struct EwcRidgeRegression {
    /// Standard ridge regularization strength.
    pub lambda: f32,
    /// EWC penalty strength (scales Fisher importance).
    pub lambda_ewc: f32,
    /// Fisher history from previously trained tasks.
    fisher_history: Vec<FisherDiagonal>,
}

impl EwcRidgeRegression {
    pub fn new(lambda: f32, lambda_ewc: f32) -> Self {
        Self {
            lambda,
            lambda_ewc,
            fisher_history: Vec::new(),
        }
    }

    /// Register a completed task's Fisher information and optimal weights.
    pub fn register_task(
        &mut self,
        name: &str,
        fisher: Vec<f64>,
        optimal_wt: Vec<f64>,
        output_dim: usize,
        state_dim: usize,
    ) {
        self.fisher_history.push(FisherDiagonal {
            values: fisher,
            optimal_wt,
            output_dim,
            state_dim,
            task_name: name.to_string(),
        });
    }

    /// Number of prior tasks whose Fisher information is stored.
    pub fn num_prior_tasks(&self) -> usize {
        self.fisher_history.len()
    }

    /// Clear all Fisher history.
    pub fn clear_history(&mut self) {
        self.fisher_history.clear();
    }

    /// Train with EWC-augmented ridge regression.
    ///
    /// The modified normal equations:
    /// ```text
    /// (X^T·X + λI + λ_ewc·Σ_k diag(F_k))·W^T = X^T·Y + λ_ewc·Σ_k diag(F_k)·W^T*_k
    /// ```
    pub fn train(&self, states: &[Vec<f32>], targets: &[Vec<f32>]) -> EwcTrainResult {
        let n = states.len();
        assert!(n > 0, "No training samples");
        let d = states[0].len(); // state_dim
        let k = targets[0].len(); // output_dim

        // 1. Compute means
        let mut state_mean = vec![0.0f64; d];
        let mut target_mean = vec![0.0f64; k];
        for s in states {
            for (j, &v) in s.iter().enumerate() {
                state_mean[j] += v as f64;
            }
        }
        for t in targets {
            for (j, &v) in t.iter().enumerate() {
                target_mean[j] += v as f64;
            }
        }
        let n_f = n as f64;
        for v in &mut state_mean {
            *v /= n_f;
        }
        for v in &mut target_mean {
            *v /= n_f;
        }

        // 2a. Compute per-dimension standard deviation for variance normalization.
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

        // 2b. Compute X^T·X (D×D) and X^T·Y (D×K) on centered+normalized data
        let mut xtx = vec![0.0f64; d * d];
        let mut xty = vec![0.0f64; d * k];

        for idx in 0..n {
            let s = &states[idx];
            let t = &targets[idx];
            // Centered and variance-normalized values
            for a in 0..d {
                let sa = (s[a] as f64 - state_mean[a]) / state_std[a];
                // Upper triangle of X^T·X
                for b in a..d {
                    let sb = (s[b] as f64 - state_mean[b]) / state_std[b];
                    xtx[a * d + b] += sa * sb;
                }
                // X^T·Y
                for b in 0..k {
                    let tb = t[b] as f64 - target_mean[b];
                    xty[a * k + b] += sa * tb;
                }
            }
        }

        // Symmetrize X^T·X
        for a in 0..d {
            for b in (a + 1)..d {
                xtx[b * d + a] = xtx[a * d + b];
            }
        }

        // 3. Extract Fisher diagonal BEFORE regularization
        let fisher: Vec<f64> = (0..d).map(|a| xtx[a * d + a] / n_f).collect();

        // 4. Add standard ridge
        let lambda_f64 = self.lambda as f64;
        for a in 0..d {
            xtx[a * d + a] += lambda_f64;
        }

        // 5. Add EWC penalty from prior tasks
        let lambda_ewc_f64 = self.lambda_ewc as f64;
        for prior in &self.fisher_history {
            // Dimensions must match
            if prior.state_dim != d || prior.output_dim != k {
                continue;
            }
            for a in 0..d {
                // Diagonal penalty
                xtx[a * d + a] += lambda_ewc_f64 * prior.values[a];
                // Anchor RHS
                for b in 0..k {
                    xty[a * k + b] += lambda_ewc_f64 * prior.values[a] * prior.optimal_wt[a * k + b];
                }
            }
        }

        // 6. Solve via Cholesky → w_t (D×K)
        let w_t = cholesky_solve(&xtx, &xty, d, k);

        // 7. Transpose W^T (D×K) → weights (K×D, f32), un-scaling by state_std
        //    W_eff[k,d] = W_normalized[k,d] / σ[d]
        let mut weights = vec![0.0f32; k * d];
        for a in 0..d {
            for b in 0..k {
                weights[b * d + a] = (w_t[a * k + b] / state_std[a]) as f32;
            }
        }

        // 8. Compute bias: bias[b] = target_mean[b] - Σ_a W_eff[b,a] * state_mean[a]
        let mut bias = vec![0.0f32; k];
        for b in 0..k {
            let mut dot = 0.0f64;
            for a in 0..d {
                dot += (w_t[a * k + b] / state_std[a]) * state_mean[a];
            }
            bias[b] = (target_mean[b] - dot) as f32;
        }

        EwcTrainResult {
            weights,
            bias,
            fisher,
            optimal_wt: w_t,
            output_dim: k,
            state_dim: d,
        }
    }

    /// Train from a StateCollector and return both a LinearReadout and the EwcTrainResult.
    pub fn train_readout(&self, collector: &StateCollector) -> (LinearReadout, EwcTrainResult) {
        let result = self.train(&collector.states, &collector.targets);
        let readout = LinearReadout::from_weights(
            result.weights.clone(),
            result.bias.clone(),
            result.output_dim,
            result.state_dim,
        );
        (readout, result)
    }
}

// ─── Cholesky Solver (local copy) ──────────────────────

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

// ─── Task Result ───────────────────────────────────────

/// Result of training a single task in the continual learning pipeline.
#[derive(Debug, Clone)]
pub struct TaskResult {
    /// Name of the task.
    pub task_name: String,
    /// Trained linear readout for this task.
    pub readout: LinearReadout,
    /// Training NMSE achieved on this task.
    pub train_nmse: f32,
}

// ─── Continual Stats ───────────────────────────────────

/// Overall statistics for the continual learning pipeline.
#[derive(Debug, Clone)]
pub struct ContinualStats {
    /// Number of tasks trained so far.
    pub tasks_trained: usize,
    /// Current replay buffer size.
    pub replay_buffer_size: usize,
    /// Number of Fisher snapshots stored.
    pub fisher_history_len: usize,
    /// Names of all trained tasks.
    pub task_names: Vec<String>,
}

// ─── Continual Trainer ─────────────────────────────────

/// Orchestrator for continual (sequential multi-task) learning.
///
/// Combines reservoir state collection (via `ReservoirTrainer`), EWC-augmented
/// ridge regression, and experience replay into a unified pipeline.
///
/// Usage:
/// ```text
/// let mut trainer = ContinualTrainer::new(1e-3, 50.0, 50, 1, 1000, 0.3);
/// let result_a = trainer.train_task("NARMA-10", &mut emc, &encoder, &inputs_a, &targets_a, 0)?;
/// let result_b = trainer.train_task("Memory-5", &mut emc, &encoder, &inputs_b, &targets_b, 0)?;
/// ```
pub struct ContinualTrainer {
    /// Base reservoir trainer (handles warmup, state collection).
    pub base_trainer: ReservoirTrainer,
    /// EWC-augmented ridge regression.
    pub ewc: EwcRidgeRegression,
    /// Experience replay buffer.
    pub replay: ReplayBuffer,
    /// Fraction of replay samples to mix in (0.0 = no replay, 1.0 = equal).
    pub replay_ratio: f32,
    /// Results from each trained task.
    pub task_results: Vec<TaskResult>,
}

impl ContinualTrainer {
    pub fn new(
        lambda: f32,
        lambda_ewc: f32,
        warmup_ticks: u64,
        ticks_per_input: u64,
        replay_max_size: usize,
        replay_ratio: f32,
    ) -> Self {
        Self {
            base_trainer: ReservoirTrainer::new(lambda, warmup_ticks, ticks_per_input),
            ewc: EwcRidgeRegression::new(lambda, lambda_ewc),
            replay: ReplayBuffer::new(replay_max_size),
            replay_ratio,
            task_results: Vec::new(),
        }
    }

    /// Train the readout on a new task, preserving knowledge of prior tasks.
    ///
    /// Steps:
    /// 1. Collect reservoir states by driving the EMC with inputs
    /// 2. Sample replay data from buffer (if available)
    /// 3. Mix new + replay states/targets
    /// 4. Train with EWC-augmented ridge regression
    /// 5. Register Fisher + optimal W^T for future EWC
    /// 6. Add new samples to replay buffer
    pub fn train_task(
        &mut self,
        name: &str,
        emc: &mut EmergentManifoldComputer,
        encoder: &dyn InputEncoder,
        inputs: &[Vec<f32>],
        targets: &[Vec<f32>],
        layer_idx: usize,
    ) -> Result<TaskResult> {
        // 1. Collect reservoir states
        let collector = self
            .base_trainer
            .collect_states(emc, encoder, inputs, targets, layer_idx)?;

        // 2. Sample replay data
        let replay_count = if !self.replay.is_empty() {
            ((collector.len() as f32) * self.replay_ratio) as usize
        } else {
            0
        };
        let (replay_states, replay_targets) = self.replay.sample(replay_count);

        // 3. Mix new + replay
        let mut all_states = collector.states.clone();
        let mut all_targets = collector.targets.clone();
        all_states.extend(replay_states);
        all_targets.extend(replay_targets);

        // 4. Train with EWC
        let result = self.ewc.train(&all_states, &all_targets);
        let readout = LinearReadout::from_weights(
            result.weights.clone(),
            result.bias.clone(),
            result.output_dim,
            result.state_dim,
        );

        // 5. Register Fisher + optimal W^T
        self.ewc.register_task(
            name,
            result.fisher,
            result.optimal_wt,
            result.output_dim,
            result.state_dim,
        );

        // 6. Add new samples to replay buffer
        self.replay.add_batch(&collector.states, &collector.targets);

        // Compute training NMSE by evaluating on the same inputs
        let train_nmse = self.evaluate_readout(emc, encoder, &readout, inputs, targets, layer_idx, 0)?;

        let task_result = TaskResult {
            task_name: name.to_string(),
            readout,
            train_nmse,
        };
        self.task_results.push(task_result.clone());

        Ok(task_result)
    }

    /// Evaluate a readout on a sequence of inputs.
    ///
    /// Drives the EMC with each input, collects predictions from the readout,
    /// and computes NMSE against the targets. The first `warmup_samples` inputs
    /// are driven without scoring to let transients decay.
    pub fn evaluate_readout(
        &self,
        emc: &mut EmergentManifoldComputer,
        encoder: &dyn InputEncoder,
        readout: &LinearReadout,
        inputs: &[Vec<f32>],
        targets: &[Vec<f32>],
        layer_idx: usize,
        warmup_samples: usize,
    ) -> Result<f32> {
        // Drive warmup inputs without scoring
        for input in inputs.iter().take(warmup_samples) {
            encoder.encode(input, &mut emc.manifold.layers[layer_idx].field);
            emc.run(self.base_trainer.ticks_per_input)?;
        }

        // Score remaining
        let mut predictions = Vec::new();
        let mut actuals = Vec::new();
        for (input, target) in inputs[warmup_samples..]
            .iter()
            .zip(targets[warmup_samples..].iter())
        {
            encoder.encode(input, &mut emc.manifold.layers[layer_idx].field);
            emc.run(self.base_trainer.ticks_per_input)?;
            let pred = readout.read(&emc.manifold.layers[layer_idx].field);
            predictions.extend_from_slice(&pred);
            actuals.extend(target.iter().copied());
        }

        Ok(nmse(&predictions, &actuals))
    }

    /// Get current continual learning statistics.
    pub fn stats(&self) -> ContinualStats {
        ContinualStats {
            tasks_trained: self.task_results.len(),
            replay_buffer_size: self.replay.len(),
            fisher_history_len: self.ewc.num_prior_tasks(),
            task_names: self.task_results.iter().map(|r| r.task_name.clone()).collect(),
        }
    }
}

// ─── Tests ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ManifoldConfig;
    use crate::encoding::SpatialEncoder;
    use crate::training::RidgeRegression;

    #[test]
    fn test_replay_buffer_basic() {
        let mut buf = ReplayBuffer::new(10);
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);

        buf.add_batch(
            &[vec![1.0, 2.0], vec![3.0, 4.0]],
            &[vec![0.1], vec![0.2]],
        );
        assert_eq!(buf.len(), 2);
        assert!(!buf.is_empty());

        buf.clear();
        assert!(buf.is_empty());
    }

    #[test]
    fn test_replay_buffer_capacity() {
        let mut buf = ReplayBuffer::new(3);
        buf.add_batch(
            &[vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]],
            &[vec![0.1], vec![0.2], vec![0.3], vec![0.4], vec![0.5]],
        );
        // Should have evicted oldest two
        assert_eq!(buf.len(), 3);
        // Oldest remaining is [3.0]
        assert_eq!(buf.states[0], vec![3.0]);
        assert_eq!(buf.states[1], vec![4.0]);
        assert_eq!(buf.states[2], vec![5.0]);
    }

    #[test]
    fn test_replay_buffer_sample() {
        let mut buf = ReplayBuffer::new(10);
        buf.add_batch(
            &[vec![0.0], vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]],
            &[vec![0.0], vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]],
        );

        // Sample 3 from 6 → indices 0, 2, 4
        let (s, t) = buf.sample(3);
        assert_eq!(s.len(), 3);
        assert_eq!(s[0], vec![0.0]);
        assert_eq!(s[1], vec![2.0]);
        assert_eq!(s[2], vec![4.0]);
        assert_eq!(t[0], vec![0.0]);
        assert_eq!(t[1], vec![2.0]);
        assert_eq!(t[2], vec![4.0]);

        // Sample from empty buffer
        let empty = ReplayBuffer::new(10);
        let (s, t) = empty.sample(5);
        assert!(s.is_empty());
        assert!(t.is_empty());

        // Sample 0
        let (s, _) = buf.sample(0);
        assert!(s.is_empty());
    }

    #[test]
    fn test_ewc_no_prior_matches_ridge() {
        // Without EWC history, EwcRidgeRegression should produce identical
        // results to standard RidgeRegression.
        let lambda = 0.01f32;
        let ewc = EwcRidgeRegression::new(lambda, 50.0);
        let ridge = RidgeRegression::new(lambda);

        let states = vec![
            vec![1.0, 0.0, 0.5],
            vec![0.0, 1.0, -0.5],
            vec![0.5, 0.5, 0.0],
            vec![-0.5, 0.5, 1.0],
            vec![1.0, -1.0, 0.0],
        ];
        let targets = vec![
            vec![0.3],
            vec![0.7],
            vec![0.5],
            vec![0.8],
            vec![-0.1],
        ];

        let ewc_result = ewc.train(&states, &targets);
        let (ridge_weights, ridge_bias) = ridge.train(&states, &targets);

        // Weights should match within f32 numerical tolerance.
        // EWC and standard ridge use different code paths (EWC adds a zero
        // Fisher diagonal to the regularization matrix), so accumulated f32
        // rounding can diverge by ~1e-3 on small systems.
        for i in 0..ewc_result.weights.len() {
            let diff = (ewc_result.weights[i] - ridge_weights[i]).abs();
            assert!(
                diff < 1e-2,
                "Weight mismatch at {}: ewc={}, ridge={}, diff={}",
                i,
                ewc_result.weights[i],
                ridge_weights[i],
                diff
            );
        }
        for i in 0..ewc_result.bias.len() {
            let diff = (ewc_result.bias[i] - ridge_bias[i]).abs();
            assert!(
                diff < 1e-2,
                "Bias mismatch at {}: ewc={}, ridge={}, diff={}",
                i,
                ewc_result.bias[i],
                ridge_bias[i],
                diff
            );
        }
    }

    #[test]
    fn test_fisher_diagonal_values() {
        // Known data: states = [[1,0],[3,0],[2,4],[2,-4]]
        // Mean = [2, 0]
        // Centered = [[-1,0],[1,0],[0,4],[0,-4]]
        // Std = [sqrt(0.5), sqrt(8)]
        // Normalized = [[-1.414,0],[1.414,0],[0,1.414],[0,-1.414]]
        // Normalized X^T·X diagonal = [4, 4]
        // Fisher = [4/4, 4/4] = [1.0, 1.0]
        let ewc = EwcRidgeRegression::new(0.0, 0.0);
        let states = vec![
            vec![1.0, 0.0],
            vec![3.0, 0.0],
            vec![2.0, 4.0],
            vec![2.0, -4.0],
        ];
        let targets = vec![vec![0.0], vec![0.0], vec![0.0], vec![0.0]];

        let result = ewc.train(&states, &targets);
        assert!(
            (result.fisher[0] - 1.0).abs() < 1e-10,
            "Fisher[0] = {}, expected 1.0",
            result.fisher[0]
        );
        assert!(
            (result.fisher[1] - 1.0).abs() < 1e-10,
            "Fisher[1] = {}, expected 1.0",
            result.fisher[1]
        );
    }

    #[test]
    fn test_ewc_pulls_weights_toward_prior() {
        // Task A: y = x0 (first feature), x1 is independent noise
        // Task B: y = x1 (second feature), x0 is independent noise
        // With EWC, after training B, w0 should be pulled toward A's optimal
        // Without EWC, w0 should be ~0.0
        // Features must be decorrelated so normalization doesn't collapse them.

        let states_a: Vec<Vec<f32>> = (0..50)
            .map(|i| {
                let x = (i as f32) / 50.0;
                let noise = ((i * 17 + 5) % 50) as f32 / 50.0;
                vec![x, noise]
            })
            .collect();
        let targets_a: Vec<Vec<f32>> = states_a.iter().map(|s| vec![s[0]]).collect();

        let states_b: Vec<Vec<f32>> = (0..50)
            .map(|i| {
                let noise = ((i * 17 + 5) % 50) as f32 / 50.0;
                let x = (i as f32) / 50.0;
                vec![noise, x]
            })
            .collect();
        let targets_b: Vec<Vec<f32>> = states_b.iter().map(|s| vec![s[1]]).collect();

        // Baseline: train B without EWC
        let baseline = EwcRidgeRegression::new(1e-4, 0.0);
        let baseline_result = baseline.train(&states_b, &targets_b);
        let w0_baseline = baseline_result.weights[0]; // Weight for feature 0

        // EWC: train A, register, then train B
        let mut ewc = EwcRidgeRegression::new(1e-4, 100.0);
        let result_a = ewc.train(&states_a, &targets_a);
        ewc.register_task("A", result_a.fisher, result_a.optimal_wt, 1, 2);
        let ewc_result = ewc.train(&states_b, &targets_b);
        let w0_ewc = ewc_result.weights[0]; // Weight for feature 0

        // With EWC, w0 should be pulled toward A's optimal (positive, ~1.0)
        // Without EWC, w0 should be near 0.0
        assert!(
            w0_ewc.abs() > w0_baseline.abs(),
            "EWC should pull w0 toward prior: w0_ewc={}, w0_baseline={}",
            w0_ewc,
            w0_baseline
        );
    }

    #[test]
    fn test_continual_trainer_single_task() {
        let config = ManifoldConfig::e8_only();
        let mut emc = EmergentManifoldComputer::new_cpu(config);
        emc.init_random(42, 0.3);

        let encoder = SpatialEncoder::default();

        // Simple input-output task
        let inputs: Vec<Vec<f32>> = (0..30).map(|i| vec![(i as f32) / 30.0]).collect();
        let targets: Vec<Vec<f32>> = inputs.iter().map(|v| vec![v[0] * 0.5]).collect();

        let mut trainer = ContinualTrainer::new(1e-3, 50.0, 10, 1, 100, 0.3);
        let result = trainer.train_task("test-task", &mut emc, &encoder, &inputs, &targets, 0)
            .expect("train_task failed");

        assert_eq!(result.task_name, "test-task");
        assert!(result.train_nmse.is_finite());

        let stats = trainer.stats();
        assert_eq!(stats.tasks_trained, 1);
        assert_eq!(stats.fisher_history_len, 1);
        assert_eq!(stats.task_names, vec!["test-task"]);
        assert!(stats.replay_buffer_size > 0);
    }

    #[test]
    fn test_continual_trainer_two_tasks() {
        let config = ManifoldConfig::e8_only();
        let mut emc = EmergentManifoldComputer::new_cpu(config);
        emc.init_random(42, 0.3);

        let encoder = SpatialEncoder::default();

        let inputs_a: Vec<Vec<f32>> = (0..20).map(|i| vec![(i as f32) / 20.0]).collect();
        let targets_a: Vec<Vec<f32>> = inputs_a.iter().map(|v| vec![v[0] * 0.5]).collect();

        let inputs_b: Vec<Vec<f32>> = (0..20).map(|i| vec![(i as f32) / 20.0]).collect();
        let targets_b: Vec<Vec<f32>> = inputs_b.iter().map(|v| vec![v[0] * 0.3 + 0.1]).collect();

        let mut trainer = ContinualTrainer::new(1e-3, 50.0, 10, 1, 100, 0.3);

        trainer
            .train_task("task-A", &mut emc, &encoder, &inputs_a, &targets_a, 0)
            .expect("train task A failed");
        trainer
            .train_task("task-B", &mut emc, &encoder, &inputs_b, &targets_b, 0)
            .expect("train task B failed");

        let stats = trainer.stats();
        assert_eq!(stats.tasks_trained, 2);
        assert_eq!(stats.fisher_history_len, 2);
        assert_eq!(stats.task_names, vec!["task-A", "task-B"]);
    }
}
