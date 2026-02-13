// Copyright (c) 2025-2026 brdigetrlol. All rights reserved.
// SPDX-License-Identifier: LicenseRef-Icarus-Proprietary
// See LICENSE in the repository root for full license terms.

//! Multi-device ensemble trainer for the EMC.
//!
//! Runs multiple EMC instances (potentially on different compute backends)
//! in parallel, concatenates their state vectors, and trains a single
//! ridge regression readout on the combined state space.
//!
//! This dramatically increases the effective reservoir dimensionality:
//! 4 EMCs × 482 dims = 1928 dims (linear), or 3856 dims (nonlinear).
//! Each backend (CUDA, CPU, NPU) introduces different numerical dynamics,
//! creating a richer feature space for the readout to exploit.

use anyhow::{Context, Result};

use crate::config::ManifoldConfig;
use crate::emc::EmergentManifoldComputer;
use crate::encoding::InputEncoder;
use crate::readout::{extract_features, feature_dim, FeatureMode, LinearReadout};
use crate::training::RidgeRegression;

/// Result from ensemble training.
#[derive(Debug, Clone)]
pub struct EnsembleTrainResult {
    /// Normalized mean squared error on training data.
    pub nmse: f32,
    /// Selected regularization lambda (meaningful only with auto_lambda).
    pub lambda: f32,
    /// Number of training samples used.
    pub num_samples: usize,
    /// Dimensionality of the concatenated state vector.
    pub state_dim: usize,
    /// Output dimensionality.
    pub output_dim: usize,
    /// Per-instance state dimensions.
    pub instance_dims: Vec<usize>,
}

/// Status of a single EMC instance within the ensemble.
#[derive(Debug, Clone)]
pub struct InstanceStatus {
    /// Human-readable backend name (e.g., "CPU", "CUDA:0", "NPU").
    pub backend_name: String,
    /// Number of lattice sites in this instance.
    pub num_sites: usize,
    /// Feature dimension for this instance (depends on feature mode).
    pub feature_dim: usize,
    /// Total ticks executed by this instance.
    pub total_ticks: u64,
    /// Backend selection from config.
    pub config_backend: crate::config::BackendSelection,
}

/// A single EMC instance within the ensemble.
struct EmcInstance {
    emc: EmergentManifoldComputer,
    config_backend: crate::config::BackendSelection,
}

/// Multi-device ensemble trainer.
///
/// Orchestrates N EMC instances running (potentially) on different backends.
/// States from all instances are concatenated into a single high-dimensional
/// vector for ridge regression training.
///
/// # Architecture
///
/// ```text
/// Input ──┬──► EMC[0] (CUDA)  ──► state_0 (482 dims) ─┐
///         ├──► EMC[1] (CPU)   ──► state_1 (482 dims) ──┤
///         ├──► EMC[2] (NPU)   ──► state_2 (482 dims) ──┼──► concat ──► Ridge ──► output
///         └──► EMC[3] (CPU)   ──► state_3 (482 dims) ──┘
/// ```
pub struct EnsembleTrainer {
    instances: Vec<EmcInstance>,
    readout: Option<LinearReadout>,
    feature_mode: FeatureMode,
    /// Collected concatenated states for training.
    collected_states: Vec<Vec<f32>>,
    /// Collected targets for training.
    collected_targets: Vec<Vec<f32>>,
}

impl EnsembleTrainer {
    /// Create a new empty ensemble trainer.
    pub fn new(feature_mode: FeatureMode) -> Self {
        Self {
            instances: Vec::new(),
            readout: None,
            feature_mode,
            collected_states: Vec::new(),
            collected_targets: Vec::new(),
        }
    }

    /// Add an EMC instance with the given configuration.
    ///
    /// The backend is determined by `config.backend`. If GPU initialization
    /// fails, falls back to CPU automatically.
    pub fn add_instance(&mut self, config: ManifoldConfig) -> Result<()> {
        let backend_sel = config.backend;
        let emc = match EmergentManifoldComputer::new(config.clone()) {
            Ok(emc) => emc,
            Err(e) => {
                eprintln!(
                    "Warning: backend {:?} init failed ({}), falling back to CPU",
                    backend_sel, e
                );
                EmergentManifoldComputer::new_cpu(config)
            }
        };
        self.instances.push(EmcInstance {
            emc,
            config_backend: backend_sel,
        });
        Ok(())
    }

    /// Add a CPU-only EMC instance.
    pub fn add_cpu_instance(&mut self, config: ManifoldConfig) -> Result<()> {
        let mut cpu_config = config;
        cpu_config.backend = crate::config::BackendSelection::Cpu;
        let emc = EmergentManifoldComputer::new_cpu(cpu_config);
        self.instances.push(EmcInstance {
            emc,
            config_backend: crate::config::BackendSelection::Cpu,
        });
        Ok(())
    }

    /// Number of EMC instances in the ensemble.
    pub fn num_instances(&self) -> usize {
        self.instances.len()
    }

    /// Initialize all instances with random field values.
    ///
    /// Each instance gets a different seed (base_seed + instance_index)
    /// to ensure diverse initial conditions.
    pub fn init_random(&mut self, base_seed: u64, amplitude: f32) {
        for (i, inst) in self.instances.iter_mut().enumerate() {
            inst.emc.init_random(base_seed + i as u64, amplitude);
        }
    }

    /// Run warmup ticks on all instances (no state collection).
    ///
    /// Warmup lets the reservoir dynamics reach a transient state
    /// before training begins. Typically 50-200 ticks.
    pub fn warmup(&mut self, ticks: u64) -> Result<()> {
        for inst in &mut self.instances {
            inst.emc.run(ticks)?;
        }
        Ok(())
    }

    /// Encode input and tick all instances, then collect concatenated state.
    ///
    /// 1. Encode `input` into layer 0 of each EMC (with leaky integration)
    /// 2. Tick each EMC for `ticks_per_input` steps
    /// 3. Extract features from each EMC and concatenate
    /// 4. Store the concatenated state
    ///
    /// If `target` is provided, store it for later training.
    pub fn drive(
        &mut self,
        input: &[f32],
        encoder: &dyn InputEncoder,
        ticks_per_input: u64,
        leak_rate: f32,
        target: Option<&[f32]>,
    ) -> Result<Vec<f32>> {
        // Encode and tick each instance
        for inst in &mut self.instances {
            if inst.emc.manifold.layers.is_empty() {
                continue;
            }
            encoder.encode_leaky(input, &mut inst.emc.manifold.layers[0].field, leak_rate);
            inst.emc.run(ticks_per_input)?;
        }

        // Extract and concatenate features
        let concat_state = self.concatenated_state();

        // Collect for training
        self.collected_states.push(concat_state.clone());
        if let Some(t) = target {
            self.collected_targets.push(t.to_vec());
        }

        Ok(concat_state)
    }

    /// Extract and concatenate the current state from all instances.
    pub fn concatenated_state(&self) -> Vec<f32> {
        let total_dim = self.total_state_dim();
        let mut concat = Vec::with_capacity(total_dim);
        for inst in &self.instances {
            if !inst.emc.manifold.layers.is_empty() {
                let features = extract_features(
                    &inst.emc.manifold.layers[0].field,
                    self.feature_mode,
                );
                concat.extend_from_slice(&features);
            }
        }
        concat
    }

    /// Total dimensionality of the concatenated state vector.
    pub fn total_state_dim(&self) -> usize {
        self.instances
            .iter()
            .map(|inst| {
                if inst.emc.manifold.layers.is_empty() {
                    0
                } else {
                    feature_dim(inst.emc.manifold.layers[0].field.num_sites, self.feature_mode)
                }
            })
            .sum()
    }

    /// Per-instance state dimensions.
    pub fn instance_state_dims(&self) -> Vec<usize> {
        self.instances
            .iter()
            .map(|inst| {
                if inst.emc.manifold.layers.is_empty() {
                    0
                } else {
                    feature_dim(inst.emc.manifold.layers[0].field.num_sites, self.feature_mode)
                }
            })
            .collect()
    }

    /// Train the ensemble readout on collected data.
    ///
    /// Uses fixed lambda ridge regression.
    pub fn train(&mut self, lambda: f32) -> Result<EnsembleTrainResult> {
        if self.collected_states.is_empty() || self.collected_targets.is_empty() {
            anyhow::bail!("No training data collected. Call drive() with targets first.");
        }
        if self.collected_states.len() != self.collected_targets.len() {
            anyhow::bail!(
                "State/target count mismatch: {} states vs {} targets",
                self.collected_states.len(),
                self.collected_targets.len()
            );
        }

        let ridge = RidgeRegression::new(lambda);
        let (weights, bias) = ridge.train(&self.collected_states, &self.collected_targets);

        let state_dim = self.collected_states[0].len();
        let output_dim = self.collected_targets[0].len();
        let num_samples = self.collected_states.len();

        // Compute NMSE on training data
        let nmse = compute_nmse(&weights, &bias, state_dim, &self.collected_states, &self.collected_targets);

        let readout = LinearReadout::from_weights_with_mode(
            weights,
            bias,
            output_dim,
            state_dim,
            self.feature_mode,
        );
        self.readout = Some(readout);

        Ok(EnsembleTrainResult {
            nmse,
            lambda,
            num_samples,
            state_dim,
            output_dim,
            instance_dims: self.instance_state_dims(),
        })
    }

    /// Train with automatic lambda selection via k-fold cross-validation.
    pub fn train_auto_lambda(&mut self, k_folds: usize) -> Result<EnsembleTrainResult> {
        if self.collected_states.is_empty() || self.collected_targets.is_empty() {
            anyhow::bail!("No training data collected. Call drive() with targets first.");
        }
        if self.collected_states.len() != self.collected_targets.len() {
            anyhow::bail!(
                "State/target count mismatch: {} states vs {} targets",
                self.collected_states.len(),
                self.collected_targets.len()
            );
        }

        let (weights, bias, selected_lambda) = RidgeRegression::train_auto_lambda(
            &self.collected_states,
            &self.collected_targets,
            k_folds,
        );

        let state_dim = self.collected_states[0].len();
        let output_dim = self.collected_targets[0].len();
        let num_samples = self.collected_states.len();

        let nmse = compute_nmse(&weights, &bias, state_dim, &self.collected_states, &self.collected_targets);

        let readout = LinearReadout::from_weights_with_mode(
            weights,
            bias,
            output_dim,
            state_dim,
            self.feature_mode,
        );
        self.readout = Some(readout);

        Ok(EnsembleTrainResult {
            nmse,
            lambda: selected_lambda,
            num_samples,
            state_dim,
            output_dim,
            instance_dims: self.instance_state_dims(),
        })
    }

    /// Predict output from the current concatenated state.
    ///
    /// Requires a trained readout (call `train()` or `train_auto_lambda()` first).
    pub fn predict(&self) -> Result<Vec<f32>> {
        let readout = self
            .readout
            .as_ref()
            .context("No trained readout. Call train() first.")?;

        let state = self.concatenated_state();
        Ok(readout.predict_from_raw_state(&state))
    }

    /// Drive input and predict in one step.
    ///
    /// Encodes input, ticks all instances, then runs readout on concatenated state.
    /// Returns the prediction vector.
    pub fn step(
        &mut self,
        input: &[f32],
        encoder: &dyn InputEncoder,
        ticks_per_input: u64,
        leak_rate: f32,
    ) -> Result<Vec<f32>> {
        // Encode and tick (no target — inference mode)
        for inst in &mut self.instances {
            if inst.emc.manifold.layers.is_empty() {
                continue;
            }
            encoder.encode_leaky(input, &mut inst.emc.manifold.layers[0].field, leak_rate);
            inst.emc.run(ticks_per_input)?;
        }

        self.predict()
    }

    /// Get status of each instance.
    pub fn instance_status(&self) -> Vec<InstanceStatus> {
        self.instances
            .iter()
            .map(|inst| {
                let num_sites = if inst.emc.manifold.layers.is_empty() {
                    0
                } else {
                    inst.emc.manifold.layers[0].field.num_sites
                };
                InstanceStatus {
                    backend_name: inst.emc.backend_name().to_string(),
                    num_sites,
                    feature_dim: feature_dim(num_sites, self.feature_mode),
                    total_ticks: inst.emc.total_ticks,
                    config_backend: inst.config_backend,
                }
            })
            .collect()
    }

    /// Number of collected training samples.
    pub fn num_samples(&self) -> usize {
        self.collected_states.len()
    }

    /// Clear collected training data (keeps the readout if trained).
    pub fn clear_collected_data(&mut self) {
        self.collected_states.clear();
        self.collected_targets.clear();
    }

    /// Check if a readout has been trained.
    pub fn is_trained(&self) -> bool {
        self.readout.is_some()
    }

    /// Get reference to the trained readout (if any).
    pub fn readout(&self) -> Option<&LinearReadout> {
        self.readout.as_ref()
    }

    /// Set the readout from an externally-trained readout (e.g., EWC regression).
    pub fn set_readout(&mut self, readout: LinearReadout) {
        self.readout = Some(readout);
    }

    /// Access collected training states (one per drive() call with a target).
    pub fn collected_states(&self) -> &[Vec<f32>] {
        &self.collected_states
    }

    /// Access collected training targets (one per drive() call with a target).
    pub fn collected_targets(&self) -> &[Vec<f32>] {
        &self.collected_targets
    }

    /// Get mutable reference to an EMC instance.
    pub fn instance_mut(&mut self, idx: usize) -> Option<&mut EmergentManifoldComputer> {
        self.instances.get_mut(idx).map(|inst| &mut inst.emc)
    }

    /// Get reference to an EMC instance.
    pub fn instance(&self, idx: usize) -> Option<&EmergentManifoldComputer> {
        self.instances.get(idx).map(|inst| &inst.emc)
    }
}

/// Compute NMSE (Normalized Mean Squared Error) for predictions.
///
/// NMSE = MSE / var(target) = sum((y - y_hat)²) / sum((y - y_mean)²)
fn compute_nmse(
    weights: &[f32],
    bias: &[f32],
    state_dim: usize,
    states: &[Vec<f32>],
    targets: &[Vec<f32>],
) -> f32 {
    let n = states.len();
    let k = bias.len();

    let mut mse = 0.0f64;
    let mut var = 0.0f64;

    // Compute target mean
    let mut target_mean = vec![0.0f64; k];
    for t in targets {
        for j in 0..k {
            target_mean[j] += t[j] as f64;
        }
    }
    for j in 0..k {
        target_mean[j] /= n as f64;
    }

    for i in 0..n {
        for j in 0..k {
            // Prediction: W · state + bias
            let mut pred = bias[j] as f64;
            for d in 0..state_dim {
                pred += weights[j * state_dim + d] as f64 * states[i][d] as f64;
            }
            let err = targets[i][j] as f64 - pred;
            mse += err * err;
            let dev = targets[i][j] as f64 - target_mean[j];
            var += dev * dev;
        }
    }

    if var.abs() < 1e-30 {
        0.0
    } else {
        (mse / var) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ManifoldConfig;
    use crate::encoding::SpatialEncoder;

    #[test]
    fn test_ensemble_creation() {
        let ensemble = EnsembleTrainer::new(FeatureMode::Linear);
        assert_eq!(ensemble.num_instances(), 0);
        assert_eq!(ensemble.total_state_dim(), 0);
        assert!(!ensemble.is_trained());
    }

    #[test]
    fn test_ensemble_add_cpu_instances() {
        let mut ensemble = EnsembleTrainer::new(FeatureMode::Linear);
        let config = ManifoldConfig::e8_only();
        ensemble.add_cpu_instance(config.clone()).unwrap();
        ensemble.add_cpu_instance(config).unwrap();

        assert_eq!(ensemble.num_instances(), 2);
        // E8: 241 sites × 2 (re+im) × 2 instances = 964 total
        assert_eq!(ensemble.total_state_dim(), 2 * 241 * 2);
    }

    #[test]
    fn test_ensemble_add_nonlinear() {
        let mut ensemble = EnsembleTrainer::new(FeatureMode::Nonlinear);
        let config = ManifoldConfig::e8_only();
        ensemble.add_cpu_instance(config).unwrap();

        // E8: 241 sites × 4 (re, im, re², im²) = 964
        assert_eq!(ensemble.total_state_dim(), 4 * 241);
    }

    #[test]
    fn test_ensemble_init_random() {
        let mut ensemble = EnsembleTrainer::new(FeatureMode::Linear);
        let config = ManifoldConfig::e8_only();
        ensemble.add_cpu_instance(config.clone()).unwrap();
        ensemble.add_cpu_instance(config).unwrap();

        ensemble.init_random(42, 0.5);

        // Verify instances got different seeds (different initial states)
        let s0 = ensemble.instance(0).unwrap().observe();
        let s1 = ensemble.instance(1).unwrap().observe();
        let diff: f32 = s0.layer_states[0]
            .values_re
            .iter()
            .zip(&s1.layer_states[0].values_re)
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 0.1, "Different seeds should produce different states");
    }

    #[test]
    fn test_ensemble_warmup() {
        let mut ensemble = EnsembleTrainer::new(FeatureMode::Linear);
        let config = ManifoldConfig::e8_only();
        ensemble.add_cpu_instance(config).unwrap();
        ensemble.init_random(42, 0.5);

        ensemble.warmup(5).unwrap();
        assert_eq!(ensemble.instance(0).unwrap().total_ticks, 5);
    }

    #[test]
    fn test_ensemble_drive_and_collect() {
        let mut ensemble = EnsembleTrainer::new(FeatureMode::Linear);
        let config = ManifoldConfig::e8_only();
        ensemble.add_cpu_instance(config).unwrap();
        ensemble.init_random(42, 0.5);
        ensemble.warmup(5).unwrap();

        let encoder = SpatialEncoder::default();
        let input = vec![0.5, -0.3, 0.7, 0.1, 0.0];
        let target = vec![1.0, 0.0];

        let state = ensemble
            .drive(&input, &encoder, 10, 0.3, Some(&target))
            .unwrap();

        assert_eq!(state.len(), 2 * 241);
        assert_eq!(ensemble.num_samples(), 1);
    }

    #[test]
    fn test_ensemble_train_and_predict() {
        let mut ensemble = EnsembleTrainer::new(FeatureMode::Linear);
        let config = ManifoldConfig::e8_only();
        ensemble.add_cpu_instance(config.clone()).unwrap();
        ensemble.add_cpu_instance(config).unwrap();
        ensemble.init_random(42, 0.5);
        ensemble.warmup(10).unwrap();

        let encoder = SpatialEncoder::default();

        // Generate some training data (simple: output = first input value)
        for i in 0..20 {
            let val = (i as f32) / 20.0;
            let input = vec![val, 0.0, 0.0, 0.0, 0.0];
            let target = vec![val];
            ensemble
                .drive(&input, &encoder, 5, 0.3, Some(&target))
                .unwrap();
        }

        let result = ensemble.train(1e-4).unwrap();
        assert_eq!(result.num_samples, 20);
        assert_eq!(result.state_dim, 2 * 241 * 2); // 2 instances × 2 × 241
        assert_eq!(result.output_dim, 1);
        assert!(result.nmse < 1.0, "NMSE should be reasonable: {}", result.nmse);
        assert!(ensemble.is_trained());

        // Now predict
        let pred = ensemble.predict().unwrap();
        assert_eq!(pred.len(), 1);
    }

    #[test]
    fn test_ensemble_train_auto_lambda() {
        let mut ensemble = EnsembleTrainer::new(FeatureMode::Linear);
        let config = ManifoldConfig::e8_only();
        ensemble.add_cpu_instance(config).unwrap();
        ensemble.init_random(42, 0.5);
        ensemble.warmup(5).unwrap();

        let encoder = SpatialEncoder::default();
        for i in 0..30 {
            let val = (i as f32) / 30.0;
            let input = vec![val, 0.0, 0.0];
            let target = vec![val * 2.0];
            ensemble
                .drive(&input, &encoder, 5, 0.3, Some(&target))
                .unwrap();
        }

        let result = ensemble.train_auto_lambda(3).unwrap();
        assert!(result.lambda > 0.0);
        assert!(result.nmse < 1.0, "Auto-lambda NMSE: {}", result.nmse);
    }

    #[test]
    fn test_ensemble_step() {
        let mut ensemble = EnsembleTrainer::new(FeatureMode::Linear);
        let config = ManifoldConfig::e8_only();
        ensemble.add_cpu_instance(config).unwrap();
        ensemble.init_random(42, 0.5);
        ensemble.warmup(5).unwrap();

        let encoder = SpatialEncoder::default();

        // Train first
        for i in 0..20 {
            let val = (i as f32) / 20.0;
            let input = vec![val, 0.0, 0.0];
            let target = vec![val];
            ensemble
                .drive(&input, &encoder, 5, 0.3, Some(&target))
                .unwrap();
        }
        ensemble.train(1e-4).unwrap();

        // Step = drive + predict in one call
        let pred = ensemble.step(&[0.5, 0.0, 0.0], &encoder, 5, 0.3).unwrap();
        assert_eq!(pred.len(), 1);
    }

    #[test]
    fn test_ensemble_instance_status() {
        let mut ensemble = EnsembleTrainer::new(FeatureMode::Linear);
        let config = ManifoldConfig::e8_only();
        ensemble.add_cpu_instance(config).unwrap();
        ensemble.init_random(42, 0.5);
        ensemble.warmup(10).unwrap();

        let status = ensemble.instance_status();
        assert_eq!(status.len(), 1);
        assert_eq!(status[0].backend_name, "CPU");
        assert_eq!(status[0].num_sites, 241);
        assert_eq!(status[0].feature_dim, 482);
        assert_eq!(status[0].total_ticks, 10);
    }

    #[test]
    fn test_ensemble_clear_data() {
        let mut ensemble = EnsembleTrainer::new(FeatureMode::Linear);
        let config = ManifoldConfig::e8_only();
        ensemble.add_cpu_instance(config).unwrap();
        ensemble.init_random(42, 0.5);

        let encoder = SpatialEncoder::default();
        ensemble
            .drive(&[0.5], &encoder, 5, 0.3, Some(&[1.0]))
            .unwrap();
        assert_eq!(ensemble.num_samples(), 1);

        ensemble.clear_collected_data();
        assert_eq!(ensemble.num_samples(), 0);
    }

    #[test]
    fn test_ensemble_train_no_data_errors() {
        let mut ensemble = EnsembleTrainer::new(FeatureMode::Linear);
        assert!(ensemble.train(1.0).is_err());
    }

    #[test]
    fn test_ensemble_predict_untrained_errors() {
        let ensemble = EnsembleTrainer::new(FeatureMode::Linear);
        assert!(ensemble.predict().is_err());
    }

    #[test]
    fn test_compute_nmse_perfect() {
        // Perfect predictions → NMSE = 0
        let states = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let targets = vec![vec![1.0], vec![0.0]];
        // W = [1, 0], b = [0] → pred = state[0]
        let weights = vec![1.0f32, 0.0];
        let bias = vec![0.0f32];
        let nmse = compute_nmse(&weights, &bias, 2, &states, &targets);
        assert!(nmse < 1e-6, "Perfect prediction NMSE should be ~0: {}", nmse);
    }
}
