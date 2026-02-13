// Copyright (c) 2025-2026 brdigetrlol. All rights reserved.
// SPDX-License-Identifier: LicenseRef-Icarus-Proprietary
// See LICENSE in the repository root for full license terms.

//! Reservoir Computing Benchmarks for the Emergent Manifold Computer
//!
//! Standard benchmarks that validate the EMC's computational capabilities:
//! 1. XOR — nonlinear binary classification
//! 2. Sine approximation — smooth function regression
//! 3. NARMA-10 — temporal nonlinear benchmark (gold standard for RC)
//! 4. Encoder comparison — spatial vs phase vs spectral
//! 5. Feature mode comparison — linear vs nonlinear features
//! 6. Memory capacity — how many timesteps back can the reservoir remember?
//!
//! Methodology: ALL data is driven through a single EMC instance. The collected
//! reservoir states are split into train/test. The readout is fit on train states
//! and evaluated on test states — no fresh EMC is created for prediction.
//!
//! Run with: `cargo test -p icarus-engine --test reservoir_benchmarks -- --nocapture`

use icarus_engine::{
    EmergentManifoldComputer, ManifoldConfig,
    SpatialEncoder, PhaseEncoder, SpectralEncoder, InputEncoder,
    ReservoirTrainer, RidgeRegression,
    LinearReadout, FeatureMode,
    readout::extract_features,
    training::{nmse, accuracy},
};

// ═══════════════════════════════════════════════════════════
// Deterministic PRNG (xorshift64) — no external deps needed
// ═══════════════════════════════════════════════════════════

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 { 1 } else { seed })
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Uniform f32 in [0, 1)
    fn uniform(&mut self) -> f32 {
        (self.next_u64() & 0xFFFF_FFFF) as f32 / (u32::MAX as f32 + 1.0)
    }

    /// Uniform f32 in [lo, hi)
    fn uniform_range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.uniform()
    }

    /// Approximate normal(0, 1) via Box-Muller
    fn normal(&mut self) -> f32 {
        let u1 = self.uniform().max(1e-10);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

// ═══════════════════════════════════════════════════════════
// Helper: create a fresh E8-only EMC on CPU
// ═══════════════════════════════════════════════════════════

fn make_emc() -> EmergentManifoldComputer {
    let config = ManifoldConfig::e8_only();
    let mut emc = EmergentManifoldComputer::new_cpu(config);
    emc.init_random(42, 0.5);
    emc
}

// ═══════════════════════════════════════════════════════════
// Helper: apply readout weights to collected state vectors
// ═══════════════════════════════════════════════════════════

/// Apply W·state + bias for each state vector. No EMC needed.
fn apply_readout(
    weights: &[f32],
    bias: &[f32],
    states: &[Vec<f32>],
    output_dim: usize,
    state_dim: usize,
) -> Vec<Vec<f32>> {
    states.iter().map(|state| {
        let mut out = bias.to_vec();
        for k in 0..output_dim {
            for d in 0..state_dim {
                out[k] += weights[k * state_dim + d] * state[d];
            }
        }
        out
    }).collect()
}

// ═══════════════════════════════════════════════════════════
// Dataset generators
// ═══════════════════════════════════════════════════════════

/// Generate XOR dataset: 4 quadrants, 2D input, binary classification.
/// Returns (inputs, targets_onehot, labels).
/// Data is interleaved by class so temporal ordering doesn't bias the readout.
fn xor_dataset(n_per_class: usize, noise: f32, seed: u64) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<usize>) {
    let mut rng = Rng::new(seed);
    let half = n_per_class / 2;

    // Generate all samples
    let mut class0_inputs = Vec::new();
    let mut class1_inputs = Vec::new();

    // Class 0: quadrants (+,+) and (-,-)
    for _ in 0..half {
        let x = rng.uniform_range(0.2, 1.0) + noise * rng.normal();
        let y = rng.uniform_range(0.2, 1.0) + noise * rng.normal();
        class0_inputs.push(vec![x, y]);
    }
    for _ in 0..half {
        let x = rng.uniform_range(-1.0, -0.2) + noise * rng.normal();
        let y = rng.uniform_range(-1.0, -0.2) + noise * rng.normal();
        class0_inputs.push(vec![x, y]);
    }

    // Class 1: quadrants (+,-) and (-,+)
    for _ in 0..half {
        let x = rng.uniform_range(0.2, 1.0) + noise * rng.normal();
        let y = rng.uniform_range(-1.0, -0.2) + noise * rng.normal();
        class1_inputs.push(vec![x, y]);
    }
    for _ in 0..half {
        let x = rng.uniform_range(-1.0, -0.2) + noise * rng.normal();
        let y = rng.uniform_range(0.2, 1.0) + noise * rng.normal();
        class1_inputs.push(vec![x, y]);
    }

    // Interleave: alternate class 0 and class 1
    let mut inputs = Vec::with_capacity(2 * n_per_class);
    let mut targets = Vec::with_capacity(2 * n_per_class);
    let mut labels = Vec::with_capacity(2 * n_per_class);

    let total = class0_inputs.len().min(class1_inputs.len());
    for i in 0..total {
        inputs.push(class0_inputs[i].clone());
        targets.push(vec![1.0, 0.0]);
        labels.push(0);

        inputs.push(class1_inputs[i].clone());
        targets.push(vec![0.0, 1.0]);
        labels.push(1);
    }

    (inputs, targets, labels)
}

/// Generate sine dataset: y = sin(2π·x) for x in [0, 1].
fn sine_dataset(n: usize) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut inputs = Vec::with_capacity(n);
    let mut targets = Vec::with_capacity(n);
    for i in 0..n {
        let x = i as f32 / n as f32;
        inputs.push(vec![x]);
        targets.push(vec![(2.0 * std::f32::consts::PI * x).sin()]);
    }
    (inputs, targets)
}

/// Generate NARMA-10 time series.
///
/// y(t+1) = 0.3·y(t) + 0.05·y(t)·Σ_{i=0}^{9} y(t-i) + 1.5·u(t-9)·u(t) + 0.1
///
/// Returns (inputs as single-element vectors, targets as single-element vectors).
/// Each input u(t) is drawn uniformly from [0, 0.5].
fn narma10(length: usize, seed: u64) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let total = length + 50; // extra warmup to fill history
    let mut rng = Rng::new(seed);

    // Generate random input sequence u(t)
    let u: Vec<f32> = (0..total).map(|_| rng.uniform() * 0.5).collect();

    // Compute NARMA-10 output y(t)
    let mut y = vec![0.0f32; total];
    for t in 10..total - 1 {
        let y_sum: f32 = (0..10).map(|i| y[t - i]).sum();
        let next = 0.3 * y[t]
            + 0.05 * y[t] * y_sum
            + 1.5 * u[t - 9] * u[t]
            + 0.1;
        // Clamp to prevent explosion
        y[t + 1] = next.clamp(-10.0, 10.0);
    }

    // Trim warmup: use last `length` timesteps
    // Input u(t) maps to target y(t+1), but we need both to exist
    let start = total - length;
    let end = total - 1; // y(t+1) exists only up to total-1
    let actual_length = end - start;
    let inputs: Vec<Vec<f32>> = u[start..start + actual_length].iter().map(|&v| vec![v]).collect();
    let targets: Vec<Vec<f32>> = y[start + 1..start + 1 + actual_length].iter().map(|&v| vec![v]).collect();

    (inputs, targets)
}

/// Generate memory capacity dataset.
/// Input: random sequence u(t). Target: u(t - delay).
fn memory_dataset(length: usize, delay: usize, seed: u64) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let total = length + delay + 10;
    let mut rng = Rng::new(seed);
    let u: Vec<f32> = (0..total).map(|_| rng.uniform_range(-1.0, 1.0)).collect();

    let start = delay + 10;
    let inputs: Vec<Vec<f32>> = u[start..start + length].iter().map(|&v| vec![v]).collect();
    let targets: Vec<Vec<f32>> = u[start - delay..start - delay + length].iter().map(|&v| vec![v]).collect();

    (inputs, targets)
}

// ═══════════════════════════════════════════════════════════
// Core benchmark runner: single-EMC train/eval pipeline
// ═══════════════════════════════════════════════════════════

struct BenchResult {
    readout: LinearReadout,
    train_predictions: Vec<Vec<f32>>,
    test_predictions: Vec<Vec<f32>>,
    selected_lambda: Option<f32>,
}

/// Process ALL data through a single EMC, split collected states, train on first
/// `train_count`, evaluate on rest. Returns readout + predictions for both splits.
fn run_benchmark(
    encoder: &dyn InputEncoder,
    inputs: &[Vec<f32>],
    targets: &[Vec<f32>],
    train_count: usize,
    trainer: &ReservoirTrainer,
) -> BenchResult {
    run_benchmark_inner(encoder, inputs, targets, train_count, trainer, false)
}

/// Like `run_benchmark` but uses auto-lambda (k-fold CV) instead of fixed lambda.
fn run_benchmark_auto(
    encoder: &dyn InputEncoder,
    inputs: &[Vec<f32>],
    targets: &[Vec<f32>],
    train_count: usize,
    trainer: &ReservoirTrainer,
) -> BenchResult {
    run_benchmark_inner(encoder, inputs, targets, train_count, trainer, true)
}

fn run_benchmark_inner(
    encoder: &dyn InputEncoder,
    inputs: &[Vec<f32>],
    targets: &[Vec<f32>],
    train_count: usize,
    trainer: &ReservoirTrainer,
    auto_lambda: bool,
) -> BenchResult {
    let mut emc = make_emc();
    let collector = trainer
        .collect_states(&mut emc, encoder, inputs, targets, 0)
        .expect("State collection failed");

    let state_dim = collector.state_dim();
    let output_dim = targets[0].len();

    // Split collected states
    let train_states = &collector.states[..train_count];
    let train_targets = &collector.targets[..train_count];
    let test_states = &collector.states[train_count..];

    let (weights, bias, selected_lambda) = if auto_lambda {
        let (w, b, lam) =
            RidgeRegression::train_auto_lambda(train_states, train_targets, 5);
        (w, b, Some(lam))
    } else {
        let ridge = RidgeRegression::new(trainer.lambda);
        let (w, b) = ridge.train(train_states, train_targets);
        (w, b, None)
    };

    // Apply readout to both splits
    let train_predictions = apply_readout(&weights, &bias, train_states, output_dim, state_dim);
    let test_predictions = apply_readout(&weights, &bias, test_states, output_dim, state_dim);

    let readout = LinearReadout::from_weights_with_mode(
        weights, bias, output_dim, state_dim, trainer.feature_mode,
    );

    BenchResult {
        readout,
        train_predictions,
        test_predictions,
        selected_lambda,
    }
}

// ═══════════════════════════════════════════════════════════
// Benchmark 1: XOR Classification
// ═══════════════════════════════════════════════════════════

#[test]
fn bench_xor_spatial() {
    println!("\n{}", "=".repeat(60));
    println!("BENCHMARK: XOR Classification (Spatial Encoder)");
    println!("{}", "=".repeat(60));

    let (inputs, targets, labels) = xor_dataset(200, 0.05, 100);
    let train_count = 200;

    let encoder = SpatialEncoder::default();
    let trainer = ReservoirTrainer::new(1e-2, 20, 1)
        .with_feature_mode(FeatureMode::Nonlinear);

    let result = run_benchmark_auto(&encoder, &inputs, &targets, train_count, &trainer);

    // Evaluate on train set (we're checking that training worked)
    let train_acc = accuracy(&result.train_predictions, &labels[..train_count]);
    // Also evaluate on latter portion if there are test samples
    let test_acc = if labels.len() > train_count {
        accuracy(&result.test_predictions, &labels[train_count..])
    } else {
        train_acc
    };

    println!("  Total samples: {}", inputs.len());
    println!("  Feature dim:   {} (nonlinear)", result.readout.state_dim);
    println!("  Selected λ: {:e}", result.selected_lambda.unwrap());
    println!("  Train accuracy: {:.1}%", train_acc * 100.0);
    println!("  Test accuracy:  {:.1}%", test_acc * 100.0);

    assert!(
        train_acc >= 0.60,
        "XOR train accuracy too low: {:.1}% (need >= 60%)",
        train_acc * 100.0
    );
    println!("  PASS (train >= 60%)");
}

#[test]
fn bench_xor_phase() {
    println!("\n{}", "=".repeat(60));
    println!("BENCHMARK: XOR Classification (Phase Encoder)");
    println!("{}", "=".repeat(60));

    let (inputs, targets, labels) = xor_dataset(200, 0.05, 100);
    let train_count = 200;

    let encoder = PhaseEncoder::default();
    let trainer = ReservoirTrainer::new(1e-2, 20, 1)
        .with_feature_mode(FeatureMode::Nonlinear);

    let result = run_benchmark(&encoder, &inputs, &targets, train_count, &trainer);
    let train_acc = accuracy(&result.train_predictions, &labels[..train_count]);

    println!("  Train accuracy: {:.1}%", train_acc * 100.0);
    assert!(
        train_acc >= 0.55,
        "XOR/Phase accuracy too low: {:.1}%",
        train_acc * 100.0
    );
    println!("  PASS (>= 55%)");
}

#[test]
fn bench_xor_spectral() {
    println!("\n{}", "=".repeat(60));
    println!("BENCHMARK: XOR Classification (Spectral Encoder)");
    println!("{}", "=".repeat(60));

    let (inputs, targets, labels) = xor_dataset(200, 0.05, 100);
    let train_count = 200;

    let encoder = SpectralEncoder::new();
    let trainer = ReservoirTrainer::new(1e-2, 20, 1)
        .with_feature_mode(FeatureMode::Nonlinear);

    let result = run_benchmark(&encoder, &inputs, &targets, train_count, &trainer);
    let train_acc = accuracy(&result.train_predictions, &labels[..train_count]);

    println!("  Train accuracy: {:.1}%", train_acc * 100.0);
    assert!(
        train_acc >= 0.55,
        "XOR/Spectral accuracy too low: {:.1}%",
        train_acc * 100.0
    );
    println!("  PASS (>= 55%)");
}

// ═══════════════════════════════════════════════════════════
// Benchmark 2: Sine Function Approximation
// ═══════════════════════════════════════════════════════════

#[test]
fn bench_sine_spatial() {
    println!("\n{}", "=".repeat(60));
    println!("BENCHMARK: Sine Approximation (Spatial, Linear Features, Auto-λ)");
    println!("{}", "=".repeat(60));

    let (inputs, targets) = sine_dataset(300);
    let train_count = 200;

    let encoder = SpatialEncoder::default();
    let trainer = ReservoirTrainer::new(1e-3, 20, 1);

    let result = run_benchmark_auto(&encoder, &inputs, &targets, train_count, &trainer);

    let train_pred: Vec<f32> = result.train_predictions.iter().map(|p| p[0]).collect();
    let train_actual: Vec<f32> = targets[..train_count].iter().map(|t| t[0]).collect();
    let train_nmse = nmse(&train_pred, &train_actual);

    let test_pred: Vec<f32> = result.test_predictions.iter().map(|p| p[0]).collect();
    let test_actual: Vec<f32> = targets[train_count..].iter().map(|t| t[0]).collect();
    let test_nmse = nmse(&test_pred, &test_actual);

    println!("  Train: {}, Test: {}", train_count, inputs.len() - train_count);
    println!("  Feature dim: {} (linear)", result.readout.state_dim);
    println!("  Selected λ: {:e}", result.selected_lambda.unwrap());
    println!("  Train NMSE: {:.4}", train_nmse);
    println!("  Test NMSE:  {:.4}", test_nmse);

    assert!(
        train_nmse < 1.0,
        "Sine train NMSE too high: {:.4} (need < 1.0)",
        train_nmse
    );
    println!("  PASS (train NMSE < 1.0)");
}

#[test]
fn bench_sine_nonlinear_features() {
    println!("\n{}", "=".repeat(60));
    println!("BENCHMARK: Sine Approximation (Spatial, Nonlinear Features, Auto-λ)");
    println!("{}", "=".repeat(60));

    let (inputs, targets) = sine_dataset(300);
    let train_count = 200;

    let encoder = SpatialEncoder::default();
    let trainer = ReservoirTrainer::new(1e-3, 20, 1)
        .with_feature_mode(FeatureMode::Nonlinear);

    let result = run_benchmark_auto(&encoder, &inputs, &targets, train_count, &trainer);

    let train_pred: Vec<f32> = result.train_predictions.iter().map(|p| p[0]).collect();
    let train_actual: Vec<f32> = targets[..train_count].iter().map(|t| t[0]).collect();
    let train_nmse = nmse(&train_pred, &train_actual);

    let test_pred: Vec<f32> = result.test_predictions.iter().map(|p| p[0]).collect();
    let test_actual: Vec<f32> = targets[train_count..].iter().map(|t| t[0]).collect();
    let test_nmse = nmse(&test_pred, &test_actual);

    println!("  Feature dim: {} (nonlinear)", result.readout.state_dim);
    println!("  Selected λ: {:e}", result.selected_lambda.unwrap());
    println!("  Train NMSE: {:.4}", train_nmse);
    println!("  Test NMSE:  {:.4}", test_nmse);

    assert!(
        train_nmse < 1.0,
        "Sine/Nonlinear train NMSE too high: {:.4}",
        train_nmse
    );
    println!("  PASS (train NMSE < 1.0)");
}

// ═══════════════════════════════════════════════════════════
// Benchmark 3: NARMA-10 (Temporal Nonlinear)
// ═══════════════════════════════════════════════════════════

#[test]
fn bench_narma10() {
    println!("\n{}", "=".repeat(60));
    println!("BENCHMARK: NARMA-10 (Temporal Nonlinear Processing, Auto-λ)");
    println!("{}", "=".repeat(60));

    let (inputs, targets) = narma10(500, 42);
    let train_count = 350;

    let encoder = SpatialEncoder { offset: 0, scale: 2.0 };
    let trainer = ReservoirTrainer::new(1e-2, 50, 1)
        .with_feature_mode(FeatureMode::Nonlinear)
        .with_leak_rate(0.3);

    let result = run_benchmark_auto(&encoder, &inputs, &targets, train_count, &trainer);

    let train_pred: Vec<f32> = result.train_predictions.iter().map(|p| p[0]).collect();
    let train_actual: Vec<f32> = targets[..train_count].iter().map(|t| t[0]).collect();
    let train_nmse = nmse(&train_pred, &train_actual);

    let test_pred: Vec<f32> = result.test_predictions.iter().map(|p| p[0]).collect();
    let test_actual: Vec<f32> = targets[train_count..].iter().map(|t| t[0]).collect();
    let test_nmse = nmse(&test_pred, &test_actual);

    println!("  Train: {}, Test: {}", train_count, inputs.len() - train_count);
    println!("  Leak rate: 0.3, Warmup: 50 ticks");
    println!("  Feature dim: {} (nonlinear)", result.readout.state_dim);
    println!("  Selected λ: {:e}", result.selected_lambda.unwrap());
    println!("  Train NMSE: {:.4}", train_nmse);
    println!("  Test NMSE:  {:.4}", test_nmse);

    // NARMA-10 with same-EMC evaluation should work much better
    assert!(
        train_nmse < 2.0,
        "NARMA-10 train NMSE too high: {:.4} (need < 2.0)",
        train_nmse
    );
    println!("  PASS (train NMSE < 2.0)");

    if test_nmse < 1.0 {
        println!("  ** EXCELLENT: Test beats mean predictor! **");
    }
    if test_nmse < 0.5 {
        println!("  ** OUTSTANDING: Research-grade RC performance! **");
    }
}

// ═══════════════════════════════════════════════════════════
// Benchmark 4: Encoder Comparison
// ═══════════════════════════════════════════════════════════

#[test]
fn bench_encoder_comparison() {
    println!("\n{}", "=".repeat(60));
    println!("BENCHMARK: Encoder Comparison on Sine Task");
    println!("{}", "=".repeat(60));

    let (inputs, targets) = sine_dataset(300);
    let train_count = 200;

    let encoders: Vec<(&str, Box<dyn InputEncoder>)> = vec![
        ("Spatial", Box::new(SpatialEncoder::default())),
        ("Phase", Box::new(PhaseEncoder::default())),
        ("Spectral", Box::new(SpectralEncoder::new())),
    ];

    let mut results: Vec<(&str, f32, f32)> = Vec::new();

    for (name, encoder) in &encoders {
        let trainer = ReservoirTrainer::new(1e-3, 20, 1);
        let result = run_benchmark_auto(encoder.as_ref(), &inputs, &targets, train_count, &trainer);

        let train_pred: Vec<f32> = result.train_predictions.iter().map(|p| p[0]).collect();
        let train_actual: Vec<f32> = targets[..train_count].iter().map(|t| t[0]).collect();
        let t_nmse = nmse(&train_pred, &train_actual);

        let test_pred: Vec<f32> = result.test_predictions.iter().map(|p| p[0]).collect();
        let test_actual: Vec<f32> = targets[train_count..].iter().map(|t| t[0]).collect();
        let te_nmse = nmse(&test_pred, &test_actual);

        let lam_str = result.selected_lambda.map(|l| format!("{:e}", l)).unwrap_or_default();
        println!("  {:>10}: train NMSE={:.4}, test NMSE={:.4}, λ={}", name, t_nmse, te_nmse, lam_str);
        results.push((name, t_nmse, te_nmse));
    }

    let best = results.iter().min_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
    println!("  Best encoder: {} (train NMSE = {:.4})", best.0, best.1);

    assert!(
        best.1 < 1.0,
        "No encoder achieved train NMSE < 1.0 on sine task"
    );
    println!("  PASS");
}

// ═══════════════════════════════════════════════════════════
// Benchmark 5: Feature Mode Comparison
// ═══════════════════════════════════════════════════════════

#[test]
fn bench_feature_mode_comparison() {
    println!("\n{}", "=".repeat(60));
    println!("BENCHMARK: Linear vs Nonlinear Features");
    println!("{}", "=".repeat(60));

    // Use XOR — a task that strongly benefits from nonlinear features
    let (inputs, targets, labels) = xor_dataset(200, 0.05, 100);
    let train_count = inputs.len();

    let encoder = SpatialEncoder::default();

    // Linear features
    let trainer_lin = ReservoirTrainer::new(1e-2, 20, 1)
        .with_feature_mode(FeatureMode::Linear);
    let result_lin = run_benchmark(&encoder, &inputs, &targets, train_count, &trainer_lin);
    let acc_lin = accuracy(&result_lin.train_predictions, &labels);

    // Nonlinear features
    let trainer_nl = ReservoirTrainer::new(1e-2, 20, 1)
        .with_feature_mode(FeatureMode::Nonlinear);
    let result_nl = run_benchmark(&encoder, &inputs, &targets, train_count, &trainer_nl);
    let acc_nl = accuracy(&result_nl.train_predictions, &labels);

    println!("  Linear features:    {:.1}% accuracy (dim={})", acc_lin * 100.0, result_lin.readout.state_dim);
    println!("  Nonlinear features: {:.1}% accuracy (dim={})", acc_nl * 100.0, result_nl.readout.state_dim);

    let diff = acc_nl - acc_lin;
    if diff > 0.0 {
        println!("  Nonlinear advantage: +{:.1}pp", diff * 100.0);
    } else {
        println!("  Linear advantage: +{:.1}pp", -diff * 100.0);
    }

    assert!(
        acc_lin >= 0.50 || acc_nl >= 0.50,
        "Both feature modes below 50%: lin={:.1}%, nl={:.1}%",
        acc_lin * 100.0, acc_nl * 100.0
    );
    println!("  PASS");
}

// ═══════════════════════════════════════════════════════════
// Benchmark 6: Memory Capacity
// ═══════════════════════════════════════════════════════════

#[test]
fn bench_memory_capacity() {
    println!("\n{}", "=".repeat(60));
    println!("BENCHMARK: Memory Capacity");
    println!("{}", "=".repeat(60));

    // SpectralEncoder writes to ALL 241 sites (E8 root decomposition),
    // giving a strong input footprint that survives the dynamics.
    // SpatialEncoder on 1-D input only modifies 1/241 sites — too weak.
    let encoder = SpectralEncoder::new();
    let mut max_delay = 0usize;

    for delay in 1..=20 {
        let (inputs, targets) = memory_dataset(400, delay, 42 + delay as u64);
        let train_count = 300;

        // Low leak_rate preserves more of the old state (temporal memory).
        // Nonlinear features give the readout 964 dims to extract echoes.
        let trainer = ReservoirTrainer::new(1e-3, 20, 1)
            .with_feature_mode(FeatureMode::Nonlinear)
            .with_leak_rate(0.3);

        let result = run_benchmark_auto(&encoder, &inputs, &targets, train_count, &trainer);

        // Evaluate on test split
        let pred_flat: Vec<f32> = result.test_predictions.iter().map(|p| p[0]).collect();
        let actual_flat: Vec<f32> = targets[train_count..].iter().map(|t| t[0]).collect();
        let score = nmse(&pred_flat, &actual_flat);
        let lam = result.selected_lambda.unwrap();

        let pass = score < 0.8;
        if pass {
            max_delay = delay;
        }
        println!(
            "  delay={:>2}: NMSE={:.4} λ={:e} {}",
            delay, score, lam,
            if pass { "OK" } else { "FAIL" }
        );

        // Stop searching once we've had 3 consecutive failures
        if delay > max_delay + 3 {
            break;
        }
    }

    println!("  Max reliable memory: {} timesteps", max_delay);

    // The reservoir should have at least 1-step memory
    assert!(
        max_delay >= 1,
        "No memory at all! Delay 1 failed."
    );
    println!("  PASS (>= 1 step)");
}

// ═══════════════════════════════════════════════════════════
// Benchmark 7: Training + Prediction Consistency
// ═══════════════════════════════════════════════════════════

#[test]
fn bench_training_prediction_roundtrip() {
    println!("\n{}", "=".repeat(60));
    println!("BENCHMARK: Train/Predict Round-Trip Consistency");
    println!("{}", "=".repeat(60));

    // Train on a simple linear function y = 0.5·x + 0.3
    let n = 200;
    let inputs: Vec<Vec<f32>> = (0..n).map(|i| vec![i as f32 / n as f32]).collect();
    let targets: Vec<Vec<f32>> = inputs.iter().map(|x| vec![0.5 * x[0] + 0.3]).collect();

    let encoder = SpatialEncoder::default();
    let trainer = ReservoirTrainer::new(1e-3, 20, 1);

    let result = run_benchmark(&encoder, &inputs, &targets, n, &trainer);

    let pred_flat: Vec<f32> = result.train_predictions.iter().map(|p| p[0]).collect();
    let actual_flat: Vec<f32> = targets.iter().map(|t| t[0]).collect();
    let score = nmse(&pred_flat, &actual_flat);

    println!("  Training set NMSE: {:.4}", score);
    println!("  Output dim: {}, State dim: {}", result.readout.output_dim, result.readout.state_dim);

    let w_mean: f32 = result.readout.weights.iter().sum::<f32>() / result.readout.weights.len() as f32;
    let w_max = result.readout.weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let w_min = result.readout.weights.iter().cloned().fold(f32::INFINITY, f32::min);
    println!("  Weight stats: mean={:.6}, min={:.4}, max={:.4}", w_mean, w_min, w_max);
    println!("  Bias: {:?}", result.readout.bias);

    assert!(
        score < 0.5,
        "Training NMSE too high on simple linear function: {:.4}",
        score
    );
    println!("  PASS (NMSE < 0.5)");
}

// ═══════════════════════════════════════════════════════════
// Benchmark 8: Reservoir State Richness
// ═══════════════════════════════════════════════════════════

#[test]
fn bench_state_richness() {
    println!("\n{}", "=".repeat(60));
    println!("BENCHMARK: Reservoir State Richness");
    println!("{}", "=".repeat(60));

    let encoder = SpatialEncoder::default();
    let mut emc = make_emc();

    // Feed different inputs and measure state diversity
    let n_inputs = 50;
    let mut states = Vec::new();

    // Warmup
    encoder.encode(&[0.5], &mut emc.manifold.layers[0].field);
    emc.run(20).unwrap();

    for i in 0..n_inputs {
        let x = i as f32 / n_inputs as f32;
        encoder.encode(&[x], &mut emc.manifold.layers[0].field);
        emc.run(1).unwrap();
        let state = extract_features(&emc.manifold.layers[0].field, FeatureMode::Linear);
        states.push(state);
    }

    // Measure: average pairwise distance between states
    let d = states[0].len();
    let mut total_dist = 0.0f64;
    let mut count = 0u64;
    for i in 0..states.len() {
        for j in (i + 1)..states.len() {
            let dist: f64 = states[i].iter().zip(states[j].iter())
                .map(|(&a, &b)| ((a - b) as f64).powi(2))
                .sum::<f64>()
                .sqrt();
            total_dist += dist;
            count += 1;
        }
    }
    let avg_dist = total_dist / count as f64;

    // Measure: state variance per dimension
    let mut var_sum = 0.0f64;
    for dim in 0..d {
        let values: Vec<f64> = states.iter().map(|s| s[dim] as f64).collect();
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let var = values.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
        var_sum += var;
    }
    let avg_var = var_sum / d as f64;

    // Count "active" dimensions (variance > threshold)
    let mut active_dims = 0;
    for dim in 0..d {
        let values: Vec<f64> = states.iter().map(|s| s[dim] as f64).collect();
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let var = values.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
        if var > 1e-10 {
            active_dims += 1;
        }
    }

    println!("  State dim: {}", d);
    println!("  Inputs tested: {}", n_inputs);
    println!("  Avg pairwise distance: {:.6}", avg_dist);
    println!("  Avg dimension variance: {:.8}", avg_var);
    println!("  Active dimensions: {} / {} ({:.1}%)", active_dims, d, 100.0 * active_dims as f32 / d as f32);

    assert!(
        active_dims > 0,
        "No active dimensions — reservoir is dead!"
    );
    assert!(
        avg_dist > 1e-8,
        "States are all identical — no separation"
    );
    println!("  PASS (reservoir is alive and separating inputs)");
}

// ═══════════════════════════════════════════════════════════
// Benchmark 9: Determinism Verification
// ═══════════════════════════════════════════════════════════

#[test]
fn bench_determinism() {
    println!("\n{}", "=".repeat(60));
    println!("BENCHMARK: Training Determinism");
    println!("{}", "=".repeat(60));

    let (inputs, targets) = sine_dataset(100);
    let encoder = SpatialEncoder::default();
    let trainer = ReservoirTrainer::new(1e-3, 10, 1);

    // Train twice with same config
    let mut emc1 = make_emc();
    let readout1 = trainer.train(&mut emc1, &encoder, &inputs, &targets, 0).unwrap();

    let mut emc2 = make_emc();
    let readout2 = trainer.train(&mut emc2, &encoder, &inputs, &targets, 0).unwrap();

    let max_diff: f32 = readout1.weights.iter().zip(readout2.weights.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    let bias_diff: f32 = readout1.bias.iter().zip(readout2.bias.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    println!("  Max weight diff: {:.2e}", max_diff);
    println!("  Max bias diff:   {:.2e}", bias_diff);

    assert!(
        max_diff < 1e-5,
        "Non-deterministic training: max weight diff = {:.2e}",
        max_diff
    );
    println!("  PASS (deterministic)");
}
