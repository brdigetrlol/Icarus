// Copyright (c) 2025-2026 brdigetrlol. All rights reserved.
// SPDX-License-Identifier: LicenseRef-Icarus-Proprietary
// See LICENSE in the repository root for full license terms.

//! Icarus Full Exhaustive Training Driver — "The Disco"
//!
//! All CPUs blazing. GPU on the big configs. Every encoder. Every parameter combination
//! we can throw at it. Claude as host, driving Icarus to its limits.
//!
//! Phase 1: Quick baselines (all tasks × all encoders × preset configs)
//! Phase 2: Exhaustive NARMA-10 coarse sweep (8 threads, ~18K combos)
//! Phase 3: Fine grid around best from Phase 2 (~3K combos)
//! Phase 4: Multi-tick & warmup sweep with best physics params
//! Phase 5: GPU large-config sweep (multi_layer, full_hierarchy)
//! Phase 6: Best params applied to all remaining tasks
//! Phase 7: Multi-seed validation of champion config

use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::thread;
use std::time::Instant;

/// Print and immediately flush stdout (critical for piped output).
macro_rules! pf {
    ($($arg:tt)*) => {{
        let mut out = std::io::stdout().lock();
        let _ = writeln!(out, $($arg)*);
        let _ = out.flush();
    }};
}

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use icarus_engine::config::{BackendSelection, ManifoldConfig};
use icarus_engine::encoding::{InputEncoder, PhaseEncoder, SpatialEncoder, SpectralEncoder};
use icarus_engine::readout::{FeatureMode, Readout};
use icarus_engine::training::{accuracy, nmse, ReservoirTrainer, RidgeRegression};
use icarus_engine::EmergentManifoldComputer;

const NUM_THREADS: usize = 8;

// ═══════════════════════════════════════════════════════
//  Data Generators
// ═══════════════════════════════════════════════════════

fn generate_narma10(length: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let total = length + 50;
    let u: Vec<f32> = (0..total).map(|_| rng.gen::<f32>() * 0.5).collect();
    let mut y = vec![0.0f32; total];

    for n in 10..total - 1 {
        let sum_y: f32 = (0..10).map(|i| y[n - i]).sum();
        y[n + 1] = 0.3 * y[n] + 0.05 * y[n] * sum_y + 1.5 * u[n - 9] * u[n] + 0.1;
        y[n + 1] = y[n + 1].clamp(-10.0, 10.0);
    }

    let offset = total - length;
    (u[offset..].to_vec(), y[offset..].to_vec())
}

fn generate_mackey_glass(length: usize, seed: u64) -> Vec<f32> {
    let tau = 17usize;
    let beta = 0.2f64;
    let gamma_mg = 0.1f64;
    let n_exp = 10.0f64;
    let total = length + tau + 200;

    let mut x = vec![0.0f64; total];
    let mut rng = StdRng::seed_from_u64(seed);
    for i in 0..=tau {
        x[i] = 1.2 + (rng.gen::<f64>() - 0.5) * 0.1;
    }

    for t in tau..total - 1 {
        let x_tau = x[t - tau];
        let dx = beta * x_tau / (1.0 + x_tau.powf(n_exp)) - gamma_mg * x[t];
        x[t + 1] = x[t] + dx;
    }

    x[total - length..].iter().map(|&v| v as f32).collect()
}

fn generate_sine_prediction(length: usize, freq: f32) -> (Vec<f32>, Vec<f32>) {
    let inputs: Vec<f32> = (0..length)
        .map(|i| (i as f32 * freq * std::f32::consts::TAU / length as f32).sin())
        .collect();
    let targets: Vec<f32> = (0..length)
        .map(|i| ((i + 1) as f32 * freq * std::f32::consts::TAU / length as f32).sin())
        .collect();
    (inputs, targets)
}

fn generate_xor_data(count: usize, seed: u64) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<usize>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut inputs = Vec::with_capacity(count);
    let mut targets = Vec::with_capacity(count);
    let mut labels = Vec::with_capacity(count);

    for _ in 0..count {
        let a: f32 = if rng.gen::<bool>() { 1.0 } else { -1.0 };
        let b: f32 = if rng.gen::<bool>() { 1.0 } else { -1.0 };
        let xor = if (a > 0.0) != (b > 0.0) { 1.0 } else { 0.0 };
        let noise_a = a + (rng.gen::<f32>() - 0.5) * 0.3;
        let noise_b = b + (rng.gen::<f32>() - 0.5) * 0.3;
        inputs.push(vec![noise_a, noise_b]);
        targets.push(vec![xor]);
        labels.push(if xor > 0.5 { 1 } else { 0 });
    }

    (inputs, targets, labels)
}

fn generate_pattern_data(
    num_classes: usize,
    dim: usize,
    samples_per_class: usize,
    noise: f32,
    seed: u64,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<usize>) {
    let mut rng = StdRng::seed_from_u64(seed);

    let prototypes: Vec<Vec<f32>> = (0..num_classes)
        .map(|_| {
            let v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            v.iter().map(|x| x / norm).collect()
        })
        .collect();

    let mut inputs = Vec::new();
    let mut targets = Vec::new();
    let mut labels = Vec::new();

    for (class_idx, proto) in prototypes.iter().enumerate() {
        for _ in 0..samples_per_class {
            let noisy: Vec<f32> = proto
                .iter()
                .map(|x| x + (rng.gen::<f32>() - 0.5) * 2.0 * noise)
                .collect();
            inputs.push(noisy);
            let mut target = vec![0.0f32; num_classes];
            target[class_idx] = 1.0;
            targets.push(target);
            labels.push(class_idx);
        }
    }

    (inputs, targets, labels)
}

// ═══════════════════════════════════════════════════════
//  Types
// ═══════════════════════════════════════════════════════

#[derive(Clone, Copy, Debug)]
enum EncoderType {
    Spatial,
    Phase,
    Spectral,
}

impl EncoderType {
    fn create(&self) -> Box<dyn InputEncoder> {
        match self {
            Self::Spatial => Box::new(SpatialEncoder::default()),
            Self::Phase => Box::new(PhaseEncoder::default()),
            Self::Spectral => Box::new(SpectralEncoder::new()),
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Self::Spatial => "spatial",
            Self::Phase => "phase",
            Self::Spectral => "spectral",
        }
    }
}

const ALL_ENCODERS: [EncoderType; 3] = [EncoderType::Spatial, EncoderType::Phase, EncoderType::Spectral];

#[derive(Clone, Copy)]
struct NarmaTrialParams {
    kw: f32,
    omega: f32,
    gamma: f32,
    rae_steps: u64,
    lambda: f32,
    amp: f32,
    warmup: u64,
    ticks_per_input: u64,
    encoder: EncoderType,
    seed: u64,
    feature_mode: FeatureMode,
    leak_rate: f32,
}

#[derive(Clone)]
struct SweepResult {
    params: NarmaTrialParams,
    nmse: f32,
    #[allow(dead_code)]
    train_ms: f64,
}

struct BaselineResult {
    encoder_name: &'static str,
    config_name: &'static str,
    task_name: &'static str,
    metric_name: &'static str,
    metric_value: f32,
    threshold: f32,
    pass: bool,
    train_ms: f64,
    total_sites: usize,
}

impl BaselineResult {
    fn print_row(&self) {
        let status = if self.pass { "PASS" } else { "FAIL" };
        pf!(
            "  {:12} {:12} {:20} {:>8} {:>8.4} {:>8} {:>8.1}ms {:>6} sites  {}",
            self.encoder_name,
            self.config_name,
            self.task_name,
            self.metric_name,
            self.metric_value,
            format!("<{:.2}", self.threshold),
            self.train_ms,
            self.total_sites,
            status,
        );
    }
}

// ═══════════════════════════════════════════════════════
//  Core Trial Runners
// ═══════════════════════════════════════════════════════

/// Run a single NARMA-10 trial with given parameters. Returns NMSE.
fn run_narma_trial(
    params: &NarmaTrialParams,
    encoder: &dyn InputEncoder,
    inputs: &[f32],
    targets: &[f32],
    train_split: usize,
) -> (f32, f64) {
    let mut config = ManifoldConfig::e8_only();
    config.layers[0].kinetic_weight = params.kw;
    config.layers[0].omega = params.omega;
    config.layers[0].gamma = params.gamma;
    config.layers[0].rae_steps_per_tick = params.rae_steps;
    config.backend = BackendSelection::Cpu;

    let train_inputs: Vec<Vec<f32>> = inputs[..train_split].iter().map(|&x| vec![x]).collect();
    let train_targets: Vec<Vec<f32>> = targets[..train_split].iter().map(|&y| vec![y]).collect();
    let test_inputs: Vec<Vec<f32>> = inputs[train_split..].iter().map(|&x| vec![x]).collect();
    let test_targets: Vec<f32> = targets[train_split..].to_vec();

    let mut emc = EmergentManifoldComputer::new_cpu(config);
    emc.init_random(params.seed, params.amp);

    let start = Instant::now();

    let trainer = ReservoirTrainer::new(params.lambda, params.warmup, params.ticks_per_input)
        .with_feature_mode(params.feature_mode)
        .with_leak_rate(params.leak_rate);
    let collector = match trainer.collect_states(&mut emc, encoder, &train_inputs, &train_targets, 0) {
        Ok(c) => c,
        Err(_) => return (f32::MAX, 0.0),
    };
    let ridge = RidgeRegression::new(params.lambda);
    let readout = ridge.train_readout(&collector);

    let train_ms = start.elapsed().as_secs_f64() * 1000.0;

    let use_leaky = params.leak_rate < 1.0;
    let mut predictions = Vec::with_capacity(test_inputs.len());
    for input in &test_inputs {
        if use_leaky {
            encoder.encode_leaky(input, &mut emc.manifold.layers[0].field, params.leak_rate);
        } else {
            encoder.encode(input, &mut emc.manifold.layers[0].field);
        }
        if emc.run(params.ticks_per_input).is_err() {
            return (f32::MAX, train_ms);
        }
        let pred = readout.read(&emc.manifold.layers[0].field);
        predictions.push(pred[0]);
    }

    let score = nmse(&predictions, &test_targets);
    (score, train_ms)
}

/// Run a timeseries trial with a specific config.
fn run_timeseries_trial(
    inputs: &[f32],
    targets: &[f32],
    train_split: usize,
    config: ManifoldConfig,
    config_name: &'static str,
    encoder: &dyn InputEncoder,
    encoder_name: &'static str,
    task_name: &'static str,
    ridge_lambda: f32,
    warmup: u64,
    ticks_per_input: u64,
    threshold: f32,
    seed: u64,
    init_amp: f32,
    leak_rate: f32,
) -> BaselineResult {
    let train_inputs: Vec<Vec<f32>> = inputs[..train_split].iter().map(|&x| vec![x]).collect();
    let train_targets: Vec<Vec<f32>> = targets[..train_split].iter().map(|&y| vec![y]).collect();
    let test_inputs: Vec<Vec<f32>> = inputs[train_split..].iter().map(|&x| vec![x]).collect();
    let test_targets: Vec<f32> = targets[train_split..].to_vec();

    let total_sites: usize = config
        .layers
        .iter()
        .map(|l| match l.layer {
            icarus_math::lattice::LatticeLayer::Analytical => 241,
            icarus_math::lattice::LatticeLayer::Creative => 1105,
            icarus_math::lattice::LatticeLayer::Associative => l.dimension * (l.dimension - 1) + 1,
            icarus_math::lattice::LatticeLayer::Sensory => 2 * l.dimension + 1,
        })
        .sum();

    let use_gpu = matches!(config.backend, BackendSelection::Gpu { .. });
    let mut emc = if use_gpu {
        match EmergentManifoldComputer::new(config) {
            Ok(e) => e,
            Err(err) => {
                eprintln!("  GPU init failed: {err}");
                return BaselineResult {
                    encoder_name, config_name, task_name, metric_name: "NMSE",
                    metric_value: f32::MAX, threshold, pass: false, train_ms: 0.0, total_sites,
                };
            }
        }
    } else {
        EmergentManifoldComputer::new_cpu(config)
    };
    emc.init_random(seed, init_amp);

    let start = Instant::now();

    let trainer = ReservoirTrainer::new(ridge_lambda, warmup, ticks_per_input)
        .with_feature_mode(FeatureMode::Nonlinear)
        .with_leak_rate(leak_rate);
    let collector = match trainer.collect_states(&mut emc, encoder, &train_inputs, &train_targets, 0) {
        Ok(c) => c,
        Err(err) => {
            eprintln!("  Collection failed: {err}");
            return BaselineResult {
                encoder_name, config_name, task_name, metric_name: "NMSE",
                metric_value: f32::MAX, threshold, pass: false,
                train_ms: start.elapsed().as_secs_f64() * 1000.0, total_sites,
            };
        }
    };
    let ridge = RidgeRegression::new(ridge_lambda);
    let readout = ridge.train_readout(&collector);

    let train_ms = start.elapsed().as_secs_f64() * 1000.0;

    let use_leaky = leak_rate < 1.0;
    let mut predictions = Vec::with_capacity(test_inputs.len());
    for input in &test_inputs {
        if use_leaky {
            encoder.encode_leaky(input, &mut emc.manifold.layers[0].field, leak_rate);
        } else {
            encoder.encode(input, &mut emc.manifold.layers[0].field);
        }
        let _ = emc.run(ticks_per_input);
        let pred = readout.read(&emc.manifold.layers[0].field);
        predictions.push(pred[0]);
    }

    let score = nmse(&predictions, &test_targets);

    BaselineResult {
        encoder_name,
        config_name,
        task_name,
        metric_name: "NMSE",
        metric_value: score,
        threshold,
        pass: score < threshold,
        train_ms,
        total_sites,
    }
}

/// Run a classification trial with a specific config.
fn run_classification_trial(
    inputs: &[Vec<f32>],
    targets: &[Vec<f32>],
    test_inputs: &[Vec<f32>],
    test_labels: &[usize],
    config: ManifoldConfig,
    config_name: &'static str,
    encoder: &dyn InputEncoder,
    encoder_name: &'static str,
    task_name: &'static str,
    ridge_lambda: f32,
    warmup: u64,
    ticks_per_input: u64,
    threshold: f32,
    seed: u64,
    init_amp: f32,
    leak_rate: f32,
) -> BaselineResult {
    let total_sites: usize = config
        .layers
        .iter()
        .map(|l| match l.layer {
            icarus_math::lattice::LatticeLayer::Analytical => 241,
            icarus_math::lattice::LatticeLayer::Creative => 1105,
            icarus_math::lattice::LatticeLayer::Associative => l.dimension * (l.dimension - 1) + 1,
            icarus_math::lattice::LatticeLayer::Sensory => 2 * l.dimension + 1,
        })
        .sum();

    let use_gpu = matches!(config.backend, BackendSelection::Gpu { .. });
    let mut emc = if use_gpu {
        match EmergentManifoldComputer::new(config) {
            Ok(e) => e,
            Err(err) => {
                eprintln!("  GPU init failed: {err}");
                return BaselineResult {
                    encoder_name, config_name, task_name, metric_name: "Acc%",
                    metric_value: 0.0, threshold: threshold * 100.0, pass: false,
                    train_ms: 0.0, total_sites,
                };
            }
        }
    } else {
        EmergentManifoldComputer::new_cpu(config)
    };
    emc.init_random(seed, init_amp);

    let start = Instant::now();

    let trainer = ReservoirTrainer::new(ridge_lambda, warmup, ticks_per_input)
        .with_feature_mode(FeatureMode::Nonlinear)
        .with_leak_rate(leak_rate);
    let collector = match trainer.collect_states(&mut emc, encoder, inputs, targets, 0) {
        Ok(c) => c,
        Err(err) => {
            eprintln!("  Collection failed: {err}");
            return BaselineResult {
                encoder_name, config_name, task_name, metric_name: "Acc%",
                metric_value: 0.0, threshold: threshold * 100.0, pass: false,
                train_ms: start.elapsed().as_secs_f64() * 1000.0, total_sites,
            };
        }
    };
    let ridge = RidgeRegression::new(ridge_lambda);
    let readout = ridge.train_readout(&collector);

    let train_ms = start.elapsed().as_secs_f64() * 1000.0;

    let use_leaky = leak_rate < 1.0;
    let mut predictions = Vec::with_capacity(test_inputs.len());
    for input in test_inputs {
        if use_leaky {
            encoder.encode_leaky(input, &mut emc.manifold.layers[0].field, leak_rate);
        } else {
            encoder.encode(input, &mut emc.manifold.layers[0].field);
        }
        let _ = emc.run(ticks_per_input);
        let pred = readout.read(&emc.manifold.layers[0].field);
        predictions.push(pred);
    }

    let acc = accuracy(&predictions, test_labels);

    BaselineResult {
        encoder_name,
        config_name,
        task_name,
        metric_name: "Acc%",
        metric_value: acc * 100.0,
        threshold: threshold * 100.0,
        pass: acc >= threshold,
        train_ms,
        total_sites,
    }
}

// ═══════════════════════════════════════════════════════
//  Parallel Sweep Infrastructure
// ═══════════════════════════════════════════════════════

fn parallel_narma_sweep(
    trials: &[NarmaTrialParams],
    inputs: &[f32],
    targets: &[f32],
    train_split: usize,
    phase_name: &str,
) -> Vec<SweepResult> {
    let total = trials.len();
    let counter = AtomicUsize::new(0);
    let results: Mutex<Vec<SweepResult>> = Mutex::new(Vec::with_capacity(total));
    let best_nmse = Mutex::new(f32::MAX);
    let start = Instant::now();

    pf!("  [{phase_name}] Launching {total} trials across {NUM_THREADS} threads...");

    thread::scope(|s| {
        for _thread_id in 0..NUM_THREADS {
            let counter = &counter;
            let results = &results;
            let best_nmse = &best_nmse;
            let trials = trials;
            let inputs = inputs;
            let targets = targets;

            s.spawn(move || {
                // Each thread creates its own encoders (SpectralEncoder is expensive to clone)
                let spatial = SpatialEncoder::default();
                let phase_enc = PhaseEncoder::default();
                let spectral = SpectralEncoder::new();

                loop {
                    let idx = counter.fetch_add(1, Ordering::Relaxed);
                    if idx >= total {
                        break;
                    }

                    let params = &trials[idx];
                    let encoder: &dyn InputEncoder = match params.encoder {
                        EncoderType::Spatial => &spatial,
                        EncoderType::Phase => &phase_enc,
                        EncoderType::Spectral => &spectral,
                    };

                    let (score, train_ms) =
                        run_narma_trial(params, encoder, inputs, targets, train_split);

                    let result = SweepResult {
                        params: *params,
                        nmse: score,
                        train_ms,
                    };

                    // Check if new best
                    {
                        let mut best = best_nmse.lock().unwrap();
                        if score < *best && score.is_finite() {
                            *best = score;
                            let rate = (idx + 1) as f64 / start.elapsed().as_secs_f64();
                            let eta = if rate > 0.0 {
                                ((total - idx - 1) as f64 / rate) as u64
                            } else {
                                0
                            };
                            pf!(
                                "  [{}/{} {:.0}/s ETA:{:02}:{:02}] {} kw={:.3} w={:.1} g={:.3} s={} a={:.1} l={:.0e} lr={:.1} → NMSE={:.4} NEW BEST",
                                idx + 1, total, rate, eta / 60, eta % 60,
                                params.encoder.name(), params.kw, params.omega, params.gamma,
                                params.rae_steps, params.amp, params.lambda, params.leak_rate, score,
                            );
                        }
                    }

                    // Progress update every 500 trials
                    if (idx + 1) % 500 == 0 {
                        let elapsed = start.elapsed().as_secs();
                        let rate = (idx + 1) as f64 / start.elapsed().as_secs_f64();
                        let eta = if rate > 0.0 {
                            ((total - idx - 1) as f64 / rate) as u64
                        } else {
                            0
                        };
                        let best = best_nmse.lock().unwrap();
                        pf!(
                            "  [{}/{} {:.0}/s ETA:{:02}:{:02}] best so far: NMSE={:.4} ({}s elapsed)",
                            idx + 1, total, rate, eta / 60, eta % 60, *best, elapsed,
                        );
                    }

                    results.lock().unwrap().push(result);
                }
            });
        }
    });

    let elapsed = start.elapsed().as_secs_f64();
    let rate = total as f64 / elapsed;
    pf!(
        "  [{phase_name}] Done: {total} trials in {:.1}s ({:.0} trials/s)",
        elapsed, rate,
    );

    let mut results = results.into_inner().unwrap();
    results.sort_by(|a, b| a.nmse.partial_cmp(&b.nmse).unwrap_or(std::cmp::Ordering::Equal));
    results
}

// ═══════════════════════════════════════════════════════
//  Config Builders
// ═══════════════════════════════════════════════════════

fn make_e8_config(kw: f32, omega: f32, gamma: f32, steps: u64) -> ManifoldConfig {
    let mut c = ManifoldConfig::e8_only();
    c.layers[0].kinetic_weight = kw;
    c.layers[0].omega = omega;
    c.layers[0].gamma = gamma;
    c.layers[0].rae_steps_per_tick = steps;
    c.backend = BackendSelection::Cpu;
    c
}

fn e8_gentle() -> ManifoldConfig {
    make_e8_config(0.5, 0.5, 0.15, 50)
}

fn e8_classify() -> ManifoldConfig {
    make_e8_config(0.5, 0.3, 0.2, 80)
}

fn multi_layer_cpu() -> ManifoldConfig {
    let mut c = ManifoldConfig::multi_layer();
    c.backend = BackendSelection::Cpu;
    c
}

fn full_hierarchy_gpu() -> ManifoldConfig {
    let mut c = ManifoldConfig::full_hierarchy();
    c.backend = BackendSelection::Gpu { device_id: 0 };
    c
}

// ═══════════════════════════════════════════════════════
//  MAIN — The Disco
// ═══════════════════════════════════════════════════════

fn main() {
    pf!();
    pf!("  ██╗ ██████╗ █████╗ ██████╗ ██╗   ██╗███████╗    ██████╗ ██╗███████╗ ██████╗ ██████╗ ");
    pf!("  ██║██╔════╝██╔══██╗██╔══██╗██║   ██║██╔════╝    ██╔══██╗██║██╔════╝██╔════╝██╔═══██╗");
    pf!("  ██║██║     ███████║██████╔╝██║   ██║███████╗    ██║  ██║██║███████╗██║     ██║   ██║");
    pf!("  ██║██║     ██╔══██║██╔══██╗██║   ██║╚════██║    ██║  ██║██║╚════██║██║     ██║   ██║");
    pf!("  ██║╚██████╗██║  ██║██║  ██║╚██████╔╝███████║    ██████╔╝██║███████║╚██████╗╚██████╔╝");
    pf!("  ╚═╝ ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝    ╚═════╝ ╚═╝╚══════╝ ╚═════╝ ╚═════╝ ");
    pf!();
    pf!("  Exhaustive Parameter Sweep — Claude as Host — HYBRID CPU+GPU");
    pf!("  Threads: {NUM_THREADS} | GPU: RTX 5070 Ti (sm_120 Blackwell) | CPU: E8 (241) | GPU: Multi-Layer (1587+)");
    pf!();

    let grand_start = Instant::now();
    let mut all_baselines: Vec<BaselineResult> = Vec::new();

    // Pre-generate NARMA data (shared across all trials)
    let (narma_inp, narma_tgt) = generate_narma10(2000, 123);
    let narma_split = 1500;

    // Shared encoder instances for baseline trials
    let spatial = SpatialEncoder::default();
    let phase_enc = PhaseEncoder::default();
    let spectral = SpectralEncoder::new();
    let encoders: Vec<(&dyn InputEncoder, &str)> = vec![
        (&spatial, "spatial"),
        (&phase_enc, "phase"),
        (&spectral, "spectral"),
    ];

    // ═══════════════════════════════════════════════════
    //  PHASE 1: Quick Baselines
    // ═══════════════════════════════════════════════════
    pf!("╔══════════════════════════════════════════════════════════════════════════════════╗");
    pf!("║  PHASE 1: Quick Baselines — All Tasks x All Encoders x Preset Configs          ║");
    pf!("╚══════════════════════════════════════════════════════════════════════════════════╝\n");

    // --- Sine wave ---
    pf!("  --- Sine Wave Prediction ---");
    {
        let (inp, tgt) = generate_sine_prediction(500, 3.0);
        for &(enc, enc_name) in &encoders {
            let r = run_timeseries_trial(
                &inp, &tgt, 350, e8_gentle(), "e8-gentle", enc, enc_name,
                "sine-prediction", 1e-4, 30, 1, 0.05, 42, 0.3, 1.0,
            );
            r.print_row();
            all_baselines.push(r);
        }
    }
    pf!();

    // --- Mackey-Glass ---
    pf!("  --- Mackey-Glass Chaotic Prediction ---");
    {
        let series = generate_mackey_glass(600, 456);
        let inputs: Vec<f32> = series[..series.len() - 1].to_vec();
        let targets: Vec<f32> = series[1..].to_vec();
        for &(enc, enc_name) in &encoders {
            let r = run_timeseries_trial(
                &inputs, &targets, 400, e8_gentle(), "e8-gentle", enc, enc_name,
                "mackey-glass", 1e-4, 30, 1, 0.10, 99, 0.3, 1.0,
            );
            r.print_row();
            all_baselines.push(r);
        }
    }
    pf!();

    // --- NARMA-10 with preset configs ---
    pf!("  --- NARMA-10 (preset configs) ---");
    {
        let configs: Vec<(ManifoldConfig, &str)> = vec![
            (make_e8_config(0.040, 8.0, 0.020, 95), "e8-tuned"),
            (e8_gentle(), "e8-gentle"),
            (multi_layer_cpu(), "multi-layer"),
        ];
        for (config, config_name) in configs {
            for &(enc, enc_name) in &encoders {
                let r = run_timeseries_trial(
                    &narma_inp, &narma_tgt, narma_split, config.clone(), config_name, enc, enc_name,
                    "narma10", 1e-3, 100, 1, 0.90, 42, 0.3, 1.0,
                );
                r.print_row();
                all_baselines.push(r);
            }
        }
    }
    pf!();

    // --- XOR ---
    pf!("  --- XOR Classification ---");
    {
        let (train_inp, train_tgt, _train_lbl) = generate_xor_data(200, 42);
        let (test_inp, _, test_lbl) = generate_xor_data(100, 99);
        for &(enc, enc_name) in &encoders {
            let r = run_classification_trial(
                &train_inp, &train_tgt, &test_inp, &test_lbl,
                e8_classify(), "e8-classify", enc, enc_name,
                "xor", 1e-3, 10, 2, 0.80, 55, 0.3, 1.0,
            );
            r.print_row();
            all_baselines.push(r);
        }
    }
    pf!();

    // --- 10-class patterns ---
    pf!("  --- 10-Class Pattern Classification ---");
    {
        let (train_inp, train_tgt, _train_lbl) = generate_pattern_data(10, 8, 20, 0.2, 789);
        let (test_inp, _, test_lbl) = generate_pattern_data(10, 8, 10, 0.2, 999);
        let configs: Vec<(ManifoldConfig, &str)> = vec![
            (e8_classify(), "e8-classify"),
            (multi_layer_cpu(), "multi-layer"),
        ];
        for (config, config_name) in configs {
            for &(enc, enc_name) in &encoders {
                let r = run_classification_trial(
                    &train_inp, &train_tgt, &test_inp, &test_lbl,
                    config.clone(), config_name, enc, enc_name,
                    "pattern-10class", 1e-3, 10, 2, 0.70, 55, 0.3, 1.0,
                );
                r.print_row();
                all_baselines.push(r);
            }
        }
    }
    pf!();

    let phase1_time = grand_start.elapsed().as_secs_f64();
    pf!("  Phase 1 complete: {} trials in {:.1}s\n", all_baselines.len(), phase1_time);

    // ═══════════════════════════════════════════════════
    //  PHASE 2: Exhaustive Coarse NARMA-10 Sweep
    // ═══════════════════════════════════════════════════
    pf!("╔══════════════════════════════════════════════════════════════════════════════════╗");
    pf!("║  PHASE 2: Exhaustive Coarse Grid — NARMA-10 — 8 Threads Blazing                ║");
    pf!("╚══════════════════════════════════════════════════════════════════════════════════╝\n");

    let coarse_kw = [0.005f32, 0.010, 0.020, 0.035, 0.040, 0.045, 0.060, 0.100, 0.250, 0.500];
    let coarse_omega = [0.1f32, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 20.0];
    let coarse_gamma = [0.005f32, 0.010, 0.015, 0.020, 0.030, 0.050, 0.100, 0.200];
    let coarse_steps: [u64; 5] = [30, 50, 80, 95, 120];
    let coarse_lambda = [1e-5f32, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2];
    let coarse_amp = [0.1f32, 0.3, 0.5];
    let coarse_leak = [0.1f32, 0.3, 0.5, 0.8, 1.0];

    let mut coarse_trials: Vec<NarmaTrialParams> = Vec::new();
    for &enc in &ALL_ENCODERS {
        for &kw in &coarse_kw {
            for &omega in &coarse_omega {
                for &gamma in &coarse_gamma {
                    for &steps in &coarse_steps {
                        for &lambda in &coarse_lambda {
                            for &amp in &coarse_amp {
                                for &leak in &coarse_leak {
                                    coarse_trials.push(NarmaTrialParams {
                                        kw, omega, gamma, rae_steps: steps, lambda, amp,
                                        warmup: 100, ticks_per_input: 1,
                                        encoder: enc, seed: 42,
                                        feature_mode: FeatureMode::Nonlinear,
                                        leak_rate: leak,
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    pf!("  Grid: {} kw x {} omega x {} gamma x {} steps x {} lambda x {} amp x {} leak x {} encoders = {} combos",
        coarse_kw.len(), coarse_omega.len(), coarse_gamma.len(),
        coarse_steps.len(), coarse_lambda.len(), coarse_amp.len(),
        coarse_leak.len(), ALL_ENCODERS.len(), coarse_trials.len());

    let coarse_results = parallel_narma_sweep(
        &coarse_trials, &narma_inp, &narma_tgt, narma_split, "COARSE",
    );

    // Print top 20 results
    pf!("\n  Top 20 from coarse sweep:");
    pf!("  {:>4}  {:8}  {:>6}  {:>6}  {:>6}  {:>4}  {:>8}  {:>4}  {:>4}  {:>8}",
        "Rank", "Encoder", "kw", "omega", "gamma", "s", "lambda", "amp", "lr", "NMSE");
    pf!("  {}", "-".repeat(78));
    for (i, r) in coarse_results.iter().take(20).enumerate() {
        pf!(
            "  {:>4}  {:8}  {:>6.3}  {:>6.1}  {:>6.3}  {:>4}  {:>8.0e}  {:>4.1}  {:>4.1}  {:>8.4}",
            i + 1, r.params.encoder.name(), r.params.kw, r.params.omega,
            r.params.gamma, r.params.rae_steps, r.params.lambda,
            r.params.amp, r.params.leak_rate, r.nmse,
        );
    }
    pf!();

    let best_coarse = &coarse_results[0];
    pf!(
        "  PHASE 2 CHAMPION: {} kw={:.3} omega={:.1} gamma={:.3} steps={} lambda={:.0e} amp={:.1} lr={:.1} → NMSE={:.4}",
        best_coarse.params.encoder.name(), best_coarse.params.kw, best_coarse.params.omega,
        best_coarse.params.gamma, best_coarse.params.rae_steps, best_coarse.params.lambda,
        best_coarse.params.amp, best_coarse.params.leak_rate, best_coarse.nmse,
    );
    pf!();

    // ═══════════════════════════════════════════════════
    //  PHASE 3: Fine Grid Around Best
    // ═══════════════════════════════════════════════════
    pf!("╔══════════════════════════════════════════════════════════════════════════════════╗");
    pf!("║  PHASE 3: Fine Grid Refinement — Zooming In On The Sweet Spot                  ║");
    pf!("╚══════════════════════════════════════════════════════════════════════════════════╝\n");

    let bp = &best_coarse.params;

    // Generate fine grid centered on best
    let fine_kw = linspace_around(bp.kw, 0.005, 0.8, 9);
    let fine_omega = linspace_around(bp.omega, 0.5, 50.0, 7);
    let fine_gamma = linspace_around(bp.gamma, 0.002, 0.5, 7);
    let fine_steps: Vec<u64> = {
        let center = bp.rae_steps as i64;
        vec![
            (center - 20).max(10) as u64,
            (center - 10).max(10) as u64,
            center as u64,
            (center + 10) as u64,
            (center + 25) as u64,
        ]
    };
    let fine_lambda = vec![
        bp.lambda * 0.1, bp.lambda * 0.3, bp.lambda * 0.5,
        bp.lambda, bp.lambda * 2.0, bp.lambda * 5.0, bp.lambda * 10.0,
    ];
    let fine_amp = vec![
        (bp.amp - 0.15).max(0.05), (bp.amp - 0.05).max(0.05),
        bp.amp, bp.amp + 0.05, bp.amp + 0.15,
    ];
    let fine_leak = vec![
        (bp.leak_rate - 0.2).max(0.05), (bp.leak_rate - 0.1).max(0.05),
        bp.leak_rate,
        (bp.leak_rate + 0.1).min(1.0), (bp.leak_rate + 0.2).min(1.0),
    ];

    let mut fine_trials: Vec<NarmaTrialParams> = Vec::new();
    // Use best encoder + also try the other two
    for &enc in &ALL_ENCODERS {
        for &kw in &fine_kw {
            for &omega in &fine_omega {
                for &gamma in &fine_gamma {
                    for &steps in &fine_steps {
                        for &lambda in &fine_lambda {
                            for &amp in &fine_amp {
                                for &leak in &fine_leak {
                                    fine_trials.push(NarmaTrialParams {
                                        kw, omega, gamma, rae_steps: steps, lambda, amp,
                                        warmup: 100, ticks_per_input: 1,
                                        encoder: enc, seed: 42,
                                        feature_mode: FeatureMode::Nonlinear,
                                        leak_rate: leak,
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    pf!("  Fine grid: {} combos (centered on Phase 2 champion)", fine_trials.len());

    let fine_results = parallel_narma_sweep(
        &fine_trials, &narma_inp, &narma_tgt, narma_split, "FINE",
    );

    pf!("\n  Top 10 from fine sweep:");
    pf!("  {:>4}  {:8}  {:>6}  {:>6}  {:>6}  {:>4}  {:>8}  {:>4}  {:>4}  {:>8}",
        "Rank", "Encoder", "kw", "omega", "gamma", "s", "lambda", "amp", "lr", "NMSE");
    pf!("  {}", "-".repeat(78));
    for (i, r) in fine_results.iter().take(10).enumerate() {
        pf!(
            "  {:>4}  {:8}  {:>6.3}  {:>6.1}  {:>6.3}  {:>4}  {:>8.0e}  {:>4.2}  {:>4.2}  {:>8.4}",
            i + 1, r.params.encoder.name(), r.params.kw, r.params.omega,
            r.params.gamma, r.params.rae_steps, r.params.lambda,
            r.params.amp, r.params.leak_rate, r.nmse,
        );
    }
    pf!();

    let best_fine = &fine_results[0];
    pf!(
        "  PHASE 3 CHAMPION: {} kw={:.4} omega={:.2} gamma={:.4} steps={} lambda={:.2e} amp={:.2} lr={:.2} → NMSE={:.4}",
        best_fine.params.encoder.name(), best_fine.params.kw, best_fine.params.omega,
        best_fine.params.gamma, best_fine.params.rae_steps, best_fine.params.lambda,
        best_fine.params.amp, best_fine.params.leak_rate, best_fine.nmse,
    );
    pf!();

    // ═══════════════════════════════════════════════════
    //  PHASE 4: Multi-Tick & Warmup Sweep
    // ═══════════════════════════════════════════════════
    pf!("╔══════════════════════════════════════════════════════════════════════════════════╗");
    pf!("║  PHASE 4: Tick Multiplier & Warmup Sweep — Temporal Dynamics Exploration        ║");
    pf!("╚══════════════════════════════════════════════════════════════════════════════════╝\n");

    let bp = &best_fine.params;
    let mut tick_trials: Vec<NarmaTrialParams> = Vec::new();
    let tick_vals: [u64; 5] = [1, 2, 3, 4, 5];
    let warmup_vals: [u64; 5] = [20, 50, 100, 200, 500];

    for &enc in &ALL_ENCODERS {
        for &ticks in &tick_vals {
            for &warmup in &warmup_vals {
                tick_trials.push(NarmaTrialParams {
                    kw: bp.kw, omega: bp.omega, gamma: bp.gamma,
                    rae_steps: bp.rae_steps, lambda: bp.lambda, amp: bp.amp,
                    warmup, ticks_per_input: ticks,
                    encoder: enc, seed: 42,
                    feature_mode: FeatureMode::Nonlinear,
                    leak_rate: bp.leak_rate,
                });
            }
        }
    }

    pf!("  Tick × Warmup grid: {} ticks x {} warmups x 3 encoders = {} combos",
        tick_vals.len(), warmup_vals.len(), tick_trials.len());

    let tick_results = parallel_narma_sweep(
        &tick_trials, &narma_inp, &narma_tgt, narma_split, "TICK",
    );

    pf!("\n  Top 10 from tick/warmup sweep:");
    pf!("  {:>4}  {:8}  {:>5}  {:>6}  {:>8}",
        "Rank", "Encoder", "Ticks", "Warmup", "NMSE");
    pf!("  {}", "-".repeat(42));
    for (i, r) in tick_results.iter().take(10).enumerate() {
        pf!(
            "  {:>4}  {:8}  {:>5}  {:>6}  {:>8.4}",
            i + 1, r.params.encoder.name(), r.params.ticks_per_input,
            r.params.warmup, r.nmse,
        );
    }
    pf!();

    let best_tick = &tick_results[0];
    pf!(
        "  PHASE 4 CHAMPION: {} ticks={} warmup={} → NMSE={:.4}",
        best_tick.params.encoder.name(), best_tick.params.ticks_per_input,
        best_tick.params.warmup, best_tick.nmse,
    );
    pf!();

    // Overall best so far (merge best from each phase)
    let candidates = [best_coarse, best_fine, best_tick];
    let overall_best = candidates
        .iter()
        .min_by(|a, b| a.nmse.partial_cmp(&b.nmse).unwrap())
        .unwrap();

    pf!(
        "  OVERALL BEST SO FAR: {} kw={:.4} omega={:.2} gamma={:.4} steps={} lambda={:.2e} amp={:.2} lr={:.2} ticks={} warmup={} → NMSE={:.4}\n",
        overall_best.params.encoder.name(), overall_best.params.kw, overall_best.params.omega,
        overall_best.params.gamma, overall_best.params.rae_steps, overall_best.params.lambda,
        overall_best.params.amp, overall_best.params.leak_rate, overall_best.params.ticks_per_input,
        overall_best.params.warmup, overall_best.nmse,
    );

    // ═══════════════════════════════════════════════════
    //  PHASE 5: GPU Large-Config Sweep
    // ═══════════════════════════════════════════════════
    pf!("╔══════════════════════════════════════════════════════════════════════════════════╗");
    pf!("║  PHASE 5: GPU Sweep — Multi-Layer & Full Hierarchy (RTX 5070 Ti)               ║");
    pf!("╚══════════════════════════════════════════════════════════════════════════════════╝\n");

    let ob = &overall_best.params;

    // Multi-layer GPU
    pf!("  --- Multi-Layer GPU (1587 sites, 3 layers) ---");
    {
        let kw_vals = [ob.kw * 0.5, ob.kw, ob.kw * 2.0];
        let omega_vals = [ob.omega * 0.5, ob.omega, ob.omega * 2.0];
        let gamma_vals = [ob.gamma * 0.5, ob.gamma, ob.gamma * 2.0];
        let lambda_vals = [ob.lambda * 0.1, ob.lambda, ob.lambda * 10.0];

        let mut gpu_best_nmse = f32::MAX;
        let mut gpu_best_desc = String::new();
        let mut gpu_tried = 0u32;

        for &(enc, enc_name) in &encoders {
            for &kw in &kw_vals {
                for &omega in &omega_vals {
                    for &gamma in &gamma_vals {
                        for &lambda in &lambda_vals {
                            gpu_tried += 1;
                            let mut config = ManifoldConfig::multi_layer();
                            config.backend = BackendSelection::Gpu { device_id: 0 };
                            config.layers[0].kinetic_weight = kw;
                            config.layers[0].omega = omega;
                            config.layers[0].gamma = gamma;
                            config.layers[0].rae_steps_per_tick = ob.rae_steps;

                            let r = run_timeseries_trial(
                                &narma_inp, &narma_tgt, narma_split,
                                config, "ml-gpu", enc, enc_name,
                                "narma10-gpu", lambda, ob.warmup, ob.ticks_per_input,
                                1.0, 42, ob.amp, ob.leak_rate,
                            );

                            if r.metric_value < gpu_best_nmse {
                                gpu_best_nmse = r.metric_value;
                                gpu_best_desc = format!(
                                    "{} kw={:.3} w={:.1} g={:.3} l={:.0e}",
                                    enc_name, kw, omega, gamma, lambda
                                );
                                pf!(
                                    "  [{}/{}] {} → NMSE={:.4} ({:.0}ms) NEW BEST",
                                    gpu_tried,
                                    kw_vals.len() * omega_vals.len() * gamma_vals.len() * lambda_vals.len() * 3,
                                    gpu_best_desc, gpu_best_nmse, r.train_ms,
                                );
                            }

                            all_baselines.push(r);
                        }
                    }
                }
            }
        }

        pf!("  Multi-Layer GPU: {} trials → BEST: {} → NMSE={:.4}", gpu_tried, gpu_best_desc, gpu_best_nmse);
    }
    pf!();

    // Multi-layer CPU comparison (baseline vs GPU)
    pf!("  --- Multi-Layer CPU Baseline (1587 sites, 3 layers) ---");
    {
        let mut config = ManifoldConfig::multi_layer();
        config.backend = BackendSelection::Cpu;
        config.layers[0].kinetic_weight = ob.kw;
        config.layers[0].omega = ob.omega;
        config.layers[0].gamma = ob.gamma;
        config.layers[0].rae_steps_per_tick = ob.rae_steps;

        for &(enc, enc_name) in &encoders {
            let r = run_timeseries_trial(
                &narma_inp, &narma_tgt, narma_split,
                config.clone(), "ml-cpu", enc, enc_name,
                "narma10-cpu", ob.lambda, ob.warmup, ob.ticks_per_input,
                1.0, 42, ob.amp, ob.leak_rate,
            );
            r.print_row();
            all_baselines.push(r);
        }
    }
    pf!();

    // Full hierarchy GPU
    pf!("  --- Full Hierarchy GPU (4 layers, metric learning) ---");
    {
        let config = full_hierarchy_gpu();
        for &(enc, enc_name) in &encoders {
            let r = run_timeseries_trial(
                &narma_inp, &narma_tgt, narma_split,
                config.clone(), "full-gpu", enc, enc_name,
                "narma10-full", ob.lambda, ob.warmup, ob.ticks_per_input,
                1.0, 42, ob.amp, ob.leak_rate,
            );
            r.print_row();
            all_baselines.push(r);
        }
    }
    pf!();

    // ═══════════════════════════════════════════════════
    //  PHASE 6: Best Params → All Tasks
    // ═══════════════════════════════════════════════════
    pf!("╔══════════════════════════════════════════════════════════════════════════════════╗");
    pf!("║  PHASE 6: Champion Config Applied To All Tasks                                 ║");
    pf!("╚══════════════════════════════════════════════════════════════════════════════════╝\n");

    let ob = &overall_best.params;

    // Sine wave with best params
    pf!("  --- Sine Wave (champion params) ---");
    {
        let (inp, tgt) = generate_sine_prediction(500, 3.0);
        let config = make_e8_config(ob.kw, ob.omega, ob.gamma, ob.rae_steps);
        for &(enc, enc_name) in &encoders {
            let r = run_timeseries_trial(
                &inp, &tgt, 350, config.clone(), "e8-champ", enc, enc_name,
                "sine-champ", ob.lambda, ob.warmup, ob.ticks_per_input, 0.01, ob.seed, ob.amp, ob.leak_rate,
            );
            r.print_row();
            all_baselines.push(r);
        }
    }
    pf!();

    // Mackey-Glass with best params
    pf!("  --- Mackey-Glass (champion params) ---");
    {
        let series = generate_mackey_glass(600, 456);
        let inputs: Vec<f32> = series[..series.len() - 1].to_vec();
        let targets: Vec<f32> = series[1..].to_vec();
        let config = make_e8_config(ob.kw, ob.omega, ob.gamma, ob.rae_steps);
        for &(enc, enc_name) in &encoders {
            let r = run_timeseries_trial(
                &inputs, &targets, 400, config.clone(), "e8-champ", enc, enc_name,
                "mg-champ", ob.lambda, ob.warmup, ob.ticks_per_input, 0.05, ob.seed, ob.amp, ob.leak_rate,
            );
            r.print_row();
            all_baselines.push(r);
        }
    }
    pf!();

    // XOR with best params
    pf!("  --- XOR Classification (champion params) ---");
    {
        let (train_inp, train_tgt, _train_lbl) = generate_xor_data(200, 42);
        let (test_inp, _, test_lbl) = generate_xor_data(100, 99);
        let config = make_e8_config(ob.kw, ob.omega, ob.gamma, ob.rae_steps);
        for &(enc, enc_name) in &encoders {
            let r = run_classification_trial(
                &train_inp, &train_tgt, &test_inp, &test_lbl,
                config.clone(), "e8-champ", enc, enc_name,
                "xor-champ", ob.lambda, ob.warmup, ob.ticks_per_input, 0.80, ob.seed, ob.amp, ob.leak_rate,
            );
            r.print_row();
            all_baselines.push(r);
        }
    }
    pf!();

    // 10-class pattern with best params
    pf!("  --- 10-Class Pattern (champion params) ---");
    {
        let (train_inp, train_tgt, _train_lbl) = generate_pattern_data(10, 8, 20, 0.2, 789);
        let (test_inp, _, test_lbl) = generate_pattern_data(10, 8, 10, 0.2, 999);
        let config = make_e8_config(ob.kw, ob.omega, ob.gamma, ob.rae_steps);
        for &(enc, enc_name) in &encoders {
            let r = run_classification_trial(
                &train_inp, &train_tgt, &test_inp, &test_lbl,
                config.clone(), "e8-champ", enc, enc_name,
                "pattern-champ", ob.lambda, ob.warmup, ob.ticks_per_input, 0.70, ob.seed, ob.amp, ob.leak_rate,
            );
            r.print_row();
            all_baselines.push(r);
        }
    }
    pf!();

    // ═══════════════════════════════════════════════════
    //  PHASE 7: Multi-Seed Validation
    // ═══════════════════════════════════════════════════
    pf!("╔══════════════════════════════════════════════════════════════════════════════════╗");
    pf!("║  PHASE 7: Multi-Seed Validation — Is the Champion Robust?                      ║");
    pf!("╚══════════════════════════════════════════════════════════════════════════════════╝\n");

    let ob = &overall_best.params;
    let seeds: [u64; 10] = [42, 123, 456, 789, 1337, 2024, 3141, 9876, 31415, 65536];

    let mut seed_trials: Vec<NarmaTrialParams> = Vec::new();
    for &seed in &seeds {
        for &enc in &ALL_ENCODERS {
            seed_trials.push(NarmaTrialParams {
                kw: ob.kw, omega: ob.omega, gamma: ob.gamma,
                rae_steps: ob.rae_steps, lambda: ob.lambda, amp: ob.amp,
                warmup: ob.warmup, ticks_per_input: ob.ticks_per_input,
                encoder: enc, seed,
                feature_mode: FeatureMode::Nonlinear,
                leak_rate: ob.leak_rate,
            });
        }
    }

    let seed_results = parallel_narma_sweep(
        &seed_trials, &narma_inp, &narma_tgt, narma_split, "SEEDS",
    );

    // Group by encoder
    pf!("\n  Multi-seed results (champion config):");
    for enc_type in &ALL_ENCODERS {
        let enc_results: Vec<&SweepResult> = seed_results
            .iter()
            .filter(|r| r.params.encoder.name() == enc_type.name())
            .collect();
        let scores: Vec<f32> = enc_results.iter().map(|r| r.nmse).collect();
        let mean = scores.iter().sum::<f32>() / scores.len() as f32;
        let std = (scores.iter().map(|s| (s - mean) * (s - mean)).sum::<f32>()
            / scores.len() as f32)
            .sqrt();
        let min = scores.iter().cloned().fold(f32::MAX, f32::min);
        let max = scores.iter().cloned().fold(f32::MIN, f32::max);
        pf!(
            "  {:8}: mean={:.4} std={:.4} min={:.4} max={:.4} ({} seeds)",
            enc_type.name(), mean, std, min, max, scores.len(),
        );
    }
    pf!();

    // Also test with different NARMA seeds
    pf!("  --- Different NARMA-10 sequences (data seed variation) ---");
    let data_seeds: [u64; 5] = [123, 456, 789, 1337, 9999];
    for &data_seed in &data_seeds {
        let (inp, tgt) = generate_narma10(2000, data_seed);
        let enc = ob.encoder.create();
        let (score, ms) = run_narma_trial(ob, enc.as_ref(), &inp, &tgt, narma_split);
        pf!(
            "  data_seed={:>5}: {} → NMSE={:.4} ({:.0}ms)",
            data_seed, ob.encoder.name(), score, ms,
        );
    }
    pf!();

    // ═══════════════════════════════════════════════════
    //  GRAND SUMMARY
    // ═══════════════════════════════════════════════════
    let total_time = grand_start.elapsed().as_secs_f64();
    let total_trials = coarse_trials.len() + fine_trials.len() + tick_trials.len()
        + seed_trials.len() + all_baselines.len() + data_seeds.len();
    let pass_count = all_baselines.iter().filter(|r| r.pass).count();

    pf!("╔══════════════════════════════════════════════════════════════════════════════════════════════╗");
    pf!("║                             GRAND SUMMARY — THE DISCO IS OVER                              ║");
    pf!("╠══════════════════════════════════════════════════════════════════════════════════════════════╣");
    pf!("║  Encoder      Config       Task                  Metric    Value  Thresh  Time     Sites   ║");
    pf!("╠══════════════════════════════════════════════════════════════════════════════════════════════╣");

    for r in &all_baselines {
        let status = if r.pass { "+" } else { "-" };
        pf!(
            "║  {:12} {:12} {:20} {:>8} {:>8.4} {:>8} {:>7.0}ms {:>5}  {} ║",
            r.encoder_name, r.config_name, r.task_name,
            r.metric_name, r.metric_value,
            format!("<{:.2}", r.threshold),
            r.train_ms, r.total_sites, status,
        );
    }

    pf!("╠══════════════════════════════════════════════════════════════════════════════════════════════╣");
    pf!("║  CHAMPION: {:76}║", format!(
        "{} kw={:.4} omega={:.2} gamma={:.4} s={} l={:.2e} amp={:.2} lr={:.2} ticks={} warmup={} → NMSE={:.4}",
        overall_best.params.encoder.name(), overall_best.params.kw, overall_best.params.omega,
        overall_best.params.gamma, overall_best.params.rae_steps, overall_best.params.lambda,
        overall_best.params.amp, overall_best.params.leak_rate, overall_best.params.ticks_per_input,
        overall_best.params.warmup, overall_best.nmse,
    ));
    pf!("╠══════════════════════════════════════════════════════════════════════════════════════════════╣");
    pf!(
        "║  {}/{} baseline tests passed | {} total trials | {:.1}s ({:.1} min){:>28}║",
        pass_count, all_baselines.len(), total_trials,
        total_time, total_time / 60.0, "",
    );
    pf!("╚══════════════════════════════════════════════════════════════════════════════════════════════╝");
}

// ═══════════════════════════════════════════════════════
//  Utility
// ═══════════════════════════════════════════════════════

/// Generate N evenly-spaced values centered on `center`, clamped to [min_val, max_val].
fn linspace_around(center: f32, min_val: f32, max_val: f32, n: usize) -> Vec<f32> {
    if n <= 1 {
        return vec![center];
    }

    // Determine a reasonable spread: 3x the center value on each side, or at least 0.5
    let spread = (center.abs() * 1.5).max(0.01);
    let lo = (center - spread).max(min_val);
    let hi = (center + spread).min(max_val);
    let step = (hi - lo) / (n - 1) as f32;

    let mut vals: Vec<f32> = (0..n).map(|i| lo + step * i as f32).collect();

    // Ensure center is included
    if !vals.iter().any(|&v| (v - center).abs() < step * 0.1) {
        vals.push(center);
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    }

    vals
}
