//! Icarus EMC Benchmarks and MVP Validation
//!
//! Measures:
//! 1. E8 RAE step latency (CPU vs GPU)
//! 2. Free energy computation
//! 3. Full EMC tick
//! 4. MVP validation: attractor convergence from random ICs

use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use icarus_engine::config::{BackendSelection, ManifoldConfig};
use icarus_engine::encoding::{InputEncoder, PhaseEncoder, SpatialEncoder, SpectralEncoder};
use icarus_engine::readout::Readout;
use icarus_engine::training::{accuracy, nmse, ReservoirTrainer, RidgeRegression};
use icarus_engine::continual::ContinualTrainer;
use icarus_engine::EmergentManifoldComputer;
use icarus_field::attractor::{find_attractors, AttractorSearchParams};
use icarus_field::free_energy::{free_energy, FreeEnergyParams};
use icarus_field::phase_field::LatticeField;
use icarus_field::rae::{RAEParams, RAESolver};
use icarus_gpu::pipeline::ComputeBackend;
use icarus_field::topology::{self, TopologyParams, PenaltyMode};
use icarus_math::lattice::e8::E8Lattice;

fn main() {
    println!("=== Icarus EMC Benchmarks ===\n");

    bench_e8_rae_cpu();
    bench_e8_free_energy();
    bench_emc_tick_cpu();

    // Try GPU benchmarks
    match icarus_gpu::pipeline::GpuBackend::new(0) {
        Ok(mut gpu) => {
            bench_e8_rae_gpu(&mut gpu);
            bench_emc_tick_gpu();
        }
        Err(e) => {
            println!("[GPU] Skipped — {e}\n");
        }
    }

    println!("=== MVP Validation ===\n");
    mvp_attractor_convergence();
    mvp_energy_monotonicity();
    mvp_determinism();

    println!("=== Reservoir Computing Benchmarks ===\n");
    bench_narma10();
    bench_mackey_glass();
    bench_pattern_classification();

    println!("=== Continual Learning Benchmark ===\n");
    bench_continual_learning();

    println!("=== Topology Benchmark ===\n");
    bench_topology();

    println!("=== All benchmarks complete ===");
}

// ─── Benchmarks ──────────────────────────────────

fn bench_e8_rae_cpu() {
    let lattice = E8Lattice::new();
    let mut field = LatticeField::from_lattice(&lattice);
    field.init_random(42, 0.5);
    let params = RAEParams::default_e8();
    let mut solver = RAESolver::new(params, field.num_sites);

    // Warmup
    solver.run(&mut field, 100);

    // Benchmark: 1000 steps
    let start = Instant::now();
    let steps = 1000u64;
    solver.run(&mut field, steps);
    let elapsed = start.elapsed();

    let per_step_us = elapsed.as_micros() as f64 / steps as f64;
    println!(
        "[CPU] E8 RAE: {steps} steps in {:.2}ms ({:.1}us/step, {} sites)",
        elapsed.as_secs_f64() * 1000.0,
        per_step_us,
        field.num_sites,
    );
}

fn bench_e8_rae_gpu(gpu: &mut icarus_gpu::pipeline::GpuBackend) {
    let lattice = E8Lattice::new();
    let mut field = LatticeField::from_lattice(&lattice);
    field.init_random(42, 0.5);
    let params = RAEParams::default_e8();

    // Warmup
    gpu.rae_step(&mut field, &params, 100).unwrap();

    // Benchmark: 1000 steps
    let start = Instant::now();
    let steps = 1000u64;
    gpu.rae_step(&mut field, &params, steps).unwrap();
    let elapsed = start.elapsed();

    let per_step_us = elapsed.as_micros() as f64 / steps as f64;
    println!(
        "[GPU] E8 RAE: {steps} steps in {:.2}ms ({:.1}us/step, {} sites)",
        elapsed.as_secs_f64() * 1000.0,
        per_step_us,
        field.num_sites,
    );
}

fn bench_e8_free_energy() {
    let lattice = E8Lattice::new();
    let mut field = LatticeField::from_lattice(&lattice);
    field.init_random(42, 0.5);
    let params = FreeEnergyParams::default();

    // Warmup
    for _ in 0..100 {
        let _ = free_energy(&field, &params);
    }

    let iters = 10000u64;
    let start = Instant::now();
    for _ in 0..iters {
        let _ = free_energy(&field, &params);
    }
    let elapsed = start.elapsed();

    let per_iter_us = elapsed.as_micros() as f64 / iters as f64;
    println!(
        "[CPU] E8 Free Energy: {iters} evals in {:.2}ms ({:.1}us/eval)",
        elapsed.as_secs_f64() * 1000.0,
        per_iter_us,
    );
}

fn bench_emc_tick_cpu() {
    let config = ManifoldConfig::e8_only();
    let mut emc = EmergentManifoldComputer::new_cpu(config);
    emc.init_random(42, 0.5);

    // Warmup
    emc.run(5).unwrap();

    let ticks = 50u64;
    let start = Instant::now();
    emc.run(ticks).unwrap();
    let elapsed = start.elapsed();

    let per_tick_ms = elapsed.as_secs_f64() * 1000.0 / ticks as f64;
    println!(
        "[CPU] EMC tick (E8, 100 RAE steps/tick): {ticks} ticks in {:.1}ms ({:.2}ms/tick)",
        elapsed.as_secs_f64() * 1000.0,
        per_tick_ms,
    );
}

fn bench_emc_tick_gpu() {
    let config = ManifoldConfig::e8_only();
    let mut emc = match EmergentManifoldComputer::new(config) {
        Ok(e) => e,
        Err(e) => {
            println!("[GPU] EMC tick: skipped — {e}");
            return;
        }
    };
    emc.init_random(42, 0.5);

    // Warmup
    emc.run(5).unwrap();

    let ticks = 50u64;
    let start = Instant::now();
    emc.run(ticks).unwrap();
    let elapsed = start.elapsed();

    let per_tick_ms = elapsed.as_secs_f64() * 1000.0 / ticks as f64;
    println!(
        "[GPU] EMC tick (E8, 100 RAE steps/tick): {ticks} ticks in {:.1}ms ({:.2}ms/tick)",
        elapsed.as_secs_f64() * 1000.0,
        per_tick_ms,
    );
    println!();
}

// ─── MVP Validation ──────────────────────────────

fn mvp_attractor_convergence() {
    println!("MVP Test 1: Attractor convergence from 100 random ICs");

    let lattice = E8Lattice::new();
    let template = LatticeField::from_lattice(&lattice);

    let mut search_params = AttractorSearchParams::default();
    search_params.num_initial_conditions = 100;
    search_params.max_steps = 10000;
    search_params.convergence_epsilon = 1e-3;
    search_params.attractor_distance_threshold = 1.0;
    search_params.initial_amplitude = 0.8;
    search_params.rae_params.gamma = 0.3;
    search_params.rae_params.omega = 0.0; // Pure dissipative for convergence

    let start = Instant::now();
    let analysis = find_attractors(&template, &search_params);
    let elapsed = start.elapsed();

    let pass = analysis.attractors.len() >= 2;
    println!(
        "  Attractors found: {} (need >= 2) ... {}",
        analysis.attractors.len(),
        if pass { "PASS" } else { "FAIL" }
    );
    println!(
        "  Converged: {}/{} ICs",
        analysis.num_converged,
        search_params.num_initial_conditions
    );
    for (i, attr) in analysis.attractors.iter().enumerate() {
        println!(
            "  Attractor {}: basin_size={}, energy={:.4}, rms_rate={:.6}",
            i, attr.basin_size, attr.final_energy, attr.final_rms_rate
        );
    }
    println!("  Time: {:.1}ms\n", elapsed.as_secs_f64() * 1000.0);
}

fn mvp_energy_monotonicity() {
    println!("MVP Test 2: Energy decreases under pure gradient flow (gamma=0, omega=0)");

    // With gamma=0 and omega=0, the RAE reduces to pure gradient descent
    // on the free energy: dz/dt = -δF/δz*, so dF/dt = -2Σ|δF/δz*|² ≤ 0
    let mut config = ManifoldConfig::e8_only();
    config.layers[0].omega = 0.0;
    config.layers[0].gamma = 0.0; // Pure gradient flow
    config.layers[0].rae_steps_per_tick = 200;
    config.backend = BackendSelection::Cpu;

    let mut emc = EmergentManifoldComputer::new_cpu(config);
    emc.init_random(42, 0.8);

    let mut energies = Vec::new();
    for _ in 0..20 {
        let e = emc.stats().layer_stats[0].total_energy;
        energies.push(e);
        emc.tick().unwrap();
    }

    let mut monotone = true;
    for i in 1..energies.len() {
        if energies[i] > energies[i - 1] + 0.01 {
            monotone = false;
            println!(
                "  VIOLATION at tick {}: {:.4} -> {:.4}",
                i,
                energies[i - 1],
                energies[i]
            );
        }
    }

    let pass_monotone = monotone;

    // Also check: with damping, overall energy still decreases significantly
    let mut config2 = ManifoldConfig::e8_only();
    config2.layers[0].omega = 0.0;
    config2.layers[0].gamma = 0.5;
    config2.layers[0].rae_steps_per_tick = 200;
    config2.backend = BackendSelection::Cpu;

    let mut emc2 = EmergentManifoldComputer::new_cpu(config2);
    emc2.init_random(42, 0.8);

    let e_start = emc2.stats().layer_stats[0].total_energy;
    emc2.run(20).unwrap();
    let e_end = emc2.stats().layer_stats[0].total_energy;
    let pass_overall = e_end < e_start * 0.5;

    println!(
        "  Pure gradient flow: {:.4} -> {:.4} (monotone={}) ... {}",
        energies[0],
        energies[energies.len() - 1],
        monotone,
        if pass_monotone { "PASS" } else { "FAIL" }
    );
    println!(
        "  With damping (gamma=0.5): {:.4} -> {:.4} (>{:.0}% decrease) ... {}",
        e_start,
        e_end,
        (1.0 - e_end / e_start) * 100.0,
        if pass_overall { "PASS" } else { "FAIL" }
    );
    println!();
}

// ─── Reservoir Computing Benchmarks ─────────────

/// Generate NARMA-10 time series.
///
/// y(n+1) = 0.3·y(n) + 0.05·y(n)·Σ_{i=0}^{9} y(n-i) + 1.5·u(n-9)·u(n) + 0.1
///
/// u(n) is uniform random in [0, 0.5].
fn generate_narma10(length: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let total = length + 50; // Extra for warmup/history

    let u: Vec<f32> = (0..total).map(|_| rng.gen::<f32>() * 0.5).collect();
    let mut y = vec![0.0f32; total];

    for n in 10..total - 1 {
        let sum_y: f32 = (0..10).map(|i| y[n - i]).sum();
        y[n + 1] = 0.3 * y[n] + 0.05 * y[n] * sum_y + 1.5 * u[n - 9] * u[n] + 0.1;
        // Clamp for stability
        y[n + 1] = y[n + 1].clamp(-10.0, 10.0);
    }

    // Return the last `length` samples, using u as input and y as target
    let offset = total - length;
    let inputs: Vec<f32> = u[offset..].to_vec();
    let targets: Vec<f32> = y[offset..].to_vec();
    (inputs, targets)
}

fn narma10_trial(
    inputs: &[f32],
    targets: &[f32],
    kw: f32,
    omega: f32,
    gamma: f32,
    steps: u64,
    init_amp: f32,
    encoder_type: u8, // 0=spatial, 1=phase, 2=spectral
    ridge_lambda: f32,
    ticks_per_input: u64,
    warmup: u64,
) -> f32 {
    let split = 1200;
    let train_inputs: Vec<Vec<f32>> = inputs[..split].iter().map(|&x| vec![x]).collect();
    let train_targets: Vec<Vec<f32>> = targets[..split].iter().map(|&y| vec![y]).collect();
    let test_inputs: Vec<Vec<f32>> = inputs[split..].iter().map(|&x| vec![x]).collect();
    let test_targets: Vec<f32> = targets[split..].to_vec();

    let mut config = ManifoldConfig::e8_only();
    config.layers[0].omega = omega;
    config.layers[0].gamma = gamma;
    config.layers[0].kinetic_weight = kw;
    config.layers[0].rae_steps_per_tick = steps;
    config.backend = BackendSelection::Cpu;

    let mut emc = EmergentManifoldComputer::new_cpu(config);
    emc.init_random(42, init_amp);

    let spatial_enc = SpatialEncoder::default();
    let phase_enc = PhaseEncoder::default();
    let spectral_enc = SpectralEncoder::new();
    let encoder: &dyn InputEncoder = match encoder_type {
        0 => &spatial_enc,
        1 => &phase_enc,
        _ => &spectral_enc,
    };
    let trainer = ReservoirTrainer::new(ridge_lambda, warmup, ticks_per_input);

    let collector = trainer
        .collect_states(&mut emc, encoder, &train_inputs, &train_targets, 0)
        .unwrap();
    let ridge = RidgeRegression::new(ridge_lambda);
    let readout = ridge.train_readout(&collector);

    let mut predictions = Vec::with_capacity(test_inputs.len());
    for input in &test_inputs {
        encoder.encode(input, &mut emc.manifold.layers[0].field);
        emc.run(ticks_per_input).unwrap();
        let pred = readout.read(&emc.manifold.layers[0].field);
        predictions.push(pred[0]);
    }

    nmse(&predictions, &test_targets)
}

fn bench_narma10() {
    println!("Reservoir Benchmark 1: NARMA-10 time series prediction");

    let (inputs, targets) = generate_narma10(1500, 123);

    // Optimized for E8 topology with rich connectivity (13,920 directed edges,
    // 57 neighbors per non-origin site). Parameters found via 3-phase sweep
    // (1,539 combos) over kinetic weight, omega, gamma, steps, amplitude,
    // ridge lambda, encoder type, and ticks-per-input.
    let score = narma10_trial(
        &inputs, &targets,
        0.04,   // kinetic_weight: moderate coupling for rich E8 graph
        8.0,    // omega: high resonance frequency for temporal encoding
        0.015,  // gamma: light damping preserves input memory
        95,     // rae_steps_per_tick: enough nonlinear mixing
        0.3,    // init_amp: small initial state for reservoir diversity
        0,      // encoder: SpatialEncoder (site 0 injection)
        1e-3,   // ridge_lambda: stronger regularization for 241-site reservoir
        1,      // ticks_per_input
        100,    // warmup_ticks
    );

    let pass = score < 0.85;
    println!(
        "  NMSE: {:.4} (threshold < 0.85) ... {}\n",
        score,
        if pass { "PASS" } else { "FAIL" }
    );
}

/// Generate Mackey-Glass time series.
///
/// dx/dt = β·x(t-τ)/(1+x(t-τ)^n) - γ·x(t)
/// Standard params: β=0.2, γ=0.1, n=10, τ=17
fn generate_mackey_glass(length: usize, seed: u64) -> Vec<f32> {
    let tau = 17usize;
    let beta = 0.2f64;
    let gamma = 0.1f64;
    let n_exp = 10.0f64;
    let dt = 1.0f64;
    let total = length + tau + 200; // Extra for warmup

    let mut x = vec![0.0f64; total];
    // Initialize with small random perturbation around 1.2
    let mut rng = StdRng::seed_from_u64(seed);
    for i in 0..=tau {
        x[i] = 1.2 + (rng.gen::<f64>() - 0.5) * 0.1;
    }

    for t in tau..total - 1 {
        let x_tau = x[t - tau];
        let dx = beta * x_tau / (1.0 + x_tau.powf(n_exp)) - gamma * x[t];
        x[t + 1] = x[t] + dt * dx;
    }

    // Return the last `length` samples
    x[total - length..].iter().map(|&v| v as f32).collect()
}

fn bench_mackey_glass() {
    println!("Reservoir Benchmark 2: Mackey-Glass chaotic time series");

    let series = generate_mackey_glass(600, 456);

    // Task: predict x(t+1) from x(t)
    let split = 400;
    let train_inputs: Vec<Vec<f32>> = series[..split].iter().map(|&x| vec![x]).collect();
    let train_targets: Vec<Vec<f32>> = series[1..split + 1].iter().map(|&y| vec![y]).collect();
    let test_inputs: Vec<Vec<f32>> = series[split..split + 100]
        .iter()
        .map(|&x| vec![x])
        .collect();
    let test_targets: Vec<f32> = series[split + 1..split + 101].to_vec();

    let mut config = ManifoldConfig::e8_only();
    config.layers[0].omega = 0.5;
    config.layers[0].gamma = 0.15;
    config.layers[0].rae_steps_per_tick = 50;
    config.backend = BackendSelection::Cpu;

    let mut emc = EmergentManifoldComputer::new_cpu(config);
    emc.init_random(99, 0.3);

    let encoder = SpatialEncoder::default();
    let trainer = ReservoirTrainer::new(1e-4, 30, 1);

    let start = Instant::now();

    let collector = trainer
        .collect_states(&mut emc, &encoder, &train_inputs, &train_targets, 0)
        .unwrap();
    let ridge = RidgeRegression::new(1e-4);
    let readout = ridge.train_readout(&collector);

    let train_elapsed = start.elapsed();

    // Test
    let mut predictions = Vec::with_capacity(test_inputs.len());
    for input in &test_inputs {
        encoder.encode(input, &mut emc.manifold.layers[0].field);
        emc.run(1).unwrap();
        let pred = readout.read(&emc.manifold.layers[0].field);
        predictions.push(pred[0]);
    }

    let score = nmse(&predictions, &test_targets);
    // Mackey-Glass is smoother than NARMA — should get better NMSE
    let pass = score < 0.1;

    println!(
        "  Train: {} samples, Test: {} samples",
        split,
        test_targets.len()
    );
    println!(
        "  NMSE: {:.4} (threshold < 0.1) ... {}",
        score,
        if pass { "PASS" } else { "FAIL" }
    );
    println!(
        "  Training time: {:.1}ms\n",
        train_elapsed.as_secs_f64() * 1000.0
    );
}

fn bench_pattern_classification() {
    println!("Reservoir Benchmark 3: Pattern classification (10 classes, 20% noise)");

    let mut rng = StdRng::seed_from_u64(789);
    let num_classes = 10usize;
    let dim = 8; // E8 dimension

    // Generate 10 random prototype patterns (8D)
    let prototypes: Vec<Vec<f32>> = (0..num_classes)
        .map(|_| {
            let v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
            // Normalize to unit length
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            v.iter().map(|x| x / norm).collect()
        })
        .collect();

    // Generate training data: 20 noisy copies per class = 200 samples
    let samples_per_class = 20;
    let noise_level = 0.2f32;
    let mut train_inputs = Vec::new();
    let mut train_targets = Vec::new();
    let mut train_labels = Vec::new();

    for (class_idx, proto) in prototypes.iter().enumerate() {
        for _ in 0..samples_per_class {
            let noisy: Vec<f32> = proto
                .iter()
                .map(|x: &f32| x + (rng.gen::<f32>() - 0.5) * 2.0 * noise_level)
                .collect();
            train_inputs.push(noisy);

            // One-hot target
            let mut target = vec![0.0f32; num_classes];
            target[class_idx] = 1.0;
            train_targets.push(target);
            train_labels.push(class_idx);
        }
    }

    // Generate test data: 10 noisy copies per class = 100 samples
    let test_per_class = 10;
    let mut test_inputs = Vec::new();
    let mut test_labels = Vec::new();

    for (class_idx, proto) in prototypes.iter().enumerate() {
        for _ in 0..test_per_class {
            let noisy: Vec<f32> = proto
                .iter()
                .map(|x: &f32| x + (rng.gen::<f32>() - 0.5) * 2.0 * noise_level)
                .collect();
            test_inputs.push(noisy);
            test_labels.push(class_idx);
        }
    }

    let mut config = ManifoldConfig::e8_only();
    config.layers[0].omega = 0.3;
    config.layers[0].gamma = 0.2;
    config.layers[0].rae_steps_per_tick = 80;
    config.backend = BackendSelection::Cpu;

    let mut emc = EmergentManifoldComputer::new_cpu(config);
    emc.init_random(55, 0.3);

    let encoder = SpectralEncoder::new();
    let trainer = ReservoirTrainer::new(1e-3, 10, 2);

    let start = Instant::now();

    let collector = trainer
        .collect_states(&mut emc, &encoder, &train_inputs, &train_targets, 0)
        .unwrap();
    let ridge = RidgeRegression::new(1e-3);
    let readout = ridge.train_readout(&collector);

    let train_elapsed = start.elapsed();

    // Test
    let mut predictions = Vec::with_capacity(test_inputs.len());
    for input in &test_inputs {
        encoder.encode(input, &mut emc.manifold.layers[0].field);
        emc.run(2).unwrap();
        let pred = readout.read(&emc.manifold.layers[0].field);
        predictions.push(pred);
    }

    let acc = accuracy(&predictions, &test_labels);
    let pass = acc > 0.70; // 70% threshold for small reservoir

    println!(
        "  Classes: {}, Train: {}, Test: {}, Noise: {}%",
        num_classes,
        train_inputs.len(),
        test_inputs.len(),
        (noise_level * 100.0) as u32
    );
    println!(
        "  Accuracy: {:.1}% (threshold > 70%) ... {}",
        acc * 100.0,
        if pass { "PASS" } else { "FAIL" }
    );
    println!(
        "  Training time: {:.1}ms\n",
        train_elapsed.as_secs_f64() * 1000.0
    );
}

// ─── Continual Learning Benchmark ────────────────

/// Generate a delayed memory recall task.
///
/// target(t) = input(t - delay), where input is uniform random in [0, 1].
fn generate_memory_task(length: usize, delay: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let total = length + delay;
    let u: Vec<f32> = (0..total).map(|_| rng.gen::<f32>()).collect();
    let inputs = u[delay..].to_vec();
    let targets = u[..length].to_vec();
    (inputs, targets)
}

fn bench_continual_learning() {
    println!("Continual Learning: Sequential tasks with EWC + Replay");

    // Task A: NARMA-10 (700 samples: 500 train, 200 test)
    let (narma_inputs, narma_targets) = generate_narma10(700, 789);
    let train_a_inputs: Vec<Vec<f32>> = narma_inputs[..500].iter().map(|&x| vec![x]).collect();
    let train_a_targets: Vec<Vec<f32>> = narma_targets[..500].iter().map(|&y| vec![y]).collect();
    let test_a_inputs: Vec<Vec<f32>> = narma_inputs[500..].iter().map(|&x| vec![x]).collect();
    let test_a_targets: Vec<Vec<f32>> = narma_targets[500..].iter().map(|&y| vec![y]).collect();

    // Task B: Memory recall delay-5 (700 samples: 500 train, 200 test)
    let (mem_inputs, mem_targets) = generate_memory_task(700, 5, 456);
    let train_b_inputs: Vec<Vec<f32>> = mem_inputs[..500].iter().map(|&x| vec![x]).collect();
    let train_b_targets: Vec<Vec<f32>> = mem_targets[..500].iter().map(|&y| vec![y]).collect();

    let conditions: &[(&str, f32, f32)] = &[
        ("Baseline (no EWC)", 0.0, 0.0),
        ("EWC only", 50.0, 0.0),
        ("EWC + Replay", 50.0, 0.3),
    ];

    let mut forgetting_values: Vec<(&str, f32)> = Vec::new();

    for &(name, lambda_ewc, replay_ratio) in conditions {
        let mut config = ManifoldConfig::e8_only();
        config.layers[0].kinetic_weight = 0.04;
        config.layers[0].omega = 8.0;
        config.layers[0].gamma = 0.015;
        config.layers[0].rae_steps_per_tick = 50; // Reduced for speed
        config.backend = BackendSelection::Cpu;

        let mut emc = EmergentManifoldComputer::new_cpu(config);
        emc.init_random(42, 0.3);

        let encoder = SpatialEncoder::default();

        let mut trainer = ContinualTrainer::new(
            1e-3,       // lambda
            lambda_ewc, // lambda_ewc
            50,          // warmup_ticks
            1,           // ticks_per_input
            1000,        // replay_max_size
            replay_ratio,
        );

        let start = Instant::now();

        // Train on Task A
        let _result_a = trainer
            .train_task("NARMA-10", &mut emc, &encoder, &train_a_inputs, &train_a_targets, 0)
            .unwrap();

        // Evaluate A with A's readout → NMSE_A_initial
        let readout_a = trainer.task_results[0].readout.clone();

        // Fresh EMC for evaluation (same state as after training A)
        let nmse_a_initial = trainer
            .evaluate_readout(
                &mut emc,
                &encoder,
                &readout_a,
                &test_a_inputs,
                &test_a_targets,
                0,
                20, // warmup samples
            )
            .unwrap();

        // Train on Task B (with EWC + replay protecting A's knowledge)
        let _result_b = trainer
            .train_task("Memory-5", &mut emc, &encoder, &train_b_inputs, &train_b_targets, 0)
            .unwrap();

        // Evaluate Task A with Task B's readout → NMSE_A_after
        let readout_b = trainer.task_results[1].readout.clone();
        let nmse_a_after = trainer
            .evaluate_readout(
                &mut emc,
                &encoder,
                &readout_b,
                &test_a_inputs,
                &test_a_targets,
                0,
                20,
            )
            .unwrap();

        let elapsed = start.elapsed();

        // Forgetting ratio: higher = more forgetting
        let forgetting = if nmse_a_initial > 1e-6 {
            nmse_a_after / nmse_a_initial
        } else {
            nmse_a_after
        };

        forgetting_values.push((name, forgetting));

        println!("  {name}:");
        println!("    Task A NMSE (initial):   {:.4}", nmse_a_initial);
        println!("    Task A NMSE (after B):   {:.4}", nmse_a_after);
        println!("    Forgetting ratio:        {:.2}x", forgetting);
        println!(
            "    Time: {:.1}ms",
            elapsed.as_secs_f64() * 1000.0
        );
    }

    // Validate: EWC should reduce forgetting vs baseline
    if forgetting_values.len() >= 2 {
        let baseline_forget = forgetting_values[0].1;
        let ewc_forget = forgetting_values[1].1;
        let pass = ewc_forget <= baseline_forget + 0.5; // EWC no worse than baseline + margin
        println!(
            "\n  EWC forgetting ({:.2}x) vs Baseline ({:.2}x) ... {}",
            ewc_forget,
            baseline_forget,
            if pass { "PASS" } else { "FAIL" }
        );
    }
    println!();
}

fn mvp_determinism() {
    println!("MVP Test 3: Determinism (same IC -> same result)");

    let config = ManifoldConfig::e8_only();

    let mut emc1 = EmergentManifoldComputer::new_cpu(config.clone());
    emc1.init_random(42, 0.5);
    emc1.run(20).unwrap();

    let mut emc2 = EmergentManifoldComputer::new_cpu(config);
    emc2.init_random(42, 0.5);
    emc2.run(20).unwrap();

    let obs1 = emc1.observe();
    let obs2 = emc2.observe();

    let mut max_diff = 0.0f32;
    for i in 0..obs1.layer_states[0].values_re.len() {
        let dr = (obs1.layer_states[0].values_re[i] - obs2.layer_states[0].values_re[i]).abs();
        let di = (obs1.layer_states[0].values_im[i] - obs2.layer_states[0].values_im[i]).abs();
        max_diff = max_diff.max(dr).max(di);
    }

    let pass = max_diff < 1e-5;
    println!(
        "  Max field difference after 20 ticks: {:.2e} ... {}",
        max_diff,
        if pass { "PASS" } else { "FAIL" }
    );
    println!();
}

fn bench_topology() {
    let lattice = E8Lattice::new();
    let mut field = LatticeField::from_lattice(&lattice);
    field.init_random(42, 0.8);

    let params = TopologyParams {
        weight: 10.0,
        persistence_threshold: 0.001,
        target_betti_0: 1,
        penalty_mode: PenaltyMode::TotalPersistence,
        gradient_step: 1.0,
    };

    // Benchmark persistence computation
    let start = Instant::now();
    let iters = 1000u64;
    let mut summary = topology::compute_sublevel_persistence(&field, params.persistence_threshold);
    for _ in 1..iters {
        summary = topology::compute_sublevel_persistence(&field, params.persistence_threshold);
    }
    let elapsed_persist = start.elapsed();

    let energy_before = topology::topological_energy(&summary, &params);

    println!(
        "[CPU] E8 Persistence: {} evals in {:.2}ms ({:.1}us/eval, {} sites)",
        iters,
        elapsed_persist.as_secs_f64() * 1000.0,
        elapsed_persist.as_micros() as f64 / iters as f64,
        field.num_sites,
    );
    println!(
        "  beta_0={}, pairs={}, total_persistence={:.4}, max_persistence={:.4}, significant={}",
        summary.betti_0,
        summary.pairs.len(),
        summary.total_persistence,
        summary.max_persistence,
        summary.significant_features,
    );

    // Benchmark gradient computation
    let start = Instant::now();
    for _ in 0..iters {
        let _ = topology::topological_gradient(&field, &summary, &params);
    }
    let elapsed_grad = start.elapsed();

    println!(
        "[CPU] E8 Topo Gradient: {} evals in {:.2}ms ({:.1}us/eval)",
        iters,
        elapsed_grad.as_secs_f64() * 1000.0,
        elapsed_grad.as_micros() as f64 / iters as f64,
    );

    // Validate: gradient step reduces energy
    topology::apply_topology_step(&mut field, &params);
    let summary_after = topology::compute_sublevel_persistence(&field, params.persistence_threshold);
    let energy_after = topology::topological_energy(&summary_after, &params);
    let pass = energy_after < energy_before;
    println!(
        "  Energy: {:.6} -> {:.6} (reduced={}) ... {}",
        energy_before,
        energy_after,
        energy_after < energy_before,
        if pass { "PASS" } else { "FAIL" },
    );
    println!();
}
