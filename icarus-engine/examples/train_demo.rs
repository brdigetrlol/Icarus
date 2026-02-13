//! Icarus EMC Training Demo — Reservoir Computing Pipeline
//!
//! Demonstrates the full train→predict pipeline on four tasks:
//! 1. Sine wave approximation (continuous function)
//! 2. XOR classification (nonlinear boolean)
//! 3. Mackey-Glass time series (chaotic dynamics, temporal memory)
//! 4. Continual learning — sequential tasks with EWC (anti-catastrophic forgetting)
//!
//! Run: cargo run --release --example train_demo -p icarus-engine

use icarus_engine::{
    EmergentManifoldComputer, ManifoldConfig,
    SpectralEncoder,
    InputEncoder, ReservoirTrainer, FeatureMode,
    EwcRidgeRegression,
    readout::Readout,
    training::{nmse, RidgeRegression},
};

fn main() {
    println!("╔══════════════════════════════════════════════════╗");
    println!("║     Icarus EMC — Reservoir Computing Demo       ║");
    println!("╠══════════════════════════════════════════════════╣");
    println!("║ E8 lattice: 241 sites (analytical layer)        ║");
    println!("║ State dim:  482 features (re + im per site)     ║");
    println!("║ Readout:    Ridge regression (closed-form)      ║");
    println!("╚══════════════════════════════════════════════════╝");
    println!();

    task_sine_approximation();
    task_xor_classification();
    task_mackey_glass();
    task_continual_learning();
}

// ─── Task 1: Sine Wave Approximation ──────────────────

fn task_sine_approximation() {
    println!("━━━ Task 1: Sine Wave Approximation ━━━");
    println!("  y = sin(2*pi*x) for x in [0, 1]");
    println!("  Spectral encoder + nonlinear features + leaky integration");
    println!();

    let config = ManifoldConfig::e8_only();
    let mut emc = EmergentManifoldComputer::new_cpu(config);
    emc.init_random(42, 0.5);

    let encoder = SpectralEncoder::new();

    // Encode scalar x as a rich 8D vector for E8 spectral decomposition
    let x_to_e8 = |x: f32| -> Vec<f32> {
        vec![
            x,
            x * x,
            x * x * x,
            (std::f32::consts::PI * x).sin(),
            (std::f32::consts::PI * x).cos(),
            (-x).exp(),
            1.0 - x,
            x * (1.0 - x),
        ]
    };

    // Generate training data: 100 samples as a temporal sequence
    let n_train = 100;
    let train_inputs: Vec<Vec<f32>> = (0..n_train)
        .map(|i| x_to_e8(i as f32 / n_train as f32))
        .collect();
    let train_targets: Vec<Vec<f32>> = (0..n_train)
        .map(|i| {
            let x = i as f32 / n_train as f32;
            vec![(2.0 * std::f32::consts::PI * x).sin()]
        })
        .collect();

    // Generate test data: 25 samples at offset positions
    let n_test = 25;
    let test_inputs: Vec<Vec<f32>> = (0..n_test)
        .map(|i| x_to_e8((2 * i + 1) as f32 / (2 * n_test) as f32))
        .collect();
    let test_targets: Vec<Vec<f32>> = (0..n_test)
        .map(|i| {
            let x = (2 * i + 1) as f32 / (2 * n_test) as f32;
            vec![(2.0 * std::f32::consts::PI * x).sin()]
        })
        .collect();

    // Train with leaky integration and nonlinear features
    let trainer = ReservoirTrainer::new(1e-3, 15, 2)
        .with_leak_rate(0.5)
        .with_feature_mode(FeatureMode::Nonlinear);
    let readout = trainer.train(&mut emc, &encoder, &train_inputs, &train_targets, 0)
        .expect("Training failed");

    println!("  Trained readout: {} weights, {} bias terms",
        readout.weights.len(), readout.bias.len());
    println!("  State dim: {} (nonlinear: 4*241), Output dim: {}",
        readout.state_dim, readout.output_dim);

    // Predict on test data — replay training tail for context, then test
    emc.init_random(42, 0.5);
    // Warm up: replay last 20 training inputs to establish reservoir state
    let warmup_start = if n_train > 20 { n_train - 20 } else { 0 };
    for inp in &train_inputs[warmup_start..] {
        encoder.encode_leaky(inp, &mut emc.manifold.layers[0].field, 0.5);
        emc.run(2).unwrap();
    }

    let mut predictions = Vec::new();
    for input in &test_inputs {
        encoder.encode_leaky(input, &mut emc.manifold.layers[0].field, 0.5);
        emc.run(2).unwrap();
        let pred = readout.read(&emc.manifold.layers[0].field);
        predictions.push(pred[0]);
    }

    let actuals: Vec<f32> = test_targets.iter().map(|t| t[0]).collect();
    let score = nmse(&predictions, &actuals);

    println!();
    println!("  {:>6} {:>10} {:>10}", "x", "predicted", "actual");
    println!("  {:>6} {:>10} {:>10}", "------", "----------", "----------");
    for i in 0..n_test.min(10) {
        let x = (2 * i + 1) as f32 / (2 * n_test) as f32;
        println!("  {:>6.3} {:>10.4} {:>10.4}", x, predictions[i], actuals[i]);
    }
    if n_test > 10 {
        println!("  ... ({} more samples)", n_test - 10);
    }

    println!();
    println!("  NMSE: {:.6} {}", score,
        if score < 0.1 { "(excellent)" }
        else if score < 0.5 { "(good)" }
        else if score < 1.0 { "(better than mean)" }
        else { "(poor)" });
    println!();
}

// ─── Task 2: XOR Classification ───────────────────────

fn task_xor_classification() {
    println!("━━━ Task 2: XOR Classification ━━━");
    println!("  XOR(a, b) — nonlinear binary function");
    println!("  Spectral encoder + nonlinear features + temporal presentation");
    println!();

    let config = ManifoldConfig::e8_only();
    let mut emc = EmergentManifoldComputer::new_cpu(config);
    emc.init_random(123, 0.5);

    let encoder = SpectralEncoder::new();

    // Encode XOR inputs as rich 8D vectors for spectral decomposition
    let xor_to_e8 = |a: f32, b: f32| -> Vec<f32> {
        vec![
            a, b,
            a * b,          // interaction term
            a + b,          // sum
            (a - b).abs(),  // absolute difference (= XOR for binary)
            a * a,          // quadratic
            b * b,
            (a + b) * 0.5,  // mean
        ]
    };

    // Generate diverse training data with small perturbations
    // This creates unique reservoir states for each presentation
    let mut train_inputs = Vec::new();
    let mut train_targets = Vec::new();
    let base_patterns: Vec<(f32, f32, f32)> = vec![
        (0.0, 0.0, 0.0), // 0 XOR 0 = 0
        (0.0, 1.0, 1.0), // 0 XOR 1 = 1
        (1.0, 0.0, 1.0), // 1 XOR 0 = 1
        (1.0, 1.0, 0.0), // 1 XOR 1 = 0
    ];

    // Present as temporal sequence: cycling through patterns with small jitter
    // The reservoir's fading memory creates different states each cycle
    for cycle in 0..30 {
        for &(a, b, xor) in &base_patterns {
            let jitter = (cycle as f32) * 0.001; // tiny jitter for state diversity
            train_inputs.push(xor_to_e8(a + jitter, b + jitter));
            train_targets.push(vec![xor]);
        }
    }

    // Nonlinear features + leaky integration
    let trainer = ReservoirTrainer::new(1e-2, 20, 3)
        .with_feature_mode(FeatureMode::Nonlinear)
        .with_leak_rate(0.4);

    let readout = trainer.train(&mut emc, &encoder, &train_inputs, &train_targets, 0)
        .expect("Training failed");

    println!("  Trained readout: state_dim={} (nonlinear: 4*241)", readout.state_dim);

    // Test on clean base patterns
    emc.init_random(123, 0.5);
    // Warmup: present a few cycles first
    for &(a, b, _) in base_patterns.iter().cycle().take(8) {
        encoder.encode_leaky(&xor_to_e8(a, b), &mut emc.manifold.layers[0].field, 0.4);
        emc.run(3).unwrap();
    }

    let mut correct = 0;
    println!();
    println!("  {:>5} {:>5} {:>10} {:>8} {:>8}", "a", "b", "output", "predict", "actual");
    println!("  {:>5} {:>5} {:>10} {:>8} {:>8}", "-----", "-----", "----------", "--------", "--------");

    for &(a, b, xor) in &base_patterns {
        encoder.encode_leaky(&xor_to_e8(a, b), &mut emc.manifold.layers[0].field, 0.4);
        emc.run(3).unwrap();
        let pred = readout.read(&emc.manifold.layers[0].field);
        let predicted_class = if pred[0] > 0.5 { 1.0 } else { 0.0 };
        if (predicted_class - xor).abs() < 0.01 { correct += 1; }
        println!("  {:>5.0} {:>5.0} {:>10.4} {:>8.0} {:>8.0}",
            a, b, pred[0], predicted_class, xor);
    }

    println!();
    println!("  Accuracy: {}/{} ({:.0}%)", correct, 4, correct as f32 / 4.0 * 100.0);
    println!();
}

// ─── Task 3: Mackey-Glass Time Series ─────────────────

fn task_mackey_glass() {
    println!("━━━ Task 3: Mackey-Glass Time Series (simplified) ━━━");
    println!("  One-step-ahead prediction on chaotic dynamics");
    println!();

    // Generate Mackey-Glass-like series using the logistic map (chaotic for r > 3.57)
    let r = 3.9;
    let n_total = 200;
    let mut series = Vec::with_capacity(n_total);
    let mut x = 0.1f32;
    for _ in 0..n_total {
        series.push(x);
        x = r * x * (1.0 - x);
    }

    // Create input-target pairs: predict x[t+1] from [x[t-3], x[t-2], x[t-1], x[t]]
    let lookback = 4;
    let skip = lookback; // skip first few for lookback window
    let mut inputs = Vec::new();
    let mut targets = Vec::new();
    for t in skip..(n_total - 1) {
        let mut inp = vec![0.0f32; 8];
        for j in 0..lookback {
            inp[j] = series[t - lookback + 1 + j];
            inp[j + lookback] = series[t - lookback + 1 + j]; // mirror into dims 4-7
        }
        inputs.push(inp);
        targets.push(vec![series[t + 1]]);
    }

    let n_train = inputs.len() * 3 / 4;
    let train_inputs = &inputs[..n_train];
    let train_targets = &targets[..n_train];
    let test_inputs = &inputs[n_train..];
    let test_targets = &targets[n_train..];

    println!("  Series length: {}, Lookback: {}", n_total, lookback);
    println!("  Train: {} samples, Test: {} samples", n_train, test_inputs.len());

    let config = ManifoldConfig::e8_only();
    let mut emc = EmergentManifoldComputer::new_cpu(config);
    emc.init_random(77, 0.3);

    let encoder = SpectralEncoder::new();

    // Use leaky integration for temporal memory
    let trainer = ReservoirTrainer::new(1e-3, 15, 1)
        .with_leak_rate(0.3)
        .with_feature_mode(FeatureMode::Nonlinear);

    let readout = trainer.train(
        &mut emc,
        &encoder,
        &train_inputs.to_vec(),
        &train_targets.to_vec(),
        0,
    ).expect("Training failed");

    println!("  Readout: state_dim={}, output_dim={}", readout.state_dim, readout.output_dim);

    // Predict on test set
    // Re-init and warm up with training data tail
    emc.init_random(77, 0.3);
    let warmup_start = if n_train > 20 { n_train - 20 } else { 0 };
    for inp in &train_inputs[warmup_start..] {
        encoder.encode_leaky(inp, &mut emc.manifold.layers[0].field, 0.3);
        emc.run(1).unwrap();
    }

    let mut predictions = Vec::new();
    for inp in test_inputs {
        encoder.encode_leaky(inp, &mut emc.manifold.layers[0].field, 0.3);
        emc.run(1).unwrap();
        let pred = readout.read(&emc.manifold.layers[0].field);
        predictions.push(pred[0]);
    }

    let actuals: Vec<f32> = test_targets.iter().map(|t| t[0]).collect();
    let score = nmse(&predictions, &actuals);

    println!();
    println!("  {:>6} {:>10} {:>10}", "step", "predicted", "actual");
    println!("  {:>6} {:>10} {:>10}", "------", "----------", "----------");
    for i in 0..test_inputs.len().min(15) {
        println!("  {:>6} {:>10.6} {:>10.6}", n_train + i, predictions[i], actuals[i]);
    }
    if test_inputs.len() > 15 {
        println!("  ... ({} more samples)", test_inputs.len() - 15);
    }

    println!();
    println!("  NMSE: {:.6} {}", score,
        if score < 0.1 { "(excellent)" }
        else if score < 0.5 { "(good)" }
        else if score < 1.0 { "(better than mean)" }
        else { "(poor)" });
    println!();
}

// ─── Task 4: Continual Learning (EWC) ────────────────

fn task_continual_learning() {
    println!("━━━ Task 4: Continual Learning (EWC) ━━━");
    println!("  Train 3 functions sequentially, measure forgetting.");
    println!("  Task-conditioned inputs let the readout distinguish tasks.");
    println!("  EWC + experience replay vs naive sequential training.");
    println!();

    let encoder = SpectralEncoder::new();

    // Task-conditioned 8D input: 5 x-features + 3-dim one-hot task ID.
    // The one-hot ensures different tasks produce different reservoir states,
    // so a single linear readout can learn task-specific mappings.
    let make_input = |x: f32, task: usize| -> Vec<f32> {
        vec![
            x,
            x * x,
            (2.0 * std::f32::consts::PI * x).sin(),
            (2.0 * std::f32::consts::PI * x).cos(),
            (-x).exp(),
            if task == 0 { 1.0 } else { 0.0 },
            if task == 1 { 1.0 } else { 0.0 },
            if task == 2 { 1.0 } else { 0.0 },
        ]
    };

    // ── Define 3 tasks (500 samples each, overdetermined with 482 features) ──
    let n = 500;
    let xs: Vec<f32> = (0..n).map(|i| i as f32 / n as f32).collect();

    let inputs_a: Vec<Vec<f32>> = xs.iter().map(|&x| make_input(x, 0)).collect();
    let targets_a: Vec<Vec<f32>> = xs.iter().map(|&x| {
        vec![(2.0 * std::f32::consts::PI * x).sin()]
    }).collect();

    let inputs_b: Vec<Vec<f32>> = xs.iter().map(|&x| make_input(x, 1)).collect();
    let targets_b: Vec<Vec<f32>> = xs.iter().map(|&x| {
        vec![(2.0 * std::f32::consts::PI * x).cos()]
    }).collect();

    let inputs_c: Vec<Vec<f32>> = xs.iter().map(|&x| make_input(x, 2)).collect();
    let targets_c: Vec<Vec<f32>> = xs.iter().map(|&x| {
        vec![4.0 * x * (1.0 - x)]
    }).collect();

    // ── Collect reservoir states (fresh EMC per task, same seed) ──
    // Independent collection ensures clean, uncontaminated states per task.
    println!("  Collecting reservoir states (fresh EMC per task)...");

    let trainer = ReservoirTrainer::new(1e-4, 15, 2)
        .with_leak_rate(0.3)
        .with_feature_mode(FeatureMode::Linear);

    let mut emc_a = EmergentManifoldComputer::new_cpu(ManifoldConfig::e8_only());
    emc_a.init_random(42, 0.5);
    let coll_a = trainer.collect_states(&mut emc_a, &encoder, &inputs_a, &targets_a, 0)
        .expect("Collect A failed");

    let mut emc_b = EmergentManifoldComputer::new_cpu(ManifoldConfig::e8_only());
    emc_b.init_random(42, 0.5);
    let coll_b = trainer.collect_states(&mut emc_b, &encoder, &inputs_b, &targets_b, 0)
        .expect("Collect B failed");

    let mut emc_c = EmergentManifoldComputer::new_cpu(ManifoldConfig::e8_only());
    emc_c.init_random(42, 0.5);
    let coll_c = trainer.collect_states(&mut emc_c, &encoder, &inputs_c, &targets_c, 0)
        .expect("Collect C failed");

    let d = coll_a.states[0].len();
    println!("    Task A (sine):      {} states x {} features", n, d);
    println!("    Task B (cosine):    {} states x {} features", n, d);
    println!("    Task C (quadratic): {} states x {} features", n, d);
    println!("    System: {} total samples > {} features (overdetermined)", 3 * n, d);
    println!();

    // ── Method 1: Individual baselines (per-task ceiling) ──
    let ridge = RidgeRegression::new(1e-4);
    let (ind_w_a, ind_b_a) = ridge.train(&coll_a.states, &coll_a.targets);
    let (ind_w_b, ind_b_b) = ridge.train(&coll_b.states, &coll_b.targets);
    let (ind_w_c, ind_b_c) = ridge.train(&coll_c.states, &coll_c.targets);

    // ── Method 2: Joint training (multi-task ceiling — all data at once) ──
    let mut all_states: Vec<Vec<f32>> = Vec::with_capacity(3 * n);
    let mut all_targets: Vec<Vec<f32>> = Vec::with_capacity(3 * n);
    all_states.extend_from_slice(&coll_a.states);
    all_targets.extend_from_slice(&coll_a.targets);
    all_states.extend_from_slice(&coll_b.states);
    all_targets.extend_from_slice(&coll_b.targets);
    all_states.extend_from_slice(&coll_c.states);
    all_targets.extend_from_slice(&coll_c.targets);
    let (joint_w, joint_b) = ridge.train(&all_states, &all_targets);

    // ── Method 3: EWC + Full Experience Replay (sequential) ──
    let mut ewc = EwcRidgeRegression::new(1e-4, 0.1);

    // Task A: train normally (no prior tasks)
    let res_a = ewc.train(&coll_a.states, &coll_a.targets);
    ewc.register_task("sine", res_a.fisher, res_a.optimal_wt, res_a.output_dim, res_a.state_dim);

    // Task B: EWC penalty + full replay of A
    let mut states_ab = coll_b.states.clone();
    let mut targets_ab = coll_b.targets.clone();
    states_ab.extend_from_slice(&coll_a.states);
    targets_ab.extend_from_slice(&coll_a.targets);
    let res_b = ewc.train(&states_ab, &targets_ab);
    ewc.register_task("cosine", res_b.fisher, res_b.optimal_wt, res_b.output_dim, res_b.state_dim);

    // Task C: EWC penalty + full replay of A and B
    let mut states_abc = coll_c.states.clone();
    let mut targets_abc = coll_c.targets.clone();
    states_abc.extend_from_slice(&coll_a.states);
    targets_abc.extend_from_slice(&coll_a.targets);
    states_abc.extend_from_slice(&coll_b.states);
    targets_abc.extend_from_slice(&coll_b.targets);
    let res_c = ewc.train(&states_abc, &targets_abc);

    // ── Method 4: Naive (only last task — catastrophic forgetting) ──
    let (naive_w, naive_b) = ridge.train(&coll_c.states, &coll_c.targets);

    // ── Evaluation ──
    println!("  Results (NMSE -- lower is better):");
    println!();
    println!("  {:>16} {:>10} {:>10} {:>10} {:>10}",
        "Task", "Individual", "Joint", "EWC+Replay", "Naive");
    println!("  {:>16} {:>10} {:>10} {:>10} {:>10}",
        "----------------", "----------", "----------", "----------", "----------");

    let ind_models = [
        (&ind_w_a[..], &ind_b_a[..]),
        (&ind_w_b[..], &ind_b_b[..]),
        (&ind_w_c[..], &ind_b_c[..]),
    ];
    let task_data: [(&str, &[Vec<f32>], &[Vec<f32>]); 3] = [
        ("A: sin(2*pi*x)", &coll_a.states, &coll_a.targets),
        ("B: cos(2*pi*x)", &coll_b.states, &coll_b.targets),
        ("C: 4x(1-x)",     &coll_c.states, &coll_c.targets),
    ];

    for (i, (name, states, targets)) in task_data.iter().enumerate() {
        let actuals: Vec<f32> = targets.iter().map(|t| t[0]).collect();
        let i_score = nmse(&predict_from_states(ind_models[i].0, ind_models[i].1, states), &actuals);
        let j_score = nmse(&predict_from_states(&joint_w, &joint_b, states), &actuals);
        let e_score = nmse(&predict_from_states(&res_c.weights, &res_c.bias, states), &actuals);
        let n_score = nmse(&predict_from_states(&naive_w, &naive_b, states), &actuals);
        println!("  {:>16} {:>10.6} {:>10.6} {:>10.6} {:>10.6}",
            name, i_score, j_score, e_score, n_score);
    }

    println!();
    println!("  Individual = per-task ceiling (each task trained alone)");
    println!("  Joint      = multi-task ceiling (all 1500 samples at once)");
    println!("  EWC+Replay = sequential training: A -> B+A_replay -> C+AB_replay");
    println!("  Naive      = trained on C only -- catastrophic forgetting on A, B");
    println!();
    println!("  Key: EWC+Replay ~= Joint >> Naive on tasks A, B (knowledge retained)");
    println!("       Naive ~= Individual on C only (last task fits, rest forgotten)");
    println!();
    println!("━━━ Demo Complete ━━━");
}

/// Apply linear readout weights W*s + b to stored state vectors.
/// weights: K x D row-major, bias: length K. Assumes K=1 for this demo.
fn predict_from_states(
    weights: &[f32],
    bias: &[f32],
    states: &[Vec<f32>],
) -> Vec<f32> {
    let d = states[0].len();
    states.iter().map(|s| {
        let mut val = bias[0];
        for j in 0..d {
            val += weights[j] * s[j];
        }
        val
    }).collect()
}
