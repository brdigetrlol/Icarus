// Copyright (c) 2025-2026 brdigetrlol. All rights reserved.
// SPDX-License-Identifier: LicenseRef-Icarus-Proprietary
// See LICENSE in the repository root for full license terms.

//! Tool definitions for the Icarus MCP server.

use mcp_core::protocol::Tool;
use mcp_core::tools::ToolBuilder;
use serde_json::json;

/// Build all Icarus MCP tool definitions.
pub fn all_tools() -> Vec<Tool> {
    vec![
        icarus_init(),
        icarus_step(),
        icarus_observe(),
        icarus_inject(),
        icarus_stats(),
        icarus_encode(),
        icarus_readout(),
        icarus_train(),
        icarus_predict(),
        // NPU bridge tools
        icarus_npu_ping(),
        icarus_npu_device_info(),
        icarus_npu_matmul(),
        icarus_npu_matvec(),
        icarus_npu_benchmark(),
        // Autonomous mode tools
        icarus_auto_start(),
        icarus_auto_stop(),
        icarus_auto_status(),
        icarus_auto_events(),
        // Visualization
        icarus_visualize(),
        // Ensemble tools
        icarus_ensemble_init(),
        icarus_ensemble_status(),
        // Game tools
        icarus_game_start(),
        icarus_game_stop(),
        icarus_game_status(),
    ]
}

fn icarus_auto_start() -> Tool {
    ToolBuilder::new(
        "icarus_auto_start",
        "Start the autonomous EMC tick loop. The EMC runs continuously in the background \
         with configurable stop conditions. Use icarus_auto_status to monitor and \
         icarus_auto_stop to halt.",
    )
    .number_param(
        "max_ticks_per_second",
        "Target tick rate (default: 60, null for uncapped)",
        false,
    )
    .raw_param(
        "stop_conditions",
        json!({
            "type": "array",
            "description": "Array of stop conditions, e.g. [{\"type\": \"max_ticks\", \"value\": 10000}, {\"type\": \"convergence_stable\", \"value\": 50}, {\"type\": \"energy_below\", \"value\": 0.01}, {\"type\": \"time_limit_secs\", \"value\": 300}, {\"type\": \"manual\"}]",
            "items": { "type": "object" }
        }),
        false,
    )
    .integer_param(
        "snapshot_interval",
        "Publish snapshot every N ticks (default: 1)",
        false,
    )
    .build()
}

fn icarus_auto_stop() -> Tool {
    ToolBuilder::new(
        "icarus_auto_stop",
        "Stop the autonomous EMC tick loop. Returns final state summary including \
         tick count, energy, and convergence trend.",
    )
    .build()
}

fn icarus_auto_status() -> Tool {
    ToolBuilder::new(
        "icarus_auto_status",
        "Query the current state of the autonomous tick loop. Returns state \
         (idle/running/paused/completed/error), current tick, energy, convergence trend, \
         and pending event count. Zero-cost: reads from published snapshot.",
    )
    .build()
}

fn icarus_auto_events() -> Tool {
    ToolBuilder::new(
        "icarus_auto_events",
        "Poll and drain recent events from the autonomous tick loop. Events include \
         convergence changes, attractor transitions, energy thresholds, and tick milestones.",
    )
    .integer_param("limit", "Maximum events to return (default: 50)", false)
    .build()
}

fn icarus_npu_ping() -> Tool {
    ToolBuilder::new(
        "icarus_npu_ping",
        "Ping the Intel NPU bridge running on the Windows host. \
         Verifies connectivity and measures round-trip latency over TCP.",
    )
    .build()
}

fn icarus_npu_device_info() -> Tool {
    ToolBuilder::new(
        "icarus_npu_device_info",
        "Get device information from the Intel NPU bridge. \
         Lists available OpenVINO devices (CPU, GPU, NPU) on the Windows host.",
    )
    .build()
}

fn icarus_npu_matmul() -> Tool {
    ToolBuilder::new(
        "icarus_npu_matmul",
        "Run matrix multiplication C[M,N] = A[M,K] * B[K,N] on the Intel NPU \
         via the Windows bridge. Returns the result matrix and execution time.",
    )
    .integer_param("m", "Number of rows in A / result (required)", true)
    .integer_param("k", "Shared dimension: columns of A, rows of B (required)", true)
    .integer_param("n", "Number of columns in B / result (required)", true)
    .raw_param(
        "a",
        json!({
            "type": "array",
            "description": "Matrix A as flat row-major array of M*K floats",
            "items": { "type": "number" }
        }),
        true,
    )
    .raw_param(
        "b",
        json!({
            "type": "array",
            "description": "Matrix B as flat row-major array of K*N floats",
            "items": { "type": "number" }
        }),
        true,
    )
    .build()
}

fn icarus_npu_matvec() -> Tool {
    ToolBuilder::new(
        "icarus_npu_matvec",
        "Run matrix-vector multiplication y[M] = W[M,N] * x[N] on the Intel NPU \
         via the Windows bridge. Returns the result vector and execution time.",
    )
    .integer_param("m", "Number of rows in W / length of result (required)", true)
    .integer_param("n", "Number of columns in W / length of x (required)", true)
    .raw_param(
        "w",
        json!({
            "type": "array",
            "description": "Weight matrix W as flat row-major array of M*N floats",
            "items": { "type": "number" }
        }),
        true,
    )
    .raw_param(
        "x",
        json!({
            "type": "array",
            "description": "Input vector x as array of N floats",
            "items": { "type": "number" }
        }),
        true,
    )
    .build()
}

fn icarus_npu_benchmark() -> Tool {
    ToolBuilder::new(
        "icarus_npu_benchmark",
        "Run a benchmark suite on the Intel NPU via the Windows bridge. \
         Tests matrix multiplication at various sizes (64x64 through 512x512) \
         and reports timing and GFLOPS.",
    )
    .build()
}

fn icarus_init() -> Tool {
    ToolBuilder::new(
        "icarus_init",
        "Initialize the Emergent Manifold Computer. Must be called before any other icarus tool. \
         Creates the EMC with the specified preset configuration and optional random initialization.",
    )
    .enum_param(
        "preset",
        "Configuration preset",
        &["e8_only", "full_hierarchy"],
        false,
    )
    .enum_param(
        "backend",
        "Compute backend (default: gpu)",
        &["gpu", "cpu"],
        false,
    )
    .number_param("seed", "Random seed for field initialization (default: 42)", false)
    .number_param("amplitude", "Initial field amplitude (default: 0.5)", false)
    .build()
}

fn icarus_step() -> Tool {
    ToolBuilder::new(
        "icarus_step",
        "Advance the EMC simulation by N ticks. Each tick runs RAE dynamics, \
         inter-layer transfer (if enabled), metric learning, and all cognitive agents.",
    )
    .integer_param("num_ticks", "Number of ticks to execute (default: 1)", false)
    .build()
}

fn icarus_observe() -> Tool {
    ToolBuilder::new(
        "icarus_observe",
        "Observe the current state of the EMC. Returns field values, energy, \
         and per-layer snapshots. Use layer_index to observe a specific layer.",
    )
    .integer_param("layer_index", "Layer index to observe (default: 0 = analytical)", false)
    .integer_param("max_sites", "Maximum sites to include in response (default: 50)", false)
    .build()
}

fn icarus_inject() -> Tool {
    ToolBuilder::new(
        "icarus_inject",
        "Inject data into the EMC at specific lattice sites. \
         Each injection specifies a site index and complex value (re, im).",
    )
    .integer_param("layer_index", "Target layer index (default: 0)", false)
    .raw_param(
        "sites",
        json!({
            "type": "array",
            "description": "Array of [site_index, re, im] triples to inject",
            "items": {
                "type": "array",
                "items": { "type": "number" },
                "minItems": 3,
                "maxItems": 3
            }
        }),
        true,
    )
    .number_param("strength", "Injection strength 0.0-1.0 (default: 1.0 = full replacement)", false)
    .build()
}

fn icarus_stats() -> Tool {
    ToolBuilder::new(
        "icarus_stats",
        "Get comprehensive statistics about the EMC: tick count, energy breakdown, \
         site counts, memory usage, backend info, and per-layer metrics.",
    )
    .build()
}

fn icarus_encode() -> Tool {
    ToolBuilder::new(
        "icarus_encode",
        "Encode input data into the EMC's complex phase field using one of three strategies: \
         'spatial' (direct amplitude injection), 'phase' (unit circle mapping), or \
         'spectral' (E8 root vector decomposition). Optionally run N ticks after encoding \
         to let the dynamics process the input.",
    )
    .enum_param(
        "encoder",
        "Encoding strategy (default: spatial)",
        &["spatial", "phase", "spectral"],
        false,
    )
    .raw_param(
        "input",
        json!({
            "type": "array",
            "description": "Input values to encode into the field",
            "items": { "type": "number" }
        }),
        true,
    )
    .integer_param("layer_index", "Target layer index (default: 0)", false)
    .integer_param("offset", "Starting site index for spatial/phase encoding (default: 0)", false)
    .number_param("scale", "Scale factor for spatial encoding (default: 1.0)", false)
    .integer_param("ticks_after", "Number of ticks to run after encoding (default: 0)", false)
    .build()
}

fn icarus_readout() -> Tool {
    ToolBuilder::new(
        "icarus_readout",
        "Read output from the EMC's phase field state. Returns the raw state vector \
         [re_0..re_N, im_0..im_N] or a subset of it. Useful for extracting the reservoir \
         state after encoding input and running dynamics.",
    )
    .integer_param("layer_index", "Layer to read from (default: 0)", false)
    .integer_param("max_values", "Maximum number of state values to return (default: 100)", false)
    .enum_param(
        "format",
        "Output format: 'complex' returns (re, im, amplitude, phase) per site, \
         'flat' returns raw [re..., im...] vector (default: complex)",
        &["complex", "flat"],
        false,
    )
    .build()
}

fn icarus_train() -> Tool {
    ToolBuilder::new(
        "icarus_train",
        "Train a linear readout on the EMC reservoir via ridge regression. \
         Runs the full pipeline: warmup → encode each input → tick → collect state → \
         ridge regression. Stores the trained model for use with icarus_predict. \
         Returns training metrics including per-output NMSE.",
    )
    .enum_param(
        "encoder",
        "Encoding strategy for inputs (default: spatial)",
        &["spatial", "phase", "spectral"],
        false,
    )
    .raw_param(
        "inputs",
        json!({
            "type": "array",
            "description": "Training inputs: array of arrays, each inner array is one input sample",
            "items": {
                "type": "array",
                "items": { "type": "number" }
            }
        }),
        true,
    )
    .raw_param(
        "targets",
        json!({
            "type": "array",
            "description": "Training targets: array of arrays, each inner array is the desired output for the corresponding input",
            "items": {
                "type": "array",
                "items": { "type": "number" }
            }
        }),
        true,
    )
    .integer_param("layer_index", "Target layer index (default: 0)", false)
    .number_param("lambda", "Ridge regression regularization strength (default: 1e-4). Ignored when auto_lambda is true.", false)
    .bool_param("auto_lambda", "If true, use k-fold cross-validation to automatically select the best lambda. Recommended for best generalization.", false)
    .integer_param("k_folds", "Number of CV folds for auto_lambda (default: 5, min: 2)", false)
    .integer_param("warmup_ticks", "Warmup ticks before state collection (default: 10)", false)
    .integer_param("ticks_per_input", "EMC ticks per input sample (default: 1)", false)
    .number_param("leak_rate", "Leaking rate 0.0-1.0 (default: 1.0 = full overwrite). Lower values blend new input with existing reservoir state, preserving temporal memory across steps.", false)
    .enum_param(
        "feature_mode",
        "Feature extraction mode (default: linear). 'nonlinear' appends squared terms, doubling state dimensionality for better nonlinear separation.",
        &["linear", "nonlinear"],
        false,
    )
    .build()
}

fn icarus_predict() -> Tool {
    ToolBuilder::new(
        "icarus_predict",
        "Run inference using a trained linear readout. Encodes each input into the EMC, \
         runs dynamics, and applies the trained readout weights. Requires icarus_train \
         to have been called first.",
    )
    .enum_param(
        "encoder",
        "Encoding strategy (must match training, default: spatial)",
        &["spatial", "phase", "spectral"],
        false,
    )
    .raw_param(
        "inputs",
        json!({
            "type": "array",
            "description": "Input samples to predict on: array of arrays",
            "items": {
                "type": "array",
                "items": { "type": "number" }
            }
        }),
        true,
    )
    .integer_param("layer_index", "Layer to read from (default: 0, must match training)", false)
    .integer_param("ticks_per_input", "EMC ticks per input (default: 1, must match training)", false)
    .number_param("leak_rate", "Leaking rate 0.0-1.0 (default: 1.0, must match training)", false)
    .build()
}

fn icarus_ensemble_init() -> Tool {
    ToolBuilder::new(
        "icarus_ensemble_init",
        "Initialize a multi-instance ensemble trainer for reservoir computing. \
         Creates N independent EMC instances (CPU backend, E8 lattice each) that \
         run in parallel. Their concatenated states feed a single ridge regression \
         readout. Use with the icarus_game tools for imitation learning.",
    )
    .integer_param(
        "num_instances",
        "Number of EMC instances to create (default: 2)",
        false,
    )
    .enum_param(
        "feature_mode",
        "Feature extraction mode (default: nonlinear). \
         'nonlinear' appends squared terms for better separation.",
        &["linear", "nonlinear"],
        false,
    )
    .number_param("seed", "Base random seed (default: 42)", false)
    .number_param(
        "amplitude",
        "Initial field amplitude for all instances (default: 0.1)",
        false,
    )
    .integer_param(
        "warmup_ticks",
        "Warmup ticks to run before training (default: 100)",
        false,
    )
    .build()
}

fn icarus_ensemble_status() -> Tool {
    ToolBuilder::new(
        "icarus_ensemble_status",
        "Get the status of the ensemble trainer. Returns per-instance backend info, \
         state dimensions, total concatenated dimension, training status (NMSE, samples, \
         confidence), and readout info. Requires icarus_ensemble_init first.",
    )
    .build()
}

fn icarus_game_start() -> Tool {
    ToolBuilder::new(
        "icarus_game_start",
        "Launch the Icarus procedural 3D game server. Spawns the icarus-game binary \
         as a subprocess serving a browser-based Three.js world on the given port. \
         The game trains an ensemble of EMC instances by observing player actions \
         (imitation learning). Open http://localhost:<port> in a browser to play.",
    )
    .integer_param("port", "HTTP port to serve on (default: 3000)", false)
    .integer_param("seed", "World generation seed (default: 42)", false)
    .build()
}

fn icarus_game_stop() -> Tool {
    ToolBuilder::new(
        "icarus_game_stop",
        "Stop the running Icarus game server. Kills the subprocess and frees the port.",
    )
    .build()
}

fn icarus_game_status() -> Tool {
    ToolBuilder::new(
        "icarus_game_status",
        "Query the running Icarus game server for training status. Returns tick count, \
         time of day, training samples, NMSE, confidence, whether the readout is trained, \
         player position, agent state, and per-backend status. Requires the game to be \
         running (icarus_game_start).",
    )
    .build()
}

fn icarus_visualize() -> Tool {
    ToolBuilder::new(
        "icarus_visualize",
        "Generate an interactive HTML visualization of the current EMC state. \
         Writes a self-contained HTML file with embedded Three.js WebGL and Canvas 2D \
         charts. Supports multiple visualization modes: lattice_field (3D point cloud), \
         energy_landscape (height map), phase_portrait (phase arrows + Kuramoto), \
         neuro_dashboard (neuromodulator gauges), timeseries (multi-line chart), \
         and dashboard (combined multi-panel). Returns the output file path.",
    )
    .enum_param(
        "mode",
        "Visualization mode (default: dashboard)",
        &[
            "dashboard",
            "lattice_field",
            "energy_landscape",
            "phase_portrait",
            "neuro_dashboard",
            "timeseries",
        ],
        false,
    )
    .string_param(
        "output_path",
        "Output file path (default: /tmp/icarus-viz-<mode>.html)",
        false,
    )
    .integer_param("layer_index", "Layer index for lattice/energy/phase renderers (default: 0)", false)
    .enum_param(
        "theme",
        "Color theme (default: dark)",
        &["dark", "light"],
        false,
    )
    .enum_param(
        "color_mode",
        "Color mapping for lattice_field: amplitude, phase, energy, convergence (default: amplitude)",
        &["amplitude", "phase", "energy", "convergence"],
        false,
    )
    .raw_param(
        "series",
        json!({
            "type": "array",
            "description": "Series to plot in timeseries mode: energy, amplitude, coherence, valence, arousal, dopamine, norepinephrine (default: all four main)",
            "items": { "type": "string" }
        }),
        false,
    )
    .build()
}
