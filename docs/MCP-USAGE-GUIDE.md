# Icarus MCP Usage Guide

Complete reference for the Icarus Emergent Manifold Computer MCP tools with JSON examples.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Tool Reference](#tool-reference)
3. [Complete Workflows](#complete-workflows)
4. [Encoding Strategies](#encoding-strategies)
5. [Reservoir Computing Pipeline](#reservoir-computing-pipeline)
6. [Error Handling](#error-handling)
7. [Performance Tips](#performance-tips)

---

## Quick Start

### Minimal Example: CPU Backend

```json
{
  "name": "icarus_init",
  "arguments": {
    "preset": "e8_only",
    "backend": "cpu",
    "seed": 42,
    "amplitude": 0.5
  }
}
```

**Response:**
```json
{
  "status": "initialized",
  "preset": "e8_only",
  "backend": "CPU",
  "layers": [
    {
      "layer": "Analytical",
      "num_sites": 241,
      "initial_energy": 30.125
    }
  ],
  "total_sites": 241,
  "memory_bytes": 36584,
  "seed": 42,
  "amplitude": 0.5
}
```

---

## Tool Reference

### 1. icarus_init

**Initialize the Emergent Manifold Computer**. Must be called before any other tool.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `preset` | enum | No | `"e8_only"` | Configuration preset: `"e8_only"` or `"full_hierarchy"` |
| `backend` | enum | No | `"gpu"` | Compute backend: `"gpu"` or `"cpu"` |
| `seed` | number | No | `42` | Random seed for field initialization |
| `amplitude` | number | No | `0.5` | Initial field amplitude (0.0-1.0) |

#### Examples

**GPU Backend (auto-fallback to CPU if GPU unavailable):**
```json
{
  "name": "icarus_init",
  "arguments": {
    "preset": "e8_only",
    "backend": "gpu"
  }
}
```

**CPU Backend with custom seed:**
```json
{
  "name": "icarus_init",
  "arguments": {
    "preset": "e8_only",
    "backend": "cpu",
    "seed": 12345,
    "amplitude": 0.3
  }
}
```

**Full hierarchy (4 layers: E8, Leech, HCP-64D, HCP-32D):**
```json
{
  "name": "icarus_init",
  "arguments": {
    "preset": "full_hierarchy",
    "backend": "gpu",
    "seed": 42,
    "amplitude": 0.5
  }
}
```

**Response (full_hierarchy):**
```json
{
  "status": "initialized",
  "preset": "full_hierarchy",
  "backend": "CUDA GPU 0",
  "layers": [
    {
      "layer": "Analytical",
      "num_sites": 241,
      "initial_energy": 30.125
    },
    {
      "layer": "Creative",
      "num_sites": 1105,
      "initial_energy": 138.125
    },
    {
      "layer": "Associative",
      "num_sites": 4033,
      "initial_energy": 504.125
    },
    {
      "layer": "Sensory",
      "num_sites": 65,
      "initial_energy": 8.125
    }
  ],
  "total_sites": 5444,
  "memory_bytes": 1398784,
  "seed": 42,
  "amplitude": 0.5
}
```

---

### 2. icarus_step

**Advance the EMC simulation** by N ticks. Each tick runs RAE dynamics, inter-layer transfer (if enabled), metric learning, and cognitive agents.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `num_ticks` | integer | No | `1` | Number of ticks to execute |

#### Examples

**Single tick:**
```json
{
  "name": "icarus_step",
  "arguments": {
    "num_ticks": 1
  }
}
```

**100 ticks (for settling dynamics):**
```json
{
  "name": "icarus_step",
  "arguments": {
    "num_ticks": 100
  }
}
```

**Response:**
```json
{
  "tick": 100,
  "ticks_executed": 100,
  "layers": [
    {
      "layer": "Analytical",
      "total_energy": 28.456,
      "kinetic_energy": 14.228,
      "potential_energy": 14.228,
      "mean_amplitude": 0.482
    }
  ]
}
```

---

### 3. icarus_observe

**Observe the current EMC state**. Returns field values, energy, and per-site snapshots.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `layer_index` | integer | No | `0` | Layer to observe (0 = Analytical) |
| `max_sites` | integer | No | `50` | Maximum sites to include in response |

#### Examples

**Observe first layer (default):**
```json
{
  "name": "icarus_observe",
  "arguments": {
    "layer_index": 0,
    "max_sites": 10
  }
}
```

**Response:**
```json
{
  "tick": 100,
  "layer": "Analytical",
  "layer_index": 0,
  "total_sites": 241,
  "showing": 10,
  "energy": 28.456,
  "sites": [
    {
      "site": 0,
      "re": "0.123456",
      "im": "0.234567",
      "amplitude": "0.265165",
      "phase": "1.0808"
    },
    {
      "site": 1,
      "re": "-0.456789",
      "im": "0.123456",
      "amplitude": "0.473181",
      "phase": "2.8760"
    }
  ]
}
```

**Observe second layer:**
```json
{
  "name": "icarus_observe",
  "arguments": {
    "layer_index": 1,
    "max_sites": 20
  }
}
```

---

### 4. icarus_inject

**Inject data into the EMC** at specific lattice sites. Each injection specifies a site index and complex value (re, im).

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `layer_index` | integer | No | `0` | Target layer index |
| `sites` | array | **Yes** | - | Array of `[site_index, re, im]` triples |
| `strength` | number | No | `1.0` | Injection strength (0.0-1.0): 1.0 = full replacement, 0.5 = 50% blend |

#### Examples

**Inject at single site (full replacement):**
```json
{
  "name": "icarus_inject",
  "arguments": {
    "layer_index": 0,
    "sites": [
      [0, 1.0, 0.0]
    ],
    "strength": 1.0
  }
}
```

**Inject at multiple sites:**
```json
{
  "name": "icarus_inject",
  "arguments": {
    "layer_index": 0,
    "sites": [
      [0, 1.0, 0.0],
      [1, 0.5, 0.866],
      [2, -0.5, 0.866],
      [10, 0.707, 0.707]
    ]
  }
}
```

**Partial injection (50% blend with existing state):**
```json
{
  "name": "icarus_inject",
  "arguments": {
    "layer_index": 0,
    "sites": [
      [5, 1.0, 0.0]
    ],
    "strength": 0.5
  }
}
```

**Response:**
```json
{
  "status": "injected",
  "layer_index": 0,
  "sites_injected": 4,
  "strength": 1.0
}
```

---

### 5. icarus_stats

**Get comprehensive EMC statistics**: tick count, energy breakdown, site counts, memory usage, backend info, and per-layer metrics.

#### Parameters

None.

#### Example

```json
{
  "name": "icarus_stats",
  "arguments": {}
}
```

**Response:**
```json
{
  "tick": 150,
  "backend": "CUDA GPU 0",
  "total_sites": 241,
  "memory_bytes": 36584,
  "layers": [
    {
      "layer": "Analytical",
      "num_sites": 241,
      "total_energy": 28.456,
      "kinetic_energy": 14.228,
      "potential_energy": 14.228,
      "mean_amplitude": 0.482
    }
  ]
}
```

---

### 6. icarus_encode

**Encode input data into the EMC's complex phase field** using one of three strategies: `spatial`, `phase`, or `spectral`.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `encoder` | enum | No | `"spatial"` | Encoding strategy: `"spatial"`, `"phase"`, or `"spectral"` |
| `input` | array | **Yes** | - | Input values (floats) |
| `layer_index` | integer | No | `0` | Target layer index |
| `offset` | integer | No | `0` | Starting site index (spatial/phase only) |
| `scale` | number | No | `1.0` | Scale factor (spatial only) |
| `ticks_after` | integer | No | `0` | Number of ticks to run after encoding |

#### Examples

**Spatial encoding (direct amplitude injection):**
```json
{
  "name": "icarus_encode",
  "arguments": {
    "encoder": "spatial",
    "input": [0.5, -0.3, 1.0, 0.0, 0.2],
    "layer_index": 0,
    "offset": 0,
    "scale": 1.0,
    "ticks_after": 10
  }
}
```

**Phase encoding (unit circle mapping):**
```json
{
  "name": "icarus_encode",
  "arguments": {
    "encoder": "phase",
    "input": [0.0, 0.5, 1.0, -0.5, -1.0],
    "layer_index": 0,
    "offset": 0,
    "ticks_after": 5
  }
}
```

**Spectral encoding (E8 root vector decomposition):**
```json
{
  "name": "icarus_encode",
  "arguments": {
    "encoder": "spectral",
    "input": [1.0, 0.5, -0.3, 0.8, 0.0, 0.0, 0.0, 0.0],
    "layer_index": 0,
    "ticks_after": 20
  }
}
```

**Response:**
```json
{
  "status": "encoded",
  "encoder": "spatial",
  "input_length": 5,
  "layer_index": 0,
  "offset": 0,
  "ticks_after": 10,
  "tick": 10,
  "layer_energy": 29.342,
  "layer_mean_amplitude": 0.495
}
```

---

### 7. icarus_readout

**Read output from the EMC's phase field state**. Returns the raw state vector or a subset.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `layer_index` | integer | No | `0` | Layer to read from |
| `max_values` | integer | No | `100` | Maximum number of state values to return |
| `format` | enum | No | `"complex"` | Output format: `"complex"` (per-site) or `"flat"` (raw [re..., im...]) |

#### Examples

**Complex format (per-site breakdown):**
```json
{
  "name": "icarus_readout",
  "arguments": {
    "layer_index": 0,
    "max_values": 10,
    "format": "complex"
  }
}
```

**Response (complex):**
```json
{
  "format": "complex",
  "layer_index": 0,
  "num_sites": 241,
  "showing": 10,
  "sites": [
    {
      "site": 0,
      "re": "0.123456",
      "im": "0.234567",
      "amplitude": "0.265165",
      "phase": "1.0808"
    },
    {
      "site": 1,
      "re": "-0.456789",
      "im": "0.123456",
      "amplitude": "0.473181",
      "phase": "2.8760"
    }
  ]
}
```

**Flat format (raw state vector for ML):**
```json
{
  "name": "icarus_readout",
  "arguments": {
    "layer_index": 0,
    "max_values": 100,
    "format": "flat"
  }
}
```

**Response (flat):**
```json
{
  "format": "flat",
  "layer_index": 0,
  "num_sites": 241,
  "state_dim": 482,
  "showing": 200,
  "values": [
    "0.123456", "0.234567", "-0.456789", "0.123456", "..."
  ]
}
```

---

### 8. icarus_train

**Train a linear readout on the EMC reservoir** via ridge regression. Runs the full pipeline: warmup → encode each input → tick → collect state → ridge regression.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `encoder` | enum | No | `"spatial"` | Encoding strategy |
| `inputs` | array | **Yes** | - | Training inputs: array of arrays |
| `targets` | array | **Yes** | - | Training targets: array of arrays (same length as inputs) |
| `layer_index` | integer | No | `0` | Target layer |
| `lambda` | number | No | `1e-4` | Ridge regression regularization strength |
| `warmup_ticks` | integer | No | `10` | Warmup ticks before state collection |
| `ticks_per_input` | integer | No | `1` | EMC ticks per input sample |

#### Examples

**Simple regression task (1D → 1D):**
```json
{
  "name": "icarus_train",
  "arguments": {
    "encoder": "spatial",
    "inputs": [
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ],
    "targets": [
      [0.0],
      [0.2],
      [0.4],
      [0.6],
      [0.8]
    ],
    "layer_index": 0,
    "lambda": 0.0001,
    "warmup_ticks": 10,
    "ticks_per_input": 1
  }
}
```

**Multi-output classification (8D → 3D):**
```json
{
  "name": "icarus_train",
  "arguments": {
    "encoder": "spectral",
    "inputs": [
      [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ],
    "targets": [
      [1.0, 0.0, 0.0],
      [0.0, 1.0, 0.0],
      [0.0, 0.0, 1.0]
    ],
    "layer_index": 0,
    "lambda": 0.001,
    "warmup_ticks": 20,
    "ticks_per_input": 5
  }
}
```

**Time series prediction (8D → 2D):**
```json
{
  "name": "icarus_train",
  "arguments": {
    "encoder": "phase",
    "inputs": [
      [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
      [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
      [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ],
    "targets": [
      [0.8, 0.9],
      [0.9, 1.0],
      [1.0, 1.1]
    ],
    "lambda": 0.0001,
    "warmup_ticks": 15,
    "ticks_per_input": 3
  }
}
```

**Response:**
```json
{
  "status": "trained",
  "encoder": "spatial",
  "num_samples": 5,
  "state_dim": 482,
  "output_dim": 1,
  "lambda": 0.0001,
  "warmup_ticks": 10,
  "ticks_per_input": 1,
  "layer_index": 0,
  "weight_norm": "12.456789",
  "train_nmse": ["0.034567"]
}
```

---

### 9. icarus_predict

**Run inference using a trained linear readout**. Requires `icarus_train` to have been called first.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `encoder` | enum | No | `"spatial"` | Encoding strategy (must match training) |
| `inputs` | array | **Yes** | - | Input samples: array of arrays |
| `layer_index` | integer | No | `0` | Layer to read from (must match training) |
| `ticks_per_input` | integer | No | `1` | EMC ticks per input (must match training) |

#### Examples

**Predict on test samples:**
```json
{
  "name": "icarus_predict",
  "arguments": {
    "encoder": "spatial",
    "inputs": [
      [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ],
    "layer_index": 0,
    "ticks_per_input": 1
  }
}
```

**Response:**
```json
{
  "status": "predicted",
  "num_inputs": 2,
  "output_dim": 1,
  "predictions": [
    ["1.0"],
    ["1.2"]
  ]
}
```

---

## Complete Workflows

### Workflow 1: Basic Simulation

Step-by-step EMC simulation with observation.

```json
// Step 1: Initialize
{
  "name": "icarus_init",
  "arguments": {
    "preset": "e8_only",
    "backend": "cpu",
    "seed": 42,
    "amplitude": 0.5
  }
}

// Step 2: Run dynamics for 100 ticks
{
  "name": "icarus_step",
  "arguments": {
    "num_ticks": 100
  }
}

// Step 3: Observe the settled state
{
  "name": "icarus_observe",
  "arguments": {
    "layer_index": 0,
    "max_sites": 20
  }
}

// Step 4: Get full statistics
{
  "name": "icarus_stats",
  "arguments": {}
}
```

---

### Workflow 2: Manual Data Injection

Inject custom patterns and watch them evolve.

```json
// Step 1: Initialize
{
  "name": "icarus_init",
  "arguments": {
    "preset": "e8_only",
    "backend": "cpu"
  }
}

// Step 2: Inject a circular pattern (6 sites on unit circle)
{
  "name": "icarus_inject",
  "arguments": {
    "layer_index": 0,
    "sites": [
      [1, 1.0, 0.0],
      [2, 0.5, 0.866],
      [3, -0.5, 0.866],
      [4, -1.0, 0.0],
      [5, -0.5, -0.866],
      [6, 0.5, -0.866]
    ],
    "strength": 1.0
  }
}

// Step 3: Run dynamics to see how pattern evolves
{
  "name": "icarus_step",
  "arguments": {
    "num_ticks": 50
  }
}

// Step 4: Observe evolved pattern
{
  "name": "icarus_observe",
  "arguments": {
    "layer_index": 0,
    "max_sites": 10
  }
}
```

---

### Workflow 3: Encode-Tick-Readout Cycle

Drive the EMC with encoded inputs and extract reservoir states.

```json
// Step 1: Initialize
{
  "name": "icarus_init",
  "arguments": {
    "preset": "e8_only",
    "backend": "gpu"
  }
}

// Step 2: Encode input using spectral encoder
{
  "name": "icarus_encode",
  "arguments": {
    "encoder": "spectral",
    "input": [1.0, 0.5, -0.3, 0.8, 0.2, -0.1, 0.0, 0.0],
    "layer_index": 0,
    "ticks_after": 10
  }
}

// Step 3: Read out the reservoir state
{
  "name": "icarus_readout",
  "arguments": {
    "layer_index": 0,
    "max_values": 50,
    "format": "flat"
  }
}
```

---

### Workflow 4: Complete Reservoir Computing Pipeline

Train a readout, then predict on test data.

```json
// Step 1: Initialize
{
  "name": "icarus_init",
  "arguments": {
    "preset": "e8_only",
    "backend": "cpu",
    "seed": 123,
    "amplitude": 0.5
  }
}

// Step 2: Train on XOR-like problem (8D → 1D)
{
  "name": "icarus_train",
  "arguments": {
    "encoder": "spatial",
    "inputs": [
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ],
    "targets": [
      [0.0],
      [1.0],
      [1.0],
      [0.0]
    ],
    "lambda": 0.001,
    "warmup_ticks": 20,
    "ticks_per_input": 5
  }
}

// Step 3: Predict on test samples
{
  "name": "icarus_predict",
  "arguments": {
    "encoder": "spatial",
    "inputs": [
      [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ],
    "ticks_per_input": 5
  }
}
```

---

## Encoding Strategies

### Spatial Encoding

**Direct amplitude injection**: `input[i] → field.set(offset + i, scale × input[i], 0.0)`

**When to use:**
- Input dimensionality ≤ number of lattice sites
- Simple regression/classification tasks
- When input values are already meaningful amplitudes

**Example:**
```json
{
  "name": "icarus_encode",
  "arguments": {
    "encoder": "spatial",
    "input": [0.5, -0.3, 1.0],
    "offset": 10,
    "scale": 2.0
  }
}
```
Result: Sites 10, 11, 12 get values (1.0, 0.0), (-0.6, 0.0), (2.0, 0.0).

---

### Phase Encoding

**Unit circle mapping**: `input[i] → (cos(π·x), sin(π·x))`

**When to use:**
- Inputs normalized to [-1, 1]
- When amplitude uniformity is desired (all sites have |z| = 1)
- Periodic or circular data

**Example:**
```json
{
  "name": "icarus_encode",
  "arguments": {
    "encoder": "phase",
    "input": [0.0, 0.5, 1.0, -0.5]
  }
}
```
Result:
- input[0] = 0.0 → θ=0 → (1, 0)
- input[1] = 0.5 → θ=π/2 → (0, 1)
- input[2] = 1.0 → θ=π → (-1, 0)
- input[3] = -0.5 → θ=-π/2 → (0, -1)

---

### Spectral Encoding

**E8 root vector basis decomposition**: Projects input onto 240 E8 root vectors.

**When to use:**
- Principled crystallographic encoding
- Exploiting E8 symmetry structure
- Input dimensionality ≤ 8
- When you want information distributed across all nearest neighbors

**Math:**
- Input is 8D vector `v`
- For each of 240 root vectors `r_j`: `c_j = (v · r_j) / |r_j|²`
- Site 0 (origin) gets `|v|` (input norm)
- Sites 1..240 get `c_j` as real amplitudes

**Example:**
```json
{
  "name": "icarus_encode",
  "arguments": {
    "encoder": "spectral",
    "input": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  }
}
```
Result: Site 0 gets 1.0, sites with roots having nonzero first component get their projection coefficients.

---

## Reservoir Computing Pipeline

### Overview

The EMC acts as a **reservoir**: a high-dimensional dynamical system that transforms temporal input sequences into rich state representations. A simple **linear readout** is trained on these states.

**Pipeline:**
1. **Warmup**: Run EMC for `warmup_ticks` to let transients decay
2. **Drive**: For each input sample, encode → tick × `ticks_per_input` → collect state
3. **Train**: Ridge regression on collected (state, target) pairs
4. **Predict**: Encode new inputs, apply trained readout weights

---

### Training Example: NARMA-10 Task

NARMA-10 is a benchmark nonlinear system identification task.

```json
// Generate NARMA-10 data externally, then:

{
  "name": "icarus_train",
  "arguments": {
    "encoder": "spatial",
    "inputs": [
      /* 100 samples of 8D inputs */
      [0.123, 0.456, ..., 0.789],
      [0.234, 0.567, ..., 0.890],
      /* ... */
    ],
    "targets": [
      /* 100 corresponding 1D targets */
      [0.456],
      [0.567],
      /* ... */
    ],
    "lambda": 0.0001,
    "warmup_ticks": 50,
    "ticks_per_input": 3
  }
}
```

**Hyperparameter tuning:**
- `lambda`: Higher = more regularization (smoother, less overfitting)
- `warmup_ticks`: Longer = better transient decay (especially for time series)
- `ticks_per_input`: Longer = more mixing dynamics per input (richer states)

---

### Prediction Example

```json
{
  "name": "icarus_predict",
  "arguments": {
    "encoder": "spatial",
    "inputs": [
      /* Test samples */
      [0.111, 0.222, ..., 0.333],
      [0.444, 0.555, ..., 0.666]
    ],
    "ticks_per_input": 3
  }
}
```

**Response:**
```json
{
  "status": "predicted",
  "num_inputs": 2,
  "output_dim": 1,
  "predictions": [
    ["0.450123"],
    ["0.560234"]
  ]
}
```

---

### Evaluating Performance

The training response includes **per-output NMSE** (Normalized Mean Squared Error):

```json
{
  "train_nmse": ["0.034567"]
}
```

**NMSE interpretation:**
- < 0.1: Excellent
- 0.1 - 0.5: Good
- 0.5 - 1.0: Acceptable (better than predicting mean)
- \> 1.0: Poor (worse than predicting mean)

---

## Error Handling

### Common Errors

**Error: EMC not initialized**
```json
{
  "error": "EMC not initialized. Call icarus_init first."
}
```
**Solution:** Call `icarus_init` before any other tool.

---

**Error: Layer index out of range**
```json
{
  "error": "Layer index 2 out of range (have 1 layers)"
}
```
**Solution:** Check how many layers your preset has:
- `e8_only`: 1 layer (index 0 only)
- `full_hierarchy`: 4 layers (indices 0-3)

---

**Error: Invalid sites parameter**
```json
{
  "error": "missing 'sites' parameter"
}
```
**Solution:** Ensure `icarus_inject` includes the required `sites` array.

---

**Error: Inputs/targets length mismatch**
```json
{
  "error": "inputs (10) and targets (8) must have equal length"
}
```
**Solution:** Ensure `inputs` and `targets` arrays have the same length in `icarus_train`.

---

**Error: No trained model**
```json
{
  "error": "No trained model. Call icarus_train first."
}
```
**Solution:** Call `icarus_train` before `icarus_predict`.

---

**Error: Empty training data**
```json
{
  "error": "inputs must not be empty"
}
```
**Solution:** Provide at least one training sample.

---

## Performance Tips

### Backend Selection

**GPU backend:**
- Use for large hierarchies (`full_hierarchy`)
- Requires CUDA GPU available
- Auto-falls back to CPU if GPU init fails

**CPU backend:**
- Use for `e8_only` (fast enough on CPU)
- More portable, no GPU dependency

---

### Hyperparameter Tuning

**Ridge regression lambda:**
- Start with `1e-4`
- Increase if overfitting (train NMSE << test NMSE)
- Decrease if underfitting (train NMSE high)

**Warmup ticks:**
- Start with `10` for simple tasks
- Use `50-100` for time series tasks
- Higher = better transient decay, but slower training

**Ticks per input:**
- Start with `1`
- Increase for tasks needing more mixing (e.g., chaotic systems)
- Higher = richer states, but slower

---

### Encoding Strategy Selection

| Task Type | Recommended Encoder |
|-----------|-------------------|
| Simple regression | Spatial |
| Time series | Phase or Spatial |
| Classification | Spectral or Spatial |
| High-dimensional input (>8D) | Spatial with offset |
| Exploiting E8 symmetry | Spectral |

---

### Memory Constraints

**E8-only (1 layer):**
- 241 sites × (2 + 36 metric) × 4 bytes = ~36 KB
- Fits easily in memory

**Full hierarchy (4 layers):**
- ~1.4 MB total
- GPU: Check `vram_budget_bytes` in config
- CPU: No issue

---

## Advanced Usage

### Multi-Layer Training

Train readout on a higher layer for different computational properties:

```json
{
  "name": "icarus_train",
  "arguments": {
    "encoder": "spatial",
    "inputs": [ /* ... */ ],
    "targets": [ /* ... */ ],
    "layer_index": 1,  // Train on Creative layer (24D Leech)
    "lambda": 0.001,
    "warmup_ticks": 30,
    "ticks_per_input": 5
  }
}
```

**Prediction must match:**
```json
{
  "name": "icarus_predict",
  "arguments": {
    "encoder": "spatial",
    "inputs": [ /* ... */ ],
    "layer_index": 1,  // Same layer as training
    "ticks_per_input": 5  // Same ticks as training
  }
}
```

---

### Custom Warmup with Manual Encoding

For fine-grained control over the reservoir state:

```json
// Step 1: Initialize
{
  "name": "icarus_init",
  "arguments": { "preset": "e8_only", "backend": "cpu" }
}

// Step 2: Manually encode warmup input
{
  "name": "icarus_encode",
  "arguments": {
    "encoder": "spatial",
    "input": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    "ticks_after": 50
  }
}

// Step 3: Now train (will run its own warmup, but reservoir is pre-warmed)
{
  "name": "icarus_train",
  "arguments": {
    "encoder": "spatial",
    "inputs": [ /* ... */ ],
    "targets": [ /* ... */ ],
    "warmup_ticks": 10  // Can use fewer ticks now
  }
}
```

---

### Observing Energy Dynamics

Track how energy evolves during settling:

```json
// Step 1: Initialize with high amplitude
{
  "name": "icarus_init",
  "arguments": {
    "preset": "e8_only",
    "backend": "cpu",
    "amplitude": 1.0
  }
}

// Step 2: Check initial energy
{
  "name": "icarus_stats",
  "arguments": {}
}
// Note: total_energy, kinetic_energy, potential_energy

// Step 3: Run dynamics
{
  "name": "icarus_step",
  "arguments": { "num_ticks": 100 }
}

// Step 4: Check settled energy
{
  "name": "icarus_stats",
  "arguments": {}
}
// Energy should have decreased (damping) and stabilized
```

---

## Full Example: End-to-End Workflow

**Task:** Learn a simple function `f(x) = 2x` using the EMC as a reservoir.

```json
// 1. Initialize
{
  "name": "icarus_init",
  "arguments": {
    "preset": "e8_only",
    "backend": "cpu",
    "seed": 42,
    "amplitude": 0.5
  }
}

// 2. Train
{
  "name": "icarus_train",
  "arguments": {
    "encoder": "spatial",
    "inputs": [
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ],
    "targets": [
      [0.0],
      [1.0],
      [2.0],
      [3.0],
      [4.0]
    ],
    "lambda": 0.0001,
    "warmup_ticks": 10,
    "ticks_per_input": 1
  }
}

// Expected response:
// {
//   "status": "trained",
//   "num_samples": 5,
//   "state_dim": 482,
//   "output_dim": 1,
//   "train_nmse": ["< 0.01"]  // Should be very low for linear function
// }

// 3. Predict on test samples
{
  "name": "icarus_predict",
  "arguments": {
    "encoder": "spatial",
    "inputs": [
      [2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ],
    "ticks_per_input": 1
  }
}

// Expected response:
// {
//   "status": "predicted",
//   "predictions": [
//     ["~5.0"],
//     ["~6.0"]
//   ]
// }
```

---

## Summary

The Icarus MCP provides 9 tools for interacting with the Emergent Manifold Computer:

1. **icarus_init**: Initialize the EMC
2. **icarus_step**: Advance simulation
3. **icarus_observe**: Inspect field state
4. **icarus_inject**: Manually inject data
5. **icarus_stats**: Get comprehensive statistics
6. **icarus_encode**: Encode inputs (spatial/phase/spectral)
7. **icarus_readout**: Extract reservoir states
8. **icarus_train**: Train linear readout via ridge regression
9. **icarus_predict**: Run inference with trained model

**Typical workflow:**
1. Initialize with `icarus_init`
2. Train a readout with `icarus_train` (handles encoding, ticking, state collection, regression)
3. Predict on new data with `icarus_predict`
4. Optionally observe/inject/step for fine-grained control

**Key concepts:**
- **Encoding strategies**: Spatial (direct), Phase (unit circle), Spectral (E8 basis)
- **Reservoir computing**: High-dimensional dynamics + linear readout
- **Ridge regression**: Closed-form training with regularization (lambda)
- **Hyperparameters**: warmup_ticks, ticks_per_input, lambda

For questions or advanced usage, refer to the Icarus source code at `/root/workspace-v2/Icarus/`.
