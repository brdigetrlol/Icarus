# Icarus Training Parameter Optimization - Audit Summary

**Generated:** 2026-02-06  
**Sources:** TEST-GAP-ANALYSIS.md, GPU-AUDIT.md, MATH-REVIEW.md, ARCHITECTURE.md, MCP-USAGE-GUIDE.md

---

## Executive Summary

The Icarus project has **strong mathematical foundations** and **verified GPU/CPU numerical parity** (tolerance <1e-4), but significant optimization opportunities exist for training parameter exploration. This document distills actionable insights for improving NMSE scores and system performance.

---

## 1. Parameter Ranges: Tested vs Untested

### Tested (Verified via 248 unit tests + benchmarks)

| Parameter | Tested Range | Status | Notes |
|-----------|--------------|--------|-------|
| **RAE timestep (dt)** | 0.001 - 1.0 | ✅ Verified | Euler: CFL-limited (dt < 2/K ≈ 0.0083 for E8); Semi-implicit: unconditionally stable up to dt=1.0 |
| **Damping (γ)** | 0.0 - 10.0 | ✅ Verified | γ=0 → energy conservation; γ→∞ → overdamped (pure gradient flow) |
| **Coupling (K)** | 0.0 - 240 | ✅ Verified | K=0 → no diffusion; K=240 (E8 kissing) → max connectivity |
| **Resonance (ω)** | 0.0 - 10.0 | ✅ Verified | ω=0 → real dynamics; ω>0 → oscillatory attractors |
| **Target amplitude (A)** | 0.1 - 2.0 | ✅ Verified | Double-well minimum at |z|=A |
| **Potential weight** | 0.1 - 10.0 | ✅ Verified | Scales well depth; V(r²) = w·(r²-A²)²/4 |
| **Kinetic weight** | 0.1 - 10.0 | ✅ Verified | Scales diffusion strength |

### Untested (High-priority for NMSE improvement)

| Parameter | Untested Range | Priority | Expected Impact |
|-----------|----------------|----------|-----------------|
| **RAE steps per tick** | >100 | HIGH | Benchmark used 5-50; longer mixing may improve temporal kernel memory |
| **Warmup ticks** | >500 | HIGH | Benchmark tested 10-500; longer warmup for Leech/HCP may reduce transient artifacts |
| **Encoder combinations** | Mixed | MEDIUM | Only tested single encoders; hybrid spatial+phase+spectral may capture richer features |
| **Multi-layer configs** | 4+ layers | HIGH | Only E8 + 3-layer (E8+Leech+HCP) tested; deeper hierarchies untested |
| **Transfer coupling strength** | 0.1-1.0 | MEDIUM | Currently 0.1 blend factor; stronger coupling may improve inter-layer coherence |
| **Adaptive timestep targets** | 0.001-1.0 | HIGH | Benchmark used fixed dt; adaptive dt (energy-based) may improve convergence |
| **Regularization (λ)** | >1e-3 | CRITICAL | Benchmark sweep: 1e-6 to 1e-3; higher λ may prevent overfitting |

### Critical Gaps (Zero Test Coverage)

1. **Agent-modulated dynamics** - All 6 cognitive agents (Perception, WorldModel, Memory, Action, Learning, Planning) have **0 tests**
2. **GPU/CPU backend parity** - Multi-layer (E8+Leech+HCP) parity untested (only single-kernel parity verified)
3. **FP16 quantization** - Implemented but not integrated into kernels; potential 2× memory savings untested
4. **Metric learning convergence** - Geometrodynamic updates implemented but no convergence tests

---

## 2. NMSE Improvement Recommendations

### Current Performance (NARMA-10 benchmark)

- **Best NMSE:** ~0.3-0.5 (target: <0.85)
- **Parameter sweep:** 1539 combinations tested
- **Key finding:** Spatial encoder + λ=1e-4 + warmup=100 + ticks_per_input=10 performed best

### Actionable Optimizations (Priority Order)

#### Priority 1: Regularization Tuning (Expected: 10-20% NMSE reduction)

```json
{
  "tool": "icarus_train",
  "lambda": [1e-3, 5e-3, 1e-2],  // Higher λ to prevent overfitting
  "warmup_ticks": [200, 500, 1000],  // Longer transient decay
  "ticks_per_input": [20, 50, 100]   // Better mixing
}
```

**Rationale:** Ridge regression tests show λ > 0 reduces weight magnitude; benchmark used λ ≤ 1e-3; higher values unexplored.

#### Priority 2: Multi-Layer Hierarchical Encoding (Expected: 15-30% NMSE reduction)

```json
{
  "preset": "full_hierarchy",  // E8 (241) + Leech (1105) + HCP-64D (4033) + HCP-32D (65)
  "encoder": "spectral",       // E8 root decomposition (240 modes)
  "layer_index": 0,            // Encode into analytical layer (E8)
  "ticks_after": 50            // Allow inter-layer transfer
}
```

**Rationale:** ARCHITECTURE.md shows inter-layer transfer operators (E8→Leech→HCP) provide hierarchical abstractions; untested for reservoir tasks.

#### Priority 3: Adaptive Timestep + Semi-Implicit Integration (Expected: 5-15% NMSE reduction)

```json
{
  "solver": "semi_implicit",   // Unconditionally stable (no CFL limit)
  "adaptive_dt": true,         // Energy-based dt adjustment
  "dt_base": 0.05,             // 6× larger than Euler CFL (0.0083)
  "energy_target_delta": 0.01  // Tighter energy control
}
```

**Rationale:** Semi-implicit tested stable at dt=1.0 (120× CFL); adaptive controller can exploit this for faster convergence.

#### Priority 4: Hybrid Encoder Strategy (Expected: 10-25% NMSE reduction)

```json
{
  "encoders": [
    {"type": "spectral", "layer": 0, "weight": 0.5},  // E8 modes
    {"type": "phase", "layer": 1, "weight": 0.3},     // Leech oscillators
    {"type": "spatial", "layer": 3, "weight": 0.2}    // HCP-32D direct
  ]
}
```

**Rationale:** TEST-GAP-ANALYSIS shows each encoder has different characteristics; blending may capture complementary features.

#### Priority 5: Longer RAE Mixing Cycles (Expected: 5-10% NMSE reduction)

```json
{
  "rae_steps_per_tick": 10,      // Current default: 1-5
  "ticks_per_input": 100,        // Benchmark max: 50
  "warmup_ticks": 1000           // Benchmark max: 500
}
```

**Rationale:** Spectral methods tests show 10K+ steps reach attractors; short mixing may miss long-timescale features.

---

## 3. GPU vs CPU Guidance for Config Sizes

### Decision Matrix

| Config | Sites | VRAM (FP32) | VRAM (FP16) | Backend | Speedup | Notes |
|--------|-------|-------------|-------------|---------|---------|-------|
| **E8 only** | 241 | 0.5 MB | 0.25 MB | Either | 1× (CPU) | No GPU benefit; allocation overhead dominates |
| **E8 + Leech** | 1,346 | 10 MB | 5 MB | GPU | 5-10× | GPU beneficial; use FP32 (FP16 untested) |
| **Full hierarchy** | 5,444 | 40 MB | 20 MB | GPU | 10-20× | GPU required for real-time; consider FP16 for >8GB models |
| **E8 + Leech + HCP-128D** | 10K+ | 100+ MB | 50+ MB | GPU | 20-50× | **Must use buffer pooling** (9.8MB alloc/free per tick kills perf) |

### GPU Optimization Checklist

- ✅ **Use GPU if:** Total sites >1000 OR ticks >1000 OR training samples >100
- ⚠️ **Use CPU if:** E8-only + few ticks (<100) OR debugging numerical issues
- ❌ **Avoid GPU for:** Single-tick operations (launch overhead ~5μs)

### GPU-Specific Recommendations (from GPU-AUDIT.md)

1. **Shared memory for Laplacian** - Untapped 3-5× speedup (E8: 240 neighbors, high reuse)
2. **Two-kernel reduction** - Eliminate CPU-side final sum (5-10× speedup for energy computation)
3. **Tiled transfer kernels** - Coalesce memory access for Leech→HCP transfer (2-4× speedup)
4. **Multi-stream overlap** - Pipeline RAE step + energy + metric update (1.5-2× throughput)

---

## 4. Math/Numerical Stability Warnings

### Critical Warnings (May Cause Silent Failure)

1. **NaN Propagation in RAE Integration**
   - **Risk:** HIGH - No tests for NaN input → integration silently corrupts field
   - **Mitigation:** Add NaN checks in encoder input; clamp potential gradient when |z|² → ∞
   - **Affected:** `rae.rs` lines 242-293, `encoding.rs` all encoders

2. **Singular Metric Tensor (det(g)=0)**
   - **Risk:** HIGH - Division by zero in metric inverse (Gauss-Jordan)
   - **Mitigation:** Eigenvalue pinning already implemented (0.5 ≤ λ ≤ 2.0), but untested in edge cases
   - **Affected:** `metric.rs` lines 91-156 (inverse computation)

3. **Overflow in Long RAE Runs (dt → large)**
   - **Risk:** HIGH - No tests for dt > 1.0 (even semi-implicit has accuracy limits)
   - **Mitigation:** Clamp dt ≤ 1.0 for semi-implicit; dt ≤ CFL/2 for Euler
   - **Affected:** `rae.rs` adaptive timestep controller

4. **FP32 Accumulation in Leech Laplacian (K=1104)**
   - **Risk:** MEDIUM - 1104-term sum → relative error ~1e-4 (marginal for 1e-4 tolerance)
   - **Mitigation:** Use Kahan summation for K > 500 (GPU kernels currently use naive sum)
   - **Affected:** `icarus-gpu/kernels/rae.rs` lines 50-56

### Moderate Warnings (May Degrade Performance)

5. **Stochastic Rounding Not Integrated**
   - **Risk:** LOW - FP16 quantization implemented (unbiased SR) but kernels still use FP32
   - **Mitigation:** Add FP16 kernel variants for memory-bound ops (reduction, transfer)
   - **Affected:** `memory.rs` lines 98-205 (unused code)

6. **Ridge Regression Singular Matrix (λ=0 or T=0)**
   - **Risk:** MEDIUM - No tests for empty training set or zero regularization
   - **Mitigation:** Enforce λ ≥ 1e-8; reject T < num_features
   - **Affected:** `training.rs` lines 45-120

7. **Out-of-Bounds Site Indices**
   - **Risk:** MEDIUM - No bounds checking in `phase_field.rs` `get()`/`set()`
   - **Mitigation:** Add debug assertions or explicit bounds checks
   - **Affected:** `phase_field.rs` lines 80-120, `perception.rs` injection

---

## 5. Encoder-Specific Optimization Hints

### Spatial Encoder (Default)

```json
{
  "encoder": "spatial",
  "scale": 1.0,     // Current default
  "offset": 0.0
}
```

**Characteristics:**
- ✅ Direct amplitude injection: z_i = scale·input[i]
- ✅ Fast (no computation)
- ⚠️ Limited expressiveness: no phase information

**Optimization Tips:**
- Use for **low-dimensional inputs** (<100 features)
- Set scale = 1/max(|input|) for amplitude normalization
- Combine with phase encoder for complex features

**Performance:** 241 sites (E8) → ~0.1ms CPU / ~0.05ms GPU

### Phase Encoder

```json
{
  "encoder": "phase",
  "offset": 0,
  "input": [-1.0, 0.0, 1.0]  // Maps to [0°, 180°, 360°]
}
```

**Characteristics:**
- ✅ Unit amplitude: |z_i| = 1 always
- ✅ Full phase space: z_i = cos(π·x) + i·sin(π·x)
- ⚠️ Loses magnitude information

**Optimization Tips:**
- Use for **periodic/oscillatory signals** (time series, audio)
- Combine with spectral encoder for frequency+phase decomposition
- Best with ω > 0 (resonance) to leverage phase dynamics

**Performance:** 241 sites → ~0.5ms CPU / ~0.2ms GPU (trig functions)

### Spectral Encoder (Advanced)

```json
{
  "encoder": "spectral",
  "input": [1.0, 0.5, -0.3, 0.8, 0.0, 0.0, 0.0, 0.0]  // 8D for E8
}
```

**Characteristics:**
- ✅ E8 root decomposition: projects onto 240 symmetry modes
- ✅ Richest feature space: captures geometric patterns
- ⚠️ Computationally expensive: 240 dot products
- ⚠️ Requires E8 layer (240 root vectors)

**Optimization Tips:**
- Use for **high-dimensional structured data** (images, graphs)
- Input length must match lattice dimension (8 for E8)
- Pad/truncate inputs to 8D
- Best with multi-layer configs (spectral features → Leech → HCP)

**Performance:** 241 sites → ~5ms CPU / ~1ms GPU (240 root projections)

**Mathematical Insight (MATH-REVIEW.md):**
- Root vectors have norm² = 2 (physical) = 8 (doubled coords)
- Type 1: 112 vectors (±2,±2,0,...) - **edge modes**
- Type 2: 128 vectors (±1)^8 even parity - **bulk modes**
- Projection: ⟨input, root_k⟩ = Σᵢ inputᵢ·rootᵢ,k

---

## 6. Quick Wins (Immediate Implementation)

### Win #1: Increase Regularization (1 line change)

**File:** `icarus-mcp/server.rs` line ~450 (train handler)

```rust
// BEFORE
let lambda = args.get("lambda").unwrap_or(1e-6);

// AFTER
let lambda = args.get("lambda").unwrap_or(1e-3);  // 1000× stronger regularization
```

**Expected Impact:** 10-15% NMSE reduction (prevents overfitting on small training sets)

### Win #2: Longer Warmup (1 line change)

**File:** `icarus-mcp/server.rs` line ~445

```rust
// BEFORE
let warmup = args.get("warmup_ticks").unwrap_or(100);

// AFTER
let warmup = args.get("warmup_ticks").unwrap_or(500);  // 5× longer transient decay
```

**Expected Impact:** 5-10% NMSE reduction (reduces initial condition artifacts)

### Win #3: Semi-Implicit Solver (Config change)

**MCP Tool:**
```json
{
  "tool": "icarus_config_set",
  "args": {
    "solver_method": "semi_implicit",
    "dt": 0.05  // 6× larger than Euler CFL
  }
}
```

**Expected Impact:** 2-3× faster training (unconditionally stable larger timesteps)

### Win #4: Enable Adaptive Timestep (Config change)

**File:** `icarus-engine/src/config.rs` line ~180

```rust
// BEFORE
adaptive_dt: false,

// AFTER
adaptive_dt: true,
energy_target_delta: 0.01,  // Tighter energy control
```

**Expected Impact:** 5-10% NMSE reduction (better convergence near attractors)

---

## 7. Parameter Sweep Template (Copy-Paste Ready)

### For NARMA-10 Task (Benchmark-Compatible)

```python
# Grid search over untested ranges
sweep_params = {
    "encoder": ["spatial", "phase", "spectral"],
    "lambda": [1e-3, 5e-3, 1e-2, 5e-2],  # Higher than benchmark (1e-6 to 1e-3)
    "warmup_ticks": [200, 500, 1000],     # Higher than benchmark (10 to 500)
    "ticks_per_input": [20, 50, 100],     # Higher than benchmark (5 to 50)
    "rae_steps_per_tick": [5, 10, 20],    # Higher than benchmark (1 to 5)
    "preset": ["e8_only", "full_hierarchy"]
}

# Total combinations: 3 × 4 × 3 × 3 × 3 × 2 = 648 experiments
# Estimated runtime: 648 × 30s = 5.4 hours (E8-only, CPU)
```

### For Multi-Layer Exploration (New Territory)

```python
multi_layer_sweep = {
    "preset": "full_hierarchy",  # E8 + Leech + HCP-64D + HCP-32D
    "encoder": "spectral",       # E8 root decomposition
    "encode_layer": [0, 1, 2, 3],  # Encode into which layer?
    "readout_layer": [0, 1, 2, 3], # Read from which layer?
    "transfer_strength": [0.1, 0.3, 0.5],  # Inter-layer coupling
    "lambda": [1e-3, 1e-2],
    "warmup_ticks": 500,
    "ticks_per_input": 50
}

# Total combinations: 4 × 4 × 3 × 2 = 96 experiments
# Estimated runtime: 96 × 120s = 3.2 hours (GPU required)
```

---

## 8. Open Questions / Research Directions

### High-Priority (Blocking NMSE <0.1)

1. **Why does spatial encoder outperform spectral?** - Counterintuitive (spectral has 240 modes vs spatial's direct injection)
   - **Hypothesis:** Spectral modes require longer mixing (ticks_per_input >100)
   - **Test:** Sweep ticks_per_input ∈ [100, 500] for spectral encoder

2. **What is optimal layer for encoding/readout?** - Benchmark only tested layer 0 (E8)
   - **Hypothesis:** Leech (layer 1) may provide better temporal integration (1105 sites vs 241)
   - **Test:** Encode into layer 1, readout from layer 1; compare NMSE

3. **Can metric learning improve NMSE?** - Geometrodynamic updates implemented but untested
   - **Hypothesis:** Adaptive metric may learn task-specific geometries
   - **Test:** Enable metric learning (β=0.01) during training; measure convergence

### Medium-Priority (NMSE <0.5 → <0.1)

4. **FP16 quantization impact on NMSE?** - 2× memory savings, but accuracy loss unknown
   - **Test:** GPU kernels with FP16 vs FP32; measure NMSE degradation

5. **Agent-modulated dynamics?** - All 6 agents have zero tests; potential for online adaptation
   - **Test:** Enable Learning agent (metric updates) during training; measure NMSE vs static metric

6. **Bidirectional transfer?** - Currently E8→Leech→HCP (unidirectional)
   - **Test:** Add Leech→E8 and HCP→Leech backward transfer; measure feedback effects

---

## 9. Critical Path for NMSE <0.1

**Recommended Sequence (6 weeks):**

1. **Week 1-2:** Regularization + warmup tuning (λ ∈ [1e-3, 5e-2], warmup ∈ [200, 1000])
   - **Target:** NMSE 0.3 → 0.2

2. **Week 3:** Semi-implicit + adaptive timestep integration
   - **Target:** 2× training speedup + NMSE 0.2 → 0.15

3. **Week 4:** Multi-layer encoding (spectral → E8, readout from Leech)
   - **Target:** NMSE 0.15 → 0.10

4. **Week 5:** Hybrid encoder strategy (spatial + phase + spectral blend)
   - **Target:** NMSE 0.10 → 0.08

5. **Week 6:** Metric learning enablement + long RAE mixing (ticks_per_input = 200)
   - **Target:** NMSE 0.08 → 0.05

---

## 10. Tool Usage Examples (MCP)

### Baseline (Current Best)

```json
{
  "tool": "icarus_train",
  "args": {
    "preset": "e8_only",
    "encoder": "spatial",
    "lambda": 1e-4,
    "warmup_ticks": 100,
    "ticks_per_input": 10,
    "training_samples": 2000,
    "backend": "cpu"
  }
}
```

### Optimized (Recommended)

```json
{
  "tool": "icarus_train",
  "args": {
    "preset": "full_hierarchy",
    "encoder": "spectral",
    "encode_layer": 0,
    "readout_layer": 1,
    "lambda": 5e-3,
    "warmup_ticks": 500,
    "ticks_per_input": 50,
    "rae_steps_per_tick": 10,
    "training_samples": 2000,
    "backend": "gpu",
    "solver_method": "semi_implicit",
    "adaptive_dt": true,
    "dt": 0.05
  }
}
```

---

## References

- **TEST-GAP-ANALYSIS.md** - 248 tests, coverage gaps, NARMA-10 sweep (1539 configs)
- **GPU-AUDIT.md** - GPU/CPU parity verified (<1e-4), optimization opportunities (shared mem, reduction, buffer pool)
- **MATH-REVIEW.md** - E8 roots (240 vectors), RAE stability (CFL vs semi-implicit), spectral methods (IMEX Crank-Nicolson)
- **ARCHITECTURE.md** - Multi-layer hierarchy (E8+Leech+HCP), transfer operators, cognitive agents (6 untested modules)
- **MCP-USAGE-GUIDE.md** - Tool schemas, encoding strategies, reservoir training workflow

---

**Next Steps:**
1. Run regularization sweep (λ ∈ [1e-3, 5e-2])
2. Enable semi-implicit solver + adaptive dt
3. Test multi-layer spectral encoding
4. Profile GPU kernels for shared memory optimization
5. Add NaN guards to RAE integration

