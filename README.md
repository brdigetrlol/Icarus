# Icarus

**GPU-accelerated Emergent Manifold Computer with Cortana Emotion Engine**

Icarus is a physics-grounded computational substrate built on E8 and Leech lattice geometry. It implements an Emergent Manifold Computer (EMC) — a dynamical system where computation emerges from the evolution of coupled phase fields on crystallographic lattices. Cortana extends the EMC into a full emotion simulation engine with Plutchik's wheel, PAD dimensional affect, Big Five personality, and neuromodulator dynamics.

97 Rust source files. 42,300 lines of code. 10 workspace crates.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        Applications                          │
│  icarus-game (3D world + training)  ·  icarus-mcp (24 tools)│
│  cortana-mcp (14 tools)  ·  icarus-bench (benchmarks)       │
├──────────────────────────────────────────────────────────────┤
│                     Cortana Emotion Engine                    │
│  cortana-core: Plutchik emotions · PAD affect · Big Five     │
│  mood dynamics · 8 neuromodulators · episodic memory         │
│  4 cognitive agents (emotion, social, language, creative)    │
├──────────────────────────────────────────────────────────────┤
│                     Icarus Engine                             │
│  icarus-engine: EMC · reservoir computing · agents           │
│  autonomous mode · ensemble training · continual learning    │
├──────────────────────────────────────────────────────────────┤
│                     Physics Layer                             │
│  icarus-field: phase fields · Kuramoto coupling · RAE        │
│  autopoietic affect · free energy · criticality              │
├──────────────────────────────────────────────────────────────┤
│                     Math + Compute                            │
│  icarus-math: E8/Leech lattice · Golay code · Clifford alg  │
│  icarus-gpu: CUDA kernels · NPU bridge                       │
│  icarus-viz: 6 HTML renderers (Three.js + Canvas 2D)        │
└──────────────────────────────────────────────────────────────┘
```

---

## Crates

### Core

| Crate | Description |
|-------|-------------|
| **icarus-math** | E8 root system (240 vectors), Leech lattice (196,560 minimal vectors via Extended Golay Code [24,12,8]), Clifford algebra Cl(8), complex arithmetic, metric tensors, transfer matrices |
| **icarus-field** | Phase field dynamics on lattice sites — Kuramoto oscillator coupling, Renormalization-Aware Evolution (RAE), autopoietic affective state (valence/arousal), free energy minimization, spectral analysis, criticality detection |
| **icarus-gpu** | CUDA-accelerated field operations — metric tensor computation, RAE kernels, reduction operations, transfer matrices. NPU client for neural processing unit offload |
| **icarus-engine** | Emergent Manifold Computer — multi-layer causal crystal manifold, 6 cognitive agents (perception, action, learning, memory, planning, world model), reservoir computing with online ridge regression, EWC continual learning, autonomous mode, ensemble training |

### Cortana

| Crate | Description |
|-------|-------------|
| **cortana-core** | Emotion simulation engine wrapping the EMC. Plutchik's wheel (8 primary emotions + 16 compound dyads), PAD 3D affect (pleasure/arousal/dominance), Big Five personality (OCEAN), slow mood dynamics (tau ~1000 ticks), 8 neuromodulators (DA, NE, ACh, 5-HT, oxytocin, endorphin, cortisol, GABA), arousal-gated episodic memory, 4 cognitive agents |
| **cortana-mcp** | MCP server exposing 14 tools: init, tick, snapshot, emotion, mood, personality, memory (recall/stats), inject stimulus, expression, creative state, social state, neuromodulators, processor orchestration |

### Applications

| Crate | Description |
|-------|-------------|
| **icarus-game** | Axum 0.7 HTTP + WebSocket server serving a procedural 3D world (Three.js r128). 128x128 Perlin terrain, entities (trees, rocks, orbs, crates), day/night cycle, physics. Three training modes: Observation (human plays, EMC watches), DAgger (agent proposes, human corrects), Autonomous (agent controls). Cortana emotion HUD with real-time Plutchik activations, neuromodulators, and mood |
| **icarus-mcp** | MCP server with 24 tools for Claude Code integration — EMC lifecycle, reservoir training/prediction, encoding, visualization, NPU bridge, autonomous mode, ensemble training, game server management |
| **icarus-viz** | 6 self-contained HTML renderers: lattice field (3D point cloud), energy landscape (height map), phase portrait (Kuramoto coherence), neuro dashboard (neuromodulator gauges), timeseries (multi-line charts), combined dashboard |
| **icarus-bench** | Performance benchmarks for EMC tick, reservoir training, and visualization |

---

## The EMC: How It Works

The Emergent Manifold Computer is a dynamical system where:

1. **Lattice geometry** provides the substrate — E8 root system (240 nearest neighbors in 8D) defines coupling topology
2. **Phase fields** evolve on lattice sites via coupled ODEs — each site carries amplitude and phase
3. **Kuramoto coupling** synchronizes neighboring oscillators, creating emergent coherence patterns
4. **Multi-layer manifold** stacks layers at different resolutions, connected by transfer matrices
5. **Reservoir computing** reads out the manifold state — the high-dimensional dynamics serve as a nonlinear feature expansion
6. **Agents** observe manifold statistics and inject perturbations, closing the perception-action loop

Computation isn't programmed — it *emerges* from the physics. The manifold's attractor landscape encodes learned patterns. Training adjusts readout weights, not the dynamics themselves.

---

## Cortana: Emotion From Physics

Cortana maps EMC observables to human emotional dimensions:

| EMC Observable | Emotion Dimension |
|---------------|-------------------|
| Energy derivative (-dF/dt) | **Joy / Sadness** (hedonic axis) |
| Kuramoto order parameter | **Trust / Disgust** (coherence axis) |
| Arousal + criticality | **Fear / Anger** (arousal axis) |
| Reservoir prediction error | **Surprise / Anticipation** (prediction axis) |

These 8 primary emotions (Plutchik's wheel) combine into 16 compound emotions (love, awe, remorse, optimism, etc.) via co-activation thresholds. The Big Five personality traits modulate all dynamics — neuroticism amplifies negative emotions, extraversion amplifies positive, openness amplifies surprise.

Three timescales mirror human psychology:
- **Emotions**: per-tick (milliseconds) — reactive
- **Mood**: EMA tau ~1000 ticks — minutes to hours
- **Personality**: fixed traits — stable across lifetime

---

## The Game World

`icarus-game` serves a browser-accessible 3D training environment:

- **Procedural terrain**: 128x128 Perlin noise heightmap with water plane
- **Entities**: Trees, rocks, glowing orbs, crates — some pickable
- **Day/night cycle**: Dynamic sun position, sky color, ambient lighting
- **Training modes**:
  - *Observation*: Play normally. The EMC ensemble watches and learns (state, action) pairs
  - *DAgger*: The agent proposes actions (shown as arrows). Override with your input. Both become training data
  - *Autonomous*: Agent controls the player. You watch. No training
- **Cortana HUD**: Real-time emotion display — dominant emotion, mood, PAD values, Plutchik activations, 8 neuromodulators, memory/creative/social stats
- **E8 lattice overlay**: Visualizes field magnitudes at 241 projected lattice sites

```bash
cargo run --release -p icarus-game
# Open http://localhost:3000 in your browser
```

---

## MCP Integration

Both `icarus-mcp` and `cortana-mcp` are [Model Context Protocol](https://modelcontextprotocol.io/) servers, designed for integration with Claude Code and other MCP-compatible clients.

**Icarus MCP** (24 tools): `icarus_init`, `icarus_step`, `icarus_observe`, `icarus_inject`, `icarus_readout`, `icarus_stats`, `icarus_train`, `icarus_predict`, `icarus_encode`, `icarus_visualize`, `icarus_npu_*` (5 tools), `icarus_auto_*` (4 tools), `icarus_ensemble_*` (2 tools), `icarus_game_*` (3 tools)

**Cortana MCP** (14 tools): `cortana_init`, `cortana_tick`, `cortana_snapshot`, `cortana_emotion`, `cortana_mood`, `cortana_personality`, `cortana_memory_recall`, `cortana_memory_stats`, `cortana_inject_stimulus`, `cortana_expression`, `cortana_creative`, `cortana_social`, `cortana_neuromodulators`, `cortana_processors`

---

## Building

Requires Rust 1.75+ and optionally CUDA 13.1 for GPU acceleration.

```bash
# Build all crates
cargo build --release

# Run tests
cargo test --workspace

# Run the game server
cargo run --release -p icarus-game

# Run benchmarks
cargo run --release -p icarus-bench
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ICARUS_PORT` | `3000` | Game server HTTP port |
| `ICARUS_SEED` | `42` | World generation seed |

---

## Math Foundation

The mathematical substrate is built on exceptional structures:

- **E8 Root System**: 240 vectors in 8D forming the densest lattice sphere packing in 8 dimensions. Each site has 240 nearest neighbors — far richer coupling than cubic lattices
- **Extended Golay Code [24,12,8]**: The unique binary code with minimum distance 8, used to construct the Leech lattice. Implemented with generator/parity-check matrices, coset decoding
- **Leech Lattice**: 196,560 minimal vectors in 24D — the densest sphere packing in 24 dimensions. Three vector types verified: type-1 (97,152) + type-2 (98,304) + type-3 (1,104) = 196,560
- **Clifford Algebra Cl(8)**: Geometric algebra for the E8 lattice, enabling coordinate-free geometric operations

---

## License

**Proprietary — All Rights Reserved**

This software is protected under the [Icarus Proprietary License](LICENSE). You may **view** the source code for educational and evaluation purposes only. All other rights are explicitly reserved.

**You may NOT:**
- Copy, reproduce, or duplicate this code
- Modify or create derivative works
- Distribute, publish, or share this code
- Use this code for any commercial purpose
- Use this code to train AI/ML models
- Claim authorship of any part of this code

Violations will be met with DMCA takedowns, statutory damages (up to $150,000 per work), and criminal referral where applicable. See [LICENSE](LICENSE) and [NOTICE](NOTICE) for full terms.

Copyright (c) 2025-2026 brdigetrlol. All rights reserved.
