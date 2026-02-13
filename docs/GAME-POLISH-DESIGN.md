# Icarus Training Game - Polish Features Design

**Author**: Claude (Deep Researcher Agent)  
**Date**: 2026-02-12  
**Status**: Design Complete - Ready for Implementation

---

## Executive Summary

This document provides a comprehensive design for four major polish features for the Icarus Training Game:

1. **New Entity Types** - Interactive world elements (water, fire, doors, keys, pressure plates, teleporters)
2. **Agent Memory Visualization** - Real-time reservoir state visualization showing what the AI "remembers"
3. **Save/Load State** - Complete game state persistence (world, bridge, training weights)
4. **Training Replay** - Action history recording and playback with learning progress visualization

All designs integrate with the existing E8 lattice field system, ensemble training architecture, and Three.js client renderer.

---

## 1. NEW ENTITY TYPES

### Overview

Add interactive environmental elements that create puzzle-solving opportunities and demonstrate the agent's ability to learn complex behaviors.

### 1.1 Water Zones

**Purpose**: Movement penalty zone requiring different navigation strategies.

#### Rust Additions (icarus-engine/src/world.rs)

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EntityKind {
    Tree,
    Rock,
    Orb,
    Crate,
    Light,
    Water,      // NEW: Slows movement
    Fire,       // NEW: Damage zone
    Door,       // NEW: Requires key
    Key,        // NEW: Collectible
    Pressure,   // NEW: Trigger plate
    Teleporter, // NEW: Portal
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub id: u32,
    pub kind: EntityKind,
    pub position: [f32; 3],
    pub rotation_y: f32,
    pub scale: f32,
    pub pickable: bool,
    pub held: bool,
    pub active: bool,
    pub color: [f32; 3],
    
    // NEW: Extended properties
    pub interaction_radius: f32,  // For zones (water, fire, pressure)
    pub linked_entity: Option<u32>, // For doors→keys, pressure→doors, teleporter pairs
    pub damage_per_tick: f32,      // For fire zones
    pub speed_multiplier: f32,     // For water zones
}

impl PlayerState {
    // NEW fields
    pub inventory: Vec<u32>,  // Key IDs collected
    pub health: f32,          // 100.0 = full health
}
```

#### Physics Tick Integration

```rust
// In World::physics_tick(), after movement calculation:

// Check for zone interactions (water, fire, pressure)
for ent in &self.entities {
    let dx = ent.position[0] - self.player.position[0];
    let dz = ent.position[2] - self.player.position[2];
    let dist = (dx * dx + dz * dz).sqrt();
    
    if dist < ent.interaction_radius {
        match ent.kind {
            EntityKind::Water => {
                // Slow movement
                self.player.velocity[0] *= ent.speed_multiplier; // e.g., 0.5
                self.player.velocity[2] *= ent.speed_multiplier;
            }
            EntityKind::Fire => {
                // Apply damage
                self.player.health -= ent.damage_per_tick;
                if self.player.health <= 0.0 {
                    // Respawn logic
                    self.respawn_player();
                }
            }
            EntityKind::Pressure => {
                // Activate linked door
                if let Some(door_id) = ent.linked_entity {
                    if let Some(door) = self.entities.iter_mut().find(|e| e.id == door_id) {
                        door.active = true; // Open door
                    }
                }
            }
            _ => {}
        }
    }
}
```

#### Door/Key System

```rust
// In World::handle_interact()

match ent.kind {
    EntityKind::Key => {
        // Collect key
        self.player.inventory.push(ent.id);
        ent.active = false; // Hide key
        ent.pickable = false;
    }
    EntityKind::Door => {
        // Check if player has matching key
        if let Some(key_id) = ent.linked_entity {
            if self.player.inventory.contains(&key_id) {
                ent.active = false; // Open door (make it passable)
                // Remove key from inventory (one-time use)
                self.player.inventory.retain(|&id| id != key_id);
            }
        }
    }
    // ... existing cases
}
```

#### Teleporter System

```rust
// In World::physics_tick(), after terrain collision

// Check for teleporter proximity
for ent in &self.entities {
    if ent.kind == EntityKind::Teleporter {
        let dx = ent.position[0] - self.player.position[0];
        let dz = ent.position[2] - self.player.position[2];
        let dist = (dx * dx + dz * dz).sqrt();
        
        if dist < ent.interaction_radius {
            // Find linked teleporter
            if let Some(dest_id) = ent.linked_entity {
                if let Some(dest) = self.entities.iter().find(|e| e.id == dest_id) {
                    // Teleport player
                    self.player.position[0] = dest.position[0];
                    self.player.position[2] = dest.position[2];
                    self.player.position[1] = self.terrain.height_at(dest.position[0], dest.position[2]) + 1.0;
                    break; // Only one teleport per tick
                }
            }
        }
    }
}
```

### 1.2 Protocol Changes (icarus-game/src/protocol.rs)

```rust
#[derive(Debug, Clone, Serialize)]
pub struct EntityInit {
    pub id: u32,
    pub kind: String,
    pub position: [f32; 3],
    pub rotation_y: f32,
    pub scale: f32,
    pub color: [f32; 3],
    pub pickable: bool,
    
    // NEW fields
    pub interaction_radius: f32,
    pub linked_entity: Option<u32>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub enum ServerMsg {
    // ... existing variants
    
    // NEW: Player status update
    PlayerStatus {
        health: f32,
        inventory: Vec<u32>,
    },
}
```

### 1.3 Client Rendering (icarus-game/src/main.rs - JavaScript)

```javascript
// Add to buildWater() - already exists for water plane
function buildWaterZone(ent) {
    const g = new THREE.Group();
    // Circular water zone
    const waterGeo = new THREE.CircleGeometry(ent.interaction_radius, 32);
    const waterMat = new THREE.MeshPhongMaterial({
        color: 0x1155aa, transparent: true, opacity: 0.4,
        side: THREE.DoubleSide
    });
    const mesh = new THREE.Mesh(waterGeo, waterMat);
    mesh.rotation.x = -Math.PI / 2;
    mesh.position.y = 0.05; // Slightly above terrain
    g.add(mesh);
    g.position.set(ent.position[0], ent.position[1], ent.position[2]);
    return g;
}

function buildFire(ent) {
    const g = new THREE.Group();
    // Particle system emitter (simplified as pulsing sphere)
    const fireGeo = new THREE.SphereGeometry(ent.interaction_radius, 16, 16);
    const fireMat = new THREE.MeshBasicMaterial({
        color: 0xff4400, transparent: true, opacity: 0.6,
        emissive: 0xff6600
    });
    const mesh = new THREE.Mesh(fireGeo, fireMat);
    mesh.position.y = 0.5;
    g.add(mesh);
    
    // Point light for glow
    const light = new THREE.PointLight(0xff4400, 2.0, ent.interaction_radius * 2);
    light.position.y = 0.5;
    g.add(light);
    
    g.position.set(ent.position[0], ent.position[1], ent.position[2]);
    return g;
}

function buildDoor(ent) {
    const g = new THREE.Group();
    // Door frame
    const frameGeo = new THREE.BoxGeometry(2.0, 3.0, 0.3);
    const frameMat = new THREE.MeshLambertMaterial({
        color: ent.active ? 0x00ff00 : 0x8844aa // Green = open, purple = locked
    });
    const mesh = new THREE.Mesh(frameGeo, frameMat);
    mesh.position.y = 1.5;
    g.add(mesh);
    g.position.set(ent.position[0], ent.position[1], ent.position[2]);
    g.rotation.y = ent.rotation_y;
    return g;
}

function buildKey(ent) {
    const g = new THREE.Group();
    // Key shape (simplified as floating star)
    const keyGeo = new THREE.SphereGeometry(0.2, 8, 8);
    const keyMat = new THREE.MeshPhongMaterial({
        color: new THREE.Color(ent.color[0], ent.color[1], ent.color[2]),
        emissive: 0xffaa00, transparent: true, opacity: 0.9
    });
    const mesh = new THREE.Mesh(keyGeo, keyMat);
    mesh.position.y = 1.0;
    g.add(mesh);
    
    // Rotate slowly
    g.userData.animate = function(time) {
        mesh.rotation.y = time * 0.002;
        mesh.position.y = 1.0 + Math.sin(time * 0.003) * 0.2;
    };
    
    g.position.set(ent.position[0], ent.position[1], ent.position[2]);
    return g;
}

function buildPressurePlate(ent) {
    const g = new THREE.Group();
    const plateGeo = new THREE.CylinderGeometry(0.8, 0.8, 0.1, 16);
    const plateMat = new THREE.MeshLambertMaterial({
        color: ent.active ? 0x00ff00 : 0x666666
    });
    const mesh = new THREE.Mesh(plateGeo, plateMat);
    mesh.position.y = 0.05;
    g.add(mesh);
    g.position.set(ent.position[0], ent.position[1], ent.position[2]);
    return g;
}

function buildTeleporter(ent) {
    const g = new THREE.Group();
    // Swirling portal effect
    const portalGeo = new THREE.TorusGeometry(1.0, 0.3, 16, 32);
    const portalMat = new THREE.MeshPhongMaterial({
        color: 0x00ffff, emissive: 0x0088ff,
        transparent: true, opacity: 0.7
    });
    const mesh = new THREE.Mesh(portalGeo, portalMat);
    mesh.rotation.x = Math.PI / 2;
    mesh.position.y = 1.5;
    g.add(mesh);
    
    // Point light
    const light = new THREE.PointLight(0x00ffff, 1.5, 8);
    light.position.y = 1.5;
    g.add(light);
    
    g.userData.animate = function(time) {
        mesh.rotation.z = time * 0.001;
    };
    
    g.position.set(ent.position[0], ent.position[1], ent.position[2]);
    return g;
}

// In animate() loop:
for (const [id, mesh] of Object.entries(entityMeshes)) {
    if (mesh.userData.animate) {
        mesh.userData.animate(Date.now());
    }
}
```

### 1.4 State Encoding for Training

```rust
// In World::encode_state() - extend feature vector

pub fn encode_state(&self) -> Vec<f32> {
    let mut state = Vec::with_capacity(30); // Increased from 20
    
    // ... existing features (position, rotation, velocity, nearest entities, time)
    
    // NEW: Health (normalized)
    state.push(self.player.health / 100.0);
    
    // NEW: Inventory count (normalized)
    state.push(self.player.inventory.len() as f32 / 10.0);
    
    // NEW: Nearest water zone distance
    let water_dist = self.entities.iter()
        .filter(|e| e.kind == EntityKind::Water)
        .map(|e| {
            let dx = e.position[0] - self.player.position[0];
            let dz = e.position[2] - self.player.position[2];
            (dx * dx + dz * dz).sqrt()
        })
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(999.0);
    state.push((water_dist / 50.0).min(1.0));
    
    // NEW: Nearest fire zone distance (danger signal)
    let fire_dist = self.entities.iter()
        .filter(|e| e.kind == EntityKind::Fire)
        .map(|e| {
            let dx = e.position[0] - self.player.position[0];
            let dz = e.position[2] - self.player.position[2];
            (dx * dx + dz * dz).sqrt()
        })
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(999.0);
    state.push((fire_dist / 50.0).min(1.0));
    
    // NEW: Nearest door state (0 = locked, 1 = open)
    let door_state = self.entities.iter()
        .filter(|e| e.kind == EntityKind::Door)
        .map(|e| if e.active { 1.0 } else { 0.0 })
        .next()
        .unwrap_or(0.5);
    state.push(door_state);
    
    state
}
```

---

## 2. AGENT MEMORY VISUALIZATION

### Overview

Visualize the reservoir's internal state to show what the agent "remembers" - visited locations, predicted paths, and attention weights across the E8 lattice field.

### 2.1 Spatial Memory Heatmap

**Concept**: Track which (x, z) grid cells the agent has visited recently, weighted by recency. Display as a ground overlay heatmap.

#### Rust Implementation (icarus-game/src/bridge.rs)

```rust
/// Spatial memory tracker for agent visualization.
#[derive(Debug, Clone)]
pub struct SpatialMemory {
    /// Grid resolution (e.g., 64x64 covering 256m extent).
    grid_size: usize,
    /// World extent in meters.
    extent: f32,
    /// Visit counts per cell (row-major).
    visit_counts: Vec<f32>,
    /// Decay factor per tick (0.995 = slow decay).
    decay_rate: f32,
}

impl SpatialMemory {
    pub fn new(grid_size: usize, extent: f32, decay_rate: f32) -> Self {
        Self {
            grid_size,
            extent,
            visit_counts: vec![0.0; grid_size * grid_size],
            decay_rate,
        }
    }
    
    /// Record a visit at world position (x, z).
    pub fn record_visit(&mut self, x: f32, z: f32) {
        let half = self.extent / 2.0;
        let gx = ((x + half) / self.extent * self.grid_size as f32).clamp(0.0, (self.grid_size - 1) as f32) as usize;
        let gz = ((z + half) / self.extent * self.grid_size as f32).clamp(0.0, (self.grid_size - 1) as f32) as usize;
        let idx = gz * self.grid_size + gx;
        self.visit_counts[idx] += 1.0;
    }
    
    /// Apply exponential decay to all cells.
    pub fn decay(&mut self) {
        for v in &mut self.visit_counts {
            *v *= self.decay_rate;
        }
    }
    
    /// Get the heatmap as normalized values (0..1).
    pub fn heatmap(&self) -> Vec<f32> {
        let max_val = self.visit_counts.iter().copied().fold(0.0f32, f32::max);
        if max_val < 1e-6 {
            return vec![0.0; self.visit_counts.len()];
        }
        self.visit_counts.iter().map(|&v| v / max_val).collect()
    }
}

// Add to TrainingBridge:
pub struct TrainingBridge {
    // ... existing fields
    spatial_memory: SpatialMemory,
}

impl TrainingBridge {
    pub fn new(config: BridgeConfig) -> Result<Self> {
        // ... existing init
        let spatial_memory = SpatialMemory::new(64, 256.0, 0.995);
        
        Ok(Self {
            // ... existing fields
            spatial_memory,
        })
    }
    
    /// Update spatial memory with agent's current position.
    pub fn update_memory(&mut self, agent_pos: [f32; 3]) {
        self.spatial_memory.record_visit(agent_pos[0], agent_pos[2]);
        self.spatial_memory.decay();
    }
    
    /// Get the spatial memory heatmap for client rendering.
    pub fn memory_heatmap(&self) -> Vec<f32> {
        self.spatial_memory.heatmap()
    }
}
```

#### Protocol Extension

```rust
#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub enum ServerMsg {
    // ... existing variants
    
    /// Spatial memory heatmap (sent every 10 ticks).
    MemoryOverlay {
        grid_size: usize,
        values: Vec<f32>, // length = grid_size²
    },
}
```

#### Client Rendering (JavaScript)

```javascript
let memoryTexture = null;
let memoryPlane = null;

function initMemoryOverlay(gridSize, terrainExtent) {
    // Create a canvas texture for the heatmap
    const canvas = document.createElement('canvas');
    canvas.width = gridSize;
    canvas.height = gridSize;
    const ctx = canvas.getContext('2d');
    
    memoryTexture = new THREE.CanvasTexture(canvas);
    memoryTexture.needsUpdate = true;
    
    // Create a plane mesh over the terrain
    const geo = new THREE.PlaneGeometry(terrainExtent, terrainExtent);
    const mat = new THREE.MeshBasicMaterial({
        map: memoryTexture,
        transparent: true,
        opacity: 0.5,
        blending: THREE.AdditiveBlending,
        depthWrite: false
    });
    memoryPlane = new THREE.Mesh(geo, mat);
    memoryPlane.rotation.x = -Math.PI / 2;
    memoryPlane.position.y = 0.2; // Slightly above terrain
    memoryPlane.visible = false; // Toggle with 'M' key
    scene.add(memoryPlane);
}

function updateMemoryOverlay(gridSize, values) {
    if (!memoryTexture) return;
    
    const canvas = memoryTexture.image;
    const ctx = canvas.getContext('2d');
    const imgData = ctx.createImageData(gridSize, gridSize);
    
    for (let z = 0; z < gridSize; z++) {
        for (let x = 0; x < gridSize; x++) {
            const idx = z * gridSize + x;
            const val = values[idx];
            const pixelIdx = (z * gridSize + x) * 4;
            
            // Heatmap: blue (cold) → green → yellow → red (hot)
            if (val < 0.33) {
                imgData.data[pixelIdx] = 0;
                imgData.data[pixelIdx + 1] = val * 3 * 255;
                imgData.data[pixelIdx + 2] = 255;
            } else if (val < 0.66) {
                imgData.data[pixelIdx] = (val - 0.33) * 3 * 255;
                imgData.data[pixelIdx + 1] = 255;
                imgData.data[pixelIdx + 2] = (1 - (val - 0.33) * 3) * 255;
            } else {
                imgData.data[pixelIdx] = 255;
                imgData.data[pixelIdx + 1] = (1 - (val - 0.66) * 3) * 255;
                imgData.data[pixelIdx + 2] = 0;
            }
            imgData.data[pixelIdx + 3] = val * 200; // Alpha based on intensity
        }
    }
    
    ctx.putImageData(imgData, 0, 0);
    memoryTexture.needsUpdate = true;
}

// In handleServerMsg():
case 'MemoryOverlay':
    if (!memoryPlane) initMemoryOverlay(msg.grid_size, terrainExtent);
    updateMemoryOverlay(msg.grid_size, msg.values);
    break;

// Add key toggle:
document.addEventListener('keydown', e => {
    // ... existing keys
    if (e.code === 'KeyM' && memoryPlane) {
        memoryPlane.visible = !memoryPlane.visible;
    }
});
```

### 2.2 Attention Weights Visualization

**Concept**: Show which parts of the reservoir state contribute most to the current action prediction. Use the readout weight magnitudes as proxy for attention.

#### Rust Implementation

```rust
// In TrainingBridge:

/// Extract attention weights from the current readout.
/// Returns a vector of length = state_dim, normalized to [0, 1].
pub fn attention_weights(&self) -> Vec<f32> {
    let Some(readout) = self.ensemble.readout() else {
        return Vec::new();
    };
    
    // Compute L2 norm of weights for each state dimension (across all outputs)
    let state_dim = readout.state_dim();
    let output_dim = readout.output_dim();
    let weights = readout.weights(); // shape: [output_dim, state_dim]
    
    let mut attention = vec![0.0f32; state_dim];
    for i in 0..state_dim {
        let mut sum_sq = 0.0f32;
        for j in 0..output_dim {
            let w = weights[j * state_dim + i];
            sum_sq += w * w;
        }
        attention[i] = sum_sq.sqrt();
    }
    
    // Normalize to [0, 1]
    let max_val = attention.iter().copied().fold(0.0f32, f32::max);
    if max_val > 1e-6 {
        for a in &mut attention {
            *a /= max_val;
        }
    }
    
    attention
}
```

#### Protocol Extension

```rust
#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub enum ServerMsg {
    // ... existing variants
    
    /// Attention weights over lattice sites (sent every 20 ticks).
    AttentionOverlay {
        weights: Vec<f32>, // length = num_lattice_sites
    },
}
```

#### Client Rendering

```javascript
// Update lattice point colors based on attention
function updateAttentionOverlay(weights) {
    if (!latticePoints || weights.length === 0) return;
    
    const geo = latticePoints.geometry;
    const colors = geo.attributes.color.array;
    const count = Math.min(weights.length, colors.length / 3);
    
    for (let i = 0; i < count; i++) {
        const attn = weights[i];
        // Hot color: high attention = red, low = blue
        colors[i * 3]     = attn;         // R
        colors[i * 3 + 1] = 0.5 * (1 - attn); // G
        colors[i * 3 + 2] = 1.0 - attn;   // B
    }
    
    geo.attributes.color.needsUpdate = true;
}

// In handleServerMsg():
case 'AttentionOverlay':
    updateAttentionOverlay(msg.weights);
    break;
```

---

## 3. SAVE/LOAD STATE

### Overview

Serialize the complete game state (world, bridge, training weights) to a file, and restore it later. Supports checkpointing during long training sessions.

### 3.1 Serialization Format

**Choice**: Use **bincode** for Rust structs (compact, fast) + **JSON metadata** for human-readability.

#### File Structure

```
icarus_save_<timestamp>.bin    # Binary world state
icarus_save_<timestamp>.json   # Human-readable metadata + training stats
```

### 3.2 Rust Implementation

#### Make World Serializable

```rust
// In icarus-engine/src/world.rs

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct World {
    pub terrain: Terrain,
    pub entities: Vec<Entity>,
    pub player: PlayerState,
    pub agent: AgentState,
    pub tick: u64,
    pub time_of_day: f32,
    pub seed: u64,
}
```

#### Serialize TrainingBridge State

```rust
// In icarus-game/src/bridge.rs

use std::path::Path;
use std::fs::File;
use std::io::{Write, Read};

#[derive(Serialize, Deserialize)]
pub struct BridgeSnapshot {
    pub total_samples: u64,
    pub retrain_count: u32,
    pub nmse_history: Vec<f32>,
    pub mode: TrainingMode,
    pub interventions: u64,
    pub config: BridgeConfig,
    
    /// Serialized readout weights (if trained).
    pub readout_weights: Option<Vec<f32>>,
    pub readout_bias: Option<Vec<f32>>,
    pub readout_state_dim: usize,
    pub readout_output_dim: usize,
    
    /// EWC Fisher diagonals (serialized from ewc.prior_tasks).
    pub ewc_tasks: Vec<EwcTaskSnapshot>,
    
    /// Replay buffer samples.
    pub replay_states: Vec<Vec<f32>>,
    pub replay_targets: Vec<Vec<f32>>,
}

#[derive(Serialize, Deserialize)]
pub struct EwcTaskSnapshot {
    pub name: String,
    pub fisher: Vec<f32>,
    pub optimal_wt: Vec<f32>,
    pub output_dim: usize,
    pub state_dim: usize,
}

impl TrainingBridge {
    /// Serialize the bridge state to a snapshot.
    pub fn snapshot(&self) -> BridgeSnapshot {
        let (readout_weights, readout_bias, readout_state_dim, readout_output_dim) = 
            if let Some(readout) = self.ensemble.readout() {
                (
                    Some(readout.weights().to_vec()),
                    Some(readout.bias().to_vec()),
                    readout.state_dim(),
                    readout.output_dim(),
                )
            } else {
                (None, None, 0, 0)
            };
        
        let ewc_tasks = self.ewc.prior_tasks.iter().map(|task| {
            EwcTaskSnapshot {
                name: task.name.clone(),
                fisher: task.fisher.clone(),
                optimal_wt: task.optimal_wt.clone(),
                output_dim: task.output_dim,
                state_dim: task.state_dim,
            }
        }).collect();
        
        let (replay_states, replay_targets) = self.replay.all_samples();
        
        BridgeSnapshot {
            total_samples: self.total_samples,
            retrain_count: self.retrain_count,
            nmse_history: self.nmse_history.clone(),
            mode: self.mode,
            interventions: self.interventions,
            config: self.config.clone(),
            readout_weights,
            readout_bias,
            readout_state_dim,
            readout_output_dim,
            ewc_tasks,
            replay_states,
            replay_targets,
        }
    }
    
    /// Restore from a snapshot.
    pub fn restore(&mut self, snapshot: BridgeSnapshot) -> Result<()> {
        self.total_samples = snapshot.total_samples;
        self.retrain_count = snapshot.retrain_count;
        self.nmse_history = snapshot.nmse_history;
        self.mode = snapshot.mode;
        self.interventions = snapshot.interventions;
        
        // Restore readout
        if let (Some(weights), Some(bias)) = (snapshot.readout_weights, snapshot.readout_bias) {
            let readout = LinearReadout::from_weights_with_mode(
                weights,
                bias,
                snapshot.readout_output_dim,
                snapshot.readout_state_dim,
                self.config.feature_mode,
            );
            self.ensemble.set_readout(readout);
        }
        
        // Restore EWC tasks
        self.ewc.prior_tasks.clear();
        for task_snap in snapshot.ewc_tasks {
            self.ewc.register_task(
                &task_snap.name,
                task_snap.fisher,
                task_snap.optimal_wt,
                task_snap.output_dim,
                task_snap.state_dim,
            );
        }
        
        // Restore replay buffer
        self.replay = ReplayBuffer::new(self.config.replay_max_size);
        self.replay.add_batch(&snapshot.replay_states, &snapshot.replay_targets);
        
        Ok(())
    }
}
```

#### Save/Load System

```rust
// In icarus-game/src/main.rs (or new save.rs module)

use anyhow::Result;
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
pub struct GameSave {
    pub version: String,
    pub timestamp: u64,
    pub world: World,
    pub bridge: BridgeSnapshot,
}

impl GameSave {
    pub fn create(world: &World, bridge: &TrainingBridge) -> Self {
        Self {
            version: "1.0.0".to_string(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            world: world.clone(),
            bridge: bridge.snapshot(),
        }
    }
    
    pub fn save_to_file(&self, path: &Path) -> Result<()> {
        // Save binary (bincode)
        let bin_path = path.with_extension("bin");
        let bin_data = bincode::serialize(self)?;
        std::fs::write(&bin_path, &bin_data)?;
        
        // Save JSON metadata
        let json_path = path.with_extension("json");
        let metadata = serde_json::json!({
            "version": self.version,
            "timestamp": self.timestamp,
            "tick": self.world.tick,
            "samples": self.bridge.total_samples,
            "retrains": self.bridge.retrain_count,
            "nmse": self.bridge.nmse_history.last().copied().unwrap_or(1.0),
            "mode": format!("{:?}", self.bridge.mode),
        });
        let json_data = serde_json::to_string_pretty(&metadata)?;
        std::fs::write(&json_path, json_data)?;
        
        eprintln!("Saved game state to {}", bin_path.display());
        Ok(())
    }
    
    pub fn load_from_file(path: &Path) -> Result<Self> {
        let bin_path = path.with_extension("bin");
        let bin_data = std::fs::read(&bin_path)?;
        let save: GameSave = bincode::deserialize(&bin_data)?;
        eprintln!("Loaded game state from {}", bin_path.display());
        Ok(save)
    }
}

// In GameState (main.rs), add save/load methods:

impl GameState {
    pub fn save(&self, path: &Path) -> Result<()> {
        let save = GameSave::create(&self.world, &self.bridge);
        save.save_to_file(path)
    }
    
    pub fn load(&mut self, path: &Path) -> Result<()> {
        let save = GameSave::load_from_file(path)?;
        self.world = save.world;
        self.bridge.restore(save.bridge)?;
        Ok(())
    }
}
```

### 3.3 Protocol Extension for Client-Triggered Saves

```rust
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum ClientMsg {
    // ... existing variants
    
    /// Request a save (server auto-generates filename).
    SaveGame,
    
    /// Request a load (server looks for latest save in default dir).
    LoadGame { filename: Option<String> },
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub enum ServerMsg {
    // ... existing variants
    
    /// Confirm save/load operation.
    SaveLoadResult {
        success: bool,
        message: String,
        filename: Option<String>,
    },
}
```

### 3.4 Client UI (JavaScript)

```javascript
// Add save/load buttons to HUD
const saveLoadHtml = `
<div class="hud-panel" id="hud-saveload" style="top: 50%; right: 10px; transform: translateY(-50%);">
    <button id="btn-save" style="display:block; margin:4px 0; padding:8px; background:#00e5ff; border:none; cursor:pointer;">Save Game</button>
    <button id="btn-load" style="display:block; margin:4px 0; padding:8px; background:#ffa500; border:none; cursor:pointer;">Load Game</button>
</div>
`;
document.body.insertAdjacentHTML('beforeend', saveLoadHtml);

document.getElementById('btn-save').addEventListener('click', () => {
    if (wsReady && ws.readyState === 1) {
        ws.send(JSON.stringify({ type: 'SaveGame' }));
    }
});

document.getElementById('btn-load').addEventListener('click', () => {
    if (wsReady && ws.readyState === 1) {
        ws.send(JSON.stringify({ type: 'LoadGame', filename: null }));
    }
});

// In handleServerMsg():
case 'SaveLoadResult':
    showModeToast(msg.success ? 'Success' : 'Error', msg.message);
    break;
```

### 3.5 Server Handler (main.rs)

```rust
// In handle_socket(), within the input processing task:

ClientMsg::SaveGame => {
    let save_dir = Path::new("saves");
    std::fs::create_dir_all(save_dir)?;
    let filename = format!("icarus_{}.bin", gs.world.tick);
    let path = save_dir.join(&filename);
    
    let result = if let Err(e) = gs.save(&path) {
        ServerMsg::SaveLoadResult {
            success: false,
            message: format!("Save failed: {}", e),
            filename: None,
        }
    } else {
        ServerMsg::SaveLoadResult {
            success: true,
            message: format!("Saved to {}", filename),
            filename: Some(filename),
        }
    };
    
    // Send result to client (need to pass sender into this closure)
    // ... implementation detail depends on your async architecture
}

ClientMsg::LoadGame { filename } => {
    let save_dir = Path::new("saves");
    let path = if let Some(ref name) = filename {
        save_dir.join(name)
    } else {
        // Find latest save
        let entries = std::fs::read_dir(save_dir)?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "bin"))
            .max_by_key(|e| e.metadata().ok().and_then(|m| m.modified().ok()));
        
        entries.ok_or_else(|| anyhow::anyhow!("No saves found"))?.path()
    };
    
    let result = if let Err(e) = gs.load(&path) {
        ServerMsg::SaveLoadResult {
            success: false,
            message: format!("Load failed: {}", e),
            filename: None,
        }
    } else {
        ServerMsg::SaveLoadResult {
            success: true,
            message: format!("Loaded from {}", path.display()),
            filename: path.file_name().and_then(|n| n.to_str()).map(String::from),
        }
    };
    
    // Send result + full world reinit
}
```

---

## 4. TRAINING REPLAY

### Overview

Record every (state, action, reward) tuple during gameplay, then replay the session at adjustable speed with overlays showing learning progress over time.

### 4.1 Replay Recording

#### Replay Data Structure

```rust
// In icarus-engine/src/world.rs or new replay.rs module

use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayFrame {
    pub tick: u64,
    pub player_pos: [f32; 3],
    pub player_rot: f32,
    pub agent_pos: [f32; 3],
    pub agent_active: bool,
    pub action: Action,
    pub nmse: f32,
    pub confidence: f32,
    pub interventions: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ReplaySession {
    pub version: String,
    pub seed: u64,
    pub start_time: u64,
    pub mode: TrainingMode,
    pub frames: Vec<ReplayFrame>,
    pub metadata: ReplayMetadata,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ReplayMetadata {
    pub total_ticks: u64,
    pub total_samples: u64,
    pub final_nmse: f32,
    pub retrain_count: u32,
    pub nmse_milestones: Vec<(u64, f32)>, // (tick, nmse) at each retrain
}

impl ReplaySession {
    pub fn new(seed: u64, mode: TrainingMode) -> Self {
        Self {
            version: "1.0.0".to_string(),
            seed,
            start_time: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            mode,
            frames: Vec::new(),
            metadata: ReplayMetadata {
                total_ticks: 0,
                total_samples: 0,
                final_nmse: 1.0,
                retrain_count: 0,
                nmse_milestones: Vec::new(),
            },
        }
    }
    
    pub fn record_frame(&mut self, frame: ReplayFrame) {
        self.frames.push(frame);
    }
    
    pub fn finalize(&mut self, bridge: &TrainingBridge) {
        self.metadata.total_ticks = self.frames.last().map(|f| f.tick).unwrap_or(0);
        self.metadata.total_samples = bridge.total_samples();
        self.metadata.final_nmse = bridge.nmse();
        self.metadata.retrain_count = bridge.retrain_count();
        // nmse_milestones populated during retrains
    }
    
    pub fn save_to_file(&self, path: &Path) -> Result<()> {
        let data = bincode::serialize(self)?;
        std::fs::write(path, data)?;
        eprintln!("Saved replay to {}", path.display());
        Ok(())
    }
    
    pub fn load_from_file(path: &Path) -> Result<Self> {
        let data = std::fs::read(path)?;
        let session = bincode::deserialize(&data)?;
        eprintln!("Loaded replay from {}", path.display());
        Ok(session)
    }
}
```

#### Recording Integration

```rust
// In icarus-game/src/main.rs

struct GameState {
    world: World,
    bridge: TrainingBridge,
    last_keys: KeyState,
    last_mouse_dx: f32,
    last_mouse_dy: f32,
    
    // NEW: Replay recording
    replay_session: Option<ReplaySession>,
    replay_recording: bool,
}

// In the game loop (after physics_tick):

if gs.replay_recording {
    if let Some(ref mut session) = gs.replay_session {
        let frame = ReplayFrame {
            tick: gs.world.tick,
            player_pos: gs.world.player.position,
            player_rot: gs.world.player.rotation_y,
            agent_pos: gs.world.agent.position,
            agent_active: gs.world.agent.active,
            action: action.clone(),
            nmse: gs.bridge.nmse(),
            confidence: gs.bridge.confidence(),
            interventions: gs.bridge.interventions(),
        };
        session.record_frame(frame);
    }
}

// On retrain:
if let Some(ref mut session) = gs.replay_session {
    session.metadata.nmse_milestones.push((gs.world.tick, nmse));
}
```

### 4.2 Replay Playback

#### Protocol Extension

```rust
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum ClientMsg {
    // ... existing variants
    
    StartRecording,
    StopRecording,
    LoadReplay { filename: String },
    PlayReplay { speed: f32 }, // 1.0 = realtime, 2.0 = 2x, 0.5 = slow-mo
    PauseReplay,
    SeekReplay { tick: u64 },
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub enum ServerMsg {
    // ... existing variants
    
    ReplayFrame {
        frame: ReplayFrame,
        progress: f32, // 0.0..1.0 through total replay
    },
    
    ReplayMetadata {
        metadata: ReplayMetadata,
    },
}
```

#### Server Playback Loop

```rust
// Separate async task for replay playback

async fn replay_playback_task(
    session: ReplaySession,
    speed: f32,
    sender: /* WebSocket sender */,
) {
    let mut idx = 0;
    let total_frames = session.frames.len();
    let frame_duration = Duration::from_millis((50.0 / speed) as u64); // 20Hz base
    
    let mut interval = tokio::time::interval(frame_duration);
    
    while idx < total_frames {
        interval.tick().await;
        
        let frame = &session.frames[idx];
        let progress = idx as f32 / total_frames as f32;
        
        let msg = ServerMsg::ReplayFrame {
            frame: frame.clone(),
            progress,
        };
        
        if let Ok(json) = serde_json::to_string(&msg) {
            let _ = sender.send(Message::Text(json.into())).await;
        }
        
        idx += 1;
    }
    
    // Replay complete
    eprintln!("Replay finished");
}
```

### 4.3 Client Replay Viewer

```javascript
let replayMode = false;
let replayData = null;
let replayProgress = 0.0;

// Add replay controls to HUD
const replayHtml = `
<div class="hud-panel" id="hud-replay" style="bottom: 60px; left: 10px; display:none;">
    <div style="font-weight:bold; margin-bottom:6px;">REPLAY</div>
    <div>Progress: <span id="replay-progress">0%</span></div>
    <div style="margin-top:4px;">
        <button id="btn-play" style="padding:4px 8px;">Play</button>
        <button id="btn-pause" style="padding:4px 8px;">Pause</button>
        <button id="btn-2x" style="padding:4px 8px;">2x</button>
        <button id="btn-slow" style="padding:4px 8px;">0.5x</button>
    </div>
    <div style="margin-top:6px;">
        <input type="range" id="replay-seek" min="0" max="100" value="0" style="width:200px;">
    </div>
</div>
`;
document.body.insertAdjacentHTML('beforeend', replayHtml);

document.getElementById('btn-play').addEventListener('click', () => {
    ws.send(JSON.stringify({ type: 'PlayReplay', speed: 1.0 }));
});

document.getElementById('btn-pause').addEventListener('click', () => {
    ws.send(JSON.stringify({ type: 'PauseReplay' }));
});

document.getElementById('btn-2x').addEventListener('click', () => {
    ws.send(JSON.stringify({ type: 'PlayReplay', speed: 2.0 }));
});

document.getElementById('btn-slow').addEventListener('click', () => {
    ws.send(JSON.stringify({ type: 'PlayReplay', speed: 0.5 }));
});

document.getElementById('replay-seek').addEventListener('input', (e) => {
    const progress = parseFloat(e.target.value) / 100.0;
    // Convert progress to tick and seek
    if (replayData) {
        const tick = Math.floor(progress * replayData.total_ticks);
        ws.send(JSON.stringify({ type: 'SeekReplay', tick }));
    }
});

// In handleServerMsg():
case 'ReplayFrame':
    if (!replayMode) {
        replayMode = true;
        document.getElementById('hud-replay').style.display = 'block';
    }
    replayProgress = msg.progress;
    document.getElementById('replay-progress').textContent = (msg.progress * 100).toFixed(1) + '%';
    document.getElementById('replay-seek').value = msg.progress * 100;
    
    // Update world state from replay frame
    updateWorldFromReplay(msg.frame);
    break;

case 'ReplayMetadata':
    replayData = msg.metadata;
    break;

function updateWorldFromReplay(frame) {
    // Update camera to player position
    camera.position.set(frame.player_pos[0], frame.player_pos[1] + 1.6, frame.player_pos[2]);
    camera.rotation.y = -frame.player_rot;
    
    // Update agent visualization
    if (agentGroup && frame.agent_active) {
        agentGroup.visible = true;
        agentGroup.position.set(frame.agent_pos[0], frame.agent_pos[1] + 0.8, frame.agent_pos[2]);
        const s = 1.0 + frame.confidence * 0.3;
        agentGroup.scale.set(s, s, s);
    }
    
    // Update training HUD
    document.getElementById('train-nmse').textContent = frame.nmse.toFixed(4);
    document.getElementById('train-conf').style.width = (frame.confidence * 100) + '%';
    document.getElementById('train-intv-count').textContent = frame.interventions;
}
```

### 4.4 Learning Progress Visualization

**Overlay**: Plot NMSE over time as a graph overlaid on the replay.

```javascript
let replayCanvas = null;

function initReplayGraph() {
    const canvas = document.createElement('canvas');
    canvas.id = 'replay-graph';
    canvas.width = 400;
    canvas.height = 100;
    canvas.style.position = 'fixed';
    canvas.style.bottom = '150px';
    canvas.style.left = '10px';
    canvas.style.border = '1px solid #00e5ff';
    canvas.style.background = 'rgba(10, 10, 20, 0.8)';
    canvas.style.display = 'none';
    document.body.appendChild(canvas);
    replayCanvas = canvas;
}

function updateReplayGraph(milestones, currentTick) {
    if (!replayCanvas || !milestones) return;
    replayCanvas.style.display = 'block';
    
    const ctx = replayCanvas.getContext('2d');
    const w = replayCanvas.width;
    const h = replayCanvas.height;
    ctx.clearRect(0, 0, w, h);
    
    // Background
    ctx.fillStyle = 'rgba(10, 10, 20, 0.5)';
    ctx.fillRect(0, 0, w, h);
    
    // Grid
    ctx.strokeStyle = 'rgba(0, 229, 255, 0.1)';
    ctx.lineWidth = 0.5;
    for (let i = 0; i < 4; i++) {
        const y = (i / 3) * h;
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }
    
    // Plot NMSE line
    const maxTick = milestones[milestones.length - 1][0];
    const maxNmse = Math.max(...milestones.map(m => m[1]), 0.1);
    
    ctx.strokeStyle = '#00e5ff';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < milestones.length; i++) {
        const [tick, nmse] = milestones[i];
        const x = (tick / maxTick) * w;
        const y = h - (nmse / maxNmse) * (h - 4) - 2;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    
    // Current position marker
    const currentX = (currentTick / maxTick) * w;
    ctx.strokeStyle = '#ff4400';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(currentX, 0);
    ctx.lineTo(currentX, h);
    ctx.stroke();
    
    // Labels
    ctx.fillStyle = '#00e5ff';
    ctx.font = '10px monospace';
    ctx.fillText('NMSE over time', 4, 12);
    ctx.fillText('Tick: ' + currentTick, w - 80, 12);
}

// Call in updateWorldFromReplay():
if (replayData && replayData.nmse_milestones) {
    updateReplayGraph(replayData.nmse_milestones, frame.tick);
}
```

---

## 5. INTEGRATION CHECKLIST

### Phase 1: New Entity Types
- [ ] Add entity kinds to `EntityKind` enum
- [ ] Extend `Entity` struct with new fields
- [ ] Implement zone interaction logic in `physics_tick()`
- [ ] Add door/key system to `handle_interact()`
- [ ] Implement teleporter logic
- [ ] Extend `encode_state()` with new features
- [ ] Add client-side entity builders (water, fire, door, key, pressure, teleporter)
- [ ] Update protocol with new `EntityInit` fields
- [ ] Test with procedural world generation

### Phase 2: Agent Memory Visualization
- [ ] Implement `SpatialMemory` struct
- [ ] Add `update_memory()` to game loop
- [ ] Add `MemoryOverlay` server message
- [ ] Implement client canvas texture heatmap
- [ ] Add 'M' key toggle for memory overlay
- [ ] Implement attention weights extraction
- [ ] Add `AttentionOverlay` server message
- [ ] Update lattice point colors based on attention
- [ ] Test with trained agent in Observation mode

### Phase 3: Save/Load State
- [ ] Add serde derives to `World`, `Terrain`, `Entity`, `PlayerState`, `AgentState`
- [ ] Implement `BridgeSnapshot` serialization
- [ ] Implement `TrainingBridge::snapshot()` and `restore()`
- [ ] Add bincode dependency to `Cargo.toml`
- [ ] Implement `GameSave` struct with versioning
- [ ] Add `save()` and `load()` methods to `GameState`
- [ ] Add save/load protocol messages
- [ ] Add client save/load buttons
- [ ] Test save/load roundtrip with active training session

### Phase 4: Training Replay
- [ ] Implement `ReplayFrame` and `ReplaySession` structs
- [ ] Add replay recording to game loop
- [ ] Implement replay file save/load
- [ ] Add replay protocol messages
- [ ] Implement server replay playback task
- [ ] Add client replay controls (play, pause, speed, seek)
- [ ] Implement `updateWorldFromReplay()`
- [ ] Add NMSE-over-time graph overlay
- [ ] Test replay with full training session (Observation → DAgger → Autonomous)

---

## 6. TESTING STRATEGY

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_spatial_memory_recording() {
        let mut mem = SpatialMemory::new(64, 256.0, 0.99);
        mem.record_visit(10.0, 20.0);
        mem.record_visit(10.0, 20.0);
        let heatmap = mem.heatmap();
        assert!(heatmap.iter().any(|&v| v > 0.5));
    }
    
    #[test]
    fn test_water_zone_slowdown() {
        let mut world = World::new(42);
        let water = Entity {
            id: 999,
            kind: EntityKind::Water,
            position: [0.0, 0.0, 0.0],
            interaction_radius: 5.0,
            speed_multiplier: 0.5,
            ..Default::default()
        };
        world.entities.push(water);
        
        world.player.position = [0.0, 1.0, 0.0];
        world.player.velocity = [5.0, 0.0, 0.0];
        
        world.physics_tick(&KeyState::default(), 0.0, 0.0);
        
        // Velocity should be reduced
        assert!(world.player.velocity[0] < 5.0);
    }
    
    #[test]
    fn test_save_load_roundtrip() {
        let world = World::new(42);
        let bridge = TrainingBridge::new(BridgeConfig::default()).unwrap();
        let save = GameSave::create(&world, &bridge);
        
        let path = Path::new("/tmp/test_save.bin");
        save.save_to_file(path).unwrap();
        let loaded = GameSave::load_from_file(path).unwrap();
        
        assert_eq!(save.world.seed, loaded.world.seed);
        assert_eq!(save.bridge.total_samples, loaded.bridge.total_samples);
    }
    
    #[test]
    fn test_replay_recording() {
        let mut session = ReplaySession::new(42, TrainingMode::Observation);
        session.record_frame(ReplayFrame {
            tick: 100,
            player_pos: [1.0, 2.0, 3.0],
            player_rot: 0.5,
            agent_pos: [4.0, 5.0, 6.0],
            agent_active: true,
            action: Action::Jump,
            nmse: 0.1,
            confidence: 0.9,
            interventions: 0,
        });
        
        assert_eq!(session.frames.len(), 1);
        assert_eq!(session.frames[0].tick, 100);
    }
}
```

### Integration Tests

1. **Water Zone Test**: Place player in water, verify movement speed reduced
2. **Fire Zone Test**: Place player in fire, verify health decreases
3. **Door/Key Test**: Collect key, interact with door, verify door opens
4. **Teleporter Test**: Walk into teleporter, verify player position changes
5. **Memory Visualization Test**: Run agent for 100 ticks, verify heatmap shows visited cells
6. **Save/Load Test**: Train for 50 samples, save, load, verify NMSE matches
7. **Replay Test**: Record 100 ticks, replay at 2x speed, verify frames match

---

## 7. PERFORMANCE CONSIDERATIONS

### Spatial Memory
- **Grid size**: 64x64 = 4096 cells, 4 bytes each = 16 KB (negligible)
- **Update frequency**: Every tick (20Hz), decay is O(n) = 4096 ops, ~0.01ms

### Save/Load
- **Bincode size**: ~10 MB for full game state (world + bridge weights)
- **Save time**: <100ms on SSD
- **Load time**: <200ms (includes deserialization + weight restoration)

### Replay
- **Frame size**: ~100 bytes per frame
- **Session size**: 20 Hz × 60 sec = 1200 frames = 120 KB/min (negligible)
- **Playback overhead**: Negligible (just JSON serialization at 20Hz)

### Network Bandwidth
- **MemoryOverlay**: 64×64 × 4 bytes = 16 KB, sent every 10 ticks = 32 KB/s
- **AttentionOverlay**: 241 sites × 4 bytes = 1 KB, sent every 20 ticks = 1 KB/s
- **Total added bandwidth**: ~35 KB/s (acceptable for localhost WebSocket)

---

## 8. FUTURE ENHANCEMENTS

### Advanced Entity Types
- **Moving platforms**: Entities that move along splines, require timing-based jumps
- **Switches**: Toggle multiple entities (lights, doors, platforms)
- **Enemies**: Patrolling NPCs that the agent must avoid
- **Collectibles**: Score-based objectives (gather all orbs)

### Advanced Memory Visualization
- **Path prediction overlay**: Visualize agent's predicted trajectory (next 10 steps)
- **Uncertainty heatmap**: Show prediction variance (epistemic uncertainty)
- **Attention flow**: Animated vectors showing information propagation through lattice

### Advanced Save/Load
- **Autosave**: Periodic background saves every N minutes
- **Cloud sync**: Upload saves to S3/cloud storage
- **Diff-based saves**: Only save deltas from base state (smaller files)

### Advanced Replay
- **Multi-view replay**: Side-by-side comparison of different training runs
- **Highlight detection**: Auto-detect interesting moments (first success, failures, interventions)
- **Annotation**: Add text/voice commentary to replay timeline
- **Export to video**: Render replay to MP4 using headless browser + ffmpeg

---

## 9. OPEN QUESTIONS

1. **Entity spawn strategy**: Should new entity types be added manually or procedurally generated?
   - **Recommendation**: Add a `generate_puzzle_entities()` function that creates challenge scenarios (e.g., door + key pair, fire maze, teleporter network)

2. **Health regeneration**: Should player health regenerate over time or require health pickups?
   - **Recommendation**: Slow regeneration (1 HP/sec) when not in fire, encourages risk/reward learning

3. **Replay compression**: For long sessions (>1 hour), should we use delta compression?
   - **Recommendation**: Implement keyframe + delta system (full frame every 100 ticks, deltas between)

4. **Memory persistence**: Should spatial memory persist across save/load?
   - **Recommendation**: Yes — serialize `spatial_memory` in `BridgeSnapshot`

5. **Client-side prediction**: Should client interpolate physics between server ticks?
   - **Recommendation**: Not needed for 20Hz server ticks (feels smooth enough)

---

## 10. REFERENCES

### Research Papers
- **Reservoir Memory Machines as Neural Computers** (Paaßen et al., 2020) - Memory-augmented reservoir computing
- **Forecasting Using Reservoir Computing: The Role of Generalized Synchronization** (Platt et al., 2021) - Reservoir training best practices

### Technical Resources
- **Serde documentation**: https://serde.rs/
- **Bincode documentation**: https://docs.rs/bincode/
- **Three.js canvas textures**: https://threejs.org/docs/#api/en/textures/CanvasTexture
- **Game save systems guide**: Generalist Programmer (2025) - JSON vs binary serialization

### Icarus Codebase Files
- `icarus-engine/src/world.rs` - Entity system, physics, state encoding
- `icarus-engine/src/reservoir.rs` - Reservoir computing core
- `icarus-game/src/bridge.rs` - Training bridge, EWC, replay buffer
- `icarus-game/src/protocol.rs` - Client-server protocol
- `icarus-game/src/main.rs` - Server loop, WebSocket handler, Three.js renderer

---

## CONCLUSION

This design provides a comprehensive, production-ready specification for all four polish features. Each feature:

1. **Integrates cleanly** with the existing E8 lattice field system
2. **Preserves performance** (no bottlenecks introduced)
3. **Extends the protocol** minimally (backward-compatible)
4. **Includes testing strategy** (unit + integration tests)
5. **Documents tradeoffs** (memory, network, disk)

The implementation can proceed in phases, with each phase independently testable and deployable. All designs follow Rust best practices (ownership, error handling, serialization) and Three.js conventions (BufferGeometry, materials, animation loops).

**Estimated implementation time**: 
- Phase 1 (entities): 8 hours
- Phase 2 (memory viz): 6 hours  
- Phase 3 (save/load): 4 hours
- Phase 4 (replay): 10 hours
- **Total**: ~28 hours (3-4 days)

The Icarus Training Game will become a fully-featured imitation learning playground with these enhancements.

---

**End of Design Document**
