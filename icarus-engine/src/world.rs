//! Procedural game world simulation for Icarus.
//!
//! Server-side physics simulation with Perlin noise terrain, entity system,
//! player/agent state, and a day/night cycle. All geometry is procedurally
//! generated from a seed — no external assets.
//!
//! The world runs at 20Hz physics ticks and encodes player actions for
//! the ensemble trainer's imitation learning pipeline.

use serde::{Deserialize, Serialize};

// ─── Terrain ────────────────────────────────────────

/// Procedural terrain generated from multi-octave Perlin noise.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Terrain {
    /// Heightmap grid (size × size), row-major.
    pub heightmap: Vec<f32>,
    /// Grid resolution (heightmap is size × size).
    pub size: usize,
    /// World-space extent (terrain spans -extent/2..+extent/2 on X and Z).
    pub extent: f32,
    /// Random seed used for generation.
    pub seed: u64,
}

impl Terrain {
    /// Generate terrain from multi-octave Perlin noise.
    pub fn generate(size: usize, extent: f32, seed: u64) -> Self {
        let mut heightmap = vec![0.0f32; size * size];
        let scale = extent / size as f32;

        for z in 0..size {
            for x in 0..size {
                let wx = (x as f32 - size as f32 / 2.0) * scale;
                let wz = (z as f32 - size as f32 / 2.0) * scale;
                heightmap[z * size + x] = perlin_fbm(wx * 0.05, wz * 0.05, seed, 4, 0.5);
            }
        }

        Self {
            heightmap,
            size,
            extent,
            seed,
        }
    }

    /// Sample terrain height at world position (x, z) via bilinear interpolation.
    pub fn height_at(&self, x: f32, z: f32) -> f32 {
        let half = self.extent / 2.0;
        let gx = ((x + half) / self.extent * self.size as f32).max(0.0);
        let gz = ((z + half) / self.extent * self.size as f32).max(0.0);

        let x0 = (gx as usize).min(self.size - 2);
        let z0 = (gz as usize).min(self.size - 2);
        let fx = gx - x0 as f32;
        let fz = gz - z0 as f32;

        let h00 = self.heightmap[z0 * self.size + x0];
        let h10 = self.heightmap[z0 * self.size + x0 + 1];
        let h01 = self.heightmap[(z0 + 1) * self.size + x0];
        let h11 = self.heightmap[(z0 + 1) * self.size + x0 + 1];

        let h0 = h00 + fx * (h10 - h00);
        let h1 = h01 + fx * (h11 - h01);
        h0 + fz * (h1 - h0)
    }

    /// Return the heightmap as a 2D grid (row-major) for serialization.
    pub fn grid(&self) -> Vec<Vec<f32>> {
        let mut grid = Vec::with_capacity(self.size);
        for z in 0..self.size {
            let start = z * self.size;
            grid.push(self.heightmap[start..start + self.size].to_vec());
        }
        grid
    }
}

// ─── Entities ───────────────────────────────────────

/// Kind of entity in the game world.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EntityKind {
    Tree,
    Rock,
    Orb,
    Crate,
    Light,
}

/// A game entity with position, appearance, and interaction state.
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
}

// ─── Player & Agent State ───────────────────────────

/// Player state: position, velocity, rotation, and interaction state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlayerState {
    pub position: [f32; 3],
    pub velocity: [f32; 3],
    pub rotation_y: f32,
    pub pitch: f32,
    pub on_ground: bool,
    pub held_entity: Option<u32>,
    pub speed: f32,
}

impl Default for PlayerState {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            velocity: [0.0, 0.0, 0.0],
            rotation_y: 0.0,
            pitch: 0.0,
            on_ground: true,
            held_entity: None,
            speed: 5.0,
        }
    }
}

/// Icarus agent state: position, velocity, learning confidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentState {
    pub position: [f32; 3],
    pub velocity: [f32; 3],
    pub rotation_y: f32,
    pub confidence: f32,
    pub active: bool,
}

impl Default for AgentState {
    fn default() -> Self {
        Self {
            position: [5.0, 0.0, 5.0],
            velocity: [0.0, 0.0, 0.0],
            rotation_y: 0.0,
            confidence: 0.0,
            active: false,
        }
    }
}

// ─── Actions & Input ────────────────────────────────

/// Player action — decoded from keyboard/mouse or predicted by Icarus.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Action {
    Move { dx: f32, dz: f32 },
    Jump,
    PickUp { entity_id: u32 },
    Drop,
    Push { direction: [f32; 2] },
    ToggleLight { entity_id: u32 },
    Idle,
}

/// Keyboard state from the client.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct KeyState {
    pub forward: bool,
    pub backward: bool,
    pub left: bool,
    pub right: bool,
    pub jump: bool,
    pub interact: bool,
}

// ─── World ──────────────────────────────────────────

/// The complete game world state.
#[derive(Debug, Clone)]
pub struct World {
    pub terrain: Terrain,
    pub entities: Vec<Entity>,
    pub player: PlayerState,
    pub agent: AgentState,
    pub tick: u64,
    pub time_of_day: f32,
    pub seed: u64,
}

impl World {
    /// Create a new world from a seed. Generates terrain, entities, and spawns player/agent.
    pub fn new(seed: u64) -> Self {
        let terrain = Terrain::generate(128, 256.0, seed);
        let entities = generate_entities(&terrain, seed);

        let spawn_y = terrain.height_at(0.0, 0.0) + 1.0;
        let mut player = PlayerState::default();
        player.position = [0.0, spawn_y, 0.0];

        let agent_y = terrain.height_at(5.0, 5.0) + 1.0;
        let mut agent = AgentState::default();
        agent.position = [5.0, agent_y, 5.0];

        Self {
            terrain,
            entities,
            player,
            agent,
            tick: 0,
            time_of_day: 0.25, // morning
            seed,
        }
    }

    /// Run one physics tick (1/20th of a second).
    ///
    /// Updates player position from input, applies gravity, checks terrain collision,
    /// handles entity interaction, and advances the day/night cycle.
    pub fn physics_tick(&mut self, keys: &KeyState, mouse_dx: f32, mouse_dy: f32) {
        let dt = 1.0 / 20.0; // 20Hz

        // ── Player rotation from mouse ──
        self.player.rotation_y += mouse_dx * 0.003;
        self.player.pitch = (self.player.pitch - mouse_dy * 0.003).clamp(-1.4, 1.4);

        // ── Player movement from keys ──
        let mut move_x = 0.0f32;
        let mut move_z = 0.0f32;
        if keys.forward {
            move_z -= 1.0;
        }
        if keys.backward {
            move_z += 1.0;
        }
        if keys.left {
            move_x -= 1.0;
        }
        if keys.right {
            move_x += 1.0;
        }

        // Normalize and rotate by player heading
        let len = (move_x * move_x + move_z * move_z).sqrt();
        if len > 0.01 {
            move_x /= len;
            move_z /= len;
            let cos_r = self.player.rotation_y.cos();
            let sin_r = self.player.rotation_y.sin();
            let rx = move_x * cos_r + move_z * sin_r;
            let rz = -move_x * sin_r + move_z * cos_r;
            self.player.velocity[0] = rx * self.player.speed;
            self.player.velocity[2] = rz * self.player.speed;
        } else {
            self.player.velocity[0] *= 0.8; // friction
            self.player.velocity[2] *= 0.8;
        }

        // ── Jump ──
        if keys.jump && self.player.on_ground {
            self.player.velocity[1] = 6.0;
            self.player.on_ground = false;
        }

        // ── Gravity ──
        self.player.velocity[1] -= 15.0 * dt;

        // ── Integrate position ──
        self.player.position[0] += self.player.velocity[0] * dt;
        self.player.position[1] += self.player.velocity[1] * dt;
        self.player.position[2] += self.player.velocity[2] * dt;

        // ── Terrain collision ──
        let ground = self.terrain.height_at(self.player.position[0], self.player.position[2]) + 1.0;
        if self.player.position[1] <= ground {
            self.player.position[1] = ground;
            self.player.velocity[1] = 0.0;
            self.player.on_ground = true;
        }

        // ── World bounds clamping ──
        let half = self.terrain.extent / 2.0 - 2.0;
        self.player.position[0] = self.player.position[0].clamp(-half, half);
        self.player.position[2] = self.player.position[2].clamp(-half, half);

        // ── Entity interaction ──
        if keys.interact {
            self.handle_interact();
        }

        // ── Update held entity position ──
        if let Some(held_id) = self.player.held_entity {
            if let Some(ent) = self.entities.iter_mut().find(|e| e.id == held_id) {
                let cos_r = self.player.rotation_y.cos();
                let sin_r = self.player.rotation_y.sin();
                ent.position = [
                    self.player.position[0] + sin_r * 1.5,
                    self.player.position[1] + 0.5,
                    self.player.position[2] + cos_r * 1.5,
                ];
            }
        }

        // ── Agent gravity + terrain collision ──
        self.agent.velocity[1] -= 15.0 * dt;
        self.agent.position[0] += self.agent.velocity[0] * dt;
        self.agent.position[1] += self.agent.velocity[1] * dt;
        self.agent.position[2] += self.agent.velocity[2] * dt;

        let agent_ground = self.terrain.height_at(self.agent.position[0], self.agent.position[2]) + 1.0;
        if self.agent.position[1] <= agent_ground {
            self.agent.position[1] = agent_ground;
            self.agent.velocity[1] = 0.0;
        }

        // ── Day/night cycle (1 cycle = 2400 ticks = 2 minutes at 20Hz) ──
        self.time_of_day = (self.time_of_day + dt / 120.0) % 1.0;

        self.tick += 1;
    }

    /// Handle player interaction with nearby entities.
    fn handle_interact(&mut self) {
        // If holding something, drop it
        if let Some(held_id) = self.player.held_entity.take() {
            if let Some(ent) = self.entities.iter_mut().find(|e| e.id == held_id) {
                ent.held = false;
                // Place on ground
                let ground = self.terrain.height_at(ent.position[0], ent.position[2]);
                ent.position[1] = ground + ent.scale * 0.5;
            }
            return;
        }

        // Find nearest interactable entity within 3 units
        let pp = self.player.position;
        let mut best_id = None;
        let mut best_dist = 3.0f32;

        for ent in &self.entities {
            if ent.held {
                continue;
            }
            let dx = ent.position[0] - pp[0];
            let dy = ent.position[1] - pp[1];
            let dz = ent.position[2] - pp[2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            if dist < best_dist {
                best_dist = dist;
                best_id = Some(ent.id);
            }
        }

        if let Some(id) = best_id {
            if let Some(ent) = self.entities.iter_mut().find(|e| e.id == id) {
                match ent.kind {
                    EntityKind::Orb | EntityKind::Crate if ent.pickable => {
                        ent.held = true;
                        self.player.held_entity = Some(id);
                    }
                    EntityKind::Light => {
                        ent.active = !ent.active;
                    }
                    _ => {}
                }
            }
        }
    }

    /// Apply an action from the Icarus agent.
    pub fn apply_agent_action(&mut self, action: &Action) {
        let agent_speed = 4.0;
        match action {
            Action::Move { dx, dz } => {
                let cos_r = self.agent.rotation_y.cos();
                let sin_r = self.agent.rotation_y.sin();
                self.agent.velocity[0] = (dx * cos_r + dz * sin_r) * agent_speed;
                self.agent.velocity[2] = (-dx * sin_r + dz * cos_r) * agent_speed;
            }
            Action::Jump => {
                let ground = self.terrain.height_at(self.agent.position[0], self.agent.position[2]) + 1.0;
                if (self.agent.position[1] - ground).abs() < 0.2 {
                    self.agent.velocity[1] = 6.0;
                }
            }
            Action::Idle => {
                self.agent.velocity[0] *= 0.8;
                self.agent.velocity[2] *= 0.8;
            }
            _ => {} // Agent doesn't pick up/drop/toggle yet
        }
    }

    /// Encode the current world state as a feature vector for the ensemble trainer.
    ///
    /// Returns ~20 floats: player position (3), rotation (2), velocity (3),
    /// nearest 3 entities (3×3 = 9), time_of_day (1), on_ground (1), has_item (1).
    pub fn encode_state(&self) -> Vec<f32> {
        let mut state = Vec::with_capacity(20);

        // Player position (normalized to [-1, 1])
        let half = self.terrain.extent / 2.0;
        state.push(self.player.position[0] / half);
        state.push(self.player.position[1] / 20.0); // Height normalized
        state.push(self.player.position[2] / half);

        // Player rotation
        state.push(self.player.rotation_y.sin());
        state.push(self.player.rotation_y.cos());

        // Player velocity (normalized)
        state.push(self.player.velocity[0] / 10.0);
        state.push(self.player.velocity[1] / 10.0);
        state.push(self.player.velocity[2] / 10.0);

        // Nearest 3 entities (relative position)
        let pp = self.player.position;
        let mut dists: Vec<(usize, f32)> = self
            .entities
            .iter()
            .enumerate()
            .map(|(i, e)| {
                let dx = e.position[0] - pp[0];
                let dz = e.position[2] - pp[2];
                (i, (dx * dx + dz * dz).sqrt())
            })
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        for k in 0..3 {
            if k < dists.len() {
                let ent = &self.entities[dists[k].0];
                state.push((ent.position[0] - pp[0]) / 20.0);
                state.push((ent.position[1] - pp[1]) / 10.0);
                state.push((ent.position[2] - pp[2]) / 20.0);
            } else {
                state.push(0.0);
                state.push(0.0);
                state.push(0.0);
            }
        }

        // Time of day
        state.push(self.time_of_day);

        // Flags
        state.push(if self.player.on_ground { 1.0 } else { 0.0 });
        state.push(if self.player.held_entity.is_some() {
            1.0
        } else {
            0.0
        });

        state
    }

    /// Encode an action as a feature vector for training targets.
    ///
    /// Returns 8 floats: [action_type (one-hot 5), move_dx, move_dz, entity_id_norm]
    pub fn encode_action(action: &Action) -> Vec<f32> {
        let mut enc = vec![0.0f32; 8];
        match action {
            Action::Move { dx, dz } => {
                enc[0] = 1.0;
                enc[5] = *dx;
                enc[6] = *dz;
            }
            Action::Jump => {
                enc[1] = 1.0;
            }
            Action::PickUp { entity_id } => {
                enc[2] = 1.0;
                enc[7] = *entity_id as f32 / 100.0;
            }
            Action::Drop => {
                enc[3] = 1.0;
            }
            Action::Push { direction } => {
                enc[0] = 1.0; // Treated as movement
                enc[5] = direction[0];
                enc[6] = direction[1];
            }
            Action::ToggleLight { entity_id } => {
                enc[4] = 1.0;
                enc[7] = *entity_id as f32 / 100.0;
            }
            Action::Idle => {} // All zeros
        }
        enc
    }

    /// Decode a prediction vector into an Action via argmax over action types.
    pub fn decode_action(prediction: &[f32]) -> Action {
        if prediction.len() < 7 {
            return Action::Idle;
        }

        // Find argmax over first 5 values (action types)
        let mut best_idx = 0;
        let mut best_val = prediction[0];
        for i in 1..5.min(prediction.len()) {
            if prediction[i] > best_val {
                best_val = prediction[i];
                best_idx = i;
            }
        }

        match best_idx {
            0 => Action::Move {
                dx: prediction.get(5).copied().unwrap_or(0.0),
                dz: prediction.get(6).copied().unwrap_or(0.0),
            },
            1 => Action::Jump,
            2 => Action::PickUp {
                entity_id: (prediction.get(7).copied().unwrap_or(0.0) * 100.0) as u32,
            },
            3 => Action::Drop,
            4 => Action::ToggleLight {
                entity_id: (prediction.get(7).copied().unwrap_or(0.0) * 100.0) as u32,
            },
            _ => Action::Idle,
        }
    }

    /// Convert keyboard input + mouse movement into an Action.
    pub fn keys_to_action(keys: &KeyState, mouse_dx: f32) -> Action {
        if keys.jump {
            return Action::Jump;
        }

        let mut dx = 0.0f32;
        let mut dz = 0.0f32;
        if keys.forward {
            dz -= 1.0;
        }
        if keys.backward {
            dz += 1.0;
        }
        if keys.left {
            dx -= 1.0;
        }
        if keys.right {
            dx += 1.0;
        }

        // Add mouse-based rotation influence
        dx += mouse_dx * 0.01;

        let len = (dx * dx + dz * dz).sqrt();
        if len > 0.01 {
            Action::Move {
                dx: dx / len,
                dz: dz / len,
            }
        } else {
            Action::Idle
        }
    }
}

// ─── Entity Generation ──────────────────────────────

/// Procedurally generate entities placed on the terrain.
fn generate_entities(terrain: &Terrain, seed: u64) -> Vec<Entity> {
    let mut rng = SimpleRng::new(seed.wrapping_add(12345));
    let mut entities = Vec::with_capacity(91);
    let mut id = 0u32;
    let half = terrain.extent / 2.0 - 10.0;

    // 40 trees
    for _ in 0..40 {
        let x = rng.next_range(-half, half);
        let z = rng.next_range(-half, half);
        let y = terrain.height_at(x, z);
        let scale = rng.next_range(1.5, 4.0);
        entities.push(Entity {
            id,
            kind: EntityKind::Tree,
            position: [x, y, z],
            rotation_y: rng.next_f32() * std::f32::consts::TAU,
            scale,
            pickable: false,
            held: false,
            active: true,
            color: [
                0.1 + rng.next_f32() * 0.15,
                0.4 + rng.next_f32() * 0.3,
                0.05 + rng.next_f32() * 0.1,
            ],
        });
        id += 1;
    }

    // 25 rocks
    for _ in 0..25 {
        let x = rng.next_range(-half, half);
        let z = rng.next_range(-half, half);
        let y = terrain.height_at(x, z);
        let scale = rng.next_range(0.5, 2.0);
        entities.push(Entity {
            id,
            kind: EntityKind::Rock,
            position: [x, y, z],
            rotation_y: rng.next_f32() * std::f32::consts::TAU,
            scale,
            pickable: false,
            held: false,
            active: true,
            color: [
                0.4 + rng.next_f32() * 0.2,
                0.35 + rng.next_f32() * 0.15,
                0.3 + rng.next_f32() * 0.1,
            ],
        });
        id += 1;
    }

    // 8 orbs
    for i in 0..8 {
        let x = rng.next_range(-half * 0.6, half * 0.6);
        let z = rng.next_range(-half * 0.6, half * 0.6);
        let y = terrain.height_at(x, z) + 1.5;
        let hue = i as f32 / 8.0;
        let (r, g, b) = hue_to_rgb(hue);
        entities.push(Entity {
            id,
            kind: EntityKind::Orb,
            position: [x, y, z],
            rotation_y: 0.0,
            scale: 0.4,
            pickable: true,
            held: false,
            active: true,
            color: [r, g, b],
        });
        id += 1;
    }

    // 12 crates
    for _ in 0..12 {
        let x = rng.next_range(-half * 0.8, half * 0.8);
        let z = rng.next_range(-half * 0.8, half * 0.8);
        let y = terrain.height_at(x, z) + 0.5;
        entities.push(Entity {
            id,
            kind: EntityKind::Crate,
            position: [x, y, z],
            rotation_y: rng.next_f32() * std::f32::consts::TAU,
            scale: 1.0,
            pickable: true,
            held: false,
            active: true,
            color: [0.6, 0.4, 0.2],
        });
        id += 1;
    }

    // 6 lights
    for _ in 0..6 {
        let x = rng.next_range(-half * 0.7, half * 0.7);
        let z = rng.next_range(-half * 0.7, half * 0.7);
        let y = terrain.height_at(x, z) + 2.5;
        entities.push(Entity {
            id,
            kind: EntityKind::Light,
            position: [x, y, z],
            rotation_y: 0.0,
            scale: 0.3,
            pickable: false,
            held: false,
            active: true,
            color: [1.0, 0.9, 0.7],
        });
        id += 1;
    }

    entities
}

// ─── Simple RNG (xorshift64) ────────────────────────

struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() & 0xFFFFFF) as f32 / 0xFFFFFF as f32
    }

    fn next_range(&mut self, min: f32, max: f32) -> f32 {
        min + self.next_f32() * (max - min)
    }
}

// ─── Perlin Noise ───────────────────────────────────

fn hash_2d(x: i32, y: i32, seed: u64) -> u64 {
    let mut h = seed;
    h = h.wrapping_add(x as u64).wrapping_mul(6364136223846793005);
    h = h.wrapping_add(y as u64).wrapping_mul(6364136223846793005);
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51afd7ed558ccd);
    h ^= h >> 33;
    h
}

fn grad_dot(hash: u64, fx: f32, fy: f32) -> f32 {
    match hash & 3 {
        0 => fx + fy,
        1 => -fx + fy,
        2 => fx - fy,
        3 => -fx - fy,
        _ => unreachable!(),
    }
}

fn perlin_2d(x: f32, y: f32, seed: u64) -> f32 {
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;

    // Smoothstep
    let u = fx * fx * (3.0 - 2.0 * fx);
    let v = fy * fy * (3.0 - 2.0 * fy);

    let n00 = grad_dot(hash_2d(x0, y0, seed), fx, fy);
    let n10 = grad_dot(hash_2d(x0 + 1, y0, seed), fx - 1.0, fy);
    let n01 = grad_dot(hash_2d(x0, y0 + 1, seed), fx, fy - 1.0);
    let n11 = grad_dot(hash_2d(x0 + 1, y0 + 1, seed), fx - 1.0, fy - 1.0);

    let nx0 = n00 + u * (n10 - n00);
    let nx1 = n01 + u * (n11 - n01);
    nx0 + v * (nx1 - nx0)
}

fn perlin_fbm(x: f32, y: f32, seed: u64, octaves: u32, persistence: f32) -> f32 {
    let mut total = 0.0f32;
    let mut amplitude = 1.0f32;
    let mut frequency = 1.0f32;
    let mut max_amp = 0.0f32;

    for i in 0..octaves {
        total += perlin_2d(x * frequency, y * frequency, seed.wrapping_add(i as u64 * 1000)) * amplitude;
        max_amp += amplitude;
        amplitude *= persistence;
        frequency *= 2.0;
    }

    // Scale to roughly [-8, 8] for terrain heights
    total / max_amp * 8.0
}

// ─── Utility ────────────────────────────────────────

/// Convert HSV hue (0..1) to RGB with full saturation and brightness.
fn hue_to_rgb(h: f32) -> (f32, f32, f32) {
    let h6 = h * 6.0;
    let i = h6.floor() as i32;
    let f = h6 - i as f32;
    match i % 6 {
        0 => (1.0, f, 0.0),
        1 => (1.0 - f, 1.0, 0.0),
        2 => (0.0, 1.0, f),
        3 => (0.0, 1.0 - f, 1.0),
        4 => (f, 0.0, 1.0),
        5 => (1.0, 0.0, 1.0 - f),
        _ => (1.0, 1.0, 1.0),
    }
}

// ─── Tests ──────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_terrain_generation() {
        let terrain = Terrain::generate(64, 128.0, 42);
        assert_eq!(terrain.heightmap.len(), 64 * 64);
        assert_eq!(terrain.size, 64);
        assert_eq!(terrain.extent, 128.0);
    }

    #[test]
    fn test_terrain_height_at_center() {
        let terrain = Terrain::generate(64, 128.0, 42);
        let h = terrain.height_at(0.0, 0.0);
        assert!(h.is_finite());
        assert!(h.abs() < 20.0);
    }

    #[test]
    fn test_terrain_height_interpolation() {
        let terrain = Terrain::generate(64, 128.0, 42);
        let h1 = terrain.height_at(1.0, 1.0);
        let h2 = terrain.height_at(1.1, 1.0);
        // Nearby points should have similar heights
        assert!((h1 - h2).abs() < 5.0);
    }

    #[test]
    fn test_terrain_grid() {
        let terrain = Terrain::generate(32, 64.0, 7);
        let grid = terrain.grid();
        assert_eq!(grid.len(), 32);
        assert_eq!(grid[0].len(), 32);
    }

    #[test]
    fn test_world_creation() {
        let world = World::new(42);
        assert_eq!(world.terrain.size, 128);
        assert!(!world.entities.is_empty());
        assert_eq!(world.tick, 0);
        assert!(world.player.position[1] > -100.0);
    }

    #[test]
    fn test_entity_generation_counts() {
        let terrain = Terrain::generate(128, 256.0, 42);
        let entities = generate_entities(&terrain, 42);
        let trees = entities.iter().filter(|e| e.kind == EntityKind::Tree).count();
        let rocks = entities.iter().filter(|e| e.kind == EntityKind::Rock).count();
        let orbs = entities.iter().filter(|e| e.kind == EntityKind::Orb).count();
        let crates = entities.iter().filter(|e| e.kind == EntityKind::Crate).count();
        let lights = entities.iter().filter(|e| e.kind == EntityKind::Light).count();
        assert_eq!(trees, 40);
        assert_eq!(rocks, 25);
        assert_eq!(orbs, 8);
        assert_eq!(crates, 12);
        assert_eq!(lights, 6);
    }

    #[test]
    fn test_physics_tick_basic() {
        let mut world = World::new(42);
        let keys = KeyState::default();
        world.physics_tick(&keys, 0.0, 0.0);
        assert_eq!(world.tick, 1);
    }

    #[test]
    fn test_physics_tick_movement() {
        let mut world = World::new(42);
        let keys = KeyState {
            forward: true,
            ..Default::default()
        };
        let start_z = world.player.position[2];
        for _ in 0..10 {
            world.physics_tick(&keys, 0.0, 0.0);
        }
        // Player should have moved
        assert!((world.player.position[2] - start_z).abs() > 0.1);
    }

    #[test]
    fn test_physics_tick_jump() {
        let mut world = World::new(42);
        let keys = KeyState {
            jump: true,
            ..Default::default()
        };
        world.physics_tick(&keys, 0.0, 0.0);
        assert!(!world.player.on_ground || world.player.velocity[1] > 0.0);
    }

    #[test]
    fn test_day_night_cycle() {
        let mut world = World::new(42);
        let t0 = world.time_of_day;
        let keys = KeyState::default();
        for _ in 0..100 {
            world.physics_tick(&keys, 0.0, 0.0);
        }
        assert!(world.time_of_day != t0);
    }

    #[test]
    fn test_encode_state() {
        let world = World::new(42);
        let state = world.encode_state();
        assert_eq!(state.len(), 20);
        for v in &state {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_encode_action_move() {
        let enc = World::encode_action(&Action::Move { dx: 0.5, dz: -0.3 });
        assert_eq!(enc.len(), 8);
        assert!((enc[0] - 1.0).abs() < 1e-6); // Move type
        assert!((enc[5] - 0.5).abs() < 1e-6);
        assert!((enc[6] - (-0.3)).abs() < 1e-6);
    }

    #[test]
    fn test_encode_action_jump() {
        let enc = World::encode_action(&Action::Jump);
        assert_eq!(enc.len(), 8);
        assert!((enc[1] - 1.0).abs() < 1e-6); // Jump type
    }

    #[test]
    fn test_decode_action_roundtrip() {
        let original = Action::Move { dx: 0.7, dz: -0.3 };
        let encoded = World::encode_action(&original);
        let decoded = World::decode_action(&encoded);
        if let Action::Move { dx, dz } = decoded {
            assert!((dx - 0.7).abs() < 1e-5);
            assert!((dz - (-0.3)).abs() < 1e-5);
        } else {
            panic!("Expected Move action");
        }
    }

    #[test]
    fn test_keys_to_action() {
        let keys = KeyState {
            forward: true,
            ..Default::default()
        };
        let action = World::keys_to_action(&keys, 0.0);
        match action {
            Action::Move { dx: _, dz } => assert!(dz < 0.0),
            _ => panic!("Expected Move action"),
        }
    }

    #[test]
    fn test_keys_to_action_idle() {
        let keys = KeyState::default();
        let action = World::keys_to_action(&keys, 0.0);
        assert!(matches!(action, Action::Idle));
    }

    #[test]
    fn test_perlin_deterministic() {
        let a = perlin_2d(1.5, 2.3, 42);
        let b = perlin_2d(1.5, 2.3, 42);
        assert!((a - b).abs() < 1e-10);
    }

    #[test]
    fn test_perlin_different_seeds() {
        let a = perlin_2d(1.5, 2.3, 42);
        let b = perlin_2d(1.5, 2.3, 99);
        // Different seeds should (almost certainly) produce different values
        assert!((a - b).abs() > 1e-10);
    }
}
