// Copyright (c) 2025-2026 brdigetrlol. All rights reserved.
// SPDX-License-Identifier: LicenseRef-Icarus-Proprietary
// See LICENSE in the repository root for full license terms.

//! Client-server protocol for the Icarus game.
//!
//! All messages are JSON-serialized over WebSocket.

use std::fmt;

use icarus_engine::world::{Action, AgentState, Entity, EntityKind, KeyState, PlayerState};
use serde::{Deserialize, Serialize};

// ─── Training Mode ───────────────────────────────────

/// Training mode for the game session.
///
/// - **Observation**: Human plays, EMC observes (state, action) pairs. Default mode.
/// - **DAgger**: Agent proposes actions; human overrides by pressing keys.
///   Both agent predictions and human corrections become training data.
/// - **Autonomous**: Agent controls the player. No training. Human watches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrainingMode {
    Observation,
    DAgger,
    Autonomous,
}

impl TrainingMode {
    /// Cycle to the next mode.
    #[allow(dead_code)]
    pub fn next(self) -> Self {
        match self {
            Self::Observation => Self::DAgger,
            Self::DAgger => Self::Autonomous,
            Self::Autonomous => Self::Observation,
        }
    }
}

impl fmt::Display for TrainingMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Observation => write!(f, "Observation"),
            Self::DAgger => write!(f, "DAgger"),
            Self::Autonomous => write!(f, "Autonomous"),
        }
    }
}

// ─── Client → Server ────────────────────────────────

/// Messages sent from the browser client to the game server.
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum ClientMsg {
    /// Player input (keyboard + mouse delta). Sent every frame (~60Hz).
    Input {
        keys: KeyState,
        mouse_dx: f32,
        mouse_dy: f32,
    },
    /// Explicit action (e.g., interact button).
    PlayerAction(Action),
    /// Client signals it is ready to receive the initial world state.
    Ready,
    /// Client requests a mode switch.
    ModeSwitch { mode: TrainingMode },
}

// ─── Server → Client ────────────────────────────────

/// Messages sent from the game server to the browser client.
#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub enum ServerMsg {
    /// Initial world state sent after client says Ready.
    InitWorld {
        terrain_grid: Vec<Vec<f32>>,
        terrain_extent: f32,
        entities: Vec<EntityInit>,
        seed: u64,
        /// E8 lattice site positions projected to 3D (241 points).
        lattice_positions: Vec<[f32; 3]>,
    },
    /// Per-tick world state update (20Hz).
    WorldState {
        player: PlayerState,
        agent: AgentState,
        entities: Vec<EntityUpdate>,
        time_of_day: f32,
        tick: u64,
        /// Predicted action direction vector (for arrow overlay).
        agent_action_vec: Option<[f32; 3]>,
    },
    /// Training progress update (sent periodically).
    TrainingUpdate {
        mode: TrainingMode,
        nmse: f32,
        samples: u64,
        confidence: f32,
        backends: Vec<BackendStatus>,
        trained: bool,
        interventions: u64,
        /// Number of online RLS weight updates since last batch retrain.
        online_updates: u64,
        /// Number of EWC prior tasks registered (0 = plain ridge).
        ewc_tasks: usize,
        /// Number of samples in the replay buffer.
        replay_size: usize,
        /// Number of completed batch retrains.
        retrain_count: u32,
        /// NMSE history across all retrains (for graph).
        nmse_history: Vec<f32>,
        /// Recent event log entries (convergence, attractor transitions).
        events: Vec<String>,
    },
    /// E8 lattice field magnitudes for overlay rendering (sent every ~10 ticks).
    LatticeOverlay {
        /// Field magnitude at each of 241 sites: sqrt(re² + im²).
        values: Vec<f32>,
    },
    /// Mode change confirmation sent to client.
    ModeChanged {
        mode: TrainingMode,
        message: String,
    },
    /// Cortana emotion engine state update (sent periodically).
    CortanaUpdate {
        dominant_emotion: String,
        mood: String,
        pleasure: f32,
        arousal: f32,
        dominance: f32,
        /// Plutchik 8 primary emotion activations [joy, sadness, trust, disgust, fear, anger, surprise, anticipation].
        plutchik: [f32; 8],
        emotional_intensity: f32,
        memory_count: usize,
        creative_drive: f32,
        social_bonding: f32,
        /// Neuromodulator levels: [DA, NE, ACh, 5-HT, oxytocin, endorphin, cortisol, GABA].
        neuromodulators: [f32; 8],
    },
}

/// Compact entity initialization data sent once.
#[derive(Debug, Clone, Serialize)]
pub struct EntityInit {
    pub id: u32,
    pub kind: String,
    pub position: [f32; 3],
    pub rotation_y: f32,
    pub scale: f32,
    pub color: [f32; 3],
    pub pickable: bool,
}

impl From<&Entity> for EntityInit {
    fn from(e: &Entity) -> Self {
        Self {
            id: e.id,
            kind: match e.kind {
                EntityKind::Tree => "tree",
                EntityKind::Rock => "rock",
                EntityKind::Orb => "orb",
                EntityKind::Crate => "crate",
                EntityKind::Light => "light",
            }
            .to_string(),
            position: e.position,
            rotation_y: e.rotation_y,
            scale: e.scale,
            color: e.color,
            pickable: e.pickable,
        }
    }
}

/// Per-tick entity state delta (only mutable fields).
#[derive(Debug, Clone, Serialize)]
pub struct EntityUpdate {
    pub id: u32,
    pub position: [f32; 3],
    pub held: bool,
    pub active: bool,
}

impl From<&Entity> for EntityUpdate {
    fn from(e: &Entity) -> Self {
        Self {
            id: e.id,
            position: e.position,
            held: e.held,
            active: e.active,
        }
    }
}

/// Status of a single compute backend in the ensemble.
#[derive(Debug, Clone, Serialize)]
pub struct BackendStatus {
    pub name: String,
    pub state_dim: usize,
    pub ticks: u64,
}
