// Copyright (c) 2025-2026 brdigetrlol. All rights reserved.
// SPDX-License-Identifier: LicenseRef-Icarus-Proprietary
// See LICENSE in the repository root for full license terms.

//! Autonomous execution types for Icarus EMC.
//!
//! Defines configuration, state, events, and snapshots used by the
//! actor-model background task that drives the EMC tick loop.

use crate::agents::action::ActionOutput;
use crate::agents::planning::ConvergenceTrend;
use crate::emc::LayerStateSnapshot;
use crate::manifold::LayerStats;
use icarus_field::autopoiesis::AffectiveState;
use serde::Serialize;

// ---------------------------------------------------------------------------
// Stop conditions
// ---------------------------------------------------------------------------

/// Conditions under which the autonomous tick loop should halt.
#[derive(Debug, Clone, Serialize)]
pub enum StopCondition {
    /// Stop after this many total ticks.
    MaxTicks(u64),
    /// Stop when convergence trend is `Stable` for N consecutive ticks.
    ConvergenceStable(u32),
    /// Stop when total energy across all layers drops below threshold.
    EnergyBelow(f32),
    /// Stop after N seconds of wall-clock time.
    TimeLimitSecs(u64),
    /// Only stops via explicit `AutoStop` command.
    Manual,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the autonomous tick loop.
#[derive(Debug, Clone, Serialize)]
pub struct AutonomousConfig {
    /// Target ticks per second. `None` = uncapped (run as fast as possible).
    pub max_ticks_per_second: Option<f64>,
    /// Conditions that will cause the loop to stop automatically.
    pub stop_conditions: Vec<StopCondition>,
    /// Publish a snapshot every N ticks (1 = every tick).
    pub snapshot_interval: u32,
    /// Capacity of the event ring buffer.
    pub event_buffer_size: usize,
}

impl Default for AutonomousConfig {
    fn default() -> Self {
        Self {
            max_ticks_per_second: Some(60.0),
            stop_conditions: vec![StopCondition::Manual],
            snapshot_interval: 1,
            event_buffer_size: 256,
        }
    }
}

// ---------------------------------------------------------------------------
// State machine
// ---------------------------------------------------------------------------

/// Current state of the autonomous execution loop.
#[derive(Debug, Clone, Serialize)]
pub enum AutonomousState {
    Idle,
    Running,
    Paused,
    Completed { reason: String },
    Error { message: String },
}

impl AutonomousState {
    pub fn is_active(&self) -> bool {
        matches!(self, Self::Running | Self::Paused)
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Running => "running",
            Self::Paused => "paused",
            Self::Completed { .. } => "completed",
            Self::Error { .. } => "error",
        }
    }
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

/// Types of events emitted by the autonomous tick loop.
#[derive(Debug, Clone, Serialize)]
pub enum AutoEventType {
    /// Loop started with config summary.
    Started,
    /// Convergence trend changed (e.g. Diverging → Converging).
    ConvergenceDetected { trend: String },
    /// An attractor transition was detected on a layer.
    AttractorTransition {
        layer: usize,
        from_site: usize,
        to_site: usize,
    },
    /// Total energy crossed a configured threshold.
    EnergyThreshold { energy: f32 },
    /// Tick milestone (every 100 / 1000 / 10000 ticks).
    TickMilestone { tick: u64 },
    /// Loop stopped (normal or by command).
    Stopped { reason: String },
    /// An error occurred during ticking.
    Error { message: String },
}

/// A single event emitted by the autonomous tick loop.
#[derive(Debug, Clone, Serialize)]
pub struct AutoEvent {
    pub tick: u64,
    pub timestamp: String,
    pub event_type: AutoEventType,
}

// ---------------------------------------------------------------------------
// Snapshot
// ---------------------------------------------------------------------------

/// A read-only snapshot of EMC state published by the background actor.
///
/// Combines data from `emc.stats()`, `emc.observe()`, planning-agent
/// convergence trend, and action-agent output. Query tools read this
/// directly via `Arc<RwLock>` without going through the command channel.
#[derive(Debug, Clone, Serialize)]
pub struct EmcSnapshot {
    /// Current tick count.
    pub tick: u64,
    /// ISO-8601 timestamp when the snapshot was taken.
    pub timestamp: String,
    /// Per-layer statistics (energy, amplitude, site count).
    pub layer_stats: Vec<LayerStats>,
    /// Total number of lattice sites across all layers.
    pub total_sites: usize,
    /// Name of the compute backend (e.g. "cuda", "cpu").
    pub backend_name: String,
    /// Estimated memory usage in bytes.
    pub memory_bytes: usize,
    /// Current affective state (valence, arousal, coherence).
    pub affective_state: AffectiveState,
    /// Convergence trend from the planning agent.
    pub convergence_trend: Option<ConvergenceTrend>,
    /// Most recent action-agent output.
    pub action_output: Option<ActionOutput>,
    /// Full per-layer state (amplitudes, phases, energies) for observe queries.
    pub layer_states: Vec<LayerStateSnapshot>,
}
