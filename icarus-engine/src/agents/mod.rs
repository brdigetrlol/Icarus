// Copyright (c) 2025-2026 brdigetrlol. All rights reserved.
// SPDX-License-Identifier: LicenseRef-Icarus-Proprietary
// See LICENSE in the repository root for full license terms.

//! Cognitive agents for the Icarus EMC
//!
//! Six specialized agents operate on the crystal manifold during each tick:
//! - **Perception**: injects external data into the manifold
//! - **WorldModel**: coordinates inter-layer information transfer
//! - **Planning**: monitors the energy landscape and convergence trends
//! - **Memory**: stores and retrieves field state snapshots
//! - **Action**: extracts output signals from the field state
//! - **Learning**: coordinates metric tensor evolution

pub mod perception;
pub mod world_model;
pub mod planning;
pub mod memory;
pub mod action;
pub mod learning;

use anyhow::Result;
use icarus_gpu::pipeline::ComputeBackend;

use crate::config::AgentConfig;
use crate::manifold::CausalCrystalManifold;

/// Phase of the EMC tick cycle
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TickPhase {
    /// Before manifold dynamics (perception injection, planning prep)
    Pre,
    /// After manifold dynamics (action extraction, memory storage, learning)
    Post,
}

/// Trait for EMC cognitive agents.
///
/// Each agent implements domain-specific logic that operates on the
/// crystal manifold during designated tick phases.
pub trait Agent: Send {
    /// Human-readable agent name (for logging)
    fn name(&self) -> &str;

    /// Which tick phases this agent participates in
    fn phases(&self) -> &[TickPhase];

    /// Execute the agent's logic for one tick phase
    fn tick(
        &mut self,
        phase: TickPhase,
        manifold: &mut CausalCrystalManifold,
        backend: &mut dyn ComputeBackend,
    ) -> Result<()>;
}

/// Orchestrates all cognitive agents during the EMC tick cycle.
pub struct AgentOrchestrator {
    agents: Vec<Box<dyn Agent>>,
}

impl AgentOrchestrator {
    /// Create the orchestrator, instantiating all enabled agents.
    pub fn new(config: &AgentConfig) -> Self {
        let mut agents: Vec<Box<dyn Agent>> = Vec::new();

        if config.enable_perception {
            agents.push(Box::new(perception::PerceptionAgent::new()));
        }
        if config.enable_world_model {
            agents.push(Box::new(world_model::WorldModelAgent::new()));
        }
        if config.enable_planning {
            agents.push(Box::new(planning::PlanningAgent::new()));
        }
        if config.enable_memory {
            agents.push(Box::new(memory::MemoryAgent::new(config.memory_capacity)));
        }
        if config.enable_action {
            agents.push(Box::new(action::ActionAgent::new()));
        }
        if config.enable_learning {
            agents.push(Box::new(learning::LearningAgent::new()));
        }

        Self { agents }
    }

    /// Run all agents in the pre-dynamics phase.
    pub fn pre_tick(
        &mut self,
        manifold: &mut CausalCrystalManifold,
        backend: &mut dyn ComputeBackend,
    ) -> Result<()> {
        for agent in &mut self.agents {
            if agent.phases().contains(&TickPhase::Pre) {
                agent.tick(TickPhase::Pre, manifold, backend)?;
            }
        }
        Ok(())
    }

    /// Run all agents in the post-dynamics phase.
    pub fn post_tick(
        &mut self,
        manifold: &mut CausalCrystalManifold,
        backend: &mut dyn ComputeBackend,
    ) -> Result<()> {
        for agent in &mut self.agents {
            if agent.phases().contains(&TickPhase::Post) {
                agent.tick(TickPhase::Post, manifold, backend)?;
            }
        }
        Ok(())
    }

    /// Number of active agents.
    pub fn num_agents(&self) -> usize {
        self.agents.len()
    }

    /// Get names of all active agents.
    pub fn agent_names(&self) -> Vec<&str> {
        self.agents.iter().map(|a| a.name()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{AgentConfig, ManifoldConfig};
    use icarus_gpu::pipeline::CpuBackend;

    #[test]
    fn test_orchestrator_default_agents() {
        let config = AgentConfig::default();
        let orch = AgentOrchestrator::new(&config);
        // Default: perception, planning, memory, action (4 agents)
        assert_eq!(orch.num_agents(), 4);
        let names = orch.agent_names();
        assert!(names.contains(&"perception"));
        assert!(names.contains(&"planning"));
        assert!(names.contains(&"memory"));
        assert!(names.contains(&"action"));
    }

    #[test]
    fn test_orchestrator_all_agents() {
        let config = AgentConfig {
            enable_perception: true,
            enable_world_model: true,
            enable_planning: true,
            enable_memory: true,
            enable_action: true,
            enable_learning: true,
            memory_capacity: 32,
        };
        let orch = AgentOrchestrator::new(&config);
        assert_eq!(orch.num_agents(), 6);
    }

    #[test]
    fn test_orchestrator_tick_cycle() {
        let agent_config = AgentConfig::default();
        let mut orch = AgentOrchestrator::new(&agent_config);

        let manifest_config = ManifoldConfig::e8_only();
        let mut manifold = CausalCrystalManifold::new(manifest_config);
        manifold.init_random(42, 0.5);
        let mut backend = CpuBackend;

        orch.pre_tick(&mut manifold, &mut backend).unwrap();
        orch.post_tick(&mut manifold, &mut backend).unwrap();
    }
}
