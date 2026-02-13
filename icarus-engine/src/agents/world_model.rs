//! World Model Agent — Inter-layer information transfer coordination
//!
//! The world model agent oversees information flow between lattice layers.
//! Primary inter-layer transfer is handled by the manifold itself;
//! this agent provides supplementary coordination and quality monitoring.

use anyhow::Result;
use icarus_gpu::pipeline::ComputeBackend;

use crate::agents::{Agent, TickPhase};
use crate::manifold::CausalCrystalManifold;

/// World model agent: coordinates inter-layer information flow.
pub struct WorldModelAgent {
    /// Number of transfer cycles executed
    pub transfer_count: u64,
}

impl WorldModelAgent {
    pub fn new() -> Self {
        Self { transfer_count: 0 }
    }
}

impl Agent for WorldModelAgent {
    fn name(&self) -> &str {
        "world_model"
    }

    fn phases(&self) -> &[TickPhase] {
        &[TickPhase::Post]
    }

    fn tick(
        &mut self,
        _phase: TickPhase,
        _manifold: &mut CausalCrystalManifold,
        _backend: &mut dyn ComputeBackend,
    ) -> Result<()> {
        // Primary inter-layer transfer is handled by manifold.tick().
        // The world model agent tracks transfer statistics and can
        // adaptively adjust blend factors in future iterations.
        self.transfer_count += 1;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ManifoldConfig;
    use crate::manifold::CausalCrystalManifold;
    use icarus_gpu::pipeline::CpuBackend;

    #[test]
    fn test_world_model_new() {
        let agent = WorldModelAgent::new();
        assert_eq!(agent.transfer_count, 0);
        assert_eq!(agent.name(), "world_model");
        assert_eq!(agent.phases(), &[TickPhase::Post]);
    }

    #[test]
    fn test_transfer_count_increments() {
        let config = ManifoldConfig::e8_only();
        let mut manifold = CausalCrystalManifold::new(config);
        manifold.init_random(42, 0.5);
        let mut backend = CpuBackend;
        let mut agent = WorldModelAgent::new();

        for i in 1..=10 {
            agent.tick(TickPhase::Post, &mut manifold, &mut backend).unwrap();
            assert_eq!(agent.transfer_count, i);
        }
    }
}
