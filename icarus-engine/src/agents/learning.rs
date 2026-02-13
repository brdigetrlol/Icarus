//! Learning Agent — Metric tensor evolution coordination
//!
//! Provides supplementary metric tensor management beyond the primary
//! geometrodynamic learning handled by manifold.tick(). Monitors metric
//! health and applies emergency regularization when needed.

use anyhow::Result;
use icarus_gpu::pipeline::ComputeBackend;

use crate::agents::{Agent, TickPhase};
use crate::manifold::CausalCrystalManifold;

/// Learning agent: coordinates metric tensor evolution.
pub struct LearningAgent {
    /// Run metric health checks every N ticks
    pub check_interval: u64,
    /// Internal tick counter
    tick_counter: u64,
}

impl LearningAgent {
    pub fn new() -> Self {
        Self {
            check_interval: 5,
            tick_counter: 0,
        }
    }
}

impl Agent for LearningAgent {
    fn name(&self) -> &str {
        "learning"
    }

    fn phases(&self) -> &[TickPhase] {
        &[TickPhase::Post]
    }

    fn tick(
        &mut self,
        _phase: TickPhase,
        manifold: &mut CausalCrystalManifold,
        _backend: &mut dyn ComputeBackend,
    ) -> Result<()> {
        self.tick_counter += 1;

        if self.tick_counter % self.check_interval != 0 {
            return Ok(());
        }

        // Monitor metric health on layers with metric learning enabled.
        // If any diagonal element deviates too far from identity,
        // apply eigenvalue pinning to prevent instability.
        for layer in &mut manifold.layers {
            if !layer.enable_metric_learning {
                continue;
            }

            let packed_size = layer.dim * (layer.dim + 1) / 2;
            let mut max_deviation = 0.0f32;

            for site in 0..layer.field.num_sites {
                for mu in 0..layer.dim {
                    let diag_idx = mu * (2 * layer.dim - mu - 1) / 2 + mu;
                    let idx = site * packed_size + diag_idx;
                    if idx < layer.metric.data.len() {
                        let val = layer.metric.data[idx];
                        max_deviation = max_deviation.max((val - 1.0).abs());
                    }
                }
            }

            // Emergency regularization if metric has deviated significantly
            if max_deviation > 10.0 {
                let floor = layer.geo_learner.params.eigenvalue_floor;
                for site in 0..layer.field.num_sites {
                    let mut site_metric = layer.metric.get_site(site);
                    site_metric.pin_eigenvalues(floor);
                    layer.metric.set_site(site, &site_metric);
                }
            }
        }

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
    fn test_learning_new() {
        let agent = LearningAgent::new();
        assert_eq!(agent.check_interval, 5);
        assert_eq!(agent.name(), "learning");
        assert_eq!(agent.phases(), &[TickPhase::Post]);
    }

    #[test]
    fn test_interval_skipping() {
        let mut config = ManifoldConfig::e8_only();
        config.layers[0].enable_metric_learning = true;
        let mut manifold = CausalCrystalManifold::new(config);
        manifold.init_random(42, 0.5);
        let mut backend = CpuBackend;
        let mut agent = LearningAgent::new();
        agent.check_interval = 3;

        // Ticks 1, 2 should skip (no metric check)
        agent.tick(TickPhase::Post, &mut manifold, &mut backend).unwrap();
        agent.tick(TickPhase::Post, &mut manifold, &mut backend).unwrap();
        // Tick 3 triggers check — should not panic on healthy metric
        agent.tick(TickPhase::Post, &mut manifold, &mut backend).unwrap();
    }

    #[test]
    fn test_healthy_metric_no_pinning() {
        // With default e8_only config (metric_learning=false), learning agent
        // skips metric check entirely. Enable it to test the check path.
        let mut config = ManifoldConfig::e8_only();
        config.layers[0].enable_metric_learning = true;
        let mut manifold = CausalCrystalManifold::new(config);
        manifold.init_random(42, 0.5);
        let mut backend = CpuBackend;
        let mut agent = LearningAgent::new();
        agent.check_interval = 1; // Check every tick

        // Fresh metric should be identity-like, no emergency pinning needed
        agent.tick(TickPhase::Post, &mut manifold, &mut backend).unwrap();

        // Verify metric diagonals are still ~1.0 (not pinned to floor)
        let site_metric = manifold.layers[0].metric.get_site(0);
        let diag_0 = site_metric.components[0]; // g_00
        assert!((diag_0 - 1.0).abs() < 0.1, "Metric diagonal should be ~1.0, got {diag_0}");
    }

    #[test]
    fn test_metric_learning_disabled_skips() {
        // When metric learning is disabled, learning agent should be a no-op
        let config = ManifoldConfig::e8_only(); // metric_learning = false by default
        let mut manifold = CausalCrystalManifold::new(config);
        manifold.init_random(42, 0.5);
        let mut backend = CpuBackend;
        let mut agent = LearningAgent::new();
        agent.check_interval = 1;

        // Should complete without error even with many ticks
        for _ in 0..20 {
            agent.tick(TickPhase::Post, &mut manifold, &mut backend).unwrap();
        }
    }
}
