// Copyright (c) 2025-2026 brdigetrlol. All rights reserved.
// SPDX-License-Identifier: LicenseRef-Icarus-Proprietary
// See LICENSE in the repository root for full license terms.

//! Perception Agent — External data injection into the manifold
//!
//! The perception agent is the EMC's interface to the outside world.
//! External data is queued as PerceptionInputs and injected into the
//! appropriate lattice layer during the pre-dynamics tick phase.

use anyhow::Result;
use icarus_gpu::pipeline::ComputeBackend;
use icarus_math::lattice::LatticeLayer;

use crate::agents::{Agent, TickPhase};
use crate::manifold::CausalCrystalManifold;

/// A batch of data to inject into the manifold.
#[derive(Debug, Clone)]
pub struct PerceptionInput {
    /// Target layer for injection
    pub layer: LatticeLayer,
    /// Site injections: (site_index, re, im)
    pub injections: Vec<(usize, f32, f32)>,
    /// Injection strength (0.0 = no effect, 1.0 = full replacement)
    pub strength: f32,
}

/// Perception agent: queues and injects external data into the manifold.
pub struct PerceptionAgent {
    input_queue: Vec<PerceptionInput>,
}

impl PerceptionAgent {
    pub fn new() -> Self {
        Self {
            input_queue: Vec::new(),
        }
    }

    /// Queue an input for injection on the next tick.
    pub fn queue_input(&mut self, input: PerceptionInput) {
        self.input_queue.push(input);
    }

    /// Number of pending inputs.
    pub fn pending_count(&self) -> usize {
        self.input_queue.len()
    }
}

impl Agent for PerceptionAgent {
    fn name(&self) -> &str {
        "perception"
    }

    fn phases(&self) -> &[TickPhase] {
        &[TickPhase::Pre]
    }

    fn tick(
        &mut self,
        _phase: TickPhase,
        manifold: &mut CausalCrystalManifold,
        _backend: &mut dyn ComputeBackend,
    ) -> Result<()> {
        let inputs = std::mem::take(&mut self.input_queue);

        for input in inputs {
            for layer in &mut manifold.layers {
                if layer.layer_type == input.layer {
                    for &(site, re, im) in &input.injections {
                        if site < layer.field.num_sites {
                            let (old_re, old_im) = layer.field.get(site);
                            let s = input.strength;
                            layer.field.set(
                                site,
                                old_re * (1.0 - s) + re * s,
                                old_im * (1.0 - s) + im * s,
                            );
                        }
                    }
                    break;
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

    fn setup() -> (CausalCrystalManifold, CpuBackend) {
        let config = ManifoldConfig::e8_only();
        let mut manifold = CausalCrystalManifold::new(config);
        manifold.init_random(42, 0.5);
        (manifold, CpuBackend)
    }

    #[test]
    fn test_perception_new() {
        let agent = PerceptionAgent::new();
        assert_eq!(agent.pending_count(), 0);
        assert_eq!(agent.name(), "perception");
        assert_eq!(agent.phases(), &[TickPhase::Pre]);
    }

    #[test]
    fn test_queue_and_pending_count() {
        let mut agent = PerceptionAgent::new();
        assert_eq!(agent.pending_count(), 0);

        agent.queue_input(PerceptionInput {
            layer: LatticeLayer::Analytical,
            injections: vec![(0, 1.0, 0.0)],
            strength: 1.0,
        });
        assert_eq!(agent.pending_count(), 1);

        agent.queue_input(PerceptionInput {
            layer: LatticeLayer::Analytical,
            injections: vec![(1, 0.0, 1.0)],
            strength: 0.5,
        });
        assert_eq!(agent.pending_count(), 2);
    }

    #[test]
    fn test_tick_drains_queue() {
        let (mut manifold, mut backend) = setup();
        let mut agent = PerceptionAgent::new();

        agent.queue_input(PerceptionInput {
            layer: LatticeLayer::Analytical,
            injections: vec![(0, 1.0, 0.0)],
            strength: 1.0,
        });
        assert_eq!(agent.pending_count(), 1);

        agent.tick(TickPhase::Pre, &mut manifold, &mut backend).unwrap();
        assert_eq!(agent.pending_count(), 0);
    }

    #[test]
    fn test_full_strength_injection() {
        let (mut manifold, mut backend) = setup();
        let mut agent = PerceptionAgent::new();

        agent.queue_input(PerceptionInput {
            layer: LatticeLayer::Analytical,
            injections: vec![(0, 3.0, 4.0)],
            strength: 1.0, // full replacement
        });

        agent.tick(TickPhase::Pre, &mut manifold, &mut backend).unwrap();

        let (re, im) = manifold.layers[0].field.get(0);
        assert!((re - 3.0).abs() < 1e-6);
        assert!((im - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_zero_strength_no_change() {
        let (mut manifold, mut backend) = setup();
        let (orig_re, orig_im) = manifold.layers[0].field.get(0);

        let mut agent = PerceptionAgent::new();
        agent.queue_input(PerceptionInput {
            layer: LatticeLayer::Analytical,
            injections: vec![(0, 99.0, 99.0)],
            strength: 0.0, // no effect
        });

        agent.tick(TickPhase::Pre, &mut manifold, &mut backend).unwrap();

        let (re, im) = manifold.layers[0].field.get(0);
        assert!((re - orig_re).abs() < 1e-6);
        assert!((im - orig_im).abs() < 1e-6);
    }

    #[test]
    fn test_half_strength_interpolation() {
        let (mut manifold, mut backend) = setup();
        let (orig_re, orig_im) = manifold.layers[0].field.get(0);

        let mut agent = PerceptionAgent::new();
        agent.queue_input(PerceptionInput {
            layer: LatticeLayer::Analytical,
            injections: vec![(0, 2.0, 4.0)],
            strength: 0.5,
        });

        agent.tick(TickPhase::Pre, &mut manifold, &mut backend).unwrap();

        let (re, im) = manifold.layers[0].field.get(0);
        let expected_re = orig_re * 0.5 + 2.0 * 0.5;
        let expected_im = orig_im * 0.5 + 4.0 * 0.5;
        assert!((re - expected_re).abs() < 1e-6);
        assert!((im - expected_im).abs() < 1e-6);
    }

    #[test]
    fn test_out_of_bounds_site_ignored() {
        let (mut manifold, mut backend) = setup();
        let num_sites = manifold.layers[0].field.num_sites;

        let mut agent = PerceptionAgent::new();
        agent.queue_input(PerceptionInput {
            layer: LatticeLayer::Analytical,
            injections: vec![(num_sites + 100, 1.0, 1.0)],
            strength: 1.0,
        });

        // Should not panic
        agent.tick(TickPhase::Pre, &mut manifold, &mut backend).unwrap();
    }

    #[test]
    fn test_multiple_sites_injection() {
        let (mut manifold, mut backend) = setup();
        let mut agent = PerceptionAgent::new();

        agent.queue_input(PerceptionInput {
            layer: LatticeLayer::Analytical,
            injections: vec![(0, 1.0, 0.0), (1, 0.0, 1.0), (2, -1.0, -1.0)],
            strength: 1.0,
        });

        agent.tick(TickPhase::Pre, &mut manifold, &mut backend).unwrap();

        let (re0, im0) = manifold.layers[0].field.get(0);
        let (re1, im1) = manifold.layers[0].field.get(1);
        let (re2, im2) = manifold.layers[0].field.get(2);

        assert!((re0 - 1.0).abs() < 1e-6);
        assert!((im0 - 0.0).abs() < 1e-6);
        assert!((re1 - 0.0).abs() < 1e-6);
        assert!((im1 - 1.0).abs() < 1e-6);
        assert!((re2 - (-1.0)).abs() < 1e-6);
        assert!((im2 - (-1.0)).abs() < 1e-6);
    }
}
