// Copyright (c) 2025-2026 brdigetrlol. All rights reserved.
// SPDX-License-Identifier: LicenseRef-Icarus-Proprietary
// See LICENSE in the repository root for full license terms.

//! Action Agent — Output extraction from the field state
//!
//! Extracts summary statistics from the analytical layer after each tick:
//! mean amplitude, phase distribution, dominant phase, and energy.
//! This is the EMC's primary output interface.

use anyhow::Result;
use icarus_gpu::pipeline::ComputeBackend;
use serde::Serialize;

use crate::agents::{Agent, TickPhase};
use crate::manifold::CausalCrystalManifold;

/// Summary of the EMC's current output state.
#[derive(Debug, Clone, Serialize)]
pub struct ActionOutput {
    /// Current tick
    pub tick: u64,
    /// Mean amplitude across the analytical layer
    pub mean_amplitude: f32,
    /// Phase distribution (histogram of phases in 8 bins, normalized)
    pub phase_histogram: [f32; 8],
    /// Total free energy
    pub energy: f32,
    /// Phase angle of the site with largest amplitude
    pub dominant_phase: f32,
}

/// Action agent: extracts output signals from the manifold.
pub struct ActionAgent {
    /// Most recent output
    pub last_output: Option<ActionOutput>,
}

impl ActionAgent {
    pub fn new() -> Self {
        Self { last_output: None }
    }
}

impl Agent for ActionAgent {
    fn name(&self) -> &str {
        "action"
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
        if let Some(layer) = manifold.layers.first() {
            let n = layer.field.num_sites;
            let params = &layer.solver.params.energy_params;
            let (energy, _, _) =
                icarus_field::free_energy::free_energy(&layer.field, params);

            let mut sum_amp = 0.0f32;
            let mut phase_histogram = [0.0f32; 8];
            let mut max_amp = 0.0f32;
            let mut dominant_phase = 0.0f32;

            for i in 0..n {
                let re = layer.field.values_re[i];
                let im = layer.field.values_im[i];
                let amp = (re * re + im * im).sqrt();
                sum_amp += amp;

                if amp > max_amp {
                    max_amp = amp;
                    dominant_phase = im.atan2(re);
                }

                if amp > 1e-8 {
                    let angle = im.atan2(re);
                    let normalized =
                        (angle + std::f32::consts::PI) / (2.0 * std::f32::consts::PI);
                    let bin = ((normalized * 8.0) as usize).min(7);
                    phase_histogram[bin] += 1.0;
                }
            }

            // Normalize histogram
            let hist_total: f32 = phase_histogram.iter().sum();
            if hist_total > 0.0 {
                for h in &mut phase_histogram {
                    *h /= hist_total;
                }
            }

            self.last_output = Some(ActionOutput {
                tick: manifold.tick,
                mean_amplitude: if n > 0 { sum_amp / n as f32 } else { 0.0 },
                phase_histogram,
                energy,
                dominant_phase,
            });
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
    fn test_action_new() {
        let agent = ActionAgent::new();
        assert!(agent.last_output.is_none());
        assert_eq!(agent.name(), "action");
        assert_eq!(agent.phases(), &[TickPhase::Post]);
    }

    #[test]
    fn test_tick_produces_output() {
        let (mut manifold, mut backend) = setup();
        manifold.tick = 7;
        let mut agent = ActionAgent::new();

        agent.tick(TickPhase::Post, &mut manifold, &mut backend).unwrap();

        let output = agent.last_output.as_ref().unwrap();
        assert_eq!(output.tick, 7);
        assert!(output.mean_amplitude >= 0.0);
        assert!(output.energy.is_finite());
        assert!(output.dominant_phase.is_finite());
    }

    #[test]
    fn test_phase_histogram_normalized() {
        let (mut manifold, mut backend) = setup();
        let mut agent = ActionAgent::new();

        agent.tick(TickPhase::Post, &mut manifold, &mut backend).unwrap();

        let output = agent.last_output.as_ref().unwrap();
        let hist_sum: f32 = output.phase_histogram.iter().sum();
        // Histogram should sum to ~1.0 (normalized) if any sites have amplitude > 1e-8
        if hist_sum > 0.0 {
            assert!((hist_sum - 1.0).abs() < 1e-5, "Histogram sum = {hist_sum}, expected ~1.0");
        }
    }

    #[test]
    fn test_phase_histogram_bins() {
        let (mut manifold, mut backend) = setup();
        let mut agent = ActionAgent::new();

        agent.tick(TickPhase::Post, &mut manifold, &mut backend).unwrap();

        let output = agent.last_output.as_ref().unwrap();
        // All bins should be non-negative
        for &bin in &output.phase_histogram {
            assert!(bin >= 0.0);
        }
    }

    #[test]
    fn test_dominant_phase_in_range() {
        let (mut manifold, mut backend) = setup();
        let mut agent = ActionAgent::new();

        agent.tick(TickPhase::Post, &mut manifold, &mut backend).unwrap();

        let output = agent.last_output.as_ref().unwrap();
        // atan2 returns [-pi, pi]
        assert!(output.dominant_phase >= -std::f32::consts::PI);
        assert!(output.dominant_phase <= std::f32::consts::PI);
    }

    #[test]
    fn test_output_updates_each_tick() {
        let (mut manifold, mut backend) = setup();
        let mut agent = ActionAgent::new();

        manifold.tick = 1;
        agent.tick(TickPhase::Post, &mut manifold, &mut backend).unwrap();
        assert_eq!(agent.last_output.as_ref().unwrap().tick, 1);

        manifold.tick = 2;
        agent.tick(TickPhase::Post, &mut manifold, &mut backend).unwrap();
        assert_eq!(agent.last_output.as_ref().unwrap().tick, 2);
    }
}
