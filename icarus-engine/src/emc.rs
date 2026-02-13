// Copyright (c) 2025-2026 brdigetrlol. All rights reserved.
// SPDX-License-Identifier: LicenseRef-Icarus-Proprietary
// See LICENSE in the repository root for full license terms.

//! EmergentManifoldComputer — Top-level EMC orchestrator
//!
//! The EMC wraps the crystal manifold, cognitive agents, and compute backend
//! into a single coherent system. Each call to `tick()` advances the simulation
//! by one full cycle: agents → dynamics → agents.

use anyhow::{Context, Result};
use icarus_field::autopoiesis::AffectiveState;
use serde::Serialize;
use icarus_gpu::pipeline::ComputeBackend;
use icarus_math::lattice::LatticeLayer;

use crate::agents::AgentOrchestrator;
use crate::config::{BackendSelection, ManifoldConfig};
use crate::manifold::{CausalCrystalManifold, LayerStats};

/// Statistics from the EMC at a point in time.
#[derive(Debug, Clone)]
pub struct EmcStats {
    pub tick: u64,
    pub layer_stats: Vec<LayerStats>,
    pub total_sites: usize,
    pub backend_name: String,
    pub memory_bytes: usize,
    pub affective_state: AffectiveState,
}

/// Snapshot of the EMC's full observable state.
#[derive(Debug, Clone)]
pub struct EmcObservation {
    pub tick: u64,
    pub layer_states: Vec<LayerStateSnapshot>,
    pub affective_state: AffectiveState,
}

/// Snapshot of a single layer's field state.
#[derive(Debug, Clone, Serialize)]
pub struct LayerStateSnapshot {
    pub layer: LatticeLayer,
    pub values_re: Vec<f32>,
    pub values_im: Vec<f32>,
    pub energy: f32,
}

/// The Emergent Manifold Computer.
///
/// Top-level orchestrator that combines the crystal manifold substrate,
/// cognitive agents, and GPU/CPU compute backend into a unified system.
pub struct EmergentManifoldComputer {
    /// The crystal manifold substrate
    pub manifold: CausalCrystalManifold,
    /// Cognitive agent orchestrator
    pub agents: AgentOrchestrator,
    /// Compute backend (GPU or CPU)
    backend: Box<dyn ComputeBackend>,
    /// Total ticks executed
    pub total_ticks: u64,
}

impl EmergentManifoldComputer {
    /// Create a new EMC from configuration.
    ///
    /// Initializes the GPU backend if configured; falls back to CPU on failure.
    pub fn new(config: ManifoldConfig) -> Result<Self> {
        let backend: Box<dyn ComputeBackend> = match config.backend {
            BackendSelection::Gpu { device_id } => {
                Box::new(
                    icarus_gpu::pipeline::GpuBackend::new(device_id)
                        .context("Failed to initialize GPU backend")?,
                )
            }
            BackendSelection::Cpu => Box::new(icarus_gpu::pipeline::CpuBackend),
            BackendSelection::Npu => {
                Box::new(
                    icarus_gpu::npu_backend::NpuBackend::new()
                        .context("Failed to initialize NPU backend")?,
                )
            }
        };

        let agents = AgentOrchestrator::new(&config.agents);
        let manifold = CausalCrystalManifold::new(config);

        Ok(Self {
            manifold,
            agents,
            backend,
            total_ticks: 0,
        })
    }

    /// Create an EMC with CPU backend (no GPU required).
    pub fn new_cpu(config: ManifoldConfig) -> Self {
        let agents = AgentOrchestrator::new(&config.agents);
        let mut cpu_config = config;
        cpu_config.backend = BackendSelection::Cpu;
        let manifold = CausalCrystalManifold::new(cpu_config);

        Self {
            manifold,
            agents,
            backend: Box::new(icarus_gpu::pipeline::CpuBackend),
            total_ticks: 0,
        }
    }

    /// Execute one full EMC tick:
    /// 1. Pre-dynamics agent pass (perception, planning prep)
    /// 2. Manifold dynamics (RAE + metric learning)
    /// 3. Post-dynamics agent pass (action, memory, learning)
    pub fn tick(&mut self) -> Result<()> {
        self.agents
            .pre_tick(&mut self.manifold, self.backend.as_mut())?;

        self.manifold.tick(self.backend.as_mut())?;

        self.agents
            .post_tick(&mut self.manifold, self.backend.as_mut())?;

        self.total_ticks += 1;
        Ok(())
    }

    /// Run N ticks.
    pub fn run(&mut self, num_ticks: u64) -> Result<()> {
        for _ in 0..num_ticks {
            self.tick()?;
        }
        Ok(())
    }

    /// Initialize the manifold with random field values.
    pub fn init_random(&mut self, seed: u64, amplitude: f32) {
        self.manifold.init_random(seed, amplitude);
    }

    /// Inject data into a specific layer at specific sites.
    ///
    /// Each entry in `sites` is (site_index, re, im).
    pub fn inject(
        &mut self,
        layer_idx: usize,
        sites: &[(usize, f32, f32)],
    ) -> Result<()> {
        if layer_idx >= self.manifold.layers.len() {
            anyhow::bail!(
                "Layer index {} out of range (have {} layers)",
                layer_idx,
                self.manifold.layers.len()
            );
        }
        let layer = &mut self.manifold.layers[layer_idx];
        for &(site, re, im) in sites {
            if site < layer.field.num_sites {
                layer.field.set(site, re, im);
            }
        }
        Ok(())
    }

    /// Observe the current state of the EMC.
    pub fn observe(&self) -> EmcObservation {
        let layer_states = self
            .manifold
            .layers
            .iter()
            .map(|layer| {
                let params = &layer.solver.params.energy_params;
                let (energy, _, _) =
                    icarus_field::free_energy::free_energy(&layer.field, params);
                LayerStateSnapshot {
                    layer: layer.layer_type,
                    values_re: layer.field.values_re.clone(),
                    values_im: layer.field.values_im.clone(),
                    energy,
                }
            })
            .collect();

        EmcObservation {
            tick: self.total_ticks,
            layer_states,
            affective_state: self.manifold.affective_state(),
        }
    }

    /// Get current statistics.
    pub fn stats(&self) -> EmcStats {
        EmcStats {
            tick: self.total_ticks,
            layer_stats: self.manifold.stats(),
            total_sites: self.manifold.total_sites(),
            backend_name: self.backend.name().to_string(),
            memory_bytes: self.manifold.memory_bytes(),
            affective_state: self.manifold.affective_state(),
        }
    }

    /// Get the current affective state (valence, arousal, phase coherence, neuromodulators).
    pub fn affective_state(&self) -> AffectiveState {
        self.manifold.affective_state()
    }

    /// Get the compute backend name.
    pub fn backend_name(&self) -> &str {
        self.backend.name()
    }

    /// Encode input data into a specific layer using an encoder.
    pub fn encode(
        &mut self,
        input: &[f32],
        encoder: &dyn crate::encoding::InputEncoder,
        layer_idx: usize,
    ) -> Result<()> {
        if layer_idx >= self.manifold.layers.len() {
            anyhow::bail!(
                "Layer index {} out of range (have {} layers)",
                layer_idx,
                self.manifold.layers.len()
            );
        }
        encoder.encode(input, &mut self.manifold.layers[layer_idx].field);
        Ok(())
    }

    /// Read output from a specific layer using a readout.
    pub fn readout(
        &self,
        readout: &dyn crate::readout::Readout,
        layer_idx: usize,
    ) -> Result<Vec<f32>> {
        if layer_idx >= self.manifold.layers.len() {
            anyhow::bail!(
                "Layer index {} out of range (have {} layers)",
                layer_idx,
                self.manifold.layers.len()
            );
        }
        Ok(readout.read(&self.manifold.layers[layer_idx].field))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ManifoldConfig;

    #[test]
    fn test_emc_creation_cpu() {
        let config = ManifoldConfig::e8_only();
        let emc = EmergentManifoldComputer::new_cpu(config);
        assert_eq!(emc.total_ticks, 0);
        assert_eq!(emc.manifold.layers.len(), 1);
        assert_eq!(emc.backend_name(), "CPU");
    }

    #[test]
    fn test_emc_tick() {
        let config = ManifoldConfig::e8_only();
        let mut emc = EmergentManifoldComputer::new_cpu(config);
        emc.init_random(42, 0.5);

        emc.tick().unwrap();
        assert_eq!(emc.total_ticks, 1);
    }

    #[test]
    fn test_emc_run_multiple() {
        let config = ManifoldConfig::e8_only();
        let mut emc = EmergentManifoldComputer::new_cpu(config);
        emc.init_random(42, 0.5);

        emc.run(10).unwrap();
        assert_eq!(emc.total_ticks, 10);
    }

    #[test]
    fn test_emc_inject() {
        let config = ManifoldConfig::e8_only();
        let mut emc = EmergentManifoldComputer::new_cpu(config);

        emc.inject(0, &[(0, 1.0, 0.5), (1, -0.3, 0.7)]).unwrap();

        let (re, im) = emc.manifold.layers[0].field.get(0);
        assert!((re - 1.0).abs() < 1e-6);
        assert!((im - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_emc_inject_out_of_range() {
        let config = ManifoldConfig::e8_only();
        let mut emc = EmergentManifoldComputer::new_cpu(config);
        assert!(emc.inject(5, &[(0, 1.0, 0.0)]).is_err());
    }

    #[test]
    fn test_emc_observe() {
        let config = ManifoldConfig::e8_only();
        let mut emc = EmergentManifoldComputer::new_cpu(config);
        emc.init_random(42, 0.5);

        let obs = emc.observe();
        assert_eq!(obs.layer_states.len(), 1);
        assert_eq!(obs.layer_states[0].values_re.len(), 241);
        assert_eq!(obs.layer_states[0].layer, LatticeLayer::Analytical);
    }

    #[test]
    fn test_emc_stats() {
        let config = ManifoldConfig::e8_only();
        let mut emc = EmergentManifoldComputer::new_cpu(config);
        emc.init_random(42, 0.5);

        let stats = emc.stats();
        assert_eq!(stats.total_sites, 241);
        assert_eq!(stats.backend_name, "CPU");
        assert!(stats.memory_bytes > 0);
    }

    #[test]
    fn test_emc_energy_decreases_with_damping() {
        let mut config = ManifoldConfig::e8_only();
        config.layers[0].omega = 0.0;
        config.layers[0].gamma = 0.5;
        config.layers[0].rae_steps_per_tick = 500;

        let mut emc = EmergentManifoldComputer::new_cpu(config);
        emc.init_random(42, 0.8);

        let e0 = emc.stats().layer_stats[0].total_energy;
        emc.tick().unwrap();
        let e1 = emc.stats().layer_stats[0].total_energy;
        emc.tick().unwrap();
        let e2 = emc.stats().layer_stats[0].total_energy;

        assert!(
            e1 <= e0 + 0.1,
            "Energy should decrease: {} -> {}",
            e0, e1
        );
        assert!(
            e2 <= e1 + 0.1,
            "Energy should decrease: {} -> {}",
            e1, e2
        );
    }

    #[test]
    fn test_emc_deterministic() {
        let config = ManifoldConfig::e8_only();

        let mut emc1 = EmergentManifoldComputer::new_cpu(config.clone());
        emc1.init_random(42, 0.5);
        emc1.run(5).unwrap();

        let mut emc2 = EmergentManifoldComputer::new_cpu(config);
        emc2.init_random(42, 0.5);
        emc2.run(5).unwrap();

        let obs1 = emc1.observe();
        let obs2 = emc2.observe();

        for i in 0..obs1.layer_states[0].values_re.len() {
            let diff = (obs1.layer_states[0].values_re[i]
                - obs2.layer_states[0].values_re[i])
                .abs();
            assert!(diff < 1e-5, "Mismatch at site {}: diff={}", i, diff);
        }
    }

    #[test]
    fn test_emc_multi_layer_creation() {
        let config = ManifoldConfig::multi_layer();
        let emc = EmergentManifoldComputer::new_cpu(config);
        let stats = emc.stats();
        assert_eq!(stats.total_sites, 241 + 1105 + 241);
        assert_eq!(stats.layer_stats.len(), 3);
        assert_eq!(stats.backend_name, "CPU");
    }

    #[test]
    fn test_emc_multi_layer_tick() {
        let config = ManifoldConfig::multi_layer();
        let mut emc = EmergentManifoldComputer::new_cpu(config);
        emc.init_random(42, 0.5);

        emc.tick().unwrap();
        assert_eq!(emc.total_ticks, 1);

        emc.run(4).unwrap();
        assert_eq!(emc.total_ticks, 5);
    }

    #[test]
    fn test_emc_multi_layer_observe() {
        let config = ManifoldConfig::multi_layer();
        let mut emc = EmergentManifoldComputer::new_cpu(config);
        emc.init_random(42, 0.5);

        let obs = emc.observe();
        assert_eq!(obs.layer_states.len(), 3);
        assert_eq!(obs.layer_states[0].values_re.len(), 241);
        assert_eq!(obs.layer_states[0].layer, LatticeLayer::Analytical);
        assert_eq!(obs.layer_states[1].values_re.len(), 1105);
        assert_eq!(obs.layer_states[1].layer, LatticeLayer::Creative);
        assert_eq!(obs.layer_states[2].values_re.len(), 241);
        assert_eq!(obs.layer_states[2].layer, LatticeLayer::Associative);
    }

    #[test]
    fn test_emc_multi_layer_stats() {
        let config = ManifoldConfig::multi_layer();
        let mut emc = EmergentManifoldComputer::new_cpu(config);
        emc.init_random(42, 0.5);
        emc.tick().unwrap();

        let stats = emc.stats();
        assert_eq!(stats.layer_stats.len(), 3);
        assert_eq!(stats.layer_stats[0].num_sites, 241);
        assert_eq!(stats.layer_stats[1].num_sites, 1105);
        assert_eq!(stats.layer_stats[2].num_sites, 241);
        assert!(stats.memory_bytes > 0);
    }

    #[test]
    fn test_emc_affective_state() {
        let config = ManifoldConfig::e8_only();
        let mut emc = EmergentManifoldComputer::new_cpu(config);
        emc.init_random(42, 0.5);

        // Before ticking, affective state should be at baseline
        let state = emc.affective_state();
        assert!((state.valence).abs() < 1e-6);
        assert!((state.aether.dopamine).abs() < 1e-6);

        // After ticking, affective controller should have updated
        emc.tick().unwrap();
        let state = emc.affective_state();
        assert!(state.phase_coherence >= 0.0 && state.phase_coherence <= 1.0 + 1e-6);
    }

    #[test]
    fn test_emc_stats_includes_affective() {
        let config = ManifoldConfig::e8_only();
        let mut emc = EmergentManifoldComputer::new_cpu(config);
        emc.init_random(42, 0.5);
        emc.tick().unwrap();

        let stats = emc.stats();
        assert!(stats.affective_state.phase_coherence >= 0.0);
    }

    #[test]
    fn test_emc_observe_includes_affective() {
        let config = ManifoldConfig::e8_only();
        let mut emc = EmergentManifoldComputer::new_cpu(config);
        emc.init_random(42, 0.5);
        emc.tick().unwrap();

        let obs = emc.observe();
        assert!(obs.affective_state.phase_coherence >= 0.0);
    }
}
