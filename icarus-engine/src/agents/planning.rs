//! Planning Agent — Energy landscape monitoring, convergence analysis, and geodesic tracking
//!
//! The planning agent tracks the free energy trajectory across ticks,
//! classifying the system's behavior as converging, stable, or diverging.
//! It also monitors dominant attractor sites per layer and detects attractor
//! transitions using geodesic distances on the curved manifold geometry.

use anyhow::Result;
use icarus_field::geodesic;
use serde::Serialize;
use icarus_gpu::pipeline::ComputeBackend;

use crate::agents::{Agent, TickPhase};
use crate::manifold::CausalCrystalManifold;

/// Convergence trend classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ConvergenceTrend {
    /// Energy is decreasing
    Converging,
    /// Energy is roughly stable
    Stable,
    /// Energy is increasing
    Diverging,
    /// Not enough data to determine
    Unknown,
}

/// Per-layer geodesic state tracked by the planning agent.
#[derive(Debug, Clone)]
pub struct LayerGeodesicState {
    /// Current dominant attractor site (max |z|)
    pub attractor_site: usize,
    /// Amplitude at the attractor site
    pub attractor_amplitude: f32,
    /// Geodesic distance of the last attractor transition (0 if no transition)
    pub transition_distance: f32,
    /// Whether an attractor transition occurred this tick
    pub transitioned: bool,
}

/// Planning agent: monitors energy landscape, convergence trends, and geodesic geometry.
pub struct PlanningAgent {
    /// Recent energy values for trend analysis
    energy_history: Vec<f32>,
    /// Maximum history length
    history_capacity: usize,
    /// Current convergence assessment
    pub trend: ConvergenceTrend,
    /// Previous tick's dominant attractor site per layer (None = first tick)
    prev_attractor_sites: Vec<Option<usize>>,
    /// Geodesic state per layer (populated after first tick)
    pub layer_geodesic: Vec<LayerGeodesicState>,
}

impl PlanningAgent {
    pub fn new() -> Self {
        Self {
            energy_history: Vec::new(),
            history_capacity: 100,
            trend: ConvergenceTrend::Unknown,
            prev_attractor_sites: Vec::new(),
            layer_geodesic: Vec::new(),
        }
    }

    /// Analyze the energy trend from recent history.
    fn analyze_trend(&mut self) {
        if self.energy_history.len() < 3 {
            self.trend = ConvergenceTrend::Unknown;
            return;
        }

        let n = self.energy_history.len();
        let recent = &self.energy_history[n.saturating_sub(10)..];

        if recent.len() < 2 {
            self.trend = ConvergenceTrend::Unknown;
            return;
        }

        let mid = recent.len() / 2;
        let first_avg: f32 = recent[..mid].iter().sum::<f32>() / mid as f32;
        let second_avg: f32 =
            recent[mid..].iter().sum::<f32>() / (recent.len() - mid) as f32;

        let delta = second_avg - first_avg;
        let threshold = first_avg.abs() * 0.01;

        if delta < -threshold {
            self.trend = ConvergenceTrend::Converging;
        } else if delta > threshold {
            self.trend = ConvergenceTrend::Diverging;
        } else {
            self.trend = ConvergenceTrend::Stable;
        }
    }

    /// Get the most recently recorded energy.
    pub fn current_energy(&self) -> Option<f32> {
        self.energy_history.last().copied()
    }

    /// Get the full energy history.
    pub fn energy_history(&self) -> &[f32] {
        &self.energy_history
    }

    /// Whether any layer experienced an attractor transition this tick.
    pub fn any_transition(&self) -> bool {
        self.layer_geodesic.iter().any(|s| s.transitioned)
    }

    /// Total geodesic transition distance across all layers this tick.
    pub fn total_transition_distance(&self) -> f32 {
        self.layer_geodesic.iter().map(|s| s.transition_distance).sum()
    }
}

impl Agent for PlanningAgent {
    fn name(&self) -> &str {
        "planning"
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
        // --- Energy tracking (existing) ---
        let total_energy: f32 = manifold
            .layers
            .iter()
            .map(|layer| {
                let params = &layer.solver.params.energy_params;
                let (e, _, _) =
                    icarus_field::free_energy::free_energy(&layer.field, params);
                e
            })
            .sum();

        self.energy_history.push(total_energy);
        if self.energy_history.len() > self.history_capacity {
            self.energy_history.remove(0);
        }

        self.analyze_trend();

        // --- Geodesic attractor tracking ---
        let num_layers = manifold.layers.len();

        // Initialize prev_attractor_sites on first tick
        if self.prev_attractor_sites.len() != num_layers {
            self.prev_attractor_sites = vec![None; num_layers];
        }

        self.layer_geodesic.clear();

        for (i, layer) in manifold.layers.iter().enumerate() {
            let (attractor_site, attractor_amplitude) =
                geodesic::find_max_amplitude_site(&layer.field);

            let (transitioned, transition_distance) = match self.prev_attractor_sites[i] {
                Some(prev_site) if prev_site != attractor_site => {
                    // Attractor moved — compute geodesic distance of the transition
                    let dist = geodesic::geodesic_distance(
                        &layer.field,
                        &layer.metric,
                        prev_site,
                        attractor_site,
                    )
                    .unwrap_or(f32::INFINITY);
                    (true, dist)
                }
                _ => (false, 0.0),
            };

            self.prev_attractor_sites[i] = Some(attractor_site);

            self.layer_geodesic.push(LayerGeodesicState {
                attractor_site,
                attractor_amplitude,
                transition_distance,
                transitioned,
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
    fn test_planning_new() {
        let agent = PlanningAgent::new();
        assert_eq!(agent.trend, ConvergenceTrend::Unknown);
        assert!(agent.energy_history().is_empty());
        assert!(agent.current_energy().is_none());
        assert!(!agent.any_transition());
        assert_eq!(agent.total_transition_distance(), 0.0);
        assert_eq!(agent.name(), "planning");
        assert_eq!(agent.phases(), &[TickPhase::Post]);
    }

    #[test]
    fn test_energy_tracking() {
        let (mut manifold, mut backend) = setup();
        let mut agent = PlanningAgent::new();

        agent.tick(TickPhase::Post, &mut manifold, &mut backend).unwrap();

        assert_eq!(agent.energy_history().len(), 1);
        let energy = agent.current_energy().unwrap();
        assert!(energy.is_finite());
    }

    #[test]
    fn test_trend_unknown_with_few_samples() {
        let (mut manifold, mut backend) = setup();
        let mut agent = PlanningAgent::new();

        // 1-2 samples: trend should be Unknown
        agent.tick(TickPhase::Post, &mut manifold, &mut backend).unwrap();
        assert_eq!(agent.trend, ConvergenceTrend::Unknown);

        agent.tick(TickPhase::Post, &mut manifold, &mut backend).unwrap();
        assert_eq!(agent.trend, ConvergenceTrend::Unknown);
    }

    #[test]
    fn test_trend_classification_converging() {
        let mut agent = PlanningAgent::new();
        // Inject decreasing energy history directly
        agent.energy_history = vec![10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        agent.analyze_trend();
        assert_eq!(agent.trend, ConvergenceTrend::Converging);
    }

    #[test]
    fn test_trend_classification_diverging() {
        let mut agent = PlanningAgent::new();
        agent.energy_history = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        agent.analyze_trend();
        assert_eq!(agent.trend, ConvergenceTrend::Diverging);
    }

    #[test]
    fn test_trend_classification_stable() {
        let mut agent = PlanningAgent::new();
        agent.energy_history = vec![5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0];
        agent.analyze_trend();
        assert_eq!(agent.trend, ConvergenceTrend::Stable);
    }

    #[test]
    fn test_geodesic_state_populated() {
        let (mut manifold, mut backend) = setup();
        let mut agent = PlanningAgent::new();

        agent.tick(TickPhase::Post, &mut manifold, &mut backend).unwrap();

        // Should have one geodesic state per layer (e8_only = 1 layer)
        assert_eq!(agent.layer_geodesic.len(), 1);
        let geo = &agent.layer_geodesic[0];
        assert!(geo.attractor_amplitude >= 0.0);
        // First tick: no transition (no previous attractor)
        assert!(!geo.transitioned);
        assert_eq!(geo.transition_distance, 0.0);
    }

    #[test]
    fn test_no_transition_when_attractor_stable() {
        let (mut manifold, mut backend) = setup();
        let mut agent = PlanningAgent::new();

        // Two ticks with same field state — attractor shouldn't move
        agent.tick(TickPhase::Post, &mut manifold, &mut backend).unwrap();
        agent.tick(TickPhase::Post, &mut manifold, &mut backend).unwrap();

        assert!(!agent.any_transition());
    }

    #[test]
    fn test_energy_history_capacity() {
        let (mut manifold, mut backend) = setup();
        let mut agent = PlanningAgent::new();

        // Run 150 ticks — history should cap at 100
        for i in 0..150 {
            manifold.tick = i;
            agent.tick(TickPhase::Post, &mut manifold, &mut backend).unwrap();
        }

        assert!(agent.energy_history().len() <= 100);
    }
}
