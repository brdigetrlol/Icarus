//! Runtime configuration for the Icarus EMC
//!
//! Defines configuration structures for the multi-layer manifold,
//! compute backend selection, and agent orchestration.

use icarus_field::rae::IntegratorMethod;
use icarus_field::topology::TopologyParams;
use icarus_math::lattice::LatticeLayer;
use serde::{Deserialize, Serialize};

/// Configuration for a single lattice layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerConfig {
    /// Which layer of the hierarchy
    pub layer: LatticeLayer,
    /// Lattice dimension (8 for E8, 24 for Leech, configurable for HCP/Hypercubic)
    pub dimension: usize,
    /// Number of RAE integration steps per EMC tick
    pub rae_steps_per_tick: u64,
    /// RAE time step (for Euler: must satisfy CFL dt < 2/K; for SemiImplicit: unrestricted)
    pub dt: f32,
    /// Resonance frequency omega
    pub omega: f32,
    /// Damping coefficient gamma (must be >= 0)
    pub gamma: f32,
    /// Kinetic (gradient) energy weight
    pub kinetic_weight: f32,
    /// Potential energy weight
    pub potential_weight: f32,
    /// Target amplitude for double-well potential
    pub target_amplitude: f32,
    /// Whether to evolve the metric tensor on this layer
    pub enable_metric_learning: bool,
    /// Whether to use adaptive timestep (energy-monitoring CFL controller)
    pub enable_adaptive_dt: bool,
    /// Integration method (Euler requires CFL: dt < 2/K; SemiImplicit is unconditionally stable)
    pub method: IntegratorMethod,
    /// Whether to enable edge-of-chaos criticality control (auto-tunes γ via Lyapunov exponent)
    pub enable_criticality_control: bool,
    /// Target Lyapunov exponent (0.0 = edge of chaos, negative = more ordered)
    pub criticality_target: f32,
    /// Minimum γ allowed by criticality controller
    pub criticality_gamma_min: f32,
    /// Maximum γ allowed by criticality controller
    pub criticality_gamma_max: f32,
    /// Topological regularization parameters (None = disabled)
    pub topology: Option<TopologyParams>,
}

impl LayerConfig {
    /// Compute the CFL stability limit for this layer's lattice type.
    ///
    /// Returns `2/K` where K is the kissing number (max neighbors).
    pub fn cfl_limit(&self) -> f32 {
        let kissing = match self.layer {
            LatticeLayer::Analytical => 240,
            LatticeLayer::Creative => 1104,
            LatticeLayer::Associative => self.dimension * (self.dimension - 1),
            LatticeLayer::Sensory => 2 * self.dimension,
        };
        if kissing > 0 {
            2.0 / kissing as f32
        } else {
            1.0
        }
    }
}

/// Compute backend selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackendSelection {
    /// CPU reference implementation
    Cpu,
    /// CUDA GPU acceleration
    Gpu { device_id: usize },
    /// Intel NPU via TCP bridge to Windows
    Npu,
}

/// Configuration for the cognitive agent subsystem
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// Enable the perception agent (external data injection)
    pub enable_perception: bool,
    /// Enable the world model agent (inter-layer transfer coordination)
    pub enable_world_model: bool,
    /// Enable the planning agent (energy landscape monitoring)
    pub enable_planning: bool,
    /// Enable the memory agent (field state snapshots)
    pub enable_memory: bool,
    /// Enable the action agent (output extraction)
    pub enable_action: bool,
    /// Enable the learning agent (metric evolution coordination)
    pub enable_learning: bool,
    /// Maximum number of field snapshots to retain
    pub memory_capacity: usize,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            enable_perception: true,
            enable_world_model: false,
            enable_planning: true,
            enable_memory: true,
            enable_action: true,
            enable_learning: false,
            memory_capacity: 64,
        }
    }
}

/// Full configuration for the Emergent Manifold Computer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifoldConfig {
    /// Per-layer configurations (ordered analytical → sensory)
    pub layers: Vec<LayerConfig>,
    /// Which compute backend to use
    pub backend: BackendSelection,
    /// Maximum VRAM budget in bytes (GPU backend only)
    pub vram_budget_bytes: usize,
    /// Enable inter-layer transfer operators
    pub enable_inter_layer_transfer: bool,
    /// Learning rate for transfer operator weight updates
    pub transfer_learning_rate: f32,
    /// Agent subsystem configuration
    pub agents: AgentConfig,
}

impl ManifoldConfig {
    /// Minimal E8-only configuration for MVP validation.
    ///
    /// Single analytical layer, flat metric, no inter-layer transfer.
    /// Suitable for attractor convergence testing.
    pub fn e8_only() -> Self {
        Self {
            layers: vec![LayerConfig {
                layer: LatticeLayer::Analytical,
                dimension: 8,
                rae_steps_per_tick: 100,
                dt: 0.002,
                omega: 1.0,
                gamma: 0.1,
                kinetic_weight: 0.5,
                potential_weight: 1.0,
                target_amplitude: 1.0,
                enable_metric_learning: false,
                enable_adaptive_dt: true,
                method: IntegratorMethod::SemiImplicit,
                enable_criticality_control: false,
                criticality_target: 0.0,
                criticality_gamma_min: 0.001,
                criticality_gamma_max: 1.0,
                topology: None,
            }],
            backend: BackendSelection::Gpu { device_id: 0 },
            vram_budget_bytes: 256 * 1024 * 1024,
            enable_inter_layer_transfer: false,
            transfer_learning_rate: 0.001,
            agents: AgentConfig::default(),
        }
    }

    /// Full four-layer hierarchy configuration.
    pub fn full_hierarchy() -> Self {
        Self {
            layers: vec![
                LayerConfig {
                    layer: LatticeLayer::Analytical,
                    dimension: 8,
                    rae_steps_per_tick: 100,
                    dt: 0.002,
                    omega: 1.0,
                    gamma: 0.1,
                    kinetic_weight: 0.5,
                    potential_weight: 1.0,
                    target_amplitude: 1.0,
                    enable_metric_learning: true,
                    enable_adaptive_dt: true,
                    method: IntegratorMethod::SemiImplicit,
                    enable_criticality_control: false,
                    criticality_target: 0.0,
                    criticality_gamma_min: 0.001,
                    criticality_gamma_max: 1.0,
                    topology: None,
                },
                LayerConfig {
                    layer: LatticeLayer::Creative,
                    dimension: 24,
                    rae_steps_per_tick: 50,
                    dt: 0.00001,
                    omega: 0.5,
                    gamma: 0.1,
                    kinetic_weight: 0.5,
                    potential_weight: 1.0,
                    target_amplitude: 1.0,
                    enable_metric_learning: true,
                    enable_adaptive_dt: true,
                    method: IntegratorMethod::SemiImplicit,
                    enable_criticality_control: false,
                    criticality_target: 0.0,
                    criticality_gamma_min: 0.001,
                    criticality_gamma_max: 1.0,
                    topology: None,
                },
                LayerConfig {
                    layer: LatticeLayer::Associative,
                    dimension: 64,
                    rae_steps_per_tick: 25,
                    dt: 0.004,
                    omega: 0.3,
                    gamma: 0.1,
                    kinetic_weight: 0.5,
                    potential_weight: 1.0,
                    target_amplitude: 1.0,
                    enable_metric_learning: false,
                    enable_adaptive_dt: true,
                    method: IntegratorMethod::SemiImplicit,
                    enable_criticality_control: false,
                    criticality_target: 0.0,
                    criticality_gamma_min: 0.001,
                    criticality_gamma_max: 1.0,
                    topology: None,
                },
                LayerConfig {
                    layer: LatticeLayer::Sensory,
                    dimension: 32,
                    rae_steps_per_tick: 10,
                    dt: 0.005,
                    omega: 0.1,
                    gamma: 0.2,
                    kinetic_weight: 0.5,
                    potential_weight: 1.0,
                    target_amplitude: 1.0,
                    enable_metric_learning: false,
                    enable_adaptive_dt: true,
                    method: IntegratorMethod::SemiImplicit,
                    enable_criticality_control: false,
                    criticality_target: 0.0,
                    criticality_gamma_min: 0.001,
                    criticality_gamma_max: 1.0,
                    topology: None,
                },
            ],
            backend: BackendSelection::Gpu { device_id: 0 },
            vram_budget_bytes: 1024 * 1024 * 1024,
            enable_inter_layer_transfer: true,
            transfer_learning_rate: 0.001,
            agents: AgentConfig {
                enable_perception: true,
                enable_world_model: true,
                enable_planning: true,
                enable_memory: true,
                enable_action: true,
                enable_learning: true,
                memory_capacity: 128,
            },
        }
    }

    /// Practical three-layer hierarchy for testing and moderate workloads.
    ///
    /// E8 (8D, 241 sites) → Leech/D24 (24D, 1105 sites) → HCP (16D, 33 sites)
    /// Inter-layer transfer enabled, metric learning on E8 only.
    /// Total memory ~1.5MB — runs comfortably on CPU.
    pub fn multi_layer() -> Self {
        Self {
            layers: vec![
                LayerConfig {
                    layer: LatticeLayer::Analytical,
                    dimension: 8,
                    rae_steps_per_tick: 50,
                    dt: 0.002,
                    omega: 1.0,
                    gamma: 0.1,
                    kinetic_weight: 0.5,
                    potential_weight: 1.0,
                    target_amplitude: 1.0,
                    enable_metric_learning: true,
                    enable_adaptive_dt: true,
                    method: IntegratorMethod::SemiImplicit,
                    enable_criticality_control: false,
                    criticality_target: 0.0,
                    criticality_gamma_min: 0.001,
                    criticality_gamma_max: 1.0,
                    topology: None,
                },
                LayerConfig {
                    layer: LatticeLayer::Creative,
                    dimension: 24,
                    rae_steps_per_tick: 25,
                    dt: 0.0005,
                    omega: 0.5,
                    gamma: 0.1,
                    kinetic_weight: 0.5,
                    potential_weight: 1.0,
                    target_amplitude: 1.0,
                    enable_metric_learning: false,
                    enable_adaptive_dt: true,
                    method: IntegratorMethod::SemiImplicit,
                    enable_criticality_control: false,
                    criticality_target: 0.0,
                    criticality_gamma_min: 0.001,
                    criticality_gamma_max: 1.0,
                    topology: None,
                },
                LayerConfig {
                    layer: LatticeLayer::Associative,
                    dimension: 16,
                    rae_steps_per_tick: 10,
                    dt: 0.005,
                    omega: 0.3,
                    gamma: 0.1,
                    kinetic_weight: 0.5,
                    potential_weight: 1.0,
                    target_amplitude: 1.0,
                    enable_metric_learning: false,
                    enable_adaptive_dt: true,
                    method: IntegratorMethod::SemiImplicit,
                    enable_criticality_control: false,
                    criticality_target: 0.0,
                    criticality_gamma_min: 0.001,
                    criticality_gamma_max: 1.0,
                    topology: None,
                },
            ],
            backend: BackendSelection::Cpu,
            vram_budget_bytes: 0,
            enable_inter_layer_transfer: true,
            transfer_learning_rate: 0.001,
            agents: AgentConfig::default(),
        }
    }

    /// Estimate total memory usage across all layers (bytes).
    pub fn estimated_memory_bytes(&self) -> usize {
        let mut total = 0usize;
        for layer in &self.layers {
            let sites_estimate = match layer.layer {
                LatticeLayer::Analytical => 241,
                LatticeLayer::Creative => 1_105,
                LatticeLayer::Associative => layer.dimension * (layer.dimension - 1) + 1, // A_n: n(n-1) neighbors + origin
                LatticeLayer::Sensory => 2 * layer.dimension + 1, // Z^n: 2n neighbors + origin
            };
            let packed_metric = layer.dimension * (layer.dimension + 1) / 2;
            let per_site_bytes = (2 + packed_metric) * 4;
            total += sites_estimate * per_site_bytes;
        }
        total
    }

    /// Validate CFL stability condition for all layers.
    ///
    /// For forward Euler on a graph Laplacian, dt must satisfy dt < 2/K
    /// where K is the maximum number of neighbors (kissing number).
    /// Returns a list of (layer_index, dt, cfl_limit) for violating layers.
    pub fn validate_cfl(&self) -> Vec<(usize, f32, f32)> {
        let mut violations = Vec::new();
        for (i, layer) in self.layers.iter().enumerate() {
            // Semi-implicit is unconditionally stable — no CFL constraint
            if layer.method == IntegratorMethod::SemiImplicit {
                continue;
            }
            let kissing = match layer.layer {
                LatticeLayer::Analytical => 240,                          // E8
                LatticeLayer::Creative => 1104,                           // D24
                LatticeLayer::Associative => layer.dimension * (layer.dimension - 1), // A_n
                LatticeLayer::Sensory => 2 * layer.dimension,             // Z^n
            };
            if kissing > 0 {
                let cfl_limit = 2.0 / kissing as f32;
                if layer.dt >= cfl_limit {
                    violations.push((i, layer.dt, cfl_limit));
                }
            }
        }
        violations
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_e8_only_config() {
        let config = ManifoldConfig::e8_only();
        assert_eq!(config.layers.len(), 1);
        assert_eq!(config.layers[0].layer, LatticeLayer::Analytical);
        assert_eq!(config.layers[0].dimension, 8);
        assert!(!config.enable_inter_layer_transfer);
    }

    #[test]
    fn test_full_hierarchy_config() {
        let config = ManifoldConfig::full_hierarchy();
        assert_eq!(config.layers.len(), 4);
        assert!(config.enable_inter_layer_transfer);
        assert!(config.agents.enable_learning);
    }

    #[test]
    fn test_config_serialization() {
        let config = ManifoldConfig::e8_only();
        let json = serde_json::to_string(&config).unwrap();
        let restored: ManifoldConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.layers.len(), 1);
        assert!((restored.layers[0].dt - 0.002).abs() < 1e-6);
    }

    #[test]
    fn test_memory_estimate() {
        let config = ManifoldConfig::e8_only();
        let est = config.estimated_memory_bytes();
        assert!(est > 0);
        // E8: 241 sites * (2 + 36) * 4 = 241 * 152 = ~36KB
        assert!(est < 100_000);
    }

    #[test]
    fn test_agent_config_default() {
        let ac = AgentConfig::default();
        assert!(ac.enable_perception);
        assert!(!ac.enable_world_model);
        assert!(!ac.enable_learning);
        assert_eq!(ac.memory_capacity, 64);
    }

    #[test]
    fn test_multi_layer_config() {
        let config = ManifoldConfig::multi_layer();
        assert_eq!(config.layers.len(), 3);
        assert_eq!(config.layers[0].layer, LatticeLayer::Analytical);
        assert_eq!(config.layers[1].layer, LatticeLayer::Creative);
        assert_eq!(config.layers[2].layer, LatticeLayer::Associative);
        assert!(config.enable_inter_layer_transfer);
        assert_eq!(config.layers[2].dimension, 16);
    }

    #[test]
    fn test_multi_layer_memory_reasonable() {
        let config = ManifoldConfig::multi_layer();
        let est = config.estimated_memory_bytes();
        // Should be well under 10MB for the practical 3-layer config
        assert!(est < 10_000_000, "multi_layer memory {} should be < 10MB", est);
        assert!(est > 0);
    }

    #[test]
    fn test_full_hierarchy_memory_reasonable() {
        let config = ManifoldConfig::full_hierarchy();
        let est = config.estimated_memory_bytes();
        // With dim=32 sensory, should be well under 100MB
        assert!(est < 100_000_000, "full_hierarchy memory {} should be < 100MB", est);
    }

    #[test]
    fn test_cfl_validation_e8() {
        let config = ManifoldConfig::e8_only();
        let violations = config.validate_cfl();
        assert!(violations.is_empty(), "E8 default config should pass CFL: {:?}", violations);
    }

    #[test]
    fn test_cfl_validation_multi_layer() {
        let config = ManifoldConfig::multi_layer();
        let violations = config.validate_cfl();
        assert!(violations.is_empty(), "multi_layer config should pass CFL: {:?}", violations);
    }

    #[test]
    fn test_cfl_limit_helper() {
        let config = ManifoldConfig::e8_only();
        let cfl = config.layers[0].cfl_limit();
        assert!((cfl - 2.0 / 240.0).abs() < 1e-6, "E8 CFL limit should be 2/240, got {}", cfl);

        let config = ManifoldConfig::multi_layer();
        // Creative (D24): 2/1104
        let cfl_d24 = config.layers[1].cfl_limit();
        assert!((cfl_d24 - 2.0 / 1104.0).abs() < 1e-6);
        // Associative HCP(16): 2/(16*15) = 2/240
        let cfl_hcp = config.layers[2].cfl_limit();
        assert!((cfl_hcp - 2.0 / 240.0).abs() < 1e-6);
    }

    #[test]
    fn test_cfl_validation_detects_violation() {
        let mut config = ManifoldConfig::e8_only();
        config.layers[0].dt = 0.1; // Way above CFL limit of 2/240 ≈ 0.0083
        config.layers[0].method = IntegratorMethod::Euler; // CFL only applies to Euler
        let violations = config.validate_cfl();
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].0, 0); // Layer 0
    }
}
