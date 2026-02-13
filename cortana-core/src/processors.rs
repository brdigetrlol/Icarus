// Copyright (c) 2025-2026 brdigetrlol. All rights reserved.
// SPDX-License-Identifier: LicenseRef-Icarus-Proprietary
// See LICENSE in the repository root for full license terms.

//! Processor orchestrator — capability-based routing to ecosystem processors.
//!
//! Maintains a registry of available processors and their capabilities.
//! Emotional state can influence processor selection (e.g., high curiosity
//! prefers deeper analysis, high urgency prefers speed).

use serde::{Deserialize, Serialize};

/// Capability categories that processors can provide.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Capability {
    /// Physics simulation (EMC lattice dynamics)
    PhysicsSimulation,
    /// GPU-accelerated vector/matrix compute
    GpuCompute,
    /// Neural processing (NPU bridge)
    NeuralProcessing,
    /// Vision analysis (describe, OCR, ask)
    VisionAnalysis,
    /// Image generation (text-to-image, img2img)
    ImageGeneration,
    /// Text generation via external LLM (Gemini)
    TextGenerationGemini,
    /// Text generation via external LLM (GLM)
    TextGenerationGlm,
    /// Deep reasoning and cognitive analysis
    DeepReasoning,
    /// Semantic search and knowledge retrieval
    SemanticSearch,
    /// Privacy and safety filtering
    PrivacyShield,
    /// Codebase context and understanding
    CodebaseContext,
    /// Reservoir computing (training, prediction)
    ReservoirComputing,
    /// Visualization rendering
    Visualization,
}

impl Capability {
    pub fn name(self) -> &'static str {
        match self {
            Self::PhysicsSimulation => "physics_simulation",
            Self::GpuCompute => "gpu_compute",
            Self::NeuralProcessing => "neural_processing",
            Self::VisionAnalysis => "vision_analysis",
            Self::ImageGeneration => "image_generation",
            Self::TextGenerationGemini => "text_generation_gemini",
            Self::TextGenerationGlm => "text_generation_glm",
            Self::DeepReasoning => "deep_reasoning",
            Self::SemanticSearch => "semantic_search",
            Self::PrivacyShield => "privacy_shield",
            Self::CodebaseContext => "codebase_context",
            Self::ReservoirComputing => "reservoir_computing",
            Self::Visualization => "visualization",
        }
    }
}

/// A registered processor in the ecosystem.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessorEntry {
    /// Unique identifier (e.g., "icarus-emc", "gpu-accelerator")
    pub id: String,
    /// Human-readable description
    pub description: String,
    /// MCP server name for tool invocation
    pub mcp_server: String,
    /// Capabilities this processor provides
    pub capabilities: Vec<Capability>,
    /// Whether the processor is currently available
    pub available: bool,
    /// Priority weight [0, 1] — higher = preferred when multiple match
    pub priority: f32,
}

/// Emotion-influenced routing hints.
#[derive(Debug, Clone, Default)]
pub struct RoutingHints {
    /// Prefer deep analysis (high curiosity/dopamine)
    pub prefer_depth: f32,
    /// Prefer speed (high urgency/norepinephrine)
    pub prefer_speed: f32,
    /// Prefer creative output (high openness + surprise)
    pub prefer_creativity: f32,
    /// Prefer safety (high fear/anxiety)
    pub prefer_safety: f32,
}

/// Result of processor selection.
#[derive(Debug, Clone, Serialize)]
pub struct ProcessorSelection {
    /// Selected processor ID
    pub processor_id: String,
    /// MCP server name
    pub mcp_server: String,
    /// Why this processor was selected
    pub reason: String,
    /// Match score [0, 1]
    pub score: f32,
}

/// Configuration for the processor orchestrator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessorConfig {
    /// Whether emotion-aware routing is enabled
    pub emotion_aware_routing: bool,
    /// Minimum score to consider a processor match
    pub min_match_score: f32,
}

impl Default for ProcessorConfig {
    fn default() -> Self {
        Self {
            emotion_aware_routing: true,
            min_match_score: 0.1,
        }
    }
}

/// Processor orchestrator — routes tasks to the best-fit ecosystem processor.
#[derive(Debug, Clone)]
pub struct ProcessorOrchestrator {
    processors: Vec<ProcessorEntry>,
    config: ProcessorConfig,
}

impl ProcessorOrchestrator {
    pub fn new(config: ProcessorConfig) -> Self {
        let mut orchestrator = Self {
            processors: Vec::new(),
            config,
        };
        orchestrator.register_defaults();
        orchestrator
    }

    /// Register the default ecosystem processors.
    fn register_defaults(&mut self) {
        self.processors = vec![
            ProcessorEntry {
                id: "icarus-emc".into(),
                description: "Emergent Manifold Computer — physics simulation on crystal lattices".into(),
                mcp_server: "icarus-mcp".into(),
                capabilities: vec![
                    Capability::PhysicsSimulation,
                    Capability::ReservoirComputing,
                ],
                available: true,
                priority: 0.9,
            },
            ProcessorEntry {
                id: "gpu-accelerator".into(),
                description: "CUDA-accelerated vector/matrix compute".into(),
                mcp_server: "gpu-accelerator".into(),
                capabilities: vec![Capability::GpuCompute],
                available: true,
                priority: 0.8,
            },
            ProcessorEntry {
                id: "npu-bridge".into(),
                description: "Intel NPU via TCP bridge for neural inference".into(),
                mcp_server: "icarus-mcp".into(),
                capabilities: vec![Capability::NeuralProcessing],
                available: true,
                priority: 0.7,
            },
            ProcessorEntry {
                id: "djhagno".into(),
                description: "Vision analysis and image generation (native Rust)".into(),
                mcp_server: "djhagno-unchained".into(),
                capabilities: vec![
                    Capability::VisionAnalysis,
                    Capability::ImageGeneration,
                ],
                available: true,
                priority: 0.8,
            },
            ProcessorEntry {
                id: "crouter-gemini".into(),
                description: "Google Gemini for text generation and research".into(),
                mcp_server: "cRouter".into(),
                capabilities: vec![Capability::TextGenerationGemini],
                available: true,
                priority: 0.7,
            },
            ProcessorEntry {
                id: "crouter-glm".into(),
                description: "GLM-4.7 for async text generation".into(),
                mcp_server: "cRouter".into(),
                capabilities: vec![Capability::TextGenerationGlm],
                available: true,
                priority: 0.6,
            },
            ProcessorEntry {
                id: "scry".into(),
                description: "Advanced cognitive reasoning engine (45 tools)".into(),
                mcp_server: "scry".into(),
                capabilities: vec![Capability::DeepReasoning],
                available: true,
                priority: 0.9,
            },
            ProcessorEntry {
                id: "oracle".into(),
                description: "Deep semantic search over past conversations and codebase".into(),
                mcp_server: "oracle".into(),
                capabilities: vec![Capability::SemanticSearch],
                available: true,
                priority: 0.9,
            },
            ProcessorEntry {
                id: "aegis".into(),
                description: "Privacy shield and safety filtering".into(),
                mcp_server: "aegis".into(),
                capabilities: vec![Capability::PrivacyShield],
                available: true,
                priority: 0.8,
            },
            ProcessorEntry {
                id: "simplex".into(),
                description: "Codebase understanding and intelligent context selection".into(),
                mcp_server: "simplex".into(),
                capabilities: vec![Capability::CodebaseContext],
                available: true,
                priority: 0.7,
            },
            ProcessorEntry {
                id: "icarus-viz".into(),
                description: "Visualization rendering for lattice fields and dashboards".into(),
                mcp_server: "icarus-mcp".into(),
                capabilities: vec![Capability::Visualization],
                available: true,
                priority: 0.8,
            },
        ];
    }

    /// Register a custom processor.
    pub fn register(&mut self, entry: ProcessorEntry) {
        // Replace if same ID exists
        self.processors.retain(|p| p.id != entry.id);
        self.processors.push(entry);
    }

    /// Unregister a processor by ID.
    pub fn unregister(&mut self, id: &str) -> bool {
        let before = self.processors.len();
        self.processors.retain(|p| p.id != id);
        self.processors.len() < before
    }

    /// Set processor availability.
    pub fn set_available(&mut self, id: &str, available: bool) -> bool {
        if let Some(p) = self.processors.iter_mut().find(|p| p.id == id) {
            p.available = available;
            true
        } else {
            false
        }
    }

    /// List all registered processors.
    pub fn list(&self) -> &[ProcessorEntry] {
        &self.processors
    }

    /// Find processors matching a capability.
    pub fn find_by_capability(&self, capability: Capability) -> Vec<&ProcessorEntry> {
        self.processors
            .iter()
            .filter(|p| p.available && p.capabilities.contains(&capability))
            .collect()
    }

    /// Select the best processor for a given capability with optional emotion hints.
    pub fn select(
        &self,
        capability: Capability,
        hints: Option<&RoutingHints>,
    ) -> Option<ProcessorSelection> {
        let candidates = self.find_by_capability(capability);
        if candidates.is_empty() {
            return None;
        }

        let mut best: Option<(f32, &ProcessorEntry)> = None;

        for proc in &candidates {
            let mut score = proc.priority;

            // Apply emotion-aware routing biases
            if self.config.emotion_aware_routing {
                if let Some(hints) = hints {
                    score += self.emotion_bias(proc, hints);
                }
            }

            score = score.clamp(0.0, 1.0);

            if score >= self.config.min_match_score {
                if best.map_or(true, |(bs, _)| score > bs) {
                    best = Some((score, proc));
                }
            }
        }

        best.map(|(score, proc)| ProcessorSelection {
            processor_id: proc.id.clone(),
            mcp_server: proc.mcp_server.clone(),
            reason: format!(
                "Best match for {} (score: {:.2})",
                capability.name(),
                score
            ),
            score,
        })
    }

    /// Compute emotion-based routing bias for a processor.
    fn emotion_bias(&self, proc: &ProcessorEntry, hints: &RoutingHints) -> f32 {
        let mut bias = 0.0f32;

        // Deep reasoning benefits from curiosity
        if proc.capabilities.contains(&Capability::DeepReasoning) {
            bias += hints.prefer_depth * 0.1;
        }

        // GPU compute benefits from urgency
        if proc.capabilities.contains(&Capability::GpuCompute) {
            bias += hints.prefer_speed * 0.1;
        }

        // Image generation and vision benefit from creativity
        if proc.capabilities.contains(&Capability::ImageGeneration)
            || proc.capabilities.contains(&Capability::VisionAnalysis)
        {
            bias += hints.prefer_creativity * 0.1;
        }

        // Privacy shield benefits from anxiety/caution
        if proc.capabilities.contains(&Capability::PrivacyShield) {
            bias += hints.prefer_safety * 0.15;
        }

        bias
    }

    /// Get a summary of all processors and their status.
    pub fn status_summary(&self) -> Vec<ProcessorStatus> {
        self.processors
            .iter()
            .map(|p| ProcessorStatus {
                id: p.id.clone(),
                available: p.available,
                capabilities: p.capabilities.iter().map(|c| c.name().to_string()).collect(),
            })
            .collect()
    }
}

impl Default for ProcessorOrchestrator {
    fn default() -> Self {
        Self::new(ProcessorConfig::default())
    }
}

/// Summary of a processor's status.
#[derive(Debug, Clone, Serialize)]
pub struct ProcessorStatus {
    pub id: String,
    pub available: bool,
    pub capabilities: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_processors_registered() {
        let orch = ProcessorOrchestrator::default();
        assert!(orch.list().len() >= 10);
    }

    #[test]
    fn test_find_by_capability() {
        let orch = ProcessorOrchestrator::default();
        let physics = orch.find_by_capability(Capability::PhysicsSimulation);
        assert!(!physics.is_empty());
        assert!(physics.iter().any(|p| p.id == "icarus-emc"));
    }

    #[test]
    fn test_find_vision() {
        let orch = ProcessorOrchestrator::default();
        let vision = orch.find_by_capability(Capability::VisionAnalysis);
        assert!(vision.iter().any(|p| p.id == "djhagno"));
    }

    #[test]
    fn test_select_best_match() {
        let orch = ProcessorOrchestrator::default();
        let sel = orch.select(Capability::DeepReasoning, None);
        assert!(sel.is_some());
        assert_eq!(sel.unwrap().processor_id, "scry");
    }

    #[test]
    fn test_select_with_emotion_hints() {
        let orch = ProcessorOrchestrator::default();
        let hints = RoutingHints {
            prefer_safety: 1.0,
            ..Default::default()
        };
        let sel = orch.select(Capability::PrivacyShield, Some(&hints));
        assert!(sel.is_some());
        let sel = sel.unwrap();
        assert_eq!(sel.processor_id, "aegis");
        assert!(sel.score > 0.8);
    }

    #[test]
    fn test_select_nonexistent_capability() {
        let mut orch = ProcessorOrchestrator::default();
        // Disable all processors
        for p in orch.processors.iter_mut() {
            p.available = false;
        }
        let sel = orch.select(Capability::PhysicsSimulation, None);
        assert!(sel.is_none());
    }

    #[test]
    fn test_register_custom_processor() {
        let mut orch = ProcessorOrchestrator::default();
        let before = orch.list().len();
        orch.register(ProcessorEntry {
            id: "custom-proc".into(),
            description: "Custom processor".into(),
            mcp_server: "custom".into(),
            capabilities: vec![Capability::GpuCompute],
            available: true,
            priority: 0.95,
        });
        assert_eq!(orch.list().len(), before + 1);

        // Custom should win for GPU compute due to higher priority
        let sel = orch.select(Capability::GpuCompute, None);
        assert_eq!(sel.unwrap().processor_id, "custom-proc");
    }

    #[test]
    fn test_unregister() {
        let mut orch = ProcessorOrchestrator::default();
        let before = orch.list().len();
        assert!(orch.unregister("scry"));
        assert_eq!(orch.list().len(), before - 1);
        assert!(!orch.unregister("nonexistent"));
    }

    #[test]
    fn test_set_available() {
        let mut orch = ProcessorOrchestrator::default();
        assert!(orch.set_available("scry", false));
        let reasoning = orch.find_by_capability(Capability::DeepReasoning);
        assert!(reasoning.is_empty());
    }

    #[test]
    fn test_status_summary() {
        let orch = ProcessorOrchestrator::default();
        let summary = orch.status_summary();
        assert!(summary.len() >= 10);
        assert!(summary.iter().all(|s| s.available));
    }

    #[test]
    fn test_capability_name() {
        assert_eq!(Capability::PhysicsSimulation.name(), "physics_simulation");
        assert_eq!(Capability::DeepReasoning.name(), "deep_reasoning");
    }

    #[test]
    fn test_register_replaces_existing() {
        let mut orch = ProcessorOrchestrator::default();
        let before = orch.list().len();
        orch.register(ProcessorEntry {
            id: "scry".into(),
            description: "Updated scry".into(),
            mcp_server: "scry-v2".into(),
            capabilities: vec![Capability::DeepReasoning],
            available: true,
            priority: 1.0,
        });
        assert_eq!(orch.list().len(), before);
        let scry = orch.list().iter().find(|p| p.id == "scry").unwrap();
        assert_eq!(scry.mcp_server, "scry-v2");
    }

    #[test]
    fn test_emotion_bias_depth() {
        let orch = ProcessorOrchestrator::default();
        let hints = RoutingHints {
            prefer_depth: 1.0,
            ..Default::default()
        };
        let sel = orch.select(Capability::DeepReasoning, Some(&hints));
        let sel_no_hints = orch.select(Capability::DeepReasoning, None);
        // With depth preference, score should be higher
        assert!(sel.unwrap().score >= sel_no_hints.unwrap().score);
    }
}
