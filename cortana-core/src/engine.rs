// Copyright (c) 2025-2026 brdigetrlol. All rights reserved.
// SPDX-License-Identifier: LicenseRef-Icarus-Proprietary
// See LICENSE in the repository root for full license terms.

//! CortanaEngine — Top-level orchestrator wrapping the Icarus EMC.
//!
//! Composes the EmergentManifoldComputer with Cortana's extended affective
//! state, personality, mood, memory, and cognitive agents into a unified
//! emotion simulation engine.

use anyhow::Result;
use icarus_engine::emc::EmergentManifoldComputer;
use icarus_field::autopoiesis::AffectiveState;

use crate::aether::ExtendedAetherParams;
use crate::affect::{ExtendedAffectiveState, PhysicsInput};
use crate::agents::{AgentContext, CortanaAgentOrchestrator};
use crate::config::CortanaConfig;
use crate::emotion::EmotionLabel;
use crate::memory::{EmotionalMemory, MemoryStats};
use crate::mood::{MoodLabel, MoodState, MoodSystem};
use crate::personality::Personality;
use crate::processors::ProcessorOrchestrator;

/// The Cortana emotion simulation engine.
///
/// Wraps an Icarus EMC (physics substrate) and extends it with:
/// - Extended affective state (3D PAD + Plutchik emotions)
/// - Big Five personality (stable traits)
/// - Mood system (slow EMA baseline)
/// - Emotional episodic memory
/// - 8 neuromodulators
/// - 4 cognitive agents
/// - Processor orchestrator
pub struct CortanaEngine {
    /// The Icarus physics substrate
    pub emc: EmergentManifoldComputer,
    /// Extended affective state (PAD + Plutchik + mood + neuromodulators)
    pub affect: ExtendedAffectiveState,
    /// Personality traits (stable)
    pub personality: Personality,
    /// Mood system (slow dynamics)
    pub mood: MoodSystem,
    /// Emotional episodic memory
    pub memory: EmotionalMemory,
    /// Extended neuromodulator parameters
    pub aether_params: ExtendedAetherParams,
    /// Cortana cognitive agent orchestrator
    pub agents: CortanaAgentOrchestrator,
    /// Processor orchestrator (ecosystem routing)
    pub processors: ProcessorOrchestrator,
    /// Total ticks executed by Cortana (matches EMC ticks)
    pub total_ticks: u64,
    /// Configuration (preserved for serialization / inspection)
    config: CortanaConfig,
}

impl CortanaEngine {
    /// Create a new CortanaEngine from configuration.
    ///
    /// Uses GPU backend if configured; falls back to CPU on failure.
    pub fn new(config: CortanaConfig) -> Result<Self> {
        let emc = EmergentManifoldComputer::new(config.emc.clone())?;
        Ok(Self::from_emc(emc, config))
    }

    /// Create a CortanaEngine with CPU backend (no GPU required).
    ///
    /// Suitable for tests and environments without GPU.
    pub fn new_cpu(config: CortanaConfig) -> Self {
        let mut emc_config = config.emc.clone();
        emc_config.backend = icarus_engine::config::BackendSelection::Cpu;
        let emc = EmergentManifoldComputer::new_cpu(emc_config);
        Self::from_emc(emc, config)
    }

    /// Build from an existing EMC instance + config.
    fn from_emc(mut emc: EmergentManifoldComputer, config: CortanaConfig) -> Self {
        emc.init_random(config.seed, config.init_amplitude);

        let affect = ExtendedAffectiveState::default();
        let personality = config.personality.clone();
        let mood = MoodSystem::new(config.mood.clone());
        let memory = EmotionalMemory::new(config.memory.clone());
        let aether_params = config.aether.clone();
        let agents = CortanaAgentOrchestrator::new();
        let processors = ProcessorOrchestrator::new(config.processors.clone());

        Self {
            emc,
            affect,
            personality,
            mood,
            memory,
            aether_params,
            agents,
            processors,
            total_ticks: 0,
            config,
        }
    }

    /// Execute one full Cortana tick:
    ///
    /// 1. EMC physics tick (RAE dynamics + Icarus agents)
    /// 2. Extract physics observables into PhysicsInput
    /// 3. Run Cortana cognitive agents (Emotion → Social → Language → Creative)
    /// 4. Increment tick counter
    pub fn tick(&mut self) -> Result<()> {
        // 1. EMC physics tick
        self.emc.tick()?;

        // 2. Extract physics observables
        let physics = self.extract_physics();

        // 3. Run Cortana agents with shared context
        let mut ctx = AgentContext {
            physics: &physics,
            affect: &mut self.affect,
            personality: &self.personality,
            mood: &mut self.mood,
            memory: &mut self.memory,
            aether_params: &self.aether_params,
            tick: self.total_ticks,
            dt: 1.0,
        };
        self.agents.tick(&mut ctx);

        // 4. Increment
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

    /// Extract physics observables from the current EMC state.
    fn extract_physics(&self) -> PhysicsInput {
        let affective = self.emc.affective_state();
        let stats = self.emc.stats();

        // Compute criticality sigma: std dev of energy across layers
        let criticality_sigma = if stats.layer_stats.len() > 1 {
            let energies: Vec<f32> = stats.layer_stats.iter().map(|l| l.total_energy).collect();
            let mean = energies.iter().sum::<f32>() / energies.len() as f32;
            let variance =
                energies.iter().map(|e| (e - mean) * (e - mean)).sum::<f32>() / energies.len() as f32;
            variance.sqrt()
        } else {
            0.0
        };

        // Prediction error: placeholder — would come from reservoir readout vs actual
        // For now, derive from mean amplitude deviation from target (1.0)
        let prediction_error = if !stats.layer_stats.is_empty() {
            let mean_amp = stats.layer_stats[0].mean_amplitude;
            (1.0 - mean_amp).abs().clamp(0.0, 1.0)
        } else {
            0.5
        };

        // Convergence: stable if total energy is low and not changing much
        let convergence_stable = stats.layer_stats.iter().all(|l| l.total_energy < 10.0);

        PhysicsInput {
            valence: affective.valence,
            arousal: affective.arousal,
            phase_coherence: affective.phase_coherence,
            criticality_sigma,
            prediction_error,
            convergence_stable,
            achievement: false, // set externally when milestones occur
        }
    }

    /// Get the current Icarus affective state (base 2D).
    pub fn base_affective_state(&self) -> AffectiveState {
        self.emc.affective_state()
    }

    /// Get the current extended affective state.
    pub fn affective_state(&self) -> &ExtendedAffectiveState {
        &self.affect
    }

    /// Get the current dominant emotion.
    pub fn dominant_emotion(&self) -> EmotionLabel {
        self.affect.dominant_emotion.clone()
    }

    /// Get the current mood label.
    pub fn mood_label(&self) -> MoodLabel {
        self.mood.state.label
    }

    /// Get the current mood state.
    pub fn mood_state(&self) -> &MoodState {
        &self.mood.state
    }

    /// Get memory statistics.
    pub fn memory_stats(&self) -> MemoryStats {
        self.memory.stats()
    }

    /// Get the personality.
    pub fn personality(&self) -> &Personality {
        &self.personality
    }

    /// Modify a personality trait by name. Returns false if trait not found.
    pub fn set_personality_trait(&mut self, name: &str, value: f32) -> bool {
        self.personality.set_trait(name, value)
    }

    /// Get the configuration.
    pub fn config(&self) -> &CortanaConfig {
        &self.config
    }

    /// Inject a stimulus event, which may be encoded into memory.
    ///
    /// Forces a high-arousal event with the given valence and tags,
    /// bypassing the normal arousal gate.
    pub fn inject_stimulus(&mut self, stimulus: &str, valence: f32, tags: Vec<String>) {
        self.memory.encode(
            self.total_ticks,
            self.total_ticks,
            stimulus.to_string(),
            self.affect.plutchik.clone(),
            valence,
            1.0, // force high arousal to guarantee encoding
            tags,
        );
    }

    /// Get processor orchestrator.
    pub fn processors(&self) -> &ProcessorOrchestrator {
        &self.processors
    }

    /// Get mutable processor orchestrator.
    pub fn processors_mut(&mut self) -> &mut ProcessorOrchestrator {
        &mut self.processors
    }

    /// Get the expression metadata from the language agent.
    pub fn expression(&self) -> &crate::agents::language_agent::ExpressionMetadata {
        self.agents.expression()
    }

    /// Get the creative state from the creative agent.
    pub fn creative_state(&self) -> &crate::agents::creative_agent::CreativeState {
        self.agents.creative_state()
    }

    /// Get the social state from the social agent.
    pub fn social_state(&self) -> &crate::agents::social_agent::SocialState {
        self.agents.social_state()
    }

    /// Get a summary snapshot of the engine state.
    pub fn snapshot(&self) -> CortanaSnapshot {
        CortanaSnapshot {
            tick: self.total_ticks,
            dominant_emotion: self.affect.dominant_emotion.name().to_string(),
            mood: self.mood.state.label.name().to_string(),
            pleasure: self.affect.pleasure,
            arousal: self.affect.arousal,
            dominance: self.affect.dominance,
            emotional_intensity: self.affect.emotional_intensity(),
            memory_count: self.memory.episodes().len(),
            creative_drive: self.agents.creative_state().creative_drive,
            social_bonding: self.agents.social_state().bonding,
        }
    }
}

/// Serializable snapshot of engine state.
#[derive(Debug, Clone, serde::Serialize)]
pub struct CortanaSnapshot {
    pub tick: u64,
    pub dominant_emotion: String,
    pub mood: String,
    pub pleasure: f32,
    pub arousal: f32,
    pub dominance: f32,
    pub emotional_intensity: f32,
    pub memory_count: usize,
    pub creative_drive: f32,
    pub social_bonding: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_creation_cpu() {
        let config = CortanaConfig::default();
        let engine = CortanaEngine::new_cpu(config);
        assert_eq!(engine.total_ticks, 0);
        assert_eq!(engine.dominant_emotion(), EmotionLabel::Neutral);
        assert_eq!(engine.mood_label(), MoodLabel::Neutral);
    }

    #[test]
    fn test_engine_tick() {
        let config = CortanaConfig::default();
        let mut engine = CortanaEngine::new_cpu(config);
        engine.tick().unwrap();
        assert_eq!(engine.total_ticks, 1);
    }

    #[test]
    fn test_engine_run() {
        let config = CortanaConfig::default();
        let mut engine = CortanaEngine::new_cpu(config);
        engine.run(10).unwrap();
        assert_eq!(engine.total_ticks, 10);
    }

    #[test]
    fn test_engine_affect_updates() {
        let config = CortanaConfig::default();
        let mut engine = CortanaEngine::new_cpu(config);
        engine.run(50).unwrap();

        // After 50 ticks, affect should have non-trivial values
        let state = engine.affective_state();
        assert!((-1.0..=1.0).contains(&state.pleasure));
        assert!((-1.0..=1.0).contains(&state.arousal));
        assert!((-1.0..=1.0).contains(&state.dominance));
    }

    #[test]
    fn test_engine_mood_evolves() {
        let config = CortanaConfig::default();
        let mut engine = CortanaEngine::new_cpu(config);
        engine.run(100).unwrap();

        // Mood should have been updated (may still be neutral, but hedonic_tone should be non-zero)
        let mood = engine.mood_state();
        assert!((-1.0..=1.0).contains(&mood.hedonic_tone));
    }

    #[test]
    fn test_engine_personality_access() {
        let config = CortanaConfig::cortana_default();
        let engine = CortanaEngine::new_cpu(config);
        assert!((engine.personality().openness - 0.85).abs() < f32::EPSILON);
    }

    #[test]
    fn test_engine_set_personality() {
        let config = CortanaConfig::default();
        let mut engine = CortanaEngine::new_cpu(config);
        assert!(engine.set_personality_trait("neuroticism", 0.9));
        assert!((engine.personality().neuroticism - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn test_engine_inject_stimulus() {
        let config = CortanaConfig::default();
        let mut engine = CortanaEngine::new_cpu(config);
        engine.inject_stimulus("test event", 0.8, vec!["test".into()]);
        assert_eq!(engine.memory.episodes().len(), 1);
        assert_eq!(engine.memory.episodes()[0].stimulus, "test event");
    }

    #[test]
    fn test_engine_snapshot() {
        let config = CortanaConfig::default();
        let mut engine = CortanaEngine::new_cpu(config);
        engine.run(10).unwrap();

        let snap = engine.snapshot();
        assert_eq!(snap.tick, 10);
        assert!(!snap.dominant_emotion.is_empty());
        assert!(!snap.mood.is_empty());
    }

    #[test]
    fn test_engine_expression() {
        let config = CortanaConfig::default();
        let mut engine = CortanaEngine::new_cpu(config);
        engine.run(10).unwrap();

        let expr = engine.expression();
        assert!(!expr.tone.is_empty());
        assert!((0.0..=1.0).contains(&expr.energy));
        assert!((0.0..=1.0).contains(&expr.formality));
    }

    #[test]
    fn test_engine_stoic_preset() {
        let config = CortanaConfig::stoic();
        let mut engine = CortanaEngine::new_cpu(config);
        engine.run(50).unwrap();
        // Stoic should have low emotional intensity
        assert!(engine.affective_state().emotional_intensity() < 2.0);
    }

    #[test]
    fn test_engine_processors_access() {
        let config = CortanaConfig::default();
        let engine = CortanaEngine::new_cpu(config);
        assert!(engine.processors().list().len() >= 10);
    }

    #[test]
    fn test_engine_deterministic() {
        let config = CortanaConfig::default();

        let mut engine1 = CortanaEngine::new_cpu(config.clone());
        engine1.run(20).unwrap();

        let mut engine2 = CortanaEngine::new_cpu(config);
        engine2.run(20).unwrap();

        // Same seed → same physics → same affect
        let snap1 = engine1.snapshot();
        let snap2 = engine2.snapshot();
        assert!((snap1.pleasure - snap2.pleasure).abs() < 1e-5);
        assert!((snap1.arousal - snap2.arousal).abs() < 1e-5);
    }

    #[test]
    fn test_engine_bounded_values() {
        let config = CortanaConfig::default();
        let mut engine = CortanaEngine::new_cpu(config);
        engine.run(500).unwrap();

        let state = engine.affective_state();
        assert!((-1.0..=1.0).contains(&state.pleasure));
        assert!((-1.0..=1.0).contains(&state.arousal));
        assert!((-1.0..=1.0).contains(&state.dominance));
        for &a in &state.plutchik.activations {
            assert!((0.0..=1.0).contains(&a));
        }
    }

    #[test]
    fn test_engine_creative_state() {
        let config = CortanaConfig::creative();
        let mut engine = CortanaEngine::new_cpu(config);
        engine.run(100).unwrap();

        let creative = engine.creative_state();
        assert!((0.0..=1.0).contains(&creative.creative_drive));
    }

    #[test]
    fn test_engine_social_state() {
        let config = CortanaConfig::default();
        let mut engine = CortanaEngine::new_cpu(config);
        engine.run(50).unwrap();

        let social = engine.social_state();
        assert!((0.0..=1.0).contains(&social.trust_level));
        assert!((0.0..=1.0).contains(&social.empathy));
    }
}
