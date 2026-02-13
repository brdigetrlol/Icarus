// Copyright (c) 2025-2026 brdigetrlol. All rights reserved.
// SPDX-License-Identifier: LicenseRef-Icarus-Proprietary
// See LICENSE in the repository root for full license terms.

//! Cortana cognitive agents — 4 specialized agents that process emotional state.
//!
//! These agents operate on Cortana's extended affective state, not directly on the
//! Icarus manifold. They run in sequence after each EMC tick:
//! EmotionAgent → SocialAgent → LanguageAgent → CreativeAgent.

pub mod emotion_agent;
pub mod social_agent;
pub mod language_agent;
pub mod creative_agent;

pub use emotion_agent::EmotionAgent;
pub use social_agent::SocialAgent;
pub use language_agent::LanguageAgent;
pub use creative_agent::CreativeAgent;

use crate::affect::{ExtendedAffectiveState, PhysicsInput};
use crate::aether::ExtendedAetherParams;
use crate::memory::EmotionalMemory;
use crate::mood::MoodSystem;
use crate::personality::Personality;

/// Shared context passed to all Cortana agents during a tick.
pub struct AgentContext<'a> {
    /// Current physics observables from EMC
    pub physics: &'a PhysicsInput,
    /// Extended affective state (read/write)
    pub affect: &'a mut ExtendedAffectiveState,
    /// Personality (read-only, stable traits)
    pub personality: &'a Personality,
    /// Mood system (read/write)
    pub mood: &'a mut MoodSystem,
    /// Emotional memory (read/write)
    pub memory: &'a mut EmotionalMemory,
    /// Neuromodulator parameters
    pub aether_params: &'a ExtendedAetherParams,
    /// Current tick number
    pub tick: u64,
    /// Delta time for this tick
    pub dt: f32,
}

/// Orchestrator that runs all 4 Cortana agents in sequence.
pub struct CortanaAgentOrchestrator {
    pub emotion: EmotionAgent,
    pub social: SocialAgent,
    pub language: LanguageAgent,
    pub creative: CreativeAgent,
}

impl CortanaAgentOrchestrator {
    pub fn new() -> Self {
        Self {
            emotion: EmotionAgent::new(),
            social: SocialAgent::new(),
            language: LanguageAgent::new(),
            creative: CreativeAgent::new(),
        }
    }

    /// Run all agents in sequence for one tick.
    ///
    /// Order matters: EmotionAgent computes primary emotions first,
    /// then SocialAgent processes social dynamics, LanguageAgent
    /// computes expression metadata, and CreativeAgent samples for inspiration.
    pub fn tick(&mut self, ctx: &mut AgentContext<'_>) {
        self.emotion.tick(ctx);
        self.social.tick(ctx);
        self.language.tick(ctx);
        self.creative.tick(ctx);
    }

    /// Get the current language expression metadata.
    pub fn expression(&self) -> &language_agent::ExpressionMetadata {
        &self.language.expression
    }

    /// Get the current creative state.
    pub fn creative_state(&self) -> &creative_agent::CreativeState {
        &self.creative.state
    }

    /// Get the current social state.
    pub fn social_state(&self) -> &social_agent::SocialState {
        &self.social.state
    }
}

impl Default for CortanaAgentOrchestrator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::affect::ExtendedAffectiveState;
    use crate::aether::ExtendedAetherParams;
    use crate::memory::EmotionalMemory;
    use crate::mood::MoodSystem;
    use crate::personality::Personality;

    fn test_context<'a>(
        physics: &'a PhysicsInput,
        affect: &'a mut ExtendedAffectiveState,
        personality: &'a Personality,
        mood: &'a mut MoodSystem,
        memory: &'a mut EmotionalMemory,
        aether_params: &'a ExtendedAetherParams,
    ) -> AgentContext<'a> {
        AgentContext {
            physics,
            affect,
            personality,
            mood,
            memory,
            aether_params,
            tick: 0,
            dt: 1.0,
        }
    }

    #[test]
    fn test_orchestrator_creation() {
        let orch = CortanaAgentOrchestrator::new();
        assert!(orch.expression().energy < 0.01);
    }

    #[test]
    fn test_orchestrator_tick() {
        let mut orch = CortanaAgentOrchestrator::new();
        let physics = PhysicsInput {
            valence: 2.0,
            arousal: 0.6,
            phase_coherence: 0.7,
            criticality_sigma: 0.2,
            prediction_error: 0.3,
            convergence_stable: true,
            achievement: false,
        };
        let mut affect = ExtendedAffectiveState::default();
        let personality = Personality::cortana_default();
        let mut mood = MoodSystem::default();
        let mut memory = EmotionalMemory::default();
        let aether_params = ExtendedAetherParams::default();

        let mut ctx = test_context(
            &physics,
            &mut affect,
            &personality,
            &mut mood,
            &mut memory,
            &aether_params,
        );
        orch.tick(&mut ctx);

        // After ticking, affect should have been updated
        assert!(ctx.affect.pleasure != 0.0 || ctx.affect.arousal != 0.0);
    }
}
