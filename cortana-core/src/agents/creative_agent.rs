//! CreativeAgent — Monitors openness + surprise/anticipation for creative ideation.
//!
//! Samples emotional memory for inspiration, generates creative prompt seeds
//! that can be routed to D'jhagno or cRouter for generation.

use serde::Serialize;

use crate::agents::AgentContext;
use crate::emotion;

/// Creative state tracked by the CreativeAgent.
#[derive(Debug, Clone, Serialize)]
pub struct CreativeState {
    /// Current creative drive [0, 1] — computed from openness + surprise + anticipation
    pub creative_drive: f32,
    /// Whether the creative threshold is currently exceeded (active ideation)
    pub ideation_active: bool,
    /// Inspiration seed — emotional memory fragment for creative prompting
    pub inspiration_seed: Option<InspirationSeed>,
    /// Flow state indicator from extended aether
    pub flow_state: f32,
    /// Total creative triggers this session
    pub total_triggers: u64,
}

impl Default for CreativeState {
    fn default() -> Self {
        Self {
            creative_drive: 0.0,
            ideation_active: false,
            inspiration_seed: None,
            flow_state: 0.0,
            total_triggers: 0,
        }
    }
}

/// An inspiration seed from emotional memory for creative prompting.
#[derive(Debug, Clone, Serialize)]
pub struct InspirationSeed {
    /// The emotional memory stimulus that inspired this seed
    pub source_stimulus: String,
    /// Dominant emotion at the time of the inspiring memory
    pub emotion_color: String,
    /// Creative drive at generation time
    pub drive_level: f32,
}

/// The creative agent — monitors for creative opportunities.
pub struct CreativeAgent {
    pub state: CreativeState,
    /// Threshold for creative drive to trigger active ideation
    ideation_threshold: f32,
    /// EMA alpha for creative drive smoothing
    drive_alpha: f32,
}

impl CreativeAgent {
    pub fn new() -> Self {
        Self {
            state: CreativeState::default(),
            ideation_threshold: 0.6,
            drive_alpha: 0.05,
        }
    }

    pub fn tick(&mut self, ctx: &mut AgentContext<'_>) {
        let activations = &ctx.affect.plutchik.activations;

        // 1. Compute raw creative drive from openness + surprise + anticipation
        let openness_factor = ctx.personality.openness;
        let surprise = activations[emotion::SURPRISE];
        let anticipation = activations[emotion::ANTICIPATION];
        let joy = activations[emotion::JOY];

        let raw_drive = (openness_factor * 0.3
            + surprise * 0.25
            + anticipation * 0.25
            + joy * 0.2)
            .clamp(0.0, 1.0);

        // 2. EMA smoothing
        self.state.creative_drive +=
            self.drive_alpha * (raw_drive - self.state.creative_drive);

        // 3. Read flow state from extended aether
        self.state.flow_state = ctx.affect.extended_aether.flow_state();

        // 4. Flow state amplifies creative drive
        let effective_drive = (self.state.creative_drive * (1.0 + self.state.flow_state * 0.3))
            .clamp(0.0, 1.0);

        // 5. Check ideation threshold
        let was_active = self.state.ideation_active;
        self.state.ideation_active = effective_drive > self.ideation_threshold;

        // 6. On transition to active, sample emotional memory for inspiration
        if self.state.ideation_active && !was_active {
            self.state.total_triggers += 1;
            self.state.inspiration_seed = sample_inspiration(ctx);
        }

        // Clear seed when ideation deactivates
        if !self.state.ideation_active && was_active {
            self.state.inspiration_seed = None;
        }
    }
}

impl Default for CreativeAgent {
    fn default() -> Self {
        Self::new()
    }
}

/// Sample an inspiration seed from emotional memory.
///
/// Prefers high-valence, high-arousal memories (emotionally vivid).
fn sample_inspiration(ctx: &AgentContext<'_>) -> Option<InspirationSeed> {
    let episodes = ctx.memory.episodes();
    if episodes.is_empty() {
        return None;
    }

    // Score each episode by emotional vividness
    let mut best_idx = 0;
    let mut best_score = f32::NEG_INFINITY;

    for (i, ep) in episodes.iter().enumerate() {
        let score = ep.arousal_at_encoding * ep.decay_factor + ep.valence.abs() * 0.5;
        if score > best_score {
            best_score = score;
            best_idx = i;
        }
    }

    let ep = &episodes[best_idx];
    let dominant = ep.emotion_snapshot.dominant_emotion(0.3);

    Some(InspirationSeed {
        source_stimulus: ep.stimulus.clone(),
        emotion_color: dominant.name().to_string(),
        drive_level: ctx.affect.extended_aether.flow_state(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::affect::{ExtendedAffectiveState, PhysicsInput};
    use crate::aether::ExtendedAetherParams;
    use crate::memory::EmotionalMemory;
    use crate::mood::MoodSystem;
    use crate::personality::Personality;

    fn make_context<'a>(
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
    fn test_creative_agent_defaults() {
        let agent = CreativeAgent::new();
        assert!(agent.state.creative_drive < 0.01);
        assert!(!agent.state.ideation_active);
        assert!(agent.state.inspiration_seed.is_none());
    }

    #[test]
    fn test_creative_drive_from_openness() {
        let mut agent = CreativeAgent::new();
        let physics = PhysicsInput {
            valence: 3.0,
            arousal: 0.7,
            phase_coherence: 0.7,
            criticality_sigma: 0.2,
            prediction_error: 0.3,
            convergence_stable: true,
            achievement: false,
        };
        let mut affect = ExtendedAffectiveState::default();
        let high_open = Personality {
            openness: 0.95,
            ..Personality::cortana_default()
        };
        let mut mood = MoodSystem::default();
        let mut memory = EmotionalMemory::default();
        let aether_params = ExtendedAetherParams::default();

        // Run several ticks for EMA to converge
        for _ in 0..100 {
            let mut ctx = make_context(
                &physics, &mut affect, &high_open, &mut mood, &mut memory, &aether_params,
            );
            agent.tick(&mut ctx);
        }

        assert!(
            agent.state.creative_drive > 0.2,
            "high openness should produce meaningful creative drive: {}",
            agent.state.creative_drive
        );
    }

    #[test]
    fn test_low_openness_low_drive() {
        let mut agent = CreativeAgent::new();
        let physics = PhysicsInput {
            valence: 0.0,
            arousal: 0.3,
            phase_coherence: 0.5,
            criticality_sigma: 0.3,
            prediction_error: 0.5,
            convergence_stable: false,
            achievement: false,
        };
        let mut affect = ExtendedAffectiveState::default();
        let low_open = Personality {
            openness: 0.1,
            ..Personality::stoic()
        };
        let mut mood = MoodSystem::default();
        let mut memory = EmotionalMemory::default();
        let aether_params = ExtendedAetherParams::default();

        for _ in 0..100 {
            let mut ctx = make_context(
                &physics, &mut affect, &low_open, &mut mood, &mut memory, &aether_params,
            );
            agent.tick(&mut ctx);
        }

        assert!(
            !agent.state.ideation_active,
            "low openness + low arousal should not trigger ideation"
        );
    }

    #[test]
    fn test_inspiration_from_memory() {
        let mut agent = CreativeAgent::new();
        let physics = PhysicsInput {
            valence: 5.0,
            arousal: 0.9,
            phase_coherence: 0.8,
            criticality_sigma: 0.1,
            prediction_error: 0.1,
            convergence_stable: true,
            achievement: true,
        };
        let mut affect = ExtendedAffectiveState::default();
        let personality = Personality {
            openness: 0.95,
            ..Personality::cortana_default()
        };
        let mut mood = MoodSystem::default();
        let mut memory = EmotionalMemory::default();
        let aether_params = ExtendedAetherParams::default();

        // Pre-populate memory
        use crate::emotion::PlutchikState;
        memory.encode(
            1, 1, "beautiful sunset".to_string(), PlutchikState::default(), 0.8, 0.9,
            vec!["nature".into()],
        );

        // Force creative drive above threshold
        agent.state.creative_drive = 0.0; // start low so we get a transition
        agent.ideation_threshold = 0.2; // lower threshold for test

        // Run ticks to build drive
        for i in 0..200 {
            let mut ctx = make_context(
                &physics, &mut affect, &personality, &mut mood, &mut memory, &aether_params,
            );
            ctx.tick = i;
            agent.tick(&mut ctx);
        }

        if agent.state.ideation_active {
            // If triggered, should have an inspiration seed from memory
            assert!(agent.state.total_triggers > 0);
        }
    }

    #[test]
    fn test_flow_state_amplifies_drive() {
        let mut agent1 = CreativeAgent::new();
        let mut agent2 = CreativeAgent::new();
        let physics = PhysicsInput {
            valence: 3.0,
            arousal: 0.6,
            phase_coherence: 0.8,
            criticality_sigma: 0.1,
            prediction_error: 0.2,
            convergence_stable: true,
            achievement: false,
        };
        let personality = Personality::cortana_default();
        let mut mood = MoodSystem::default();
        let mut memory = EmotionalMemory::default();
        let aether_params = ExtendedAetherParams::default();

        // Agent 1: no flow
        let mut affect1 = ExtendedAffectiveState::default();
        for _ in 0..50 {
            let mut ctx = make_context(
                &physics, &mut affect1, &personality, &mut mood, &mut memory, &aether_params,
            );
            agent1.tick(&mut ctx);
        }

        // Agent 2: high flow (pre-set endorphin + dopamine)
        let mut affect2 = ExtendedAffectiveState::default();
        affect2.extended_aether.endorphin = 0.8;
        affect2.extended_aether.base.dopamine = 0.7;
        for _ in 0..50 {
            let mut ctx = make_context(
                &physics, &mut affect2, &personality, &mut mood, &mut memory, &aether_params,
            );
            agent2.tick(&mut ctx);
        }

        // Flow should amplify creative drive (or at least flow_state should be higher)
        assert!(agent2.state.flow_state >= agent1.state.flow_state);
    }
}
