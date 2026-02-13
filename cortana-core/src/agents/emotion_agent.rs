//! EmotionAgent — Maps EMC physics to Plutchik activations, applies personality
//! modulation, detects compound emotions, updates PAD + extended aether + mood,
//! and gates memory encoding.

use crate::agents::AgentContext;

/// The emotion agent — core affect processing pipeline.
///
/// Runs each tick to:
/// 1. Update ExtendedAffectiveState from physics
/// 2. Update mood from PAD dimensions
/// 3. Gate emotional memory encoding (high-arousal events only)
pub struct EmotionAgent {
    /// Rolling tick counter for memory encoding timestamps
    _tick_count: u64,
}

impl EmotionAgent {
    pub fn new() -> Self {
        Self { _tick_count: 0 }
    }

    /// Process one tick of emotion dynamics.
    pub fn tick(&mut self, ctx: &mut AgentContext<'_>) {
        // 1. Update the full extended affective state from physics
        ctx.affect
            .update(ctx.physics, ctx.personality, ctx.aether_params, ctx.dt);

        // 2. Update mood from current PAD dimensions
        let (p, a, d) = ctx.affect.pad();
        ctx.mood.update(p, a, d);

        // 3. Store mood state in affect for unified access
        ctx.affect.mood = ctx.mood.state.clone();

        // 4. Gate memory encoding — only encode if arousal is high enough
        let arousal_bipolar = ctx.affect.arousal;
        let arousal_unipolar = (arousal_bipolar + 1.0) * 0.5; // rescale [-1,1] to [0,1]

        if arousal_unipolar >= 0.6 {
            let stimulus = format!(
                "tick:{} emotion:{} mood:{}",
                ctx.tick,
                ctx.affect.dominant_emotion.name(),
                ctx.affect.mood.label.name(),
            );

            let mut tags = vec![ctx.affect.dominant_emotion.name().to_string()];
            if ctx.affect.mood.label.is_negative() {
                tags.push("negative_mood".into());
            }
            if ctx.affect.mood.label.is_positive() {
                tags.push("positive_mood".into());
            }
            if ctx.physics.achievement {
                tags.push("achievement".into());
            }

            ctx.memory.encode(
                ctx.tick,
                ctx.tick, // use tick as timestamp proxy
                stimulus,
                ctx.affect.plutchik.clone(),
                ctx.affect.pleasure,
                arousal_unipolar,
                tags,
            );
        }

        self._tick_count += 1;
    }
}

impl Default for EmotionAgent {
    fn default() -> Self {
        Self::new()
    }
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
        tick: u64,
    ) -> AgentContext<'a> {
        AgentContext {
            physics,
            affect,
            personality,
            mood,
            memory,
            aether_params,
            tick,
            dt: 1.0,
        }
    }

    #[test]
    fn test_emotion_agent_updates_affect() {
        let mut agent = EmotionAgent::new();
        let physics = PhysicsInput {
            valence: 5.0,
            arousal: 0.7,
            phase_coherence: 0.8,
            criticality_sigma: 0.1,
            prediction_error: 0.2,
            convergence_stable: true,
            achievement: false,
        };
        let mut affect = ExtendedAffectiveState::default();
        let personality = Personality::cortana_default();
        let mut mood = MoodSystem::default();
        let mut memory = EmotionalMemory::default();
        let aether_params = ExtendedAetherParams::default();

        let mut ctx = make_context(
            &physics, &mut affect, &personality, &mut mood, &mut memory, &aether_params, 1,
        );
        agent.tick(&mut ctx);

        assert!(ctx.affect.pleasure > 0.0, "positive valence → positive pleasure");
    }

    #[test]
    fn test_emotion_agent_updates_mood() {
        let mut agent = EmotionAgent::new();
        let physics = PhysicsInput {
            valence: 8.0,
            arousal: 0.3,
            phase_coherence: 0.9,
            criticality_sigma: 0.05,
            prediction_error: 0.1,
            convergence_stable: true,
            achievement: false,
        };
        let mut affect = ExtendedAffectiveState::default();
        let personality = Personality::cortana_default();
        let mut mood = MoodSystem::default();
        let mut memory = EmotionalMemory::default();
        let aether_params = ExtendedAetherParams::default();

        // Run many ticks to let mood converge
        for i in 0..500 {
            let mut ctx = make_context(
                &physics, &mut affect, &personality, &mut mood, &mut memory, &aether_params, i,
            );
            agent.tick(&mut ctx);
        }

        assert!(mood.state.hedonic_tone > 0.0, "sustained positive → positive mood");
    }

    #[test]
    fn test_emotion_agent_gates_memory() {
        let mut agent = EmotionAgent::new();

        // Low arousal — should NOT encode
        let low_physics = PhysicsInput {
            valence: 1.0,
            arousal: 0.1,
            phase_coherence: 0.5,
            criticality_sigma: 0.3,
            prediction_error: 0.5,
            convergence_stable: false,
            achievement: false,
        };
        let mut affect = ExtendedAffectiveState::default();
        let personality = Personality::cortana_default();
        let mut mood = MoodSystem::default();
        let mut memory = EmotionalMemory::default();
        let aether_params = ExtendedAetherParams::default();

        let mut ctx = make_context(
            &low_physics, &mut affect, &personality, &mut mood, &mut memory, &aether_params, 1,
        );
        agent.tick(&mut ctx);

        let count_after_low = ctx.memory.episodes().len();

        // High arousal — should encode
        let high_physics = PhysicsInput {
            valence: 5.0,
            arousal: 0.95,
            phase_coherence: 0.8,
            criticality_sigma: 0.1,
            prediction_error: 0.1,
            convergence_stable: true,
            achievement: true,
        };
        let mut ctx = make_context(
            &high_physics, &mut affect, &personality, &mut mood, &mut memory, &aether_params, 2,
        );
        agent.tick(&mut ctx);

        let count_after_high = ctx.memory.episodes().len();
        assert!(
            count_after_high > count_after_low,
            "high arousal event should be encoded: {} vs {}",
            count_after_high,
            count_after_low
        );
    }

    #[test]
    fn test_emotion_agent_achievement_tagged() {
        let mut agent = EmotionAgent::new();
        let physics = PhysicsInput {
            valence: 5.0,
            arousal: 0.95,
            phase_coherence: 0.8,
            criticality_sigma: 0.1,
            prediction_error: 0.1,
            convergence_stable: true,
            achievement: true,
        };
        let mut affect = ExtendedAffectiveState::default();
        let personality = Personality::cortana_default();
        let mut mood = MoodSystem::default();
        let mut memory = EmotionalMemory::default();
        let aether_params = ExtendedAetherParams::default();

        let mut ctx = make_context(
            &physics, &mut affect, &personality, &mut mood, &mut memory, &aether_params, 1,
        );
        agent.tick(&mut ctx);

        if let Some(last) = ctx.memory.episodes().last() {
            assert!(last.tags.contains(&"achievement".to_string()));
        }
    }
}
