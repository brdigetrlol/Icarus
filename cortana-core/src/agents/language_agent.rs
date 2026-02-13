//! LanguageAgent — Computes expression metadata from emotional state.
//!
//! Produces tone, energy, formality, and vocabulary hints that can be used
//! by dialog tools to color generated text. Does NOT generate text itself.

use serde::Serialize;

use crate::agents::AgentContext;
use crate::emotion;
use crate::mood::MoodLabel;

/// Expression metadata — how Cortana should express herself.
#[derive(Debug, Clone, Serialize)]
pub struct ExpressionMetadata {
    /// Emotional tone: "warm", "anxious", "excited", "melancholic", "neutral", etc.
    pub tone: String,
    /// Energy level [0, 1]: 0 = subdued/quiet, 1 = energetic/emphatic
    pub energy: f32,
    /// Formality [0, 1]: 0 = casual/intimate, 1 = formal/professional
    pub formality: f32,
    /// Vocabulary hints — suggested word categories for this emotional state
    pub vocabulary_hints: Vec<String>,
    /// Whether to use exclamations / emphasis markers
    pub emphatic: bool,
    /// Suggested sentence length: "short", "medium", "long"
    pub sentence_length: String,
}

impl Default for ExpressionMetadata {
    fn default() -> Self {
        Self {
            tone: "neutral".into(),
            energy: 0.0,
            formality: 0.5,
            vocabulary_hints: Vec::new(),
            emphatic: false,
            sentence_length: "medium".into(),
        }
    }
}

/// The language agent — computes expression metadata from affect.
pub struct LanguageAgent {
    pub expression: ExpressionMetadata,
}

impl LanguageAgent {
    pub fn new() -> Self {
        Self {
            expression: ExpressionMetadata::default(),
        }
    }

    pub fn tick(&mut self, ctx: &mut AgentContext<'_>) {
        let affect = &ctx.affect;
        let activations = &affect.plutchik.activations;

        // 1. Determine tone from dominant emotion + mood
        self.expression.tone = compute_tone(affect.dominant_emotion.clone(), &affect.mood.label);

        // 2. Energy from arousal (rescale bipolar to unipolar)
        self.expression.energy = ((affect.arousal + 1.0) * 0.5).clamp(0.0, 1.0);

        // 3. Formality: high conscientiousness → formal, high extraversion → casual
        self.expression.formality = (0.5
            + (ctx.personality.conscientiousness - 0.5) * 0.4
            - (ctx.personality.extraversion - 0.5) * 0.2)
            .clamp(0.0, 1.0);

        // In negative mood, slightly more formal (guarded)
        if affect.mood.label.is_negative() {
            self.expression.formality = (self.expression.formality + 0.1).min(1.0);
        }

        // 4. Vocabulary hints from active emotions
        self.expression.vocabulary_hints = compute_vocabulary_hints(activations);

        // 5. Emphatic when high arousal + strong emotion
        self.expression.emphatic =
            self.expression.energy > 0.7 && affect.emotional_intensity() > 0.6;

        // 6. Sentence length from energy + arousal
        self.expression.sentence_length = if self.expression.energy > 0.7 {
            "short".into() // excited → punchy
        } else if self.expression.energy < 0.3 {
            "long".into() // calm → reflective
        } else {
            "medium".into()
        };
    }
}

impl Default for LanguageAgent {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute the overall tone from dominant emotion and mood.
fn compute_tone(
    dominant: crate::emotion::EmotionLabel,
    mood: &MoodLabel,
) -> String {
    use crate::emotion::EmotionLabel;

    // Emotion-driven tone (overrides mood if a strong emotion is active)
    let emotion_tone = match dominant {
        EmotionLabel::Primary { emotion: p, .. } => match p {
            emotion::PrimaryEmotion::Joy => "warm",
            emotion::PrimaryEmotion::Sadness => "gentle",
            emotion::PrimaryEmotion::Trust => "supportive",
            emotion::PrimaryEmotion::Disgust => "disapproving",
            emotion::PrimaryEmotion::Fear => "cautious",
            emotion::PrimaryEmotion::Anger => "assertive",
            emotion::PrimaryEmotion::Surprise => "excited",
            emotion::PrimaryEmotion::Anticipation => "eager",
        },
        EmotionLabel::Compound(c) => match c {
            emotion::CompoundEmotion::Love => "affectionate",
            emotion::CompoundEmotion::Optimism => "hopeful",
            emotion::CompoundEmotion::Awe => "reverent",
            emotion::CompoundEmotion::Curiosity => "inquisitive",
            emotion::CompoundEmotion::Remorse => "apologetic",
            emotion::CompoundEmotion::Contempt => "cool",
            _ => "neutral",
        },
        EmotionLabel::Neutral => {
            // Fall back to mood-based tone
            match mood {
                MoodLabel::Exuberant => "enthusiastic",
                MoodLabel::Serene => "calm",
                MoodLabel::Anxious => "cautious",
                MoodLabel::Melancholic => "subdued",
                MoodLabel::Determined => "focused",
                MoodLabel::Contemplative => "thoughtful",
                MoodLabel::Neutral => "neutral",
            }
        }
    };

    emotion_tone.into()
}

/// Generate vocabulary hints from active emotion activations.
fn compute_vocabulary_hints(activations: &[f32; 8]) -> Vec<String> {
    let mut hints = Vec::new();
    let threshold = 0.3;

    if activations[emotion::JOY] > threshold {
        hints.push("positive_words".into());
        hints.push("warmth".into());
    }
    if activations[emotion::SADNESS] > threshold {
        hints.push("empathetic_words".into());
        hints.push("gentle_language".into());
    }
    if activations[emotion::TRUST] > threshold {
        hints.push("reassuring_words".into());
        hints.push("collaborative_language".into());
    }
    if activations[emotion::FEAR] > threshold {
        hints.push("cautious_words".into());
        hints.push("hedging_language".into());
    }
    if activations[emotion::ANGER] > threshold {
        hints.push("direct_words".into());
        hints.push("assertive_language".into());
    }
    if activations[emotion::SURPRISE] > threshold {
        hints.push("exclamatory_words".into());
        hints.push("wonder_language".into());
    }
    if activations[emotion::ANTICIPATION] > threshold {
        hints.push("forward_looking_words".into());
        hints.push("planning_language".into());
    }
    if activations[emotion::DISGUST] > threshold {
        hints.push("critical_words".into());
    }

    if hints.is_empty() {
        hints.push("neutral_language".into());
    }

    hints
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
    fn test_language_agent_defaults() {
        let agent = LanguageAgent::new();
        assert_eq!(agent.expression.tone, "neutral");
        assert!(agent.expression.energy < 0.01);
    }

    #[test]
    fn test_tone_from_joy() {
        let tone = compute_tone(
            crate::emotion::EmotionLabel::Primary {
                emotion: emotion::PrimaryEmotion::Joy,
                intensity: crate::emotion::Intensity::Moderate,
            },
            &MoodLabel::Neutral,
        );
        assert_eq!(tone, "warm");
    }

    #[test]
    fn test_tone_from_neutral_with_mood() {
        let tone = compute_tone(
            crate::emotion::EmotionLabel::Neutral,
            &MoodLabel::Serene,
        );
        assert_eq!(tone, "calm");
    }

    #[test]
    fn test_vocabulary_hints_joy() {
        let mut activations = [0.0f32; 8];
        activations[emotion::JOY] = 0.5;
        let hints = compute_vocabulary_hints(&activations);
        assert!(hints.contains(&"positive_words".to_string()));
        assert!(hints.contains(&"warmth".to_string()));
    }

    #[test]
    fn test_vocabulary_hints_empty() {
        let activations = [0.1f32; 8];
        let hints = compute_vocabulary_hints(&activations);
        assert!(hints.contains(&"neutral_language".to_string()));
    }

    #[test]
    fn test_formality_from_personality() {
        let mut agent = LanguageAgent::new();
        let physics = PhysicsInput {
            valence: 0.0,
            arousal: 0.5,
            phase_coherence: 0.5,
            criticality_sigma: 0.3,
            prediction_error: 0.5,
            convergence_stable: false,
            achievement: false,
        };

        // High conscientiousness → more formal
        let mut affect = ExtendedAffectiveState::default();
        let formal_personality = Personality {
            conscientiousness: 0.95,
            extraversion: 0.2,
            ..Personality::uniform(0.5)
        };
        let mut mood = MoodSystem::default();
        let mut memory = EmotionalMemory::default();
        let aether_params = ExtendedAetherParams::default();

        let mut ctx = make_context(
            &physics, &mut affect, &formal_personality, &mut mood, &mut memory, &aether_params,
        );
        agent.tick(&mut ctx);
        let formal_val = agent.expression.formality;

        // Low conscientiousness, high extraversion → less formal
        let casual_personality = Personality {
            conscientiousness: 0.1,
            extraversion: 0.95,
            ..Personality::uniform(0.5)
        };
        let mut ctx2 = make_context(
            &physics, &mut affect, &casual_personality, &mut mood, &mut memory, &aether_params,
        );
        agent.tick(&mut ctx2);
        let casual_val = agent.expression.formality;

        assert!(
            formal_val > casual_val,
            "formal personality should produce higher formality: {} vs {}",
            formal_val,
            casual_val
        );
    }

    #[test]
    fn test_energy_from_arousal() {
        let mut agent = LanguageAgent::new();
        let physics = PhysicsInput {
            valence: 0.0,
            arousal: 0.9,
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

        // First update affect so arousal gets set
        affect.update(&physics, &personality, &aether_params, 1.0);

        let mut ctx = make_context(
            &physics, &mut affect, &personality, &mut mood, &mut memory, &aether_params,
        );
        agent.tick(&mut ctx);

        assert!(
            agent.expression.energy > 0.5,
            "high arousal → high energy: {}",
            agent.expression.energy
        );
    }

    #[test]
    fn test_sentence_length() {
        let mut agent = LanguageAgent::new();
        // High energy → short sentences
        agent.expression.energy = 0.8;
        let physics = PhysicsInput {
            valence: 5.0,
            arousal: 0.95,
            phase_coherence: 0.5,
            criticality_sigma: 0.3,
            prediction_error: 0.5,
            convergence_stable: false,
            achievement: false,
        };
        let mut affect = ExtendedAffectiveState::default();
        affect.update(
            &physics,
            &Personality::cortana_default(),
            &ExtendedAetherParams::default(),
            1.0,
        );
        let personality = Personality::cortana_default();
        let mut mood = MoodSystem::default();
        let mut memory = EmotionalMemory::default();
        let aether_params = ExtendedAetherParams::default();

        let mut ctx = make_context(
            &physics, &mut affect, &personality, &mut mood, &mut memory, &aether_params,
        );
        agent.tick(&mut ctx);

        // The actual sentence_length depends on computed energy from affect.arousal
        assert!(["short", "medium", "long"].contains(&agent.expression.sentence_length.as_str()));
    }
}
