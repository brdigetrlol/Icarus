//! Mood system — slow-moving emotional baseline that biases emotion dynamics.
//!
//! Mood operates on a much longer timescale than emotions (~1000 ticks EMA)
//! and drifts toward neutral via homeostasis (~5000 ticks). Current mood
//! biases emotion classification thresholds: negative mood lowers thresholds
//! for negative emotions.

use serde::{Deserialize, Serialize};

/// Discrete mood labels classified from PAD dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MoodLabel {
    /// High pleasure + high arousal + high dominance — energetic joy
    Exuberant,
    /// High pleasure + low arousal — calm contentment
    Serene,
    /// Low pleasure + high arousal + low dominance — worried/stressed
    Anxious,
    /// Low pleasure + low arousal — withdrawn/sad
    Melancholic,
    /// High dominance + moderate pleasure + high arousal — driven/focused
    Determined,
    /// Moderate pleasure + low arousal + moderate dominance — reflective
    Contemplative,
    /// All dimensions near zero — baseline
    Neutral,
}

impl MoodLabel {
    /// Human-readable name for the mood.
    pub fn name(self) -> &'static str {
        match self {
            Self::Exuberant => "exuberant",
            Self::Serene => "serene",
            Self::Anxious => "anxious",
            Self::Melancholic => "melancholic",
            Self::Determined => "determined",
            Self::Contemplative => "contemplative",
            Self::Neutral => "neutral",
        }
    }

    /// Whether this mood is generally positive.
    pub fn is_positive(self) -> bool {
        matches!(self, Self::Exuberant | Self::Serene | Self::Determined)
    }

    /// Whether this mood is generally negative.
    pub fn is_negative(self) -> bool {
        matches!(self, Self::Anxious | Self::Melancholic)
    }
}

/// Mood state — slow EMA of PAD dimensions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoodState {
    /// Hedonic tone — slow EMA of pleasure [-1, 1].
    pub hedonic_tone: f32,
    /// Tense energy — slow EMA of arousal [-1, 1].
    pub tense_energy: f32,
    /// Confidence — slow EMA of dominance [-1, 1].
    pub confidence: f32,
    /// Current mood classification.
    pub label: MoodLabel,
}

impl Default for MoodState {
    fn default() -> Self {
        Self {
            hedonic_tone: 0.0,
            tense_energy: 0.0,
            confidence: 0.0,
            label: MoodLabel::Neutral,
        }
    }
}

/// Configuration for the mood system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoodConfig {
    /// EMA time constant in ticks. Higher = slower mood changes.
    pub tau: f32,
    /// Homeostasis rate — drift toward neutral per tick.
    /// Effective rate = homeostasis_rate / tau_homeostasis.
    pub homeostasis_rate: f32,
    /// Mood-emotion bias strength [0, 1].
    /// How strongly current mood biases emotion thresholds.
    pub mood_emotion_bias: f32,
}

impl Default for MoodConfig {
    fn default() -> Self {
        Self {
            tau: 1000.0,
            homeostasis_rate: 0.0002,
            mood_emotion_bias: 0.3,
        }
    }
}

/// Mood system that tracks and updates mood over time.
#[derive(Debug, Clone)]
pub struct MoodSystem {
    pub state: MoodState,
    pub config: MoodConfig,
}

impl MoodSystem {
    pub fn new(config: MoodConfig) -> Self {
        Self {
            state: MoodState::default(),
            config,
        }
    }

    /// Update mood state from current PAD values.
    ///
    /// Uses exponential moving average with time constant `tau`.
    /// Also applies homeostasis drift toward neutral.
    pub fn update(&mut self, pleasure: f32, arousal: f32, dominance: f32) {
        let alpha = 1.0 / self.config.tau;

        // EMA update
        self.state.hedonic_tone += alpha * (pleasure - self.state.hedonic_tone);
        self.state.tense_energy += alpha * (arousal - self.state.tense_energy);
        self.state.confidence += alpha * (dominance - self.state.confidence);

        // Homeostasis: drift toward zero (neutral)
        let hr = self.config.homeostasis_rate;
        self.state.hedonic_tone *= 1.0 - hr;
        self.state.tense_energy *= 1.0 - hr;
        self.state.confidence *= 1.0 - hr;

        // Clamp
        self.state.hedonic_tone = self.state.hedonic_tone.clamp(-1.0, 1.0);
        self.state.tense_energy = self.state.tense_energy.clamp(-1.0, 1.0);
        self.state.confidence = self.state.confidence.clamp(-1.0, 1.0);

        // Reclassify mood
        self.state.label = classify_mood(
            self.state.hedonic_tone,
            self.state.tense_energy,
            self.state.confidence,
        );
    }

    /// Get the mood-induced bias for negative emotion thresholds.
    ///
    /// Negative mood lowers the threshold for negative emotions (easier to trigger),
    /// positive mood raises it (harder to trigger).
    /// Returns a multiplier in [1 - bias, 1 + bias].
    pub fn negative_emotion_threshold_bias(&self) -> f32 {
        // Negative hedonic tone → lower threshold (multiplier < 1)
        // Positive hedonic tone → higher threshold (multiplier > 1)
        1.0 + self.state.hedonic_tone * self.config.mood_emotion_bias
    }

    /// Get the mood-induced bias for positive emotion thresholds.
    /// Inverse of negative: positive mood lowers positive thresholds.
    pub fn positive_emotion_threshold_bias(&self) -> f32 {
        1.0 - self.state.hedonic_tone * self.config.mood_emotion_bias
    }

    /// Get a mood-congruent recall bias for emotional memory.
    ///
    /// Returns a preference weight for memories matching current mood valence.
    /// Positive mood → prefer positive memories, negative → prefer negative.
    pub fn recall_bias(&self) -> f32 {
        self.state.hedonic_tone * self.config.mood_emotion_bias
    }
}

impl Default for MoodSystem {
    fn default() -> Self {
        Self::new(MoodConfig::default())
    }
}

/// Classify mood from PAD dimensions.
fn classify_mood(hedonic: f32, energy: f32, confidence: f32) -> MoodLabel {
    let h_pos = hedonic > 0.15;
    let h_neg = hedonic < -0.15;
    let e_high = energy > 0.15;
    let e_low = energy < -0.15;
    let c_high = confidence > 0.15;
    let c_low = confidence < -0.15;

    match (h_pos, h_neg, e_high, e_low, c_high, c_low) {
        // Exuberant: happy + energized + confident
        (true, _, true, _, true, _) => MoodLabel::Exuberant,
        // Determined: confident + energized (dominance-driven)
        (_, _, true, _, true, _) if !h_neg => MoodLabel::Determined,
        // Serene: happy + calm
        (true, _, _, true, _, _) => MoodLabel::Serene,
        // Serene: happy + moderate
        (true, _, false, false, _, _) => MoodLabel::Serene,
        // Anxious: unhappy + energized + not confident
        (_, true, true, _, _, true) => MoodLabel::Anxious,
        // Anxious: unhappy + energized
        (_, true, true, _, _, _) => MoodLabel::Anxious,
        // Melancholic: unhappy + low energy
        (_, true, _, true, _, _) => MoodLabel::Melancholic,
        // Melancholic: unhappy + moderate energy
        (_, true, _, _, _, _) => MoodLabel::Melancholic,
        // Contemplative: calm + moderate hedonic
        (_, _, _, true, _, _) if !h_neg => MoodLabel::Contemplative,
        // Default
        _ => MoodLabel::Neutral,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_mood() {
        let mood = MoodState::default();
        assert_eq!(mood.label, MoodLabel::Neutral);
        assert!(mood.hedonic_tone.abs() < f32::EPSILON);
    }

    #[test]
    fn test_classify_exuberant() {
        assert_eq!(classify_mood(0.5, 0.5, 0.5), MoodLabel::Exuberant);
    }

    #[test]
    fn test_classify_serene() {
        assert_eq!(classify_mood(0.5, -0.5, 0.0), MoodLabel::Serene);
    }

    #[test]
    fn test_classify_anxious() {
        assert_eq!(classify_mood(-0.5, 0.5, -0.5), MoodLabel::Anxious);
    }

    #[test]
    fn test_classify_melancholic() {
        assert_eq!(classify_mood(-0.5, -0.5, 0.0), MoodLabel::Melancholic);
    }

    #[test]
    fn test_classify_determined() {
        assert_eq!(classify_mood(0.0, 0.5, 0.5), MoodLabel::Determined);
    }

    #[test]
    fn test_classify_contemplative() {
        assert_eq!(classify_mood(0.0, -0.5, 0.0), MoodLabel::Contemplative);
    }

    #[test]
    fn test_classify_neutral() {
        assert_eq!(classify_mood(0.0, 0.0, 0.0), MoodLabel::Neutral);
    }

    #[test]
    fn test_mood_ema_update() {
        let mut system = MoodSystem::new(MoodConfig {
            tau: 10.0, // fast for testing
            homeostasis_rate: 0.0,
            mood_emotion_bias: 0.3,
        });

        // Sustained positive input
        for _ in 0..100 {
            system.update(0.8, 0.0, 0.0);
        }
        assert!(system.state.hedonic_tone > 0.5, "should converge toward 0.8");
        assert!(system.state.label.is_positive());
    }

    #[test]
    fn test_mood_homeostasis() {
        let mut system = MoodSystem::new(MoodConfig {
            tau: 10.0,
            homeostasis_rate: 0.05, // strong for testing
            mood_emotion_bias: 0.3,
        });

        // Push to extreme
        system.state.hedonic_tone = 0.9;
        system.state.tense_energy = 0.9;
        system.state.confidence = 0.9;

        // Update with zero input — should drift toward neutral
        for _ in 0..200 {
            system.update(0.0, 0.0, 0.0);
        }

        assert!(
            system.state.hedonic_tone.abs() < 0.1,
            "should drift toward 0, got {}",
            system.state.hedonic_tone
        );
    }

    #[test]
    fn test_negative_emotion_threshold_bias() {
        let mut system = MoodSystem::default();

        // Neutral mood → bias ≈ 1.0
        assert!((system.negative_emotion_threshold_bias() - 1.0).abs() < 0.01);

        // Negative mood → lower threshold (< 1.0, easier to trigger)
        system.state.hedonic_tone = -0.8;
        assert!(system.negative_emotion_threshold_bias() < 1.0);

        // Positive mood → higher threshold (> 1.0, harder to trigger)
        system.state.hedonic_tone = 0.8;
        assert!(system.negative_emotion_threshold_bias() > 1.0);
    }

    #[test]
    fn test_mood_clamped() {
        let mut system = MoodSystem::new(MoodConfig {
            tau: 1.0, // very fast
            homeostasis_rate: 0.0,
            mood_emotion_bias: 0.3,
        });

        for _ in 0..10000 {
            system.update(10.0, 10.0, 10.0);
        }

        assert!((-1.0..=1.0).contains(&system.state.hedonic_tone));
        assert!((-1.0..=1.0).contains(&system.state.tense_energy));
        assert!((-1.0..=1.0).contains(&system.state.confidence));
    }

    #[test]
    fn test_mood_label_properties() {
        assert!(MoodLabel::Exuberant.is_positive());
        assert!(MoodLabel::Serene.is_positive());
        assert!(MoodLabel::Determined.is_positive());
        assert!(!MoodLabel::Anxious.is_positive());
        assert!(!MoodLabel::Melancholic.is_positive());
        assert!(MoodLabel::Anxious.is_negative());
        assert!(MoodLabel::Melancholic.is_negative());
        assert!(!MoodLabel::Neutral.is_positive());
        assert!(!MoodLabel::Neutral.is_negative());
    }

    #[test]
    fn test_recall_bias() {
        let mut system = MoodSystem::default();
        system.state.hedonic_tone = 0.5;
        assert!(system.recall_bias() > 0.0);

        system.state.hedonic_tone = -0.5;
        assert!(system.recall_bias() < 0.0);
    }
}
