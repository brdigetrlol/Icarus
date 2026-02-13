//! Big Five (OCEAN) personality model — stable traits that modulate emotional dynamics.

use serde::{Deserialize, Serialize};

/// Big Five personality traits. Each trait is a continuous value in [0.0, 1.0].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Personality {
    /// Openness to experience — curiosity, creativity, novelty-seeking.
    pub openness: f32,
    /// Conscientiousness — orderliness, self-discipline, planning.
    pub conscientiousness: f32,
    /// Extraversion — social energy, assertiveness, positive emotion bias.
    pub extraversion: f32,
    /// Agreeableness — empathy, cooperation, trust bias.
    pub agreeableness: f32,
    /// Neuroticism — emotional instability, anxiety, negative emotion bias.
    pub neuroticism: f32,
}

impl Personality {
    /// Create a new personality with all traits at the given value.
    pub fn uniform(value: f32) -> Self {
        let v = value.clamp(0.0, 1.0);
        Self {
            openness: v,
            conscientiousness: v,
            extraversion: v,
            agreeableness: v,
            neuroticism: v,
        }
    }

    /// Default Cortana personality — warm, curious, emotionally stable.
    pub fn cortana_default() -> Self {
        Self {
            openness: 0.85,
            conscientiousness: 0.70,
            extraversion: 0.75,
            agreeableness: 0.80,
            neuroticism: 0.25,
        }
    }

    /// Stoic personality — low emotional reactivity, high discipline.
    pub fn stoic() -> Self {
        Self {
            openness: 0.50,
            conscientiousness: 0.90,
            extraversion: 0.30,
            agreeableness: 0.60,
            neuroticism: 0.10,
        }
    }

    /// Creative personality — high openness, moderate neuroticism.
    pub fn creative() -> Self {
        Self {
            openness: 0.95,
            conscientiousness: 0.40,
            extraversion: 0.65,
            agreeableness: 0.70,
            neuroticism: 0.45,
        }
    }

    /// Anxious personality — high neuroticism, low extraversion.
    pub fn anxious() -> Self {
        Self {
            openness: 0.50,
            conscientiousness: 0.60,
            extraversion: 0.25,
            agreeableness: 0.65,
            neuroticism: 0.85,
        }
    }

    /// Clamp all traits to [0, 1].
    pub fn clamp(&mut self) {
        self.openness = self.openness.clamp(0.0, 1.0);
        self.conscientiousness = self.conscientiousness.clamp(0.0, 1.0);
        self.extraversion = self.extraversion.clamp(0.0, 1.0);
        self.agreeableness = self.agreeableness.clamp(0.0, 1.0);
        self.neuroticism = self.neuroticism.clamp(0.0, 1.0);
    }

    /// Get a trait by name (case-insensitive).
    pub fn get_trait(&self, name: &str) -> Option<f32> {
        match name.to_lowercase().as_str() {
            "openness" | "o" => Some(self.openness),
            "conscientiousness" | "c" => Some(self.conscientiousness),
            "extraversion" | "e" => Some(self.extraversion),
            "agreeableness" | "a" => Some(self.agreeableness),
            "neuroticism" | "n" => Some(self.neuroticism),
            _ => None,
        }
    }

    /// Set a trait by name (case-insensitive). Returns true if the trait was found.
    pub fn set_trait(&mut self, name: &str, value: f32) -> bool {
        let v = value.clamp(0.0, 1.0);
        match name.to_lowercase().as_str() {
            "openness" | "o" => {
                self.openness = v;
                true
            }
            "conscientiousness" | "c" => {
                self.conscientiousness = v;
                true
            }
            "extraversion" | "e" => {
                self.extraversion = v;
                true
            }
            "agreeableness" | "a" => {
                self.agreeableness = v;
                true
            }
            "neuroticism" | "n" => {
                self.neuroticism = v;
                true
            }
            _ => false,
        }
    }

    /// Compute the modulation factor for positive emotions.
    /// Extraversion amplifies positive emotions, neuroticism dampens them.
    pub fn positive_emotion_modulation(&self) -> f32 {
        1.0 + (self.extraversion - 0.5) * 0.4 - (self.neuroticism - 0.5) * 0.2
    }

    /// Compute the modulation factor for negative emotions.
    /// Neuroticism amplifies negative emotions, extraversion dampens them.
    pub fn negative_emotion_modulation(&self) -> f32 {
        1.0 + (self.neuroticism - 0.5) * 0.4 - (self.extraversion - 0.5) * 0.2
    }

    /// Modulation for trust-related emotions. Agreeableness raises trust.
    pub fn trust_modulation(&self) -> f32 {
        1.0 + (self.agreeableness - 0.5) * 0.4
    }

    /// Modulation for surprise/anticipation. Openness amplifies curiosity.
    pub fn novelty_modulation(&self) -> f32 {
        1.0 + (self.openness - 0.5) * 0.4
    }

    /// Modulation for mood stability. Conscientiousness stabilizes mood.
    pub fn mood_stability(&self) -> f32 {
        1.0 + (self.conscientiousness - 0.5) * 0.3 - (self.neuroticism - 0.5) * 0.3
    }

    /// Apply personality modulation to a Plutchik activation array in-place.
    /// `modulation_strength` controls how strongly personality biases emotions [0, 1].
    pub fn modulate_activations(&self, activations: &mut [f32; 8], modulation_strength: f32) {
        use crate::emotion::*;
        let s = modulation_strength.clamp(0.0, 1.0);

        // Joy: boosted by extraversion
        activations[JOY] *= 1.0 + s * (self.positive_emotion_modulation() - 1.0);
        // Sadness: boosted by neuroticism
        activations[SADNESS] *= 1.0 + s * (self.negative_emotion_modulation() - 1.0);
        // Trust: boosted by agreeableness
        activations[TRUST] *= 1.0 + s * (self.trust_modulation() - 1.0);
        // Disgust: inversely affected by agreeableness
        activations[DISGUST] *= 1.0 + s * (1.0 / self.trust_modulation().max(0.1) - 1.0);
        // Fear: boosted by neuroticism
        activations[FEAR] *= 1.0 + s * (self.negative_emotion_modulation() - 1.0);
        // Anger: mildly boosted by neuroticism, dampened by agreeableness
        activations[ANGER] *= 1.0
            + s * ((self.neuroticism - 0.5) * 0.3 - (self.agreeableness - 0.5) * 0.2);
        // Surprise: boosted by openness
        activations[SURPRISE] *= 1.0 + s * (self.novelty_modulation() - 1.0);
        // Anticipation: boosted by openness
        activations[ANTICIPATION] *= 1.0 + s * (self.novelty_modulation() - 1.0);

        // Clamp all to [0, 1]
        for v in activations.iter_mut() {
            *v = v.clamp(0.0, 1.0);
        }
    }
}

impl Default for Personality {
    fn default() -> Self {
        Self::cortana_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cortana_default_valid() {
        let p = Personality::cortana_default();
        assert!((0.0..=1.0).contains(&p.openness));
        assert!((0.0..=1.0).contains(&p.conscientiousness));
        assert!((0.0..=1.0).contains(&p.extraversion));
        assert!((0.0..=1.0).contains(&p.agreeableness));
        assert!((0.0..=1.0).contains(&p.neuroticism));
    }

    #[test]
    fn test_uniform() {
        let p = Personality::uniform(0.5);
        assert!((p.openness - 0.5).abs() < f32::EPSILON);
        assert!((p.neuroticism - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_uniform_clamping() {
        let p = Personality::uniform(2.0);
        assert!((p.openness - 1.0).abs() < f32::EPSILON);
        let p = Personality::uniform(-1.0);
        assert!(p.openness.abs() < f32::EPSILON);
    }

    #[test]
    fn test_get_set_trait() {
        let mut p = Personality::cortana_default();
        assert!(p.get_trait("openness").is_some());
        assert!((p.get_trait("o").unwrap() - 0.85).abs() < f32::EPSILON);

        assert!(p.set_trait("neuroticism", 0.9));
        assert!((p.neuroticism - 0.9).abs() < f32::EPSILON);

        assert!(!p.set_trait("invalid", 0.5));
        assert!(p.get_trait("invalid").is_none());
    }

    #[test]
    fn test_positive_emotion_modulation() {
        let extrovert = Personality {
            extraversion: 0.9,
            neuroticism: 0.1,
            ..Personality::uniform(0.5)
        };
        let introvert = Personality {
            extraversion: 0.1,
            neuroticism: 0.9,
            ..Personality::uniform(0.5)
        };
        assert!(extrovert.positive_emotion_modulation() > introvert.positive_emotion_modulation());
    }

    #[test]
    fn test_negative_emotion_modulation() {
        let neurotic = Personality {
            neuroticism: 0.9,
            extraversion: 0.1,
            ..Personality::uniform(0.5)
        };
        let stable = Personality {
            neuroticism: 0.1,
            extraversion: 0.9,
            ..Personality::uniform(0.5)
        };
        assert!(neurotic.negative_emotion_modulation() > stable.negative_emotion_modulation());
    }

    #[test]
    fn test_modulate_activations() {
        let p = Personality {
            neuroticism: 0.9,
            ..Personality::uniform(0.5)
        };
        let mut activations = [0.5; 8];
        let original = activations;
        p.modulate_activations(&mut activations, 1.0);
        // Fear and Sadness should be amplified by high neuroticism
        assert!(activations[crate::emotion::FEAR] > original[crate::emotion::FEAR]);
        assert!(activations[crate::emotion::SADNESS] > original[crate::emotion::SADNESS]);
    }

    #[test]
    fn test_modulate_activations_zero_strength() {
        let p = Personality::cortana_default();
        let mut activations = [0.5; 8];
        let original = activations;
        p.modulate_activations(&mut activations, 0.0);
        // With zero modulation, activations should be unchanged
        for (a, b) in activations.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_mood_stability() {
        let disciplined = Personality {
            conscientiousness: 0.9,
            neuroticism: 0.1,
            ..Personality::uniform(0.5)
        };
        let chaotic = Personality {
            conscientiousness: 0.1,
            neuroticism: 0.9,
            ..Personality::uniform(0.5)
        };
        assert!(disciplined.mood_stability() > chaotic.mood_stability());
    }

    #[test]
    fn test_all_presets_valid() {
        for p in &[
            Personality::cortana_default(),
            Personality::stoic(),
            Personality::creative(),
            Personality::anxious(),
        ] {
            assert!((0.0..=1.0).contains(&p.openness));
            assert!((0.0..=1.0).contains(&p.conscientiousness));
            assert!((0.0..=1.0).contains(&p.extraversion));
            assert!((0.0..=1.0).contains(&p.agreeableness));
            assert!((0.0..=1.0).contains(&p.neuroticism));
        }
    }

    #[test]
    fn test_trust_modulation() {
        let agreeable = Personality {
            agreeableness: 0.9,
            ..Personality::uniform(0.5)
        };
        let disagreeable = Personality {
            agreeableness: 0.1,
            ..Personality::uniform(0.5)
        };
        assert!(agreeable.trust_modulation() > disagreeable.trust_modulation());
    }
}
