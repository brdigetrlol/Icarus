//! Configuration for the Cortana emotion simulation engine.
//!
//! Aggregates all sub-configurations into a single `CortanaConfig` struct
//! with sensible presets for different personality profiles.

use icarus_engine::config::ManifoldConfig;
use serde::{Deserialize, Serialize};

use crate::aether::ExtendedAetherParams;
use crate::memory::MemoryConfig;
use crate::mood::MoodConfig;
use crate::personality::Personality;
use crate::processors::ProcessorConfig;

/// Emotion system configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionConfig {
    /// Global emotion sensitivity multiplier [0, 2].
    /// 1.0 = normal, >1 = more reactive, <1 = dampened.
    pub sensitivity: f32,
    /// Threshold for compound emotion detection [0, 1].
    /// Lower = more compounds detected.
    pub compound_threshold: f32,
    /// Weight of prediction error in fear/surprise computation [0, 1].
    pub prediction_error_weight: f32,
    /// Rate of bipolar suppression (opposing emotions inhibit each other) [0, 1].
    pub bipolar_suppression_rate: f32,
}

impl Default for EmotionConfig {
    fn default() -> Self {
        Self {
            sensitivity: 1.0,
            compound_threshold: 0.3,
            prediction_error_weight: 0.5,
            bipolar_suppression_rate: 0.3,
        }
    }
}

/// Full Cortana configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CortanaConfig {
    /// EMC manifold configuration
    pub emc: ManifoldConfig,
    /// Personality (Big Five traits)
    pub personality: Personality,
    /// Emotion system parameters
    pub emotion: EmotionConfig,
    /// Extended neuromodulator parameters
    pub aether: ExtendedAetherParams,
    /// Memory system parameters
    pub memory: MemoryConfig,
    /// Mood system parameters
    pub mood: MoodConfig,
    /// Processor orchestrator parameters
    pub processors: ProcessorConfig,
    /// Random seed for EMC initialization
    pub seed: u64,
    /// Initial field amplitude for EMC
    pub init_amplitude: f32,
}

impl Default for CortanaConfig {
    fn default() -> Self {
        Self {
            emc: ManifoldConfig::e8_only(),
            personality: Personality::cortana_default(),
            emotion: EmotionConfig::default(),
            aether: ExtendedAetherParams::default(),
            memory: MemoryConfig::default(),
            mood: MoodConfig::default(),
            processors: ProcessorConfig::default(),
            seed: 42,
            init_amplitude: 0.5,
        }
    }
}

impl CortanaConfig {
    /// Default Cortana personality — warm, curious, emotionally stable.
    /// Uses E8-only manifold for fast simulation.
    pub fn cortana_default() -> Self {
        Self::default()
    }

    /// Full multi-layer configuration with Cortana personality.
    /// Uses 3-layer manifold (E8 + Leech + HCP) for richer dynamics.
    pub fn cortana_full() -> Self {
        Self {
            emc: ManifoldConfig::multi_layer(),
            personality: Personality::cortana_default(),
            ..Self::default()
        }
    }

    /// Stoic personality — low emotional reactivity, high discipline.
    pub fn stoic() -> Self {
        Self {
            personality: Personality::stoic(),
            emotion: EmotionConfig {
                sensitivity: 0.6,
                bipolar_suppression_rate: 0.5,
                ..EmotionConfig::default()
            },
            mood: MoodConfig {
                tau: 2000.0, // slower mood changes
                homeostasis_rate: 0.0005, // faster drift to neutral
                mood_emotion_bias: 0.15, // less mood-emotion interaction
            },
            ..Self::default()
        }
    }

    /// Creative personality — high openness, more emotional variability.
    pub fn creative() -> Self {
        Self {
            personality: Personality::creative(),
            emotion: EmotionConfig {
                sensitivity: 1.3,
                compound_threshold: 0.25, // detect more compound emotions
                ..EmotionConfig::default()
            },
            mood: MoodConfig {
                tau: 500.0, // faster mood changes
                homeostasis_rate: 0.0001, // slower drift (more extreme moods)
                mood_emotion_bias: 0.4,
            },
            ..Self::default()
        }
    }

    /// Anxious personality — high neuroticism, more negative emotion sensitivity.
    pub fn anxious() -> Self {
        Self {
            personality: Personality::anxious(),
            emotion: EmotionConfig {
                sensitivity: 1.5,
                prediction_error_weight: 0.7, // more fear from uncertainty
                ..EmotionConfig::default()
            },
            mood: MoodConfig {
                tau: 800.0,
                homeostasis_rate: 0.0001, // slow recovery from negative mood
                mood_emotion_bias: 0.5, // strong mood-emotion feedback
            },
            memory: MemoryConfig {
                encoding_threshold: 0.4, // encodes more events (lower bar)
                ..MemoryConfig::default()
            },
            ..Self::default()
        }
    }

    /// Validate the configuration, returning any issues found.
    pub fn validate(&self) -> Vec<String> {
        let mut issues = Vec::new();

        if self.emotion.sensitivity < 0.0 || self.emotion.sensitivity > 5.0 {
            issues.push(format!(
                "emotion.sensitivity {} out of range [0, 5]",
                self.emotion.sensitivity
            ));
        }

        if self.mood.tau < 1.0 {
            issues.push(format!("mood.tau {} must be >= 1.0", self.mood.tau));
        }

        if self.memory.capacity == 0 {
            issues.push("memory.capacity must be > 0".into());
        }

        let cfl_violations = self.emc.validate_cfl();
        for (layer, dt, limit) in &cfl_violations {
            issues.push(format!(
                "CFL violation: layer {} dt={} exceeds limit={}",
                layer, dt, limit
            ));
        }

        issues
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = CortanaConfig::default();
        assert!(config.validate().is_empty());
        assert!((config.personality.openness - 0.85).abs() < f32::EPSILON);
    }

    #[test]
    fn test_cortana_default() {
        let config = CortanaConfig::cortana_default();
        assert!(config.validate().is_empty());
        assert_eq!(config.emc.layers.len(), 1);
    }

    #[test]
    fn test_cortana_full() {
        let config = CortanaConfig::cortana_full();
        assert!(config.validate().is_empty());
        assert_eq!(config.emc.layers.len(), 3);
    }

    #[test]
    fn test_stoic_preset() {
        let config = CortanaConfig::stoic();
        assert!(config.validate().is_empty());
        assert!(config.emotion.sensitivity < 1.0);
        assert!(config.mood.tau > 1000.0);
        assert!(config.personality.neuroticism < 0.2);
    }

    #[test]
    fn test_creative_preset() {
        let config = CortanaConfig::creative();
        assert!(config.validate().is_empty());
        assert!(config.emotion.sensitivity > 1.0);
        assert!(config.mood.tau < 1000.0);
        assert!(config.personality.openness > 0.9);
    }

    #[test]
    fn test_anxious_preset() {
        let config = CortanaConfig::anxious();
        assert!(config.validate().is_empty());
        assert!(config.emotion.sensitivity > 1.0);
        assert!(config.memory.encoding_threshold < 0.5);
        assert!(config.personality.neuroticism > 0.8);
    }

    #[test]
    fn test_validate_bad_sensitivity() {
        let mut config = CortanaConfig::default();
        config.emotion.sensitivity = -1.0;
        let issues = config.validate();
        assert!(!issues.is_empty());
    }

    #[test]
    fn test_validate_bad_tau() {
        let mut config = CortanaConfig::default();
        config.mood.tau = 0.0;
        let issues = config.validate();
        assert!(!issues.is_empty());
    }

    #[test]
    fn test_validate_zero_capacity() {
        let mut config = CortanaConfig::default();
        config.memory.capacity = 0;
        let issues = config.validate();
        assert!(!issues.is_empty());
    }

    #[test]
    fn test_config_serialization() {
        let config = CortanaConfig::cortana_default();
        let json = serde_json::to_string(&config).unwrap();
        let restored: CortanaConfig = serde_json::from_str(&json).unwrap();
        assert!((restored.personality.openness - 0.85).abs() < f32::EPSILON);
        assert!((restored.emotion.sensitivity - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_all_presets_valid() {
        for config in &[
            CortanaConfig::cortana_default(),
            CortanaConfig::cortana_full(),
            CortanaConfig::stoic(),
            CortanaConfig::creative(),
            CortanaConfig::anxious(),
        ] {
            let issues = config.validate();
            assert!(issues.is_empty(), "Preset failed validation: {:?}", issues);
        }
    }
}
