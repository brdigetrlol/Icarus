//! Plutchik's Wheel of Emotions — 8 primary emotions as 4 bipolar pairs,
//! 3 intensity levels, and 16 compound emotions (dyads) from co-activation.

use serde::{Deserialize, Serialize};

/// Index constants for the 8 primary emotions in activation arrays.
pub const JOY: usize = 0;
pub const SADNESS: usize = 1;
pub const TRUST: usize = 2;
pub const DISGUST: usize = 3;
pub const FEAR: usize = 4;
pub const ANGER: usize = 5;
pub const SURPRISE: usize = 6;
pub const ANTICIPATION: usize = 7;

/// The 8 primary emotions organized as 4 bipolar pairs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PrimaryEmotion {
    Joy,
    Sadness,
    Trust,
    Disgust,
    Fear,
    Anger,
    Surprise,
    Anticipation,
}

impl PrimaryEmotion {
    pub const ALL: [PrimaryEmotion; 8] = [
        Self::Joy,
        Self::Sadness,
        Self::Trust,
        Self::Disgust,
        Self::Fear,
        Self::Anger,
        Self::Surprise,
        Self::Anticipation,
    ];

    pub fn index(self) -> usize {
        match self {
            Self::Joy => JOY,
            Self::Sadness => SADNESS,
            Self::Trust => TRUST,
            Self::Disgust => DISGUST,
            Self::Fear => FEAR,
            Self::Anger => ANGER,
            Self::Surprise => SURPRISE,
            Self::Anticipation => ANTICIPATION,
        }
    }

    /// Return the bipolar opposite.
    pub fn opposite(self) -> Self {
        match self {
            Self::Joy => Self::Sadness,
            Self::Sadness => Self::Joy,
            Self::Trust => Self::Disgust,
            Self::Disgust => Self::Trust,
            Self::Fear => Self::Anger,
            Self::Anger => Self::Fear,
            Self::Surprise => Self::Anticipation,
            Self::Anticipation => Self::Surprise,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Joy => "Joy",
            Self::Sadness => "Sadness",
            Self::Trust => "Trust",
            Self::Disgust => "Disgust",
            Self::Fear => "Fear",
            Self::Anger => "Anger",
            Self::Surprise => "Surprise",
            Self::Anticipation => "Anticipation",
        }
    }
}

/// Intensity level (Plutchik's 3 concentric rings).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Intensity {
    /// Outer ring: serenity, acceptance, apprehension, distraction, pensiveness, boredom, annoyance, interest
    Mild,
    /// Middle ring: joy, trust, fear, surprise, sadness, disgust, anger, anticipation
    Moderate,
    /// Inner ring: ecstasy, admiration, terror, amazement, grief, loathing, rage, vigilance
    Intense,
}

impl Intensity {
    pub fn from_activation(activation: f32) -> Self {
        if activation >= 0.7 {
            Self::Intense
        } else if activation >= 0.3 {
            Self::Moderate
        } else {
            Self::Mild
        }
    }

    /// Human-readable name for each primary at each intensity level.
    pub fn name_for(self, primary: PrimaryEmotion) -> &'static str {
        match (primary, self) {
            (PrimaryEmotion::Joy, Intensity::Mild) => "Serenity",
            (PrimaryEmotion::Joy, Intensity::Moderate) => "Joy",
            (PrimaryEmotion::Joy, Intensity::Intense) => "Ecstasy",
            (PrimaryEmotion::Sadness, Intensity::Mild) => "Pensiveness",
            (PrimaryEmotion::Sadness, Intensity::Moderate) => "Sadness",
            (PrimaryEmotion::Sadness, Intensity::Intense) => "Grief",
            (PrimaryEmotion::Trust, Intensity::Mild) => "Acceptance",
            (PrimaryEmotion::Trust, Intensity::Moderate) => "Trust",
            (PrimaryEmotion::Trust, Intensity::Intense) => "Admiration",
            (PrimaryEmotion::Disgust, Intensity::Mild) => "Boredom",
            (PrimaryEmotion::Disgust, Intensity::Moderate) => "Disgust",
            (PrimaryEmotion::Disgust, Intensity::Intense) => "Loathing",
            (PrimaryEmotion::Fear, Intensity::Mild) => "Apprehension",
            (PrimaryEmotion::Fear, Intensity::Moderate) => "Fear",
            (PrimaryEmotion::Fear, Intensity::Intense) => "Terror",
            (PrimaryEmotion::Anger, Intensity::Mild) => "Annoyance",
            (PrimaryEmotion::Anger, Intensity::Moderate) => "Anger",
            (PrimaryEmotion::Anger, Intensity::Intense) => "Rage",
            (PrimaryEmotion::Surprise, Intensity::Mild) => "Distraction",
            (PrimaryEmotion::Surprise, Intensity::Moderate) => "Surprise",
            (PrimaryEmotion::Surprise, Intensity::Intense) => "Amazement",
            (PrimaryEmotion::Anticipation, Intensity::Mild) => "Interest",
            (PrimaryEmotion::Anticipation, Intensity::Moderate) => "Anticipation",
            (PrimaryEmotion::Anticipation, Intensity::Intense) => "Vigilance",
        }
    }
}

/// Compound emotions (dyads) — emergent from co-activation of two primaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CompoundEmotion {
    // Primary dyads (adjacent primaries on the wheel)
    Love,           // Joy + Trust
    Submission,     // Trust + Fear
    Awe,            // Fear + Surprise
    Disapproval,    // Surprise + Sadness
    Remorse,        // Sadness + Disgust
    Contempt,       // Disgust + Anger
    Aggressiveness, // Anger + Anticipation
    Optimism,       // Anticipation + Joy
    // Secondary dyads (one apart on the wheel)
    Guilt,          // Joy + Fear
    Curiosity,      // Trust + Surprise
    Despair,        // Fear + Sadness
    Cynicism,       // Surprise + Disgust
    Envy,           // Sadness + Anger
    Pride,          // Disgust + Anticipation
    Fatalism,       // Anger + Joy
    Hope,           // Anticipation + Trust
}

impl CompoundEmotion {
    /// The two primaries whose co-activation produces this compound.
    pub fn components(self) -> (PrimaryEmotion, PrimaryEmotion) {
        match self {
            Self::Love => (PrimaryEmotion::Joy, PrimaryEmotion::Trust),
            Self::Submission => (PrimaryEmotion::Trust, PrimaryEmotion::Fear),
            Self::Awe => (PrimaryEmotion::Fear, PrimaryEmotion::Surprise),
            Self::Disapproval => (PrimaryEmotion::Surprise, PrimaryEmotion::Sadness),
            Self::Remorse => (PrimaryEmotion::Sadness, PrimaryEmotion::Disgust),
            Self::Contempt => (PrimaryEmotion::Disgust, PrimaryEmotion::Anger),
            Self::Aggressiveness => (PrimaryEmotion::Anger, PrimaryEmotion::Anticipation),
            Self::Optimism => (PrimaryEmotion::Anticipation, PrimaryEmotion::Joy),
            Self::Guilt => (PrimaryEmotion::Joy, PrimaryEmotion::Fear),
            Self::Curiosity => (PrimaryEmotion::Trust, PrimaryEmotion::Surprise),
            Self::Despair => (PrimaryEmotion::Fear, PrimaryEmotion::Sadness),
            Self::Cynicism => (PrimaryEmotion::Surprise, PrimaryEmotion::Disgust),
            Self::Envy => (PrimaryEmotion::Sadness, PrimaryEmotion::Anger),
            Self::Pride => (PrimaryEmotion::Disgust, PrimaryEmotion::Anticipation),
            Self::Fatalism => (PrimaryEmotion::Anger, PrimaryEmotion::Joy),
            Self::Hope => (PrimaryEmotion::Anticipation, PrimaryEmotion::Trust),
        }
    }

    pub const ALL: [CompoundEmotion; 16] = [
        Self::Love,
        Self::Submission,
        Self::Awe,
        Self::Disapproval,
        Self::Remorse,
        Self::Contempt,
        Self::Aggressiveness,
        Self::Optimism,
        Self::Guilt,
        Self::Curiosity,
        Self::Despair,
        Self::Cynicism,
        Self::Envy,
        Self::Pride,
        Self::Fatalism,
        Self::Hope,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::Love => "Love",
            Self::Submission => "Submission",
            Self::Awe => "Awe",
            Self::Disapproval => "Disapproval",
            Self::Remorse => "Remorse",
            Self::Contempt => "Contempt",
            Self::Aggressiveness => "Aggressiveness",
            Self::Optimism => "Optimism",
            Self::Guilt => "Guilt",
            Self::Curiosity => "Curiosity",
            Self::Despair => "Despair",
            Self::Cynicism => "Cynicism",
            Self::Envy => "Envy",
            Self::Pride => "Pride",
            Self::Fatalism => "Fatalism",
            Self::Hope => "Hope",
        }
    }
}

/// A label covering both primary (with intensity) and compound emotions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EmotionLabel {
    Primary {
        emotion: PrimaryEmotion,
        intensity: Intensity,
    },
    Compound(CompoundEmotion),
    Neutral,
}

impl EmotionLabel {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Primary {
                emotion,
                intensity,
            } => intensity.name_for(*emotion),
            Self::Compound(c) => c.name(),
            Self::Neutral => "Neutral",
        }
    }
}

/// The full Plutchik emotional state — continuous activations for all 8 primaries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlutchikState {
    /// Activation level [0.0, 1.0] for each primary emotion.
    pub activations: [f32; 8],
}

impl PlutchikState {
    pub fn new() -> Self {
        Self {
            activations: [0.0; 8],
        }
    }

    /// Get the activation for a primary emotion.
    pub fn activation(&self, emotion: PrimaryEmotion) -> f32 {
        self.activations[emotion.index()]
    }

    /// Set the activation for a primary emotion (clamped to [0, 1]).
    pub fn set_activation(&mut self, emotion: PrimaryEmotion, value: f32) {
        self.activations[emotion.index()] = value.clamp(0.0, 1.0);
    }

    /// Get the intensity level for a primary emotion.
    pub fn intensity(&self, emotion: PrimaryEmotion) -> Intensity {
        Intensity::from_activation(self.activations[emotion.index()])
    }

    /// Find the dominant primary emotion (highest activation).
    pub fn dominant_primary(&self) -> Option<(PrimaryEmotion, f32)> {
        let mut best_idx = 0;
        let mut best_val = 0.0f32;
        for (i, &v) in self.activations.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best_idx = i;
            }
        }
        if best_val < 0.05 {
            return None;
        }
        Some((PrimaryEmotion::ALL[best_idx], best_val))
    }

    /// Detect active compound emotions from co-activation patterns.
    /// Returns compounds where both component activations exceed the threshold.
    pub fn active_compounds(&self, threshold: f32) -> Vec<(CompoundEmotion, f32)> {
        let mut result = Vec::new();
        for compound in CompoundEmotion::ALL {
            let (a, b) = compound.components();
            let act_a = self.activations[a.index()];
            let act_b = self.activations[b.index()];
            if act_a >= threshold && act_b >= threshold {
                let strength = (act_a * act_b).sqrt();
                result.push((compound, strength));
            }
        }
        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        result
    }

    /// Determine the dominant emotion label (primary or compound).
    pub fn dominant_emotion(&self, compound_threshold: f32) -> EmotionLabel {
        let compounds = self.active_compounds(compound_threshold);
        let primary = self.dominant_primary();

        match (compounds.first(), primary) {
            (Some((compound, c_strength)), Some((prim, p_activation))) => {
                // Compound wins if its strength exceeds the primary activation
                if *c_strength > p_activation * 0.8 {
                    EmotionLabel::Compound(*compound)
                } else {
                    EmotionLabel::Primary {
                        emotion: prim,
                        intensity: self.intensity(prim),
                    }
                }
            }
            (Some((compound, _)), None) => EmotionLabel::Compound(*compound),
            (None, Some((prim, _))) => EmotionLabel::Primary {
                emotion: prim,
                intensity: self.intensity(prim),
            },
            (None, None) => EmotionLabel::Neutral,
        }
    }

    /// Apply bipolar suppression: when one pole is active, suppress the opposite.
    pub fn apply_bipolar_suppression(&mut self, suppression_rate: f32) {
        let pairs = [
            (JOY, SADNESS),
            (TRUST, DISGUST),
            (FEAR, ANGER),
            (SURPRISE, ANTICIPATION),
        ];
        for (a, b) in pairs {
            let act_a = self.activations[a];
            let act_b = self.activations[b];
            if act_a > act_b {
                self.activations[b] = (act_b - act_a * suppression_rate).max(0.0);
            } else if act_b > act_a {
                self.activations[a] = (act_a - act_b * suppression_rate).max(0.0);
            }
        }
    }

    /// Compute the total emotional activation (sum of all primaries).
    pub fn total_activation(&self) -> f32 {
        self.activations.iter().sum()
    }
}

impl Default for PlutchikState {
    fn default() -> Self {
        Self::new()
    }
}

/// Map EMC physics observables to Plutchik primary emotion activations.
///
/// - `valence`: energy derivative -dF/dt (positive = energy decreasing = good)
/// - `arousal`: 1 - mean_amplitude (high when field is low-amplitude/chaotic)
/// - `phase_coherence`: Kuramoto order parameter [0, 1]
/// - `criticality_sigma`: standard deviation of criticality measure
/// - `prediction_error`: deviation of reservoir readout from actual (0 = perfect)
pub fn map_physics_to_plutchik(
    valence: f32,
    arousal: f32,
    phase_coherence: f32,
    criticality_sigma: f32,
    prediction_error: f32,
) -> PlutchikState {
    let mut state = PlutchikState::new();

    // Joy ↔ Sadness: driven by valence
    if valence > 0.0 {
        state.activations[JOY] = (valence * 2.0).clamp(0.0, 1.0);
    } else {
        state.activations[SADNESS] = (-valence * 2.0).clamp(0.0, 1.0);
    }

    // Trust ↔ Disgust: driven by phase coherence
    // High coherence → trust (system is organized, predictable)
    // Low coherence → disgust (system is disordered, unreliable)
    if phase_coherence > 0.5 {
        state.activations[TRUST] = ((phase_coherence - 0.5) * 2.0).clamp(0.0, 1.0);
    } else {
        state.activations[DISGUST] = ((0.5 - phase_coherence) * 2.0).clamp(0.0, 1.0);
    }

    // Fear ↔ Anger: driven by arousal + criticality
    // High arousal + high criticality variance → fear (unstable, out of control)
    // High arousal + low criticality variance → anger (energized but controlled)
    let threat = arousal * 0.6 + criticality_sigma * 0.4;
    if criticality_sigma > 0.3 {
        state.activations[FEAR] = (threat * 1.5).clamp(0.0, 1.0);
    } else if arousal > 0.3 {
        state.activations[ANGER] = (arousal * 1.2).clamp(0.0, 1.0);
    }

    // Surprise ↔ Anticipation: driven by prediction error
    // High prediction error → surprise (unexpected outcome)
    // Low prediction error → anticipation (expected, engaged)
    if prediction_error > 0.3 {
        state.activations[SURPRISE] = ((prediction_error - 0.3) * 1.5).clamp(0.0, 1.0);
    } else {
        state.activations[ANTICIPATION] = ((0.3 - prediction_error) * 2.0).clamp(0.0, 1.0);
    }

    state
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_primary_emotion_indices() {
        for (i, e) in PrimaryEmotion::ALL.iter().enumerate() {
            assert_eq!(e.index(), i);
        }
    }

    #[test]
    fn test_bipolar_opposites() {
        assert_eq!(PrimaryEmotion::Joy.opposite(), PrimaryEmotion::Sadness);
        assert_eq!(PrimaryEmotion::Sadness.opposite(), PrimaryEmotion::Joy);
        assert_eq!(PrimaryEmotion::Trust.opposite(), PrimaryEmotion::Disgust);
        assert_eq!(PrimaryEmotion::Fear.opposite(), PrimaryEmotion::Anger);
        assert_eq!(PrimaryEmotion::Surprise.opposite(), PrimaryEmotion::Anticipation);
    }

    #[test]
    fn test_intensity_from_activation() {
        assert_eq!(Intensity::from_activation(0.0), Intensity::Mild);
        assert_eq!(Intensity::from_activation(0.29), Intensity::Mild);
        assert_eq!(Intensity::from_activation(0.3), Intensity::Moderate);
        assert_eq!(Intensity::from_activation(0.69), Intensity::Moderate);
        assert_eq!(Intensity::from_activation(0.7), Intensity::Intense);
        assert_eq!(Intensity::from_activation(1.0), Intensity::Intense);
    }

    #[test]
    fn test_intensity_names() {
        assert_eq!(Intensity::Mild.name_for(PrimaryEmotion::Joy), "Serenity");
        assert_eq!(Intensity::Moderate.name_for(PrimaryEmotion::Joy), "Joy");
        assert_eq!(Intensity::Intense.name_for(PrimaryEmotion::Joy), "Ecstasy");
        assert_eq!(Intensity::Intense.name_for(PrimaryEmotion::Fear), "Terror");
        assert_eq!(Intensity::Mild.name_for(PrimaryEmotion::Anger), "Annoyance");
    }

    #[test]
    fn test_plutchik_state_new() {
        let state = PlutchikState::new();
        assert_eq!(state.total_activation(), 0.0);
        assert_eq!(state.dominant_primary(), None);
    }

    #[test]
    fn test_set_and_get_activation() {
        let mut state = PlutchikState::new();
        state.set_activation(PrimaryEmotion::Joy, 0.8);
        assert!((state.activation(PrimaryEmotion::Joy) - 0.8).abs() < f32::EPSILON);
        assert_eq!(state.intensity(PrimaryEmotion::Joy), Intensity::Intense);

        // Clamping
        state.set_activation(PrimaryEmotion::Fear, 1.5);
        assert!((state.activation(PrimaryEmotion::Fear) - 1.0).abs() < f32::EPSILON);
        state.set_activation(PrimaryEmotion::Anger, -0.5);
        assert!((state.activation(PrimaryEmotion::Anger)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_dominant_primary() {
        let mut state = PlutchikState::new();
        state.set_activation(PrimaryEmotion::Trust, 0.6);
        state.set_activation(PrimaryEmotion::Joy, 0.3);
        let (dom, val) = state.dominant_primary().unwrap();
        assert_eq!(dom, PrimaryEmotion::Trust);
        assert!((val - 0.6).abs() < f32::EPSILON);
    }

    #[test]
    fn test_active_compounds() {
        let mut state = PlutchikState::new();
        state.set_activation(PrimaryEmotion::Joy, 0.7);
        state.set_activation(PrimaryEmotion::Trust, 0.6);

        let compounds = state.active_compounds(0.3);
        assert!(!compounds.is_empty());
        assert_eq!(compounds[0].0, CompoundEmotion::Love);
    }

    #[test]
    fn test_compound_components() {
        let (a, b) = CompoundEmotion::Love.components();
        assert_eq!(a, PrimaryEmotion::Joy);
        assert_eq!(b, PrimaryEmotion::Trust);

        let (a, b) = CompoundEmotion::Awe.components();
        assert_eq!(a, PrimaryEmotion::Fear);
        assert_eq!(b, PrimaryEmotion::Surprise);
    }

    #[test]
    fn test_dominant_emotion_neutral() {
        let state = PlutchikState::new();
        assert_eq!(state.dominant_emotion(0.3), EmotionLabel::Neutral);
    }

    #[test]
    fn test_dominant_emotion_primary() {
        let mut state = PlutchikState::new();
        state.set_activation(PrimaryEmotion::Fear, 0.8);
        match state.dominant_emotion(0.3) {
            EmotionLabel::Primary { emotion, intensity } => {
                assert_eq!(emotion, PrimaryEmotion::Fear);
                assert_eq!(intensity, Intensity::Intense);
            }
            other => panic!("Expected Primary, got {:?}", other),
        }
    }

    #[test]
    fn test_dominant_emotion_compound() {
        let mut state = PlutchikState::new();
        state.set_activation(PrimaryEmotion::Joy, 0.7);
        state.set_activation(PrimaryEmotion::Trust, 0.7);
        match state.dominant_emotion(0.3) {
            EmotionLabel::Compound(c) => assert_eq!(c, CompoundEmotion::Love),
            other => panic!("Expected Compound Love, got {:?}", other),
        }
    }

    #[test]
    fn test_bipolar_suppression() {
        let mut state = PlutchikState::new();
        state.set_activation(PrimaryEmotion::Joy, 0.8);
        state.set_activation(PrimaryEmotion::Sadness, 0.4);
        state.apply_bipolar_suppression(0.3);
        // Sadness should be suppressed by Joy
        assert!(state.activation(PrimaryEmotion::Sadness) < 0.4);
        // Joy should remain unchanged (it was dominant)
        assert!((state.activation(PrimaryEmotion::Joy) - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn test_physics_mapping_positive_valence() {
        let state = map_physics_to_plutchik(0.5, 0.2, 0.8, 0.1, 0.1);
        assert!(state.activation(PrimaryEmotion::Joy) > 0.5);
        assert!(state.activation(PrimaryEmotion::Trust) > 0.3);
        assert!(state.activation(PrimaryEmotion::Anticipation) > 0.0);
    }

    #[test]
    fn test_physics_mapping_negative_valence() {
        let state = map_physics_to_plutchik(-0.5, 0.8, 0.2, 0.5, 0.8);
        assert!(state.activation(PrimaryEmotion::Sadness) > 0.5);
        assert!(state.activation(PrimaryEmotion::Disgust) > 0.0);
        assert!(state.activation(PrimaryEmotion::Fear) > 0.0);
        assert!(state.activation(PrimaryEmotion::Surprise) > 0.0);
    }

    #[test]
    fn test_emotion_label_names() {
        let label = EmotionLabel::Primary {
            emotion: PrimaryEmotion::Joy,
            intensity: Intensity::Intense,
        };
        assert_eq!(label.name(), "Ecstasy");

        let label = EmotionLabel::Compound(CompoundEmotion::Love);
        assert_eq!(label.name(), "Love");

        assert_eq!(EmotionLabel::Neutral.name(), "Neutral");
    }

    #[test]
    fn test_all_compounds_have_valid_components() {
        for compound in CompoundEmotion::ALL {
            let (a, b) = compound.components();
            assert_ne!(a, b, "Compound {:?} has same primary twice", compound);
        }
    }

    #[test]
    fn test_total_activation() {
        let mut state = PlutchikState::new();
        state.set_activation(PrimaryEmotion::Joy, 0.5);
        state.set_activation(PrimaryEmotion::Trust, 0.3);
        assert!((state.total_activation() - 0.8).abs() < f32::EPSILON);
    }
}
