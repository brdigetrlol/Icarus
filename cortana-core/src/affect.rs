// Copyright (c) 2025-2026 brdigetrlol. All rights reserved.
// SPDX-License-Identifier: LicenseRef-Icarus-Proprietary
// See LICENSE in the repository root for full license terms.

//! Extended affective state — 3D PAD model with Plutchik emotions.
//!
//! Extends Icarus's 2D (Valence, Arousal) affect into 3D PAD (Pleasure, Arousal, Dominance)
//! and integrates Plutchik emotion activations, mood, and extended neuromodulators.

use icarus_field::autopoiesis::AffectiveState;
use serde::Serialize;

use crate::aether::{AetherSignals, ExtendedAether, ExtendedAetherParams};
use crate::emotion::{map_physics_to_plutchik, EmotionLabel, PlutchikState};
use crate::mood::MoodState;
use crate::personality::Personality;

/// Physics observables from the EMC that drive emotion computation.
#[derive(Debug, Clone)]
pub struct PhysicsInput {
    /// Valence V = -dF/dt (positive = improving)
    pub valence: f32,
    /// Arousal A = 1 - |Ψ| (high = desynchronized)
    pub arousal: f32,
    /// Kuramoto phase coherence [0, 1]
    pub phase_coherence: f32,
    /// Standard deviation of energy across layers (criticality measure)
    pub criticality_sigma: f32,
    /// Reservoir prediction error [0, 1] (0 = perfect prediction)
    pub prediction_error: f32,
    /// Whether convergence is currently stable
    pub convergence_stable: bool,
    /// Whether an achievement event occurred this tick
    pub achievement: bool,
}

impl PhysicsInput {
    /// Extract physics input from an Icarus AffectiveState + supplementary data.
    pub fn from_affective_state(
        state: &AffectiveState,
        criticality_sigma: f32,
        prediction_error: f32,
        convergence_stable: bool,
        achievement: bool,
    ) -> Self {
        Self {
            valence: state.valence,
            arousal: state.arousal,
            phase_coherence: state.phase_coherence,
            criticality_sigma,
            prediction_error,
            convergence_stable,
            achievement,
        }
    }
}

/// Extended affective state combining PAD dimensions, Plutchik emotions,
/// mood, and neuromodulators into a unified representation.
#[derive(Debug, Clone, Serialize)]
pub struct ExtendedAffectiveState {
    /// Pleasure dimension [-1, 1] — derived from smoothed valence.
    pub pleasure: f32,
    /// Arousal dimension [-1, 1] — rescaled from [0,1] Icarus arousal to bipolar.
    pub arousal: f32,
    /// Dominance dimension [-1, 1] — sense of control/predictability.
    pub dominance: f32,
    /// Icarus base affective state (preserved).
    pub base: AffectiveState,
    /// Plutchik emotion activations (8 primary + compounds).
    pub plutchik: PlutchikState,
    /// Current mood state (slow EMA baseline).
    pub mood: MoodState,
    /// Extended neuromodulator state (8 modulators).
    pub extended_aether: ExtendedAether,
    /// Currently dominant emotion label.
    pub dominant_emotion: EmotionLabel,
}

impl ExtendedAffectiveState {
    /// Create a new extended affective state from an Icarus base state.
    pub fn new(base: AffectiveState) -> Self {
        Self {
            pleasure: 0.0,
            arousal: 0.0,
            dominance: 0.0,
            base,
            plutchik: PlutchikState::default(),
            mood: MoodState::default(),
            extended_aether: ExtendedAether::from_base(base.aether),
            dominant_emotion: EmotionLabel::Neutral,
        }
    }

    /// Update the full affective state from physics observables.
    ///
    /// This is the main entry point called each tick by the EmotionAgent:
    /// 1. Maps physics → Plutchik activations
    /// 2. Applies personality modulation
    /// 3. Detects compound emotions
    /// 4. Computes PAD dimensions
    /// 5. Steps extended neuromodulators
    /// 6. Determines dominant emotion
    pub fn update(
        &mut self,
        physics: &PhysicsInput,
        personality: &Personality,
        aether_params: &ExtendedAetherParams,
        dt: f32,
    ) {
        // 1. Map physics to raw Plutchik activations
        let mut plutchik = map_physics_to_plutchik(
            physics.valence,
            physics.arousal,
            physics.phase_coherence,
            physics.criticality_sigma,
            physics.prediction_error,
        );

        // 2. Apply personality modulation
        personality.modulate_activations(&mut plutchik.activations, 0.8);

        // 3. Apply bipolar suppression (opposing emotions inhibit each other)
        plutchik.apply_bipolar_suppression(0.3);

        // 4. Compute PAD dimensions
        self.pleasure = compute_pleasure(physics.valence);
        self.arousal = compute_arousal_bipolar(physics.arousal);
        self.dominance = compute_dominance(
            1.0 - physics.prediction_error,
            physics.criticality_sigma,
        );

        // 5. Step extended neuromodulators
        let signals = AetherSignals {
            joy: plutchik.activations[crate::emotion::JOY],
            trust: plutchik.activations[crate::emotion::TRUST],
            fear: plutchik.activations[crate::emotion::FEAR],
            anger: plutchik.activations[crate::emotion::ANGER],
            arousal: physics.arousal,
            phase_coherence: physics.phase_coherence,
            convergence_stable: physics.convergence_stable,
            achievement: physics.achievement,
        };
        self.extended_aether.base = self.base.aether;
        self.extended_aether.step(&signals, aether_params, dt);

        // 6. Apply stress suppression to positive emotions
        let suppression = self.extended_aether.stress_suppression();
        plutchik.activations[crate::emotion::JOY] *= suppression;
        plutchik.activations[crate::emotion::TRUST] *= suppression;

        // 7. Store updated state
        self.plutchik = plutchik;
        self.base.valence = physics.valence;
        self.base.arousal = physics.arousal;
        self.base.phase_coherence = physics.phase_coherence;

        // 8. Determine dominant emotion
        self.dominant_emotion = self.plutchik.dominant_emotion(0.3);
    }

    /// Get the PAD vector as a 3-tuple.
    pub fn pad(&self) -> (f32, f32, f32) {
        (self.pleasure, self.arousal, self.dominance)
    }

    /// Get the overall emotional intensity (magnitude of PAD vector).
    pub fn emotional_intensity(&self) -> f32 {
        (self.pleasure * self.pleasure
            + self.arousal * self.arousal
            + self.dominance * self.dominance)
            .sqrt()
    }

    /// Get the max activation across all primary emotions.
    pub fn peak_activation(&self) -> f32 {
        self.plutchik
            .activations
            .iter()
            .copied()
            .fold(0.0f32, f32::max)
    }
}

impl Default for ExtendedAffectiveState {
    fn default() -> Self {
        Self::new(AffectiveState::baseline())
    }
}

/// Map raw valence to pleasure [-1, 1] via tanh compression.
/// Valence can be any real number (dF/dt); tanh squashes to bounded range.
fn compute_pleasure(valence: f32) -> f32 {
    // Scale factor: valence ~±10 maps to ~±0.9
    (valence * 0.3).tanh()
}

/// Rescale Icarus arousal [0, 1] to bipolar [-1, 1].
fn compute_arousal_bipolar(arousal: f32) -> f32 {
    (arousal * 2.0 - 1.0).clamp(-1.0, 1.0)
}

/// Compute dominance from prediction accuracy and criticality.
///
/// High prediction accuracy + low criticality = high dominance (in control).
/// Low prediction accuracy + high criticality = low dominance (overwhelmed).
fn compute_dominance(prediction_accuracy: f32, criticality_sigma: f32) -> f32 {
    let raw = prediction_accuracy - criticality_sigma;
    // Sigmoid mapping to [-1, 1] with steepness 3.0
    (2.0 / (1.0 + (-3.0 * raw).exp())) - 1.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aether::ExtendedAetherParams;
    use crate::personality::Personality;

    fn default_physics() -> PhysicsInput {
        PhysicsInput {
            valence: 0.0,
            arousal: 0.5,
            phase_coherence: 0.5,
            criticality_sigma: 0.3,
            prediction_error: 0.5,
            convergence_stable: false,
            achievement: false,
        }
    }

    #[test]
    fn test_new_extended_affect() {
        let state = ExtendedAffectiveState::default();
        assert!((state.pleasure).abs() < f32::EPSILON);
        assert!((state.arousal).abs() < f32::EPSILON);
        assert!((state.dominance).abs() < f32::EPSILON);
        assert_eq!(state.dominant_emotion, EmotionLabel::Neutral);
    }

    #[test]
    fn test_pleasure_from_valence() {
        // Positive valence → positive pleasure
        assert!(compute_pleasure(5.0) > 0.5);
        // Negative valence → negative pleasure
        assert!(compute_pleasure(-5.0) < -0.5);
        // Zero valence → zero pleasure
        assert!(compute_pleasure(0.0).abs() < f32::EPSILON);
        // Bounded to [-1, 1]
        assert!(compute_pleasure(100.0) <= 1.0);
        assert!(compute_pleasure(-100.0) >= -1.0);
    }

    #[test]
    fn test_arousal_bipolar() {
        assert!((compute_arousal_bipolar(0.0) - (-1.0)).abs() < f32::EPSILON);
        assert!((compute_arousal_bipolar(0.5) - 0.0).abs() < f32::EPSILON);
        assert!((compute_arousal_bipolar(1.0) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_dominance_sigmoid() {
        // Perfect prediction, no criticality → high dominance
        let d = compute_dominance(1.0, 0.0);
        assert!(d > 0.8);
        // No prediction, high criticality → low dominance
        let d = compute_dominance(0.0, 1.0);
        assert!(d < -0.8);
        // Equal → near zero
        let d = compute_dominance(0.5, 0.5);
        assert!(d.abs() < 0.1);
    }

    #[test]
    fn test_update_positive_state() {
        let mut state = ExtendedAffectiveState::default();
        let personality = Personality::cortana_default();
        let aether_params = ExtendedAetherParams::default();
        let physics = PhysicsInput {
            valence: 5.0,
            arousal: 0.3,
            phase_coherence: 0.8,
            criticality_sigma: 0.1,
            prediction_error: 0.1,
            convergence_stable: true,
            achievement: false,
        };

        state.update(&physics, &personality, &aether_params, 1.0);

        assert!(state.pleasure > 0.0, "positive valence → positive pleasure");
        assert!(state.dominance > 0.0, "good prediction → positive dominance");
        assert!(
            state.plutchik.activations[crate::emotion::JOY] > 0.0,
            "positive valence → joy activation"
        );
    }

    #[test]
    fn test_update_negative_state() {
        let mut state = ExtendedAffectiveState::default();
        let personality = Personality::cortana_default();
        let aether_params = ExtendedAetherParams::default();
        let physics = PhysicsInput {
            valence: -5.0,
            arousal: 0.9,
            phase_coherence: 0.2,
            criticality_sigma: 0.8,
            prediction_error: 0.9,
            convergence_stable: false,
            achievement: false,
        };

        state.update(&physics, &personality, &aether_params, 1.0);

        assert!(state.pleasure < 0.0, "negative valence → negative pleasure");
        assert!(
            state.dominance < 0.0,
            "poor prediction → negative dominance"
        );
    }

    #[test]
    fn test_personality_modulates_emotions() {
        let physics = PhysicsInput {
            valence: 3.0,
            arousal: 0.5,
            phase_coherence: 0.5,
            criticality_sigma: 0.3,
            prediction_error: 0.3,
            convergence_stable: false,
            achievement: false,
        };
        let aether_params = ExtendedAetherParams::default();

        // Neurotic personality → amplified negative emotions
        let mut neurotic_state = ExtendedAffectiveState::default();
        let neurotic = Personality::anxious();
        neurotic_state.update(&physics, &neurotic, &aether_params, 1.0);

        // Stable personality → dampened negative emotions
        let mut stable_state = ExtendedAffectiveState::default();
        let stable = Personality::stoic();
        stable_state.update(&physics, &stable, &aether_params, 1.0);

        // Neurotic should have higher fear activation than stoic
        let neurotic_fear = neurotic_state.plutchik.activations[crate::emotion::FEAR];
        let stable_fear = stable_state.plutchik.activations[crate::emotion::FEAR];
        assert!(
            neurotic_fear >= stable_fear,
            "neurotic fear {neurotic_fear} should be >= stoic fear {stable_fear}"
        );
    }

    #[test]
    fn test_pad_vector() {
        let mut state = ExtendedAffectiveState::default();
        state.pleasure = 0.5;
        state.arousal = -0.3;
        state.dominance = 0.7;
        let (p, a, d) = state.pad();
        assert!((p - 0.5).abs() < f32::EPSILON);
        assert!((a - (-0.3)).abs() < f32::EPSILON);
        assert!((d - 0.7).abs() < f32::EPSILON);
    }

    #[test]
    fn test_emotional_intensity() {
        let mut state = ExtendedAffectiveState::default();
        assert!(state.emotional_intensity() < f32::EPSILON);

        state.pleasure = 0.6;
        state.arousal = 0.8;
        state.dominance = 0.0;
        let expected = (0.36 + 0.64f32).sqrt();
        assert!((state.emotional_intensity() - expected).abs() < 1e-5);
    }

    #[test]
    fn test_peak_activation() {
        let mut state = ExtendedAffectiveState::default();
        state.plutchik.activations[crate::emotion::JOY] = 0.9;
        state.plutchik.activations[crate::emotion::TRUST] = 0.3;
        assert!((state.peak_activation() - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn test_stress_suppresses_joy() {
        let personality = Personality::cortana_default();
        let aether_params = ExtendedAetherParams::default();
        let physics = PhysicsInput {
            valence: 5.0,
            arousal: 0.3,
            phase_coherence: 0.8,
            criticality_sigma: 0.1,
            prediction_error: 0.1,
            convergence_stable: true,
            achievement: false,
        };

        // Calm state
        let mut calm = ExtendedAffectiveState::default();
        calm.update(&physics, &personality, &aether_params, 1.0);
        let calm_joy = calm.plutchik.activations[crate::emotion::JOY];

        // Stressed state (high cortisol pre-set)
        let mut stressed = ExtendedAffectiveState::default();
        stressed.extended_aether.cortisol = 0.9;
        stressed.update(&physics, &personality, &aether_params, 1.0);
        let stressed_joy = stressed.plutchik.activations[crate::emotion::JOY];

        assert!(
            calm_joy > stressed_joy,
            "cortisol should suppress joy: calm {calm_joy} > stressed {stressed_joy}"
        );
    }

    #[test]
    fn test_physics_input_from_affective_state() {
        let base = AffectiveState::baseline();
        let input = PhysicsInput::from_affective_state(&base, 0.1, 0.2, true, false);
        assert!((input.valence - base.valence).abs() < f32::EPSILON);
        assert!((input.arousal - base.arousal).abs() < f32::EPSILON);
        assert!((input.criticality_sigma - 0.1).abs() < f32::EPSILON);
        assert!((input.prediction_error - 0.2).abs() < f32::EPSILON);
        assert!(input.convergence_stable);
        assert!(!input.achievement);
    }

    #[test]
    fn test_repeated_updates_stable() {
        let mut state = ExtendedAffectiveState::default();
        let personality = Personality::cortana_default();
        let aether_params = ExtendedAetherParams::default();
        let physics = default_physics();

        for _ in 0..1000 {
            state.update(&physics, &personality, &aether_params, 1.0);
        }

        // All values should remain bounded
        assert!((-1.0..=1.0).contains(&state.pleasure));
        assert!((-1.0..=1.0).contains(&state.arousal));
        assert!((-1.0..=1.0).contains(&state.dominance));
        for &a in &state.plutchik.activations {
            assert!((0.0..=1.0).contains(&a));
        }
    }
}
