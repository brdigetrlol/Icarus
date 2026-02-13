//! Extended neuromodulator system — 8 neuromodulators with coupled ODE dynamics.
//!
//! Extends Icarus's 4 neuromodulators (DA, NE, ACh, 5-HT) with 4 social/stress/reward
//! modulators (oxytocin, endorphin, cortisol, GABA).

use icarus_field::autopoiesis::CognitiveAether;
use serde::{Deserialize, Serialize};

/// Extended neuromodulator state — 8 modulators total.
#[derive(Debug, Clone, Serialize)]
pub struct ExtendedAether {
    /// Icarus base neuromodulators (dopamine, norepinephrine, acetylcholine, serotonin).
    pub base: CognitiveAether,

    /// Oxytocin — bonding, trust, social connection.
    /// Rises with sustained trust+joy co-activation.
    pub oxytocin: f32,

    /// Endorphin — reward, flow states, pain modulation.
    /// Rises with stable convergence + high coherence.
    pub endorphin: f32,

    /// Cortisol — stress, threat response, resource mobilization.
    /// Rises with sustained fear/anger + high arousal.
    pub cortisol: f32,

    /// GABA — inhibition, calm, anxiety regulation.
    /// Counterbalances cortisol; rises when serotonin high + arousal low.
    pub gaba: f32,
}

/// Parameters for extended neuromodulator ODE dynamics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtendedAetherParams {
    // Oxytocin dynamics
    pub oxytocin_decay: f32,
    pub oxytocin_trust_gain: f32,
    pub oxytocin_joy_gain: f32,

    // Endorphin dynamics
    pub endorphin_decay: f32,
    pub endorphin_flow_gain: f32,
    pub endorphin_achievement_spike: f32,

    // Cortisol dynamics
    pub cortisol_decay: f32,
    pub cortisol_threat_gain: f32,
    pub cortisol_sustained_multiplier: f32,

    // GABA dynamics
    pub gaba_decay: f32,
    pub gaba_serotonin_gain: f32,
    pub gaba_calm_gain: f32,

    // Cross-modulation
    pub cortisol_suppression_factor: f32,
}

impl Default for ExtendedAetherParams {
    fn default() -> Self {
        Self {
            oxytocin_decay: 0.01,
            oxytocin_trust_gain: 0.05,
            oxytocin_joy_gain: 0.03,

            endorphin_decay: 0.02,
            endorphin_flow_gain: 0.04,
            endorphin_achievement_spike: 0.3,

            cortisol_decay: 0.015,
            cortisol_threat_gain: 0.06,
            cortisol_sustained_multiplier: 1.5,

            gaba_decay: 0.02,
            gaba_serotonin_gain: 0.04,
            gaba_calm_gain: 0.03,

            cortisol_suppression_factor: 0.2,
        }
    }
}

/// Signals from the emotion system that drive neuromodulator dynamics.
#[derive(Debug, Clone)]
pub struct AetherSignals {
    /// Joy activation [0, 1]
    pub joy: f32,
    /// Trust activation [0, 1]
    pub trust: f32,
    /// Fear activation [0, 1]
    pub fear: f32,
    /// Anger activation [0, 1]
    pub anger: f32,
    /// Arousal level [0, 1]
    pub arousal: f32,
    /// Phase coherence [0, 1]
    pub phase_coherence: f32,
    /// Whether convergence is stable
    pub convergence_stable: bool,
    /// Whether an achievement/milestone event occurred this tick
    pub achievement: bool,
}

impl ExtendedAether {
    pub fn new(base: CognitiveAether) -> Self {
        Self {
            base,
            oxytocin: 0.0,
            endorphin: 0.0,
            cortisol: 0.0,
            gaba: 0.3,
        }
    }

    pub fn from_base(base: CognitiveAether) -> Self {
        Self::new(base)
    }

    /// Step the extended neuromodulator ODEs forward by dt.
    pub fn step(&mut self, signals: &AetherSignals, params: &ExtendedAetherParams, dt: f32) {
        // Oxytocin: dOxy/dt = -decay*Oxy + trust_gain*Trust + joy_gain*Joy
        let d_oxytocin = -params.oxytocin_decay * self.oxytocin
            + params.oxytocin_trust_gain * signals.trust
            + params.oxytocin_joy_gain * signals.joy;
        self.oxytocin = (self.oxytocin + d_oxytocin * dt).clamp(0.0, 1.0);

        // Endorphin: dEnd/dt = -decay*End + flow_gain*(coherence * convergence_stable)
        let flow_signal = if signals.convergence_stable {
            signals.phase_coherence
        } else {
            0.0
        };
        let mut d_endorphin =
            -params.endorphin_decay * self.endorphin + params.endorphin_flow_gain * flow_signal;
        if signals.achievement {
            d_endorphin += params.endorphin_achievement_spike;
        }
        self.endorphin = (self.endorphin + d_endorphin * dt).clamp(0.0, 1.0);

        // Cortisol: dCort/dt = -decay*Cort + threat_gain*(Fear + Anger) * arousal
        let threat = (signals.fear + signals.anger) * signals.arousal;
        // Sustained high cortisol amplifies itself (positive feedback loop, capped)
        let sustained = if self.cortisol > 0.5 {
            params.cortisol_sustained_multiplier
        } else {
            1.0
        };
        let d_cortisol =
            -params.cortisol_decay * self.cortisol + params.cortisol_threat_gain * threat * sustained;
        self.cortisol = (self.cortisol + d_cortisol * dt).clamp(0.0, 1.0);

        // GABA: dGABA/dt = -decay*GABA + serotonin_gain*5HT + calm_gain*(1-arousal) - cortisol_suppression*Cort
        let d_gaba = -params.gaba_decay * self.gaba
            + params.gaba_serotonin_gain * self.base.serotonin
            + params.gaba_calm_gain * (1.0 - signals.arousal)
            - params.cortisol_suppression_factor * self.cortisol;
        self.gaba = (self.gaba + d_gaba * dt).clamp(0.0, 1.0);
    }

    /// Get the cortisol-mediated suppression factor for positive neuromodulators.
    /// High cortisol suppresses dopamine, oxytocin, and endorphin responses.
    pub fn stress_suppression(&self) -> f32 {
        1.0 - self.cortisol * 0.3
    }

    /// Overall emotional regulation capacity (GABA counterbalancing cortisol).
    pub fn regulation_capacity(&self) -> f32 {
        (self.gaba - self.cortisol * 0.5 + 0.5).clamp(0.0, 1.0)
    }

    /// Social bonding strength (oxytocin + serotonin contribution).
    pub fn social_bonding(&self) -> f32 {
        (self.oxytocin * 0.7 + self.base.serotonin * 0.3).clamp(0.0, 1.0)
    }

    /// Flow state indicator (endorphin + dopamine + low cortisol).
    pub fn flow_state(&self) -> f32 {
        ((self.endorphin + self.base.dopamine) * 0.5 * (1.0 - self.cortisol * 0.5))
            .clamp(0.0, 1.0)
    }
}

impl Default for ExtendedAether {
    fn default() -> Self {
        Self::new(CognitiveAether::baseline())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_signals() -> AetherSignals {
        AetherSignals {
            joy: 0.0,
            trust: 0.0,
            fear: 0.0,
            anger: 0.0,
            arousal: 0.0,
            phase_coherence: 0.0,
            convergence_stable: false,
            achievement: false,
        }
    }

    #[test]
    fn test_new_extended_aether() {
        let aether = ExtendedAether::default();
        assert!(aether.oxytocin.abs() < f32::EPSILON);
        assert!(aether.endorphin.abs() < f32::EPSILON);
        assert!(aether.cortisol.abs() < f32::EPSILON);
        assert!(aether.gaba > 0.0);
    }

    #[test]
    fn test_oxytocin_rises_with_trust_and_joy() {
        let mut aether = ExtendedAether::default();
        let params = ExtendedAetherParams::default();
        let signals = AetherSignals {
            joy: 0.8,
            trust: 0.7,
            ..default_signals()
        };

        let initial = aether.oxytocin;
        for _ in 0..100 {
            aether.step(&signals, &params, 1.0);
        }
        assert!(aether.oxytocin > initial);
    }

    #[test]
    fn test_endorphin_rises_with_flow_state() {
        let mut aether = ExtendedAether::default();
        let params = ExtendedAetherParams::default();
        let signals = AetherSignals {
            phase_coherence: 0.9,
            convergence_stable: true,
            ..default_signals()
        };

        for _ in 0..100 {
            aether.step(&signals, &params, 1.0);
        }
        assert!(aether.endorphin > 0.1);
    }

    #[test]
    fn test_endorphin_achievement_spike() {
        let mut aether = ExtendedAether::default();
        let params = ExtendedAetherParams::default();
        let signals = AetherSignals {
            achievement: true,
            ..default_signals()
        };

        aether.step(&signals, &params, 1.0);
        assert!(aether.endorphin > 0.0);
    }

    #[test]
    fn test_cortisol_rises_with_threat() {
        let mut aether = ExtendedAether::default();
        let params = ExtendedAetherParams::default();
        let signals = AetherSignals {
            fear: 0.8,
            anger: 0.5,
            arousal: 0.9,
            ..default_signals()
        };

        for _ in 0..100 {
            aether.step(&signals, &params, 1.0);
        }
        assert!(aether.cortisol > 0.3);
    }

    #[test]
    fn test_gaba_rises_with_calm() {
        let mut aether = ExtendedAether::default();
        aether.base.serotonin = 0.8;
        let params = ExtendedAetherParams::default();
        let signals = AetherSignals {
            arousal: 0.1,
            ..default_signals()
        };

        for _ in 0..100 {
            aether.step(&signals, &params, 1.0);
        }
        assert!(aether.gaba > 0.3);
    }

    #[test]
    fn test_cortisol_suppresses_gaba() {
        let mut aether = ExtendedAether::default();
        aether.cortisol = 0.8;
        aether.gaba = 0.5;
        let params = ExtendedAetherParams::default();
        let signals = AetherSignals {
            fear: 0.8,
            arousal: 0.9,
            ..default_signals()
        };

        let initial_gaba = aether.gaba;
        aether.step(&signals, &params, 1.0);
        assert!(aether.gaba < initial_gaba);
    }

    #[test]
    fn test_stress_suppression() {
        let mut aether = ExtendedAether::default();
        assert!((aether.stress_suppression() - 1.0).abs() < f32::EPSILON);

        aether.cortisol = 1.0;
        assert!(aether.stress_suppression() < 1.0);
    }

    #[test]
    fn test_regulation_capacity() {
        let mut aether = ExtendedAether::default();
        aether.gaba = 0.8;
        aether.cortisol = 0.0;
        let high_reg = aether.regulation_capacity();

        aether.gaba = 0.1;
        aether.cortisol = 0.9;
        let low_reg = aether.regulation_capacity();

        assert!(high_reg > low_reg);
    }

    #[test]
    fn test_social_bonding() {
        let mut aether = ExtendedAether::default();
        aether.oxytocin = 0.8;
        aether.base.serotonin = 0.7;
        assert!(aether.social_bonding() > 0.5);
    }

    #[test]
    fn test_flow_state() {
        let mut aether = ExtendedAether::default();
        aether.endorphin = 0.8;
        aether.base.dopamine = 0.7;
        aether.cortisol = 0.0;
        assert!(aether.flow_state() > 0.3);

        // High cortisol reduces flow
        aether.cortisol = 0.9;
        let stressed_flow = aether.flow_state();
        aether.cortisol = 0.0;
        let calm_flow = aether.flow_state();
        assert!(calm_flow > stressed_flow);
    }

    #[test]
    fn test_all_values_clamped() {
        let mut aether = ExtendedAether::default();
        let params = ExtendedAetherParams::default();

        // Extreme positive signals
        let signals = AetherSignals {
            joy: 1.0,
            trust: 1.0,
            fear: 1.0,
            anger: 1.0,
            arousal: 1.0,
            phase_coherence: 1.0,
            convergence_stable: true,
            achievement: true,
        };

        for _ in 0..10000 {
            aether.step(&signals, &params, 1.0);
        }

        assert!((0.0..=1.0).contains(&aether.oxytocin));
        assert!((0.0..=1.0).contains(&aether.endorphin));
        assert!((0.0..=1.0).contains(&aether.cortisol));
        assert!((0.0..=1.0).contains(&aether.gaba));
    }

    #[test]
    fn test_decay_toward_zero() {
        let mut aether = ExtendedAether::default();
        aether.oxytocin = 0.5;
        aether.endorphin = 0.5;
        aether.cortisol = 0.5;
        let params = ExtendedAetherParams::default();
        let signals = default_signals();

        for _ in 0..1000 {
            aether.step(&signals, &params, 1.0);
        }

        assert!(aether.oxytocin < 0.01);
        assert!(aether.endorphin < 0.01);
        assert!(aether.cortisol < 0.01);
    }
}
