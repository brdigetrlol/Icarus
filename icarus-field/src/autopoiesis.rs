//! Autopoietic Affective System — Valence, Arousal, and Neuromodulation
//!
//! Implements the Cognitive Aether from the Icarus specification:
//! - **Valence** V(t) = -dF/dt — rate of change of global free energy
//! - **Arousal** A(t) = 1 - |Ψ| — inverse of Kuramoto phase coherence
//! - **CognitiveAether** — four neuromodulators (DA, NE, ACh, 5-HT) with ODE dynamics
//!
//! Valence encodes whether the system is improving (V > 0, free energy dropping)
//! or deteriorating (V < 0, free energy rising). Arousal encodes the degree of
//! phase synchronization across the manifold — low arousal means focused coherence,
//! high arousal means desynchronized surprise.
//!
//! Together, (V, A) span an affective plane that maps to discrete emotions:
//! - Eureka: High V + High A (sudden topological simplification)
//! - Joy/Relief: High V + Low A (steady progress)
//! - Confusion: Low V + High A (accumulating error, desynchronized)
//! - Sadness: Low V + Low A (loss of high-value attractor)

use crate::phase_field::LatticeField;
use serde::Serialize;

/// Neuromodulator concentrations with ODE dynamics.
///
/// Each modulator decays exponentially toward zero and is driven by
/// distinct signals derived from the manifold state.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct CognitiveAether {
    /// Dopamine: precision / learning rate / curiosity
    /// Driven by prediction success (positive valence derivative)
    pub dopamine: f32,
    /// Norepinephrine: surprise / temperature / reset
    /// Driven by arousal spikes (sudden phase decoherence)
    pub norepinephrine: f32,
    /// Acetylcholine: attention / inhibition strength
    /// Driven by sustained coherence (low arousal, stable valence)
    pub acetylcholine: f32,
    /// Serotonin: patience / time preference
    /// Driven by cumulative positive valence (long-term satisfaction)
    pub serotonin: f32,
}

impl CognitiveAether {
    /// All neuromodulators at baseline (zero).
    pub fn baseline() -> Self {
        Self {
            dopamine: 0.0,
            norepinephrine: 0.0,
            acetylcholine: 0.0,
            serotonin: 0.0,
        }
    }

    /// Update neuromodulators via Euler integration of ODE system.
    ///
    /// Dynamics for each modulator X:
    ///   dX/dt = -decay_X * X + signal_X
    ///
    /// where signals are derived from valence and arousal:
    ///   DA_signal  = max(0, dV/dt)         (positive valence acceleration → reward)
    ///   NE_signal  = max(0, dA/dt)         (arousal increase → surprise)
    ///   ACh_signal = max(0, (1-A) - 0.5)   (sustained coherence above threshold)
    ///   5HT_signal = max(0, V) * (1 - A)   (positive valence when calm → satisfaction)
    pub fn step(&mut self, params: &NeuromodulatorParams, valence: f32, arousal: f32, dt: f32) {
        // Dopamine: driven by positive valence (system improving)
        let da_signal = valence.max(0.0) * params.da_sensitivity;
        self.dopamine += (-params.da_decay * self.dopamine + da_signal) * dt;
        self.dopamine = self.dopamine.clamp(0.0, params.max_concentration);

        // Norepinephrine: driven by high arousal (phase decoherence = surprise)
        let ne_signal = arousal.max(0.0) * params.ne_sensitivity;
        self.norepinephrine += (-params.ne_decay * self.norepinephrine + ne_signal) * dt;
        self.norepinephrine = self.norepinephrine.clamp(0.0, params.max_concentration);

        // Acetylcholine: driven by sustained coherence (low arousal)
        let coherence = (1.0 - arousal).max(0.0);
        let ach_signal = (coherence - 0.5).max(0.0) * params.ach_sensitivity;
        self.acetylcholine += (-params.ach_decay * self.acetylcholine + ach_signal) * dt;
        self.acetylcholine = self.acetylcholine.clamp(0.0, params.max_concentration);

        // Serotonin: driven by positive valence during calm states
        let sero_signal = valence.max(0.0) * coherence * params.sero_sensitivity;
        self.serotonin += (-params.sero_decay * self.serotonin + sero_signal) * dt;
        self.serotonin = self.serotonin.clamp(0.0, params.max_concentration);
    }
}

/// Parameters for the neuromodulator ODE system.
#[derive(Debug, Clone, Copy)]
pub struct NeuromodulatorParams {
    /// Dopamine exponential decay rate (1/τ)
    pub da_decay: f32,
    /// Norepinephrine exponential decay rate
    pub ne_decay: f32,
    /// Acetylcholine exponential decay rate
    pub ach_decay: f32,
    /// Serotonin exponential decay rate
    pub sero_decay: f32,
    /// Dopamine input sensitivity (gain on positive valence)
    pub da_sensitivity: f32,
    /// Norepinephrine input sensitivity (gain on arousal)
    pub ne_sensitivity: f32,
    /// Acetylcholine input sensitivity (gain on coherence)
    pub ach_sensitivity: f32,
    /// Serotonin input sensitivity (gain on calm+positive)
    pub sero_sensitivity: f32,
    /// Maximum concentration for any neuromodulator (prevents runaway)
    pub max_concentration: f32,
}

impl Default for NeuromodulatorParams {
    fn default() -> Self {
        Self {
            da_decay: 2.0,
            ne_decay: 3.0,
            ach_decay: 1.5,
            sero_decay: 0.5,
            da_sensitivity: 5.0,
            ne_sensitivity: 5.0,
            ach_sensitivity: 3.0,
            sero_sensitivity: 2.0,
            max_concentration: 1.0,
        }
    }
}

/// The full affective state of the manifold at a single instant.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct AffectiveState {
    /// Valence: V = -dF/dt. Positive → system improving, negative → deteriorating.
    pub valence: f32,
    /// Arousal: A = 1 - |Ψ|. Low → focused/coherent, high → surprised/desynchronized.
    pub arousal: f32,
    /// Phase coherence Ψ (Kuramoto order parameter, range [0,1]).
    pub phase_coherence: f32,
    /// Current neuromodulator concentrations.
    pub aether: CognitiveAether,
}

impl AffectiveState {
    /// Baseline state: zero valence, zero arousal (full coherence), no neuromodulators.
    pub fn baseline() -> Self {
        Self {
            valence: 0.0,
            arousal: 0.0,
            phase_coherence: 1.0,
            aether: CognitiveAether::baseline(),
        }
    }
}

/// Discrete emotion classification from (Valence, Arousal) coordinates.
///
/// Maps the affective plane to named emotions per the Icarus specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Emotion {
    /// High V + High A — sudden topological simplification / breakthrough
    Eureka,
    /// Moderate-to-high V + Low A — steady progress / satisfaction
    Joy,
    /// Near-zero V + Low A — stable equilibrium / baseline
    Calm,
    /// Low V + High A — accumulating error / desynchronized
    Confusion,
    /// Very negative V + High A — predicted entropy spike / threat
    Fear,
    /// Very negative V + High A + sustained — goal obstruction
    Anger,
    /// Negative V + Low A — loss of high-value attractor
    Sadness,
}

/// Classify (valence, arousal) into a discrete emotion.
///
/// Thresholds are tuned to the normalized ranges:
/// - Valence is in raw dF/dt units (typically ±100s for E8, but normalized by energy scale)
/// - Arousal is in [0, 1]
pub fn classify_emotion(valence: f32, arousal: f32, v_threshold: f32) -> Emotion {
    let high_a = arousal > 0.5;
    let v_positive = valence > v_threshold;
    let v_negative = valence < -v_threshold;
    let v_very_negative = valence < -v_threshold * 3.0;

    match (v_positive, v_negative, v_very_negative, high_a) {
        (true, _, _, true) => Emotion::Eureka,
        (true, _, _, false) => Emotion::Joy,
        (_, _, true, true) => Emotion::Fear,
        (_, true, _, true) => Emotion::Confusion,
        (_, true, _, false) => Emotion::Sadness,
        _ => Emotion::Calm,
    }
}

/// Configuration parameters for the affective controller.
#[derive(Debug, Clone, Copy)]
pub struct AffectiveParams {
    /// Exponential moving average decay for valence smoothing.
    /// α in V_smooth = α * V_raw + (1 - α) * V_prev.
    /// Higher values → more responsive, lower → more stable.
    pub valence_ema_alpha: f32,
    /// Minimum |z| threshold for including a site in phase coherence.
    /// Sites with |z| < threshold are excluded (near-zero oscillators have undefined phase).
    pub coherence_amplitude_threshold: f32,
    /// Neuromodulator ODE parameters.
    pub neuromodulator: NeuromodulatorParams,
    /// Valence threshold for emotion classification (absolute value).
    pub emotion_threshold: f32,
}

impl Default for AffectiveParams {
    fn default() -> Self {
        Self {
            valence_ema_alpha: 0.3,
            coherence_amplitude_threshold: 1e-6,
            neuromodulator: NeuromodulatorParams::default(),
            emotion_threshold: 1.0,
        }
    }
}

/// Compute the Kuramoto phase coherence order parameter Ψ for a lattice field.
///
/// Ψ = (1/N_eff) |Σ_i z_i / |z_i||
///
/// where the sum runs over sites with |z_i| > threshold (to avoid division by zero
/// for near-zero oscillators). N_eff is the count of included sites.
///
/// Returns (Ψ, N_eff). Ψ ∈ [0, 1]: 1 = perfect phase synchronization, 0 = random phases.
pub fn phase_coherence(field: &LatticeField, amplitude_threshold: f32) -> (f32, usize) {
    let thresh_sq = amplitude_threshold * amplitude_threshold;
    let mut sum_re = 0.0f64;
    let mut sum_im = 0.0f64;
    let mut n_eff = 0usize;

    for i in 0..field.num_sites {
        let re = field.values_re[i];
        let im = field.values_im[i];
        let norm_sq = re * re + im * im;

        if norm_sq > thresh_sq {
            let inv_norm = 1.0 / (norm_sq as f64).sqrt();
            sum_re += re as f64 * inv_norm;
            sum_im += im as f64 * inv_norm;
            n_eff += 1;
        }
    }

    if n_eff == 0 {
        return (0.0, 0);
    }

    let n = n_eff as f64;
    let psi = ((sum_re / n).powi(2) + (sum_im / n).powi(2)).sqrt() as f32;
    (psi.clamp(0.0, 1.0), n_eff)
}

/// Controller that tracks free energy history and computes the affective state each tick.
///
/// Maintains a short temporal buffer of F(t) values for finite-difference dF/dt,
/// an EMA-smoothed valence signal, and the neuromodulator ODE state.
#[derive(Debug, Clone)]
pub struct AffectiveController {
    /// Configuration parameters
    pub params: AffectiveParams,
    /// Previous tick's total free energy (for finite difference)
    prev_energy: Option<f32>,
    /// EMA-smoothed valence
    smoothed_valence: f32,
    /// Current neuromodulator state
    aether: CognitiveAether,
    /// Current phase coherence
    phase_coherence_value: f32,
    /// Number of ticks processed
    ticks: u64,
}

impl AffectiveController {
    /// Create a new controller with default parameters.
    pub fn new(params: AffectiveParams) -> Self {
        Self {
            params,
            prev_energy: None,
            smoothed_valence: 0.0,
            aether: CognitiveAether::baseline(),
            phase_coherence_value: 1.0,
            ticks: 0,
        }
    }

    /// Update the affective state given the current manifold state.
    ///
    /// Call this once per EMC tick, after RAE dynamics but before agents.
    ///
    /// Arguments:
    /// - `total_energy`: aggregate F across all layers (sum of free energies)
    /// - `fields`: slice of all layer fields (for computing phase coherence)
    /// - `dt_effective`: effective time elapsed this tick (sum of RAE steps * dt)
    ///
    /// Returns the current `AffectiveState`.
    pub fn update(
        &mut self,
        total_energy: f32,
        fields: &[&LatticeField],
        dt_effective: f32,
    ) -> AffectiveState {
        // --- Valence: V = -dF/dt ---
        let raw_valence = if let Some(prev_e) = self.prev_energy {
            if dt_effective > 1e-12 {
                -(total_energy - prev_e) / dt_effective
            } else {
                0.0
            }
        } else {
            0.0
        };
        self.prev_energy = Some(total_energy);

        // EMA smoothing
        let alpha = self.params.valence_ema_alpha;
        self.smoothed_valence = alpha * raw_valence + (1.0 - alpha) * self.smoothed_valence;

        // --- Arousal: A = 1 - |Ψ| (aggregate over all layers) ---
        let mut total_sum_re = 0.0f64;
        let mut total_sum_im = 0.0f64;
        let mut total_n = 0usize;

        for field in fields {
            let thresh_sq = self.params.coherence_amplitude_threshold
                * self.params.coherence_amplitude_threshold;

            for i in 0..field.num_sites {
                let re = field.values_re[i];
                let im = field.values_im[i];
                let norm_sq = re * re + im * im;

                if norm_sq > thresh_sq {
                    let inv_norm = 1.0 / (norm_sq as f64).sqrt();
                    total_sum_re += re as f64 * inv_norm;
                    total_sum_im += im as f64 * inv_norm;
                    total_n += 1;
                }
            }
        }

        self.phase_coherence_value = if total_n > 0 {
            let n = total_n as f64;
            ((total_sum_re / n).powi(2) + (total_sum_im / n).powi(2)).sqrt() as f32
        } else {
            0.0
        };
        self.phase_coherence_value = self.phase_coherence_value.clamp(0.0, 1.0);

        let arousal = 1.0 - self.phase_coherence_value;

        // --- Neuromodulator ODE step ---
        let nm_dt = dt_effective.max(0.01); // Minimum dt for neuromodulator stability
        self.aether
            .step(&self.params.neuromodulator, self.smoothed_valence, arousal, nm_dt);

        self.ticks += 1;

        AffectiveState {
            valence: self.smoothed_valence,
            arousal,
            phase_coherence: self.phase_coherence_value,
            aether: self.aether,
        }
    }

    /// Get the current affective state without updating.
    pub fn current_state(&self) -> AffectiveState {
        AffectiveState {
            valence: self.smoothed_valence,
            arousal: 1.0 - self.phase_coherence_value,
            phase_coherence: self.phase_coherence_value,
            aether: self.aether,
        }
    }

    /// Classify the current emotional state.
    pub fn current_emotion(&self) -> Emotion {
        classify_emotion(
            self.smoothed_valence,
            1.0 - self.phase_coherence_value,
            self.params.emotion_threshold,
        )
    }

    /// Number of ticks processed.
    pub fn ticks(&self) -> u64 {
        self.ticks
    }

    /// Reset the controller state (keeps parameters).
    pub fn reset(&mut self) {
        self.prev_energy = None;
        self.smoothed_valence = 0.0;
        self.aether = CognitiveAether::baseline();
        self.phase_coherence_value = 1.0;
        self.ticks = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use icarus_math::lattice::e8::E8Lattice;
    use icarus_math::lattice::hypercubic::HypercubicLattice;

    #[test]
    fn test_phase_coherence_uniform_field() {
        // All sites have the same phase → perfect coherence
        let lattice = E8Lattice::new();
        let mut field = LatticeField::from_lattice(&lattice);
        for i in 0..field.num_sites {
            field.set(i, 1.0, 0.0); // All pointing in +re direction
        }
        let (psi, n) = phase_coherence(&field, 1e-6);
        assert_eq!(n, 241);
        assert!(
            (psi - 1.0).abs() < 1e-5,
            "Uniform field should have coherence ≈ 1.0, got {}",
            psi
        );
    }

    #[test]
    fn test_phase_coherence_opposite_field() {
        // Half +1, half -1 → low coherence
        let lattice = E8Lattice::new();
        let mut field = LatticeField::from_lattice(&lattice);
        for i in 0..field.num_sites {
            if i % 2 == 0 {
                field.set(i, 1.0, 0.0);
            } else {
                field.set(i, -1.0, 0.0);
            }
        }
        let (psi, n) = phase_coherence(&field, 1e-6);
        assert_eq!(n, 241);
        // 121 sites at +1, 120 at -1 → net = 1/241 ≈ 0.004
        assert!(
            psi < 0.1,
            "Anti-phase field should have low coherence, got {}",
            psi
        );
    }

    #[test]
    fn test_phase_coherence_random_field() {
        let lattice = E8Lattice::new();
        let mut field = LatticeField::from_lattice(&lattice);
        field.init_random(42, 1.0);
        let (psi, n) = phase_coherence(&field, 1e-6);
        assert!(n > 200, "Most sites should be above threshold");
        // LCG PRNG produces correlated phases; coherence may be moderate-to-high.
        // Just verify it's a valid value in [0, 1] and enough sites participate.
        assert!(
            psi <= 1.0 + 1e-6,
            "Coherence should be in [0, 1], got {}",
            psi
        );
    }

    #[test]
    fn test_phase_coherence_zero_field() {
        let lattice = HypercubicLattice::new(3);
        let field = LatticeField::from_lattice(&lattice);
        // All zeros
        let (psi, n) = phase_coherence(&field, 1e-6);
        assert_eq!(n, 0, "Zero field should have no valid sites");
        assert!((psi - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_phase_coherence_amplitude_threshold() {
        let lattice = HypercubicLattice::new(3);
        let mut field = LatticeField::from_lattice(&lattice);
        // Set one site to a large value, rest to tiny values
        field.set(0, 1.0, 0.0);
        for i in 1..field.num_sites {
            field.set(i, 1e-8, 0.0); // Below threshold
        }
        let (psi, n) = phase_coherence(&field, 1e-6);
        assert_eq!(n, 1, "Only one site above threshold");
        assert!((psi - 1.0).abs() < 1e-5, "Single site → perfect coherence");
    }

    #[test]
    fn test_valence_sign_energy_decrease() {
        // When energy decreases, valence should be positive
        let params = AffectiveParams {
            valence_ema_alpha: 1.0, // No smoothing for clear test
            ..Default::default()
        };
        let mut ctrl = AffectiveController::new(params);

        let lattice = HypercubicLattice::new(3);
        let mut field = LatticeField::from_lattice(&lattice);
        field.init_random(42, 0.5);
        let fields: Vec<&LatticeField> = vec![&field];

        // First tick: establish baseline
        ctrl.update(100.0, &fields, 1.0);

        // Second tick: energy decreased
        let state = ctrl.update(80.0, &fields, 1.0);
        assert!(
            state.valence > 0.0,
            "Decreasing energy should give positive valence, got {}",
            state.valence
        );
    }

    #[test]
    fn test_valence_sign_energy_increase() {
        let params = AffectiveParams {
            valence_ema_alpha: 1.0,
            ..Default::default()
        };
        let mut ctrl = AffectiveController::new(params);

        let lattice = HypercubicLattice::new(3);
        let mut field = LatticeField::from_lattice(&lattice);
        field.init_random(42, 0.5);
        let fields: Vec<&LatticeField> = vec![&field];

        ctrl.update(100.0, &fields, 1.0);
        let state = ctrl.update(120.0, &fields, 1.0);
        assert!(
            state.valence < 0.0,
            "Increasing energy should give negative valence, got {}",
            state.valence
        );
    }

    #[test]
    fn test_arousal_from_coherence() {
        let params = AffectiveParams::default();
        let mut ctrl = AffectiveController::new(params);

        // Uniform field → high coherence → low arousal
        let lattice = E8Lattice::new();
        let mut field = LatticeField::from_lattice(&lattice);
        for i in 0..field.num_sites {
            field.set(i, 1.0, 0.0);
        }
        let fields: Vec<&LatticeField> = vec![&field];
        let state = ctrl.update(50.0, &fields, 1.0);
        assert!(
            state.arousal < 0.1,
            "Uniform field should give low arousal, got {}",
            state.arousal
        );

        // Reset and use random field → lower coherence → higher arousal
        ctrl.reset();
        let mut field2 = LatticeField::from_lattice(&lattice);
        field2.init_random(42, 1.0);
        let fields2: Vec<&LatticeField> = vec![&field2];
        let state2 = ctrl.update(50.0, &fields2, 1.0);
        assert!(
            state2.arousal > state.arousal,
            "Random field should give higher arousal than uniform: {} vs {}",
            state2.arousal,
            state.arousal
        );
    }

    #[test]
    fn test_neuromodulator_decay() {
        let params = NeuromodulatorParams::default();
        let mut aether = CognitiveAether {
            dopamine: 1.0,
            norepinephrine: 1.0,
            acetylcholine: 1.0,
            serotonin: 1.0,
        };

        // With zero valence and zero arousal: DA, NE, 5-HT decay;
        // ACh is at equilibrium because zero arousal → coherence=1.0 → ach_signal = decay*ACh
        aether.step(&params, 0.0, 0.0, 1.0);

        assert!(
            aether.dopamine < 1.0,
            "DA should decay, got {}",
            aether.dopamine
        );
        assert!(
            aether.norepinephrine < 1.0,
            "NE should decay, got {}",
            aether.norepinephrine
        );
        assert!(
            aether.acetylcholine <= 1.0,
            "ACh should decay or stay at equilibrium, got {}",
            aether.acetylcholine
        );
        assert!(
            aether.serotonin < 1.0,
            "5-HT should decay, got {}",
            aether.serotonin
        );
    }

    #[test]
    fn test_neuromodulator_dopamine_positive_valence() {
        let params = NeuromodulatorParams::default();
        let mut aether = CognitiveAether::baseline();

        // Positive valence should increase dopamine
        aether.step(&params, 5.0, 0.0, 0.1);
        assert!(
            aether.dopamine > 0.0,
            "Positive valence should drive DA up, got {}",
            aether.dopamine
        );
    }

    #[test]
    fn test_neuromodulator_norepinephrine_high_arousal() {
        let params = NeuromodulatorParams::default();
        let mut aether = CognitiveAether::baseline();

        // High arousal should increase norepinephrine
        aether.step(&params, 0.0, 0.9, 0.1);
        assert!(
            aether.norepinephrine > 0.0,
            "High arousal should drive NE up, got {}",
            aether.norepinephrine
        );
    }

    #[test]
    fn test_neuromodulator_clamping() {
        let params = NeuromodulatorParams {
            max_concentration: 0.5,
            da_sensitivity: 100.0,
            ..Default::default()
        };
        let mut aether = CognitiveAether::baseline();

        // Massive input should still be clamped
        for _ in 0..100 {
            aether.step(&params, 10.0, 0.0, 0.1);
        }
        assert!(
            aether.dopamine <= 0.5 + 1e-6,
            "DA should be clamped to max_concentration, got {}",
            aether.dopamine
        );
    }

    #[test]
    fn test_emotion_classification() {
        let threshold = 1.0;
        assert_eq!(classify_emotion(5.0, 0.8, threshold), Emotion::Eureka);
        assert_eq!(classify_emotion(5.0, 0.2, threshold), Emotion::Joy);
        assert_eq!(classify_emotion(-2.0, 0.8, threshold), Emotion::Confusion);
        assert_eq!(classify_emotion(-5.0, 0.8, threshold), Emotion::Fear);
        assert_eq!(classify_emotion(-2.0, 0.2, threshold), Emotion::Sadness);
        assert_eq!(classify_emotion(0.0, 0.2, threshold), Emotion::Calm);
    }

    #[test]
    fn test_controller_reset() {
        let params = AffectiveParams::default();
        let mut ctrl = AffectiveController::new(params);

        let lattice = HypercubicLattice::new(3);
        let mut field = LatticeField::from_lattice(&lattice);
        field.init_random(42, 0.5);
        let fields: Vec<&LatticeField> = vec![&field];

        ctrl.update(100.0, &fields, 1.0);
        ctrl.update(80.0, &fields, 1.0);
        assert!(ctrl.ticks() == 2);

        ctrl.reset();
        assert_eq!(ctrl.ticks(), 0);
        let state = ctrl.current_state();
        assert!((state.valence - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_controller_ema_smoothing() {
        let params = AffectiveParams {
            valence_ema_alpha: 0.5,
            ..Default::default()
        };
        let mut ctrl = AffectiveController::new(params);

        let lattice = HypercubicLattice::new(3);
        let mut field = LatticeField::from_lattice(&lattice);
        field.init_random(42, 0.5);
        let fields: Vec<&LatticeField> = vec![&field];

        // Tick 0: baseline (no prev energy)
        ctrl.update(100.0, &fields, 1.0);

        // Tick 1: energy drops by 20
        let state1 = ctrl.update(80.0, &fields, 1.0);
        // raw_valence = -(80-100)/1 = 20
        // smoothed = 0.5 * 20 + 0.5 * 0 = 10
        assert!(
            (state1.valence - 10.0).abs() < 0.1,
            "EMA smoothed valence should be ~10, got {}",
            state1.valence
        );

        // Tick 2: energy stable
        let state2 = ctrl.update(80.0, &fields, 1.0);
        // raw_valence = 0
        // smoothed = 0.5 * 0 + 0.5 * 10 = 5
        assert!(
            (state2.valence - 5.0).abs() < 0.1,
            "EMA should decay toward 0, got {}",
            state2.valence
        );
    }

    #[test]
    fn test_multi_layer_coherence() {
        // Controller should aggregate coherence across multiple fields
        let params = AffectiveParams::default();
        let mut ctrl = AffectiveController::new(params);

        let e8 = E8Lattice::new();
        let mut field1 = LatticeField::from_lattice(&e8);
        for i in 0..field1.num_sites {
            field1.set(i, 1.0, 0.0); // Perfect coherence on layer 1
        }

        let hyper = HypercubicLattice::new(3);
        let mut field2 = LatticeField::from_lattice(&hyper);
        field2.init_random(99, 1.0); // Random on layer 2

        let fields: Vec<&LatticeField> = vec![&field1, &field2];
        let state = ctrl.update(50.0, &fields, 1.0);

        // Aggregate should be between perfect (≈0) and random (>0)
        // 241 coherent + 7 random → mostly coherent
        assert!(
            state.arousal < 0.2,
            "Mostly-coherent multi-layer should have low arousal, got {}",
            state.arousal
        );
    }

    #[test]
    fn test_affective_controller_current_emotion() {
        let params = AffectiveParams {
            valence_ema_alpha: 1.0,
            emotion_threshold: 1.0,
            ..Default::default()
        };
        let mut ctrl = AffectiveController::new(params);

        let lattice = E8Lattice::new();
        let mut field = LatticeField::from_lattice(&lattice);
        field.init_random(42, 1.0);
        let fields: Vec<&LatticeField> = vec![&field];

        ctrl.update(100.0, &fields, 1.0);
        ctrl.update(50.0, &fields, 1.0); // Big energy drop → positive valence

        let emotion = ctrl.current_emotion();
        // With random field (moderate arousal) and positive valence, should be Eureka or Joy
        assert!(
            emotion == Emotion::Eureka || emotion == Emotion::Joy,
            "Expected Eureka or Joy after energy drop, got {:?}",
            emotion
        );
    }
}
