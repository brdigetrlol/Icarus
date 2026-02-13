//! Readout mechanisms for extracting predictions from EMC state.
//!
//! Two readout strategies:
//! 1. **Linear**: `output = W · state + bias` — trainable via ridge regression
//! 2. **Direct**: Raw state extraction for external processing
//!
//! Plus `StateCollector` for gathering training data from reservoir dynamics.
//!
//! Feature modes control what state features are extracted:
//! - **Linear**: `[re, im]` — 2N features (default)
//! - **Nonlinear**: `[re, im, re², im²]` — 4N features (quadratic expansion)

use icarus_field::phase_field::LatticeField;

// ─── Feature Mode ───────────────────────────────────

/// Controls which features are extracted from the lattice field state.
///
/// Nonlinear features dramatically improve performance on tasks requiring
/// temporal memory and nonlinear computation (e.g., NARMA-10).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureMode {
    /// Linear features only: `[re_0..re_N, im_0..im_N]` — 2N features.
    Linear,
    /// Quadratic expansion: `[re, im, re², im²]` — 4N features.
    /// Gives ridge regression access to nonlinear combinations of state values.
    Nonlinear,
}

impl Default for FeatureMode {
    fn default() -> Self {
        Self::Linear
    }
}

/// Extract a feature vector from a lattice field according to the given mode.
///
/// - `Linear`: `[re_0..re_N, im_0..im_N]` — 2N features
/// - `Nonlinear`: `[re_0..re_N, im_0..im_N, re_0²..re_N², im_0²..im_N²]` — 4N features
pub fn extract_features(field: &LatticeField, mode: FeatureMode) -> Vec<f32> {
    let n = field.num_sites;
    match mode {
        FeatureMode::Linear => {
            let mut state = Vec::with_capacity(2 * n);
            state.extend_from_slice(&field.values_re);
            state.extend_from_slice(&field.values_im);
            state
        }
        FeatureMode::Nonlinear => {
            let mut state = Vec::with_capacity(4 * n);
            // Linear terms
            state.extend_from_slice(&field.values_re);
            state.extend_from_slice(&field.values_im);
            // Quadratic terms: re², im²
            for i in 0..n {
                state.push(field.values_re[i] * field.values_re[i]);
            }
            for i in 0..n {
                state.push(field.values_im[i] * field.values_im[i]);
            }
            state
        }
    }
}

/// Compute the feature dimension for a given number of sites and feature mode.
pub fn feature_dim(num_sites: usize, mode: FeatureMode) -> usize {
    match mode {
        FeatureMode::Linear => 2 * num_sites,
        FeatureMode::Nonlinear => 4 * num_sites,
    }
}

/// Trait for reading output from a lattice field state.
pub trait Readout: Send + Sync {
    /// Read output values from the field state.
    fn read(&self, field: &LatticeField) -> Vec<f32>;

    /// Name of this readout strategy.
    fn name(&self) -> &str;
}

// ─── Linear Readout ──────────────────────────────────

/// Linear readout: `output = W · features + bias`
///
/// Features are extracted from the lattice field according to `feature_mode`:
/// - `Linear`: `[re, im]` — state_dim = 2N
/// - `Nonlinear`: `[re, im, re², im²]` — state_dim = 4N
///
/// `W` has shape `(output_dim, state_dim)`, stored row-major.
/// `b` has shape `(output_dim,)`.
///
/// Trainable via `RidgeRegression` in the `training` module.
#[derive(Debug, Clone)]
pub struct LinearReadout {
    /// Weight matrix, row-major: `weights[i * state_dim + j]`
    pub weights: Vec<f32>,
    /// Bias vector
    pub bias: Vec<f32>,
    /// Output dimension
    pub output_dim: usize,
    /// State dimension (depends on feature_mode and num_sites)
    pub state_dim: usize,
    /// Feature extraction mode
    pub feature_mode: FeatureMode,
}

impl LinearReadout {
    /// Create a new linear readout with zero weights (linear feature mode).
    pub fn new(output_dim: usize, num_sites: usize) -> Self {
        let state_dim = 2 * num_sites;
        Self {
            weights: vec![0.0; output_dim * state_dim],
            bias: vec![0.0; output_dim],
            output_dim,
            state_dim,
            feature_mode: FeatureMode::Linear,
        }
    }

    /// Set trained weights and bias.
    ///
    /// `weights` must have length `output_dim × state_dim` (row-major).
    /// `bias` must have length `output_dim`.
    pub fn set_weights(&mut self, weights: Vec<f32>, bias: Vec<f32>) {
        assert_eq!(
            weights.len(),
            self.output_dim * self.state_dim,
            "Weight matrix size mismatch: expected {}×{} = {}, got {}",
            self.output_dim,
            self.state_dim,
            self.output_dim * self.state_dim,
            weights.len()
        );
        assert_eq!(
            bias.len(),
            self.output_dim,
            "Bias size mismatch: expected {}, got {}",
            self.output_dim,
            bias.len()
        );
        self.weights = weights;
        self.bias = bias;
    }

    /// Create from pre-computed weights (e.g., from ridge regression).
    pub fn from_weights(
        weights: Vec<f32>,
        bias: Vec<f32>,
        output_dim: usize,
        state_dim: usize,
    ) -> Self {
        Self::from_weights_with_mode(weights, bias, output_dim, state_dim, FeatureMode::Linear)
    }

    /// Create from pre-computed weights with a specific feature mode.
    pub fn from_weights_with_mode(
        weights: Vec<f32>,
        bias: Vec<f32>,
        output_dim: usize,
        state_dim: usize,
        feature_mode: FeatureMode,
    ) -> Self {
        assert_eq!(weights.len(), output_dim * state_dim);
        assert_eq!(bias.len(), output_dim);
        Self {
            weights,
            bias,
            output_dim,
            state_dim,
            feature_mode,
        }
    }
}

impl LinearReadout {
    /// Predict from a pre-computed state vector (e.g., concatenated ensemble state).
    ///
    /// `state` must have length `self.state_dim`. Feature extraction is assumed
    /// to already be done — this performs only `W · state + bias`.
    pub fn predict_from_raw_state(&self, state: &[f32]) -> Vec<f32> {
        let feat_len = state.len().min(self.state_dim);
        let mut output = self.bias.clone();
        for i in 0..self.output_dim {
            let row_base = i * self.state_dim;
            for j in 0..feat_len {
                output[i] += self.weights[row_base + j] * state[j];
            }
        }
        output
    }
}

impl Readout for LinearReadout {
    fn read(&self, field: &LatticeField) -> Vec<f32> {
        let state = extract_features(field, self.feature_mode);
        let feat_len = state.len().min(self.state_dim);
        let mut output = self.bias.clone();

        for i in 0..self.output_dim {
            let row_base = i * self.state_dim;
            for j in 0..feat_len {
                output[i] += self.weights[row_base + j] * state[j];
            }
        }

        output
    }

    fn name(&self) -> &str {
        "linear"
    }
}

// ─── Direct Readout ──────────────────────────────────

/// Direct readout: extracts the raw state vector `[re_0..re_N, im_0..im_N]`.
///
/// Useful when you want to process the state externally or need the full
/// reservoir state for analysis.
#[derive(Debug, Clone)]
pub struct DirectReadout;

impl Readout for DirectReadout {
    fn read(&self, field: &LatticeField) -> Vec<f32> {
        let mut state = Vec::with_capacity(2 * field.num_sites);
        state.extend_from_slice(&field.values_re);
        state.extend_from_slice(&field.values_im);
        state
    }

    fn name(&self) -> &str {
        "direct"
    }
}

// ─── State Collector ─────────────────────────────────

/// Collects state vectors from the EMC for training.
///
/// During training, the EMC is run on input sequences. After each input,
/// the field state is collected as a feature vector. The feature mode controls
/// what features are extracted:
/// - `Linear`: `[re, im]` — 2N features (default)
/// - `Nonlinear`: `[re, im, re², im²]` — 4N features (quadratic expansion)
#[derive(Debug, Clone)]
pub struct StateCollector {
    /// Collected state vectors
    pub states: Vec<Vec<f32>>,
    /// Corresponding target values
    pub targets: Vec<Vec<f32>>,
    /// Feature extraction mode
    pub feature_mode: FeatureMode,
}

impl StateCollector {
    pub fn new() -> Self {
        Self {
            states: Vec::new(),
            targets: Vec::new(),
            feature_mode: FeatureMode::Linear,
        }
    }

    /// Create a state collector with a specific feature mode.
    pub fn with_mode(mode: FeatureMode) -> Self {
        Self {
            states: Vec::new(),
            targets: Vec::new(),
            feature_mode: mode,
        }
    }

    /// Collect the current field state as a feature vector.
    pub fn collect_state(&mut self, field: &LatticeField) {
        let state = extract_features(field, self.feature_mode);
        self.states.push(state);
    }

    /// Add a target value for the most recent state.
    pub fn add_target(&mut self, target: Vec<f32>) {
        self.targets.push(target);
    }

    /// Collect state and target simultaneously.
    pub fn collect(&mut self, field: &LatticeField, target: Vec<f32>) {
        self.collect_state(field);
        self.add_target(target);
    }

    /// Number of collected samples.
    pub fn len(&self) -> usize {
        self.states.len()
    }

    /// Whether any samples have been collected.
    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    /// Clear all collected data.
    pub fn clear(&mut self) {
        self.states.clear();
        self.targets.clear();
    }

    /// State dimension (2 × num_sites from the first collected state).
    /// Returns 0 if no states have been collected.
    pub fn state_dim(&self) -> usize {
        self.states.first().map_or(0, |s| s.len())
    }
}

impl Default for StateCollector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use icarus_math::lattice::e8::E8Lattice;

    fn make_e8_field() -> LatticeField {
        let lattice = E8Lattice::new();
        LatticeField::from_lattice(&lattice)
    }

    // ─── LinearReadout ───

    #[test]
    fn test_linear_readout_zero_weights() {
        let field = make_e8_field();
        let readout = LinearReadout::new(3, 241);

        let output = readout.read(&field);
        assert_eq!(output.len(), 3);
        for &v in &output {
            assert!(v.abs() < 1e-6);
        }
    }

    #[test]
    fn test_linear_readout_identity_like() {
        let mut field = make_e8_field();
        field.set(0, 1.0, 0.0);
        field.set(1, 0.0, 2.0);

        // Single output: weight on re[0] = 1.0 and im[1] = 0.5
        let state_dim = 2 * 241;
        let mut weights = vec![0.0f32; state_dim];
        weights[0] = 1.0; // re[0]
        weights[241 + 1] = 0.5; // im[1]

        let readout = LinearReadout::from_weights(weights, vec![0.1], 1, state_dim);
        let output = readout.read(&field);

        // output[0] = 1.0*1.0 + 0.5*2.0 + 0.1 = 2.1
        assert!((output[0] - 2.1).abs() < 1e-5);
    }

    #[test]
    fn test_linear_readout_with_bias() {
        let field = make_e8_field();
        let readout = LinearReadout::from_weights(
            vec![0.0; 2 * 241 * 2],
            vec![3.0, -1.5],
            2,
            2 * 241,
        );

        let output = readout.read(&field);
        assert!((output[0] - 3.0).abs() < 1e-6);
        assert!((output[1] - (-1.5)).abs() < 1e-6);
    }

    #[test]
    fn test_linear_set_weights() {
        let mut readout = LinearReadout::new(1, 241);
        let state_dim = 2 * 241;
        let w = vec![0.1; state_dim];
        let b = vec![0.5];
        readout.set_weights(w, b);
        assert_eq!(readout.weights.len(), state_dim);
        assert!((readout.bias[0] - 0.5).abs() < 1e-6);
    }

    // ─── DirectReadout ───

    #[test]
    fn test_direct_readout() {
        let mut field = make_e8_field();
        field.set(0, 1.5, -0.3);
        field.set(5, 0.0, 2.7);

        let readout = DirectReadout;
        let state = readout.read(&field);

        assert_eq!(state.len(), 2 * 241);
        assert!((state[0] - 1.5).abs() < 1e-6); // re[0]
        assert!((state[241] - (-0.3)).abs() < 1e-6); // im[0]
        assert!((state[241 + 5] - 2.7).abs() < 1e-6); // im[5]
    }

    // ─── StateCollector ───

    #[test]
    fn test_state_collector_basic() {
        let mut field = make_e8_field();
        field.set(0, 1.0, 2.0);

        let mut collector = StateCollector::new();
        assert!(collector.is_empty());

        collector.collect(&field, vec![0.5]);

        assert_eq!(collector.len(), 1);
        assert_eq!(collector.state_dim(), 2 * 241);
        assert!((collector.states[0][0] - 1.0).abs() < 1e-6); // re[0]
        assert!((collector.states[0][241] - 2.0).abs() < 1e-6); // im[0]
        assert!((collector.targets[0][0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_state_collector_multiple() {
        let mut field = make_e8_field();
        let mut collector = StateCollector::new();

        for i in 0..5 {
            field.set(0, i as f32, 0.0);
            collector.collect(&field, vec![i as f32 * 2.0]);
        }

        assert_eq!(collector.len(), 5);

        collector.clear();
        assert!(collector.is_empty());
        assert_eq!(collector.state_dim(), 0);
    }

    // ─── Trait object tests ───

    #[test]
    fn test_readout_names() {
        assert_eq!(LinearReadout::new(1, 1).name(), "linear");
        assert_eq!(DirectReadout.name(), "direct");
    }

    #[test]
    fn test_readout_trait_object() {
        let readouts: Vec<Box<dyn Readout>> = vec![
            Box::new(LinearReadout::new(1, 241)),
            Box::new(DirectReadout),
        ];

        let field = make_e8_field();
        for r in &readouts {
            let _ = r.read(&field);
        }
    }
}
