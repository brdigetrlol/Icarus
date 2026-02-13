//! Input Encoding for the Emergent Manifold Computer
//!
//! Three encoding strategies map external data into the EMC's complex phase field:
//! 1. **Spatial**: Direct amplitude injection — `input[i] → (input[i], 0)`
//! 2. **Phase**: Unit circle mapping — `input[i] → (cos(πx), sin(πx))`
//! 3. **Spectral**: E8 root vector basis decomposition — principled crystallographic encoding

use icarus_field::phase_field::LatticeField;

/// Trait for encoding external input into a lattice field.
pub trait InputEncoder: Send + Sync {
    /// Encode input data into the given field.
    fn encode(&self, input: &[f32], field: &mut LatticeField);

    /// Name of this encoding strategy.
    fn name(&self) -> &str;

    /// Encode with leaking rate: blends new encoding with existing field state.
    ///
    /// `leak_rate` in `[0, 1]`:
    /// - `1.0` = full overwrite (identical to `encode`)
    /// - `0.0` = keep old state entirely (no input effect)
    /// - `0.3` = 70% old state + 30% new encoding (typical for temporal memory)
    ///
    /// This preserves reservoir memory across input steps, critical for tasks
    /// requiring temporal context (e.g., NARMA-10).
    fn encode_leaky(&self, input: &[f32], field: &mut LatticeField, leak_rate: f32) {
        let old_re = field.values_re.clone();
        let old_im = field.values_im.clone();
        self.encode(input, field);
        let n = field.num_sites;
        let keep = 1.0 - leak_rate;
        for i in 0..n {
            field.values_re[i] = keep * old_re[i] + leak_rate * field.values_re[i];
            field.values_im[i] = keep * old_im[i] + leak_rate * field.values_im[i];
        }
    }
}

// ─── Spatial Encoder ─────────────────────────────────

/// Spatial encoding: maps `input[i]` directly to field site `i` as real amplitude.
///
/// `input[i] → field.set(offset + i, scale * input[i], 0.0)`
///
/// Simplest encoding. Good when input dimensionality ≤ number of lattice sites.
#[derive(Debug, Clone)]
pub struct SpatialEncoder {
    /// Starting site index for injection (default 0)
    pub offset: usize,
    /// Scale factor applied to input values (default 1.0)
    pub scale: f32,
}

impl Default for SpatialEncoder {
    fn default() -> Self {
        Self {
            offset: 0,
            scale: 1.0,
        }
    }
}

impl InputEncoder for SpatialEncoder {
    fn encode(&self, input: &[f32], field: &mut LatticeField) {
        let start = self.offset.min(field.num_sites);
        let n = input.len().min(field.num_sites - start);
        for i in 0..n {
            field.set(start + i, input[i] * self.scale, 0.0);
        }
    }

    fn name(&self) -> &str {
        "spatial"
    }
}

// ─── Phase Encoder ───────────────────────────────────

/// Phase encoding: maps `input[i]` to a point on the unit circle.
///
/// `input[i] → field.set(offset + i, cos(π·input[i]), sin(π·input[i]))`
///
/// All sites have amplitude 1; information is encoded purely in phase.
/// Best for inputs normalized to `[-1, 1]`.
#[derive(Debug, Clone)]
pub struct PhaseEncoder {
    /// Starting site index for injection (default 0)
    pub offset: usize,
}

impl Default for PhaseEncoder {
    fn default() -> Self {
        Self { offset: 0 }
    }
}

impl InputEncoder for PhaseEncoder {
    fn encode(&self, input: &[f32], field: &mut LatticeField) {
        let start = self.offset.min(field.num_sites);
        let n = input.len().min(field.num_sites - start);
        for i in 0..n {
            let theta = std::f32::consts::PI * input[i];
            field.set(start + i, theta.cos(), theta.sin());
        }
    }

    fn name(&self) -> &str {
        "phase"
    }
}

// ─── Spectral Encoder ────────────────────────────────

/// Spectral encoding: decomposes input into the E8 root vector basis.
///
/// Takes the first 8 values of input (the E8 dimension), projects onto
/// each of the 240 root vectors, and injects projection coefficients
/// as real amplitudes at the corresponding lattice sites (1..=240).
///
/// `c_j = (input · r_j) / |r_j|²`
/// `field.set(j + 1, c_j, 0.0)`
///
/// Site 0 (origin) receives the input L2 norm as a global signal.
///
/// This encoding respects E8 crystallographic symmetry and distributes
/// information across all nearest-neighbor sites simultaneously.
#[derive(Debug, Clone)]
pub struct SpectralEncoder {
    /// Pre-computed root vectors as f32 (240 × 8)
    roots: Vec<[f32; 8]>,
    /// Pre-computed 1/|r_j|² for each root vector
    inv_norm_sq: Vec<f32>,
}

impl SpectralEncoder {
    /// Create a new spectral encoder with E8 root vectors.
    pub fn new() -> Self {
        let mut roots = Vec::with_capacity(240);

        // Type 1: 112 vectors — permutations of (±1, ±1, 0, 0, 0, 0, 0, 0)
        // norm² = 2
        for i in 0..8usize {
            for j in (i + 1)..8 {
                for signs in 0..4u8 {
                    let mut root = [0.0f32; 8];
                    root[i] = if signs & 1 == 0 { 1.0 } else { -1.0 };
                    root[j] = if signs & 2 == 0 { 1.0 } else { -1.0 };
                    roots.push(root);
                }
            }
        }

        // Type 2: 128 vectors — (±1)^8 with even parity
        // norm² = 8
        for pattern in 0..256u16 {
            if pattern.count_ones() % 2 == 0 {
                let mut root = [0.0f32; 8];
                for k in 0..8 {
                    root[k] = if pattern & (1 << k) == 0 { 1.0 } else { -1.0 };
                }
                roots.push(root);
            }
        }

        debug_assert_eq!(roots.len(), 240);

        let inv_norm_sq: Vec<f32> = roots
            .iter()
            .map(|r| {
                let ns: f32 = r.iter().map(|x| x * x).sum();
                1.0 / ns
            })
            .collect();

        Self { roots, inv_norm_sq }
    }

    /// Access the root vectors (for testing/inspection).
    pub fn root_vectors(&self) -> &[[f32; 8]] {
        &self.roots
    }
}

impl Default for SpectralEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl InputEncoder for SpectralEncoder {
    fn encode(&self, input: &[f32], field: &mut LatticeField) {
        // Pad input to 8D if shorter, truncate if longer
        let mut v = [0.0f32; 8];
        let n = input.len().min(8);
        v[..n].copy_from_slice(&input[..n]);

        // Project onto each root vector and inject at sites 1..=240
        for (j, (root, &inv_ns)) in self.roots.iter().zip(self.inv_norm_sq.iter()).enumerate() {
            let dot: f32 = v.iter().zip(root.iter()).map(|(a, b)| a * b).sum();
            let coeff = dot * inv_ns;

            let site = j + 1; // Site 0 is origin; root vectors map to sites 1..=240
            if site < field.num_sites {
                field.set(site, coeff, 0.0);
            }
        }

        // Site 0 (origin) gets the input L2 norm as a global signal
        let input_norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if field.num_sites > 0 {
            field.set(0, input_norm, 0.0);
        }
    }

    fn name(&self) -> &str {
        "spectral"
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

    // ─── SpatialEncoder ───

    #[test]
    fn test_spatial_basic() {
        let mut field = make_e8_field();
        let enc = SpatialEncoder::default();
        let input = vec![0.5, -0.3, 1.0];

        enc.encode(&input, &mut field);

        let (re0, im0) = field.get(0);
        let (re1, im1) = field.get(1);
        let (re2, im2) = field.get(2);

        assert!((re0 - 0.5).abs() < 1e-6);
        assert!(im0.abs() < 1e-6);
        assert!((re1 - (-0.3)).abs() < 1e-6);
        assert!(im1.abs() < 1e-6);
        assert!((re2 - 1.0).abs() < 1e-6);
        assert!(im2.abs() < 1e-6);
    }

    #[test]
    fn test_spatial_with_offset_and_scale() {
        let mut field = make_e8_field();
        let enc = SpatialEncoder {
            offset: 10,
            scale: 2.0,
        };
        let input = vec![0.5, -0.3];

        enc.encode(&input, &mut field);

        // Sites 0..9 should be untouched (zero)
        let (re0, _) = field.get(0);
        assert!(re0.abs() < 1e-6);

        // Sites 10, 11 should have scaled values
        let (re10, _) = field.get(10);
        let (re11, _) = field.get(11);
        assert!((re10 - 1.0).abs() < 1e-6); // 0.5 * 2.0
        assert!((re11 - (-0.6)).abs() < 1e-6); // -0.3 * 2.0
    }

    #[test]
    fn test_spatial_input_longer_than_field() {
        let mut field = make_e8_field();
        let enc = SpatialEncoder::default();
        let input: Vec<f32> = (0..500).map(|i| i as f32 * 0.01).collect();

        enc.encode(&input, &mut field);

        // Should only fill up to num_sites (241)
        let (re240, _) = field.get(240);
        assert!((re240 - 2.40).abs() < 1e-4);
    }

    // ─── PhaseEncoder ───

    #[test]
    fn test_phase_basic() {
        let mut field = make_e8_field();
        let enc = PhaseEncoder::default();

        // input = 0 → θ=0 → (cos 0, sin 0) = (1, 0)
        // input = 0.5 → θ=π/2 → (cos π/2, sin π/2) = (0, 1)
        // input = 1.0 → θ=π → (cos π, sin π) = (-1, 0)
        let input = vec![0.0, 0.5, 1.0];
        enc.encode(&input, &mut field);

        let (re0, im0) = field.get(0);
        assert!((re0 - 1.0).abs() < 1e-6);
        assert!(im0.abs() < 1e-6);

        let (re1, im1) = field.get(1);
        assert!(re1.abs() < 1e-5);
        assert!((im1 - 1.0).abs() < 1e-5);

        let (re2, im2) = field.get(2);
        assert!((re2 - (-1.0)).abs() < 1e-5);
        assert!(im2.abs() < 1e-5);
    }

    #[test]
    fn test_phase_unit_amplitude() {
        let mut field = make_e8_field();
        let enc = PhaseEncoder::default();
        let input: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) * 0.02).collect();

        enc.encode(&input, &mut field);

        for i in 0..100 {
            let amp_sq = field.norm_sq(i);
            assert!(
                (amp_sq - 1.0).abs() < 1e-5,
                "Site {} amplitude² = {} (should be 1.0)",
                i,
                amp_sq
            );
        }
    }

    // ─── SpectralEncoder ───

    #[test]
    fn test_spectral_root_vector_count() {
        let enc = SpectralEncoder::new();
        assert_eq!(enc.root_vectors().len(), 240);
    }

    #[test]
    fn test_spectral_root_norms() {
        let enc = SpectralEncoder::new();
        let type1_count = enc
            .root_vectors()
            .iter()
            .filter(|r| {
                let ns: f32 = r.iter().map(|x| x * x).sum();
                (ns - 2.0).abs() < 1e-6
            })
            .count();
        let type2_count = enc
            .root_vectors()
            .iter()
            .filter(|r| {
                let ns: f32 = r.iter().map(|x| x * x).sum();
                (ns - 8.0).abs() < 1e-6
            })
            .count();
        assert_eq!(type1_count, 112);
        assert_eq!(type2_count, 128);
    }

    #[test]
    fn test_spectral_basic() {
        let mut field = make_e8_field();
        let enc = SpectralEncoder::new();

        // Encode the unit vector e_0 = [1, 0, 0, 0, 0, 0, 0, 0]
        let input = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        enc.encode(&input, &mut field);

        // Origin should have norm = 1.0
        let (re0, _) = field.get(0);
        assert!((re0 - 1.0).abs() < 1e-5, "Origin should be input norm = 1.0");

        // Type 1 roots with nonzero component 0: (1,±1,0,...) → dot = 1
        // Coefficient = 1 / 2 = 0.5
        let (re1, _) = field.get(1); // First root: (+1, +1, 0, ..., 0) → dot = 1, coeff = 0.5
        assert!(
            (re1 - 0.5).abs() < 1e-5,
            "Root (1,1,0...) should have coeff 0.5, got {}",
            re1
        );
    }

    #[test]
    fn test_spectral_zero_input() {
        let mut field = make_e8_field();
        let enc = SpectralEncoder::new();

        enc.encode(&[], &mut field);

        // All sites should be zero (including origin norm = 0)
        for i in 0..field.num_sites {
            let (re, im) = field.get(i);
            assert!(
                re.abs() < 1e-6 && im.abs() < 1e-6,
                "Site {} should be zero: ({}, {})",
                i,
                re,
                im
            );
        }
    }

    #[test]
    fn test_spectral_symmetry() {
        let mut field1 = make_e8_field();
        let mut field2 = make_e8_field();
        let enc = SpectralEncoder::new();

        // Encode v and -v
        let v = vec![1.0, 2.0, -0.5, 0.3, 0.0, 0.0, 0.0, 0.0];
        let neg_v: Vec<f32> = v.iter().map(|x| -x).collect();

        enc.encode(&v, &mut field1);
        enc.encode(&neg_v, &mut field2);

        // All projection coefficients should be negated (linear projection)
        for i in 1..field1.num_sites.min(241) {
            let (re1, _) = field1.get(i);
            let (re2, _) = field2.get(i);
            assert!(
                (re1 + re2).abs() < 1e-5,
                "Site {}: {} + {} should be 0",
                i,
                re1,
                re2
            );
        }

        // Origin norms should be equal (both have same |v|)
        let (norm1, _) = field1.get(0);
        let (norm2, _) = field2.get(0);
        assert!((norm1 - norm2).abs() < 1e-5);
    }

    #[test]
    fn test_spectral_short_input() {
        let mut field = make_e8_field();
        let enc = SpectralEncoder::new();

        // 3D input should be padded to 8D with zeros
        let input = vec![1.0, 2.0, 3.0];
        enc.encode(&input, &mut field);

        let (re0, _) = field.get(0);
        let expected_norm = (1.0f32 + 4.0 + 9.0).sqrt();
        assert!((re0 - expected_norm).abs() < 1e-5);
    }

    // ─── Cross-encoder tests ───

    #[test]
    fn test_encoder_names() {
        assert_eq!(SpatialEncoder::default().name(), "spatial");
        assert_eq!(PhaseEncoder::default().name(), "phase");
        assert_eq!(SpectralEncoder::new().name(), "spectral");
    }

    #[test]
    fn test_encoder_trait_object() {
        let encoders: Vec<Box<dyn InputEncoder>> = vec![
            Box::new(SpatialEncoder::default()),
            Box::new(PhaseEncoder::default()),
            Box::new(SpectralEncoder::new()),
        ];

        let mut field = make_e8_field();
        let input = vec![0.5; 8];

        for enc in &encoders {
            enc.encode(&input, &mut field);
            // Just verify it doesn't panic
        }
    }
}
