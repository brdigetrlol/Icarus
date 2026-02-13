//! FP16 compact storage for field snapshots
//!
//! Provides memory-efficient storage of field states using IEEE 754 half-precision
//! (f16) for archival and snapshot purposes. All computation remains in f32 —
//! f16 is used only for storage compression (50% memory savings).
//!
//! Primary use case: the memory agent stores up to 128 historical field snapshots.
//! At E8 scale (241 sites), each snapshot drops from 1.9KB to 0.96KB.
//! At Leech scale (1105 sites), each snapshot drops from 8.8KB to 4.4KB.

use half::f16;
use icarus_math::lattice::LatticeLayer;

use crate::phase_field::LatticeField;

/// A compact representation of field state using FP16 storage.
///
/// Captures only the phase field values (re, im) — not the topology (CSR graph,
/// weights, displacements) which is shared across snapshots of the same lattice.
#[derive(Debug, Clone)]
pub struct CompactFieldSnapshot {
    /// Real parts stored as f16
    pub values_re: Vec<f16>,
    /// Imaginary parts stored as f16
    pub values_im: Vec<f16>,
    /// Which layer this snapshot is from
    pub layer: LatticeLayer,
    /// Number of lattice sites
    pub num_sites: usize,
}

impl CompactFieldSnapshot {
    /// Capture a field's current state as an FP16 snapshot.
    ///
    /// This is a lossy conversion — f16 has ~3.3 decimal digits of precision
    /// vs f32's ~7.2 digits. For field values in the typical range [-2, 2],
    /// the maximum round-trip error is ~0.001.
    pub fn from_field(field: &LatticeField) -> Self {
        Self {
            values_re: f32_slice_to_f16(&field.values_re),
            values_im: f32_slice_to_f16(&field.values_im),
            layer: field.layer,
            num_sites: field.num_sites,
        }
    }

    /// Restore this snapshot's values into an existing LatticeField.
    ///
    /// The field must have the same number of sites. Only values_re and values_im
    /// are overwritten — topology (CSR graph, weights) is preserved.
    ///
    /// # Panics
    /// Panics if `field.num_sites != self.num_sites`.
    pub fn restore_to_field(&self, field: &mut LatticeField) {
        assert_eq!(
            field.num_sites, self.num_sites,
            "Cannot restore snapshot ({} sites) to field ({} sites)",
            self.num_sites, field.num_sites
        );
        f16_slice_to_f32(&self.values_re, &mut field.values_re);
        f16_slice_to_f32(&self.values_im, &mut field.values_im);
    }

    /// Create a new LatticeField from this snapshot, cloning topology from a template.
    ///
    /// The template must have the same number of sites and provides the CSR
    /// neighbor topology, weights, and displacement vectors.
    ///
    /// # Panics
    /// Panics if `template.num_sites != self.num_sites`.
    pub fn to_field(&self, template: &LatticeField) -> LatticeField {
        assert_eq!(
            template.num_sites, self.num_sites,
            "Template ({} sites) doesn't match snapshot ({} sites)",
            template.num_sites, self.num_sites
        );
        let mut field = template.clone();
        self.restore_to_field(&mut field);
        field
    }

    /// Memory usage of this snapshot in bytes.
    ///
    /// Only counts the f16 value storage, not the struct overhead.
    pub fn memory_bytes(&self) -> usize {
        // 2 bytes per f16, two arrays (re + im)
        self.num_sites * 2 * 2
    }

    /// Compute the maximum absolute round-trip error against a full-precision field.
    ///
    /// Returns the maximum of |snapshot_f32[i] - field_f32[i]| across all sites
    /// and both real/imaginary components.
    pub fn max_error(&self, field: &LatticeField) -> f32 {
        assert_eq!(field.num_sites, self.num_sites);
        let mut max_err = 0.0f32;
        for i in 0..self.num_sites {
            let err_re = (self.values_re[i].to_f32() - field.values_re[i]).abs();
            let err_im = (self.values_im[i].to_f32() - field.values_im[i]).abs();
            max_err = max_err.max(err_re).max(err_im);
        }
        max_err
    }

    /// Compute the mean absolute round-trip error against a full-precision field.
    pub fn mean_error(&self, field: &LatticeField) -> f32 {
        assert_eq!(field.num_sites, self.num_sites);
        let mut total_err = 0.0f64;
        let count = (self.num_sites * 2) as f64;
        for i in 0..self.num_sites {
            total_err += (self.values_re[i].to_f32() - field.values_re[i]).abs() as f64;
            total_err += (self.values_im[i].to_f32() - field.values_im[i]).abs() as f64;
        }
        (total_err / count) as f32
    }

    /// Memory savings ratio compared to full f32 storage.
    ///
    /// Returns the fraction of memory saved (always 0.5 for f16 vs f32 values).
    pub fn savings_ratio(&self) -> f32 {
        // f16 is 2 bytes vs f32's 4 bytes = 50% savings on value storage
        0.5
    }
}

// ---------------------------------------------------------------------------
// Conversion helpers
// ---------------------------------------------------------------------------

/// Convert an f32 slice to a Vec of f16 values.
pub fn f32_slice_to_f16(data: &[f32]) -> Vec<f16> {
    data.iter().map(|&v| f16::from_f32(v)).collect()
}

/// Convert f16 values back into an existing f32 slice.
pub fn f16_slice_to_f32(src: &[f16], dst: &mut [f32]) {
    assert_eq!(src.len(), dst.len());
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d = s.to_f32();
    }
}

/// Convert an f32 slice to raw f16 bit representations (u16).
///
/// Useful for GPU transport where the kernel expects raw half-precision bits.
pub fn f32_to_f16_bits(data: &[f32]) -> Vec<u16> {
    data.iter().map(|&v| f16::from_f32(v).to_bits()).collect()
}

/// Convert raw f16 bit representations (u16) back to f32.
pub fn f16_bits_to_f32(data: &[u16]) -> Vec<f32> {
    data.iter().map(|&bits| f16::from_bits(bits).to_f32()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use icarus_math::lattice::e8::E8Lattice;
    use icarus_math::lattice::hypercubic::HypercubicLattice;

    fn make_e8_field() -> LatticeField {
        let lattice = E8Lattice::new();
        let mut field = LatticeField::from_lattice(&lattice);
        field.init_random(42, 1.5);
        field
    }

    fn make_small_field() -> LatticeField {
        let lattice = HypercubicLattice::new(3);
        let mut field = LatticeField::from_lattice(&lattice);
        field.init_random(99, 2.0);
        field
    }

    #[test]
    fn test_snapshot_round_trip() {
        let field = make_e8_field();
        let snapshot = CompactFieldSnapshot::from_field(&field);

        assert_eq!(snapshot.num_sites, 241);
        assert_eq!(snapshot.layer, LatticeLayer::Analytical);
        assert_eq!(snapshot.values_re.len(), 241);
        assert_eq!(snapshot.values_im.len(), 241);

        // Restore to a new field
        let restored = snapshot.to_field(&field);
        assert_eq!(restored.num_sites, field.num_sites);
        assert_eq!(restored.layer, field.layer);

        // Check values are close (f16 precision ~0.001 for values in [-2, 2])
        for i in 0..field.num_sites {
            let err_re = (restored.values_re[i] - field.values_re[i]).abs();
            let err_im = (restored.values_im[i] - field.values_im[i]).abs();
            assert!(err_re < 0.01, "site {} re error {} too large", i, err_re);
            assert!(err_im < 0.01, "site {} im error {} too large", i, err_im);
        }
    }

    #[test]
    fn test_snapshot_restore_in_place() {
        let field = make_e8_field();
        let snapshot = CompactFieldSnapshot::from_field(&field);

        // Create a zeroed field with same topology
        let lattice = E8Lattice::new();
        let mut target = LatticeField::from_lattice(&lattice);
        assert!((target.values_re[0]).abs() < 1e-12); // starts zeroed

        snapshot.restore_to_field(&mut target);

        // Should now have the snapshot's values
        for i in 0..field.num_sites {
            let err = (target.values_re[i] - snapshot.values_re[i].to_f32()).abs();
            assert!(err < 1e-10, "restore mismatch at site {}: {}", i, err);
        }
    }

    #[test]
    fn test_memory_savings() {
        let field = make_e8_field();
        let snapshot = CompactFieldSnapshot::from_field(&field);

        // f32 field value memory: 241 * 2 * 4 = 1928 bytes
        let f32_bytes = field.num_sites * 2 * 4;
        // f16 snapshot memory: 241 * 2 * 2 = 964 bytes
        let f16_bytes = snapshot.memory_bytes();

        assert_eq!(f32_bytes, 1928);
        assert_eq!(f16_bytes, 964);
        assert_eq!(f16_bytes * 2, f32_bytes);
        assert!((snapshot.savings_ratio() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_max_error_bound() {
        let field = make_e8_field();
        let snapshot = CompactFieldSnapshot::from_field(&field);

        let max_err = snapshot.max_error(&field);
        // For values in [-1.5, 1.5], f16 precision gives max error ~0.001
        assert!(
            max_err < 0.01,
            "max round-trip error {} should be < 0.01",
            max_err
        );
        // But error should be non-zero (f16 is lossy)
        assert!(max_err > 0.0, "error should be non-zero (lossy compression)");
    }

    #[test]
    fn test_mean_error() {
        let field = make_e8_field();
        let snapshot = CompactFieldSnapshot::from_field(&field);

        let mean_err = snapshot.mean_error(&field);
        assert!(mean_err < 0.005, "mean error {} should be < 0.005", mean_err);
        assert!(mean_err > 0.0);
        assert!(mean_err <= snapshot.max_error(&field));
    }

    #[test]
    fn test_small_field_snapshot() {
        let field = make_small_field();
        let snapshot = CompactFieldSnapshot::from_field(&field);

        assert_eq!(snapshot.num_sites, field.num_sites);
        let restored = snapshot.to_field(&field);

        for i in 0..field.num_sites {
            let err_re = (restored.values_re[i] - field.values_re[i]).abs();
            assert!(err_re < 0.02, "site {} error {}", i, err_re);
        }
    }

    #[test]
    fn test_extreme_values() {
        let lattice = HypercubicLattice::new(2);
        let mut field = LatticeField::from_lattice(&lattice);

        // Set extreme values: f16 max is ~65504, min positive is ~5.96e-8
        field.set(0, 100.0, -100.0);
        field.set(1, 0.001, -0.001);
        field.set(2, 0.0, 0.0);

        let snapshot = CompactFieldSnapshot::from_field(&field);
        let restored = snapshot.to_field(&field);

        // Large values: relative error should be small
        assert!((restored.values_re[0] - 100.0).abs() < 0.2);
        // Small values: absolute error small
        assert!((restored.values_re[1] - 0.001).abs() < 0.001);
        // Zero is exact
        assert_eq!(restored.values_re[2], 0.0);
        assert_eq!(restored.values_im[2], 0.0);
    }

    #[test]
    #[should_panic(expected = "Cannot restore snapshot")]
    fn test_restore_size_mismatch() {
        let field_e8 = make_e8_field();
        let snapshot = CompactFieldSnapshot::from_field(&field_e8);

        let lattice = HypercubicLattice::new(3);
        let mut small_field = LatticeField::from_lattice(&lattice);
        snapshot.restore_to_field(&mut small_field); // should panic
    }

    #[test]
    fn test_f32_f16_bits_round_trip() {
        let data = vec![0.0f32, 1.0, -1.0, 0.5, 3.14159, -42.0, 0.001];
        let bits = f32_to_f16_bits(&data);
        let restored = f16_bits_to_f32(&bits);

        assert_eq!(bits.len(), data.len());
        assert_eq!(restored.len(), data.len());

        for (orig, rest) in data.iter().zip(restored.iter()) {
            let err = (orig - rest).abs();
            let tol = orig.abs() * 0.01 + 0.001; // 1% relative + 0.001 absolute
            assert!(err < tol, "f16 bits round-trip: {} -> {} (err {})", orig, rest, err);
        }
    }

    #[test]
    fn test_f32_slice_to_f16_exact_values() {
        // Values exactly representable in f16
        let data = vec![0.0f32, 1.0, -1.0, 0.5, -0.5, 2.0];
        let f16_vals = f32_slice_to_f16(&data);
        for (i, (&orig, &half_val)) in data.iter().zip(f16_vals.iter()).enumerate() {
            assert_eq!(
                half_val.to_f32(),
                orig,
                "value {} should be exactly representable in f16",
                i
            );
        }
    }

    #[test]
    fn test_multiple_snapshots_memory() {
        let field = make_e8_field();

        // Simulate memory agent storing 128 snapshots
        let num_snapshots = 128;
        let f32_total = num_snapshots * field.num_sites * 2 * 4; // ~247KB
        let f16_total = num_snapshots * field.num_sites * 2 * 2; // ~123KB

        let snapshot = CompactFieldSnapshot::from_field(&field);
        let per_snapshot = snapshot.memory_bytes();

        assert_eq!(f16_total, per_snapshot * num_snapshots);
        assert_eq!(f16_total * 2, f32_total);

        // At E8 scale, 128 snapshots save ~123KB
        let savings = f32_total - f16_total;
        assert_eq!(savings, 123_392); // 241 * 2 * 2 * 128
    }
}
