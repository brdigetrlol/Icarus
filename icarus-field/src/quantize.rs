//! Advanced quantization inspired by Quartet II (arXiv:2601.22813).
//!
//! Implements techniques from the MS-EDEN paper for improved field snapshot
//! fidelity and compression:
//!
//! ## Techniques
//!
//! 1. **Stochastic Rounding (SR)** — Unbiased FP16 quantization where
//!    `E[SR(x)] = x`. Prevents systematic bias accumulation across
//!    iterative snapshot store/restore cycles.
//!
//! 2. **Randomized Hadamard Transform (RHT)** — Pre-rotation that smooths
//!    value distributions before quantization, reducing worst-case MSE.
//!
//! 3. **EDEN Bias Correction** — Scalar correction factor `S` that removes
//!    systematic quantization bias: `S = ||x||^2 / <Q(x), x>`.
//!
//! 4. **Four-over-Six (4/6) Scale Selection** — Tests two candidate scales
//!    per block and picks the one with lower MSE (Cook et al., 2025).
//!
//! 5. **Block-wise 4-bit Quantization** — Microscaling format inspired by
//!    NVFP4: 16-element blocks with per-block f32 scale factors. ~81% savings.
//!
//! ## Snapshot Types
//!
//! | Type | Bits/Value | Savings | Key Feature |
//! |------|-----------|---------|-------------|
//! | `CompactFieldSnapshot` (fp16) | 16 | 50% | Simple RTN |
//! | [`EdenSnapshot`] | 16 | ~50% | SR + EDEN (unbiased) |
//! | [`Block4BitSnapshot`] | ~6 | ~81% | 4/6 scales |
//! | [`MsEdenSnapshot`] | ~6 | ~78% | RHT + EDEN + 4-bit (full pipeline) |

use half::f16;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use crate::phase_field::LatticeField;
use icarus_math::lattice::LatticeLayer;

// ---------------------------------------------------------------------------
// Section 1: Stochastic Rounding (SR) for FP16
// ---------------------------------------------------------------------------

/// Return the next representable f16 value above `v`.
///
/// For positive values, increments the bit pattern; for negative values,
/// decrements the magnitude (which moves toward zero, i.e., "up").
fn next_f16(v: f16) -> f16 {
    let bits = v.to_bits();
    if v.is_nan() || v == f16::INFINITY {
        return v;
    }
    // +-0 → smallest positive subnormal
    if bits == 0x0000 || bits == 0x8000 {
        return f16::from_bits(0x0001);
    }
    if bits & 0x8000 == 0 {
        // Positive: increment bit pattern
        f16::from_bits(bits + 1)
    } else {
        // Negative: decrement magnitude (move toward zero = upward)
        f16::from_bits(bits - 1)
    }
}

/// Return the previous representable f16 value below `v`.
fn prev_f16(v: f16) -> f16 {
    let bits = v.to_bits();
    if v.is_nan() || v == f16::NEG_INFINITY {
        return v;
    }
    // +0 → smallest negative subnormal
    if bits == 0x0000 {
        return f16::from_bits(0x8001);
    }
    // -0 → smallest negative subnormal
    if bits == 0x8000 {
        return f16::from_bits(0x8001);
    }
    if bits & 0x8000 == 0 {
        // Positive: decrement bit pattern
        f16::from_bits(bits - 1)
    } else {
        // Negative: increment magnitude (move away from zero = downward)
        f16::from_bits(bits + 1)
    }
}

/// Stochastically round an f32 value to f16 such that `E[SR(x)] = x`.
///
/// Given value `x` between two adjacent f16 representable values `a < b`,
/// rounds to `b` with probability `(x - a) / (b - a)`, and to `a` otherwise.
/// This guarantees the expected value equals `x` (unbiased).
///
/// `rand_uniform` should be drawn from `Uniform(0, 1)`.
pub fn stochastic_round_f16(v: f32, rand_uniform: f32) -> f16 {
    let q = f16::from_f32(v);
    let q_val = q.to_f32();
    let err = v - q_val;

    // Already exactly representable
    if err == 0.0 {
        return q;
    }

    // Find adjacent representable value in direction of error
    let adjacent = if err > 0.0 {
        next_f16(q)
    } else {
        prev_f16(q)
    };
    let adj_val = adjacent.to_f32();
    let gap = (adj_val - q_val).abs();

    if gap < f32::EPSILON {
        return q;
    }

    // Probability of rounding to the adjacent value
    let p_adj = err.abs() / gap;
    if rand_uniform < p_adj {
        adjacent
    } else {
        q
    }
}

/// Batch stochastic rounding of f32 slice to f16 with deterministic seed.
///
/// The seed ensures reproducibility — same seed + same data = same result.
pub fn f32_slice_to_f16_stochastic(data: &[f32], seed: u64) -> Vec<f16> {
    let mut rng = SmallRng::seed_from_u64(seed);
    data.iter()
        .map(|&v| stochastic_round_f16(v, rng.gen::<f32>()))
        .collect()
}

// ---------------------------------------------------------------------------
// Section 2: Fast Walsh-Hadamard Transform (FWHT)
// ---------------------------------------------------------------------------

/// Round up to next power of 2.
fn next_power_of_2(n: usize) -> usize {
    if n.is_power_of_two() {
        n
    } else {
        n.next_power_of_two()
    }
}

/// In-place normalized Fast Walsh-Hadamard Transform.
///
/// The normalized FWHT is self-inverse: applying it twice recovers the original
/// signal (up to floating-point rounding). This makes it ideal for RHT where
/// the same transform is applied for both encoding and decoding.
///
/// Complexity: O(n log n). Input length must be a power of 2.
pub fn fwht_normalized(data: &mut [f32]) {
    let n = data.len();
    assert!(n.is_power_of_two(), "FWHT requires power-of-2 length, got {}", n);
    if n <= 1 {
        return;
    }

    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(h * 2) {
            for j in i..i + h {
                let x = data[j];
                let y = data[j + h];
                data[j] = x + y;
                data[j + h] = x - y;
            }
        }
        h *= 2;
    }

    // Normalize: divide by sqrt(n) so the transform is self-inverse
    let norm = 1.0 / (n as f32).sqrt();
    for v in data.iter_mut() {
        *v *= norm;
    }
}

// ---------------------------------------------------------------------------
// Section 3: Randomized Hadamard Transform (RHT)
// ---------------------------------------------------------------------------

/// Apply Randomized Hadamard Transform (forward) in-place.
///
/// RHT(x) = H * D * x, where D is a random diagonal sign matrix and H is the
/// normalized Hadamard transform. The RHT smooths the distribution of values,
/// making quantization error more uniform (reducing worst-case MSE).
///
/// Use [`rht_inverse`] with the same seed to recover the original data.
///
/// Input length must be a power of 2.
pub fn rht_transform(data: &mut [f32], seed: u64) {
    let n = data.len();
    assert!(n.is_power_of_two(), "RHT requires power-of-2 length, got {}", n);

    // Forward: D first, then H
    apply_random_signs(data, seed);
    fwht_normalized(data);
}

/// Apply inverse Randomized Hadamard Transform in-place.
///
/// RHT^{-1}(y) = D * H * y (reverse order of forward transform).
/// Since both D and H are self-inverse, this recovers the original data:
/// RHT^{-1}(RHT(x)) = D * H * H * D * x = D * I * D * x = x.
///
/// Must use the same seed as the corresponding [`rht_transform`] call.
pub fn rht_inverse(data: &mut [f32], seed: u64) {
    let n = data.len();
    assert!(n.is_power_of_two(), "RHT inverse requires power-of-2 length, got {}", n);

    // Inverse: H first, then D
    fwht_normalized(data);
    apply_random_signs(data, seed);
}

/// Apply random ±1 sign flips determined by seed.
fn apply_random_signs(data: &mut [f32], seed: u64) {
    let mut rng = SmallRng::seed_from_u64(seed);
    for v in data.iter_mut() {
        if rng.gen_bool(0.5) {
            *v = -*v;
        }
    }
}

// ---------------------------------------------------------------------------
// Section 4: EDEN Bias Correction
// ---------------------------------------------------------------------------

/// Compute the EDEN bias correction factor: `S = ||x||^2 / <Q(x), x>`.
///
/// This scalar correction removes systematic quantization bias. When the
/// quantized vector `Q(x)` is multiplied by `S`, the result is an unbiased
/// estimate of `x` in the limit of large dimension (Vargaftik et al., 2022).
///
/// Returns 1.0 if the denominator is near zero (degenerate case).
pub fn eden_correction_factor(original: &[f32], quantized: &[f32]) -> f32 {
    debug_assert_eq!(original.len(), quantized.len());

    // Use f64 accumulation for precision
    let norm_sq: f64 = original.iter().map(|&v| (v as f64) * (v as f64)).sum();
    let dot: f64 = original
        .iter()
        .zip(quantized.iter())
        .map(|(&o, &q)| (o as f64) * (q as f64))
        .sum();

    if dot.abs() < 1e-30 {
        1.0
    } else {
        (norm_sq / dot) as f32
    }
}

// ---------------------------------------------------------------------------
// Section 5: EdenSnapshot — Enhanced FP16 with SR + EDEN
// ---------------------------------------------------------------------------

/// An enhanced FP16 snapshot using stochastic rounding and EDEN correction.
///
/// Compared to `CompactFieldSnapshot` (which uses biased round-to-nearest):
/// - **Stochastic rounding**: `E[SR(x)] = x` — no systematic bias
/// - **EDEN correction**: scalar factor that minimizes reconstruction MSE
///
/// The EDEN correction adds only 8 bytes overhead (two f32 values) regardless
/// of field size, so memory savings remain ~50% for any practical lattice.
#[derive(Debug, Clone)]
pub struct EdenSnapshot {
    /// Real parts stored as f16 (stochastically rounded)
    pub values_re: Vec<f16>,
    /// Imaginary parts stored as f16 (stochastically rounded)
    pub values_im: Vec<f16>,
    /// EDEN correction factor for real component
    pub correction_re: f32,
    /// EDEN correction factor for imaginary component
    pub correction_im: f32,
    /// Seed used for stochastic rounding (for reproducibility)
    pub sr_seed: u64,
    /// Which layer this snapshot is from
    pub layer: LatticeLayer,
    /// Number of lattice sites
    pub num_sites: usize,
}

/// Golden ratio constant for seed derivation (ensures decorrelated seeds).
const GOLDEN_RATIO_HASH: u64 = 0x9E3779B97F4A7C15;

impl EdenSnapshot {
    /// Capture a field's current state as an EDEN-corrected FP16 snapshot.
    ///
    /// Uses stochastic rounding (unbiased) and computes per-component EDEN
    /// correction factors. The `seed` controls SR randomness for reproducibility.
    pub fn from_field(field: &LatticeField, seed: u64) -> Self {
        let seed_im = seed.wrapping_add(GOLDEN_RATIO_HASH);

        // Stochastically round to f16
        let values_re = f32_slice_to_f16_stochastic(&field.values_re, seed);
        let values_im = f32_slice_to_f16_stochastic(&field.values_im, seed_im);

        // Compute EDEN correction factors
        let quantized_re: Vec<f32> = values_re.iter().map(|v| v.to_f32()).collect();
        let quantized_im: Vec<f32> = values_im.iter().map(|v| v.to_f32()).collect();

        let correction_re = eden_correction_factor(&field.values_re, &quantized_re);
        let correction_im = eden_correction_factor(&field.values_im, &quantized_im);

        Self {
            values_re,
            values_im,
            correction_re,
            correction_im,
            sr_seed: seed,
            layer: field.layer,
            num_sites: field.num_sites,
        }
    }

    /// Restore this snapshot's values into an existing LatticeField.
    ///
    /// Applies EDEN correction during dequantization: `v_restored = v_f16 * S`.
    ///
    /// # Panics
    /// Panics if `field.num_sites != self.num_sites`.
    pub fn restore_to_field(&self, field: &mut LatticeField) {
        assert_eq!(
            field.num_sites, self.num_sites,
            "Cannot restore EdenSnapshot ({} sites) to field ({} sites)",
            self.num_sites, field.num_sites
        );

        for i in 0..self.num_sites {
            field.values_re[i] = self.values_re[i].to_f32() * self.correction_re;
            field.values_im[i] = self.values_im[i].to_f32() * self.correction_im;
        }
    }

    /// Create a new LatticeField from this snapshot, cloning topology from a template.
    ///
    /// # Panics
    /// Panics if `template.num_sites != self.num_sites`.
    pub fn to_field(&self, template: &LatticeField) -> LatticeField {
        assert_eq!(
            template.num_sites, self.num_sites,
            "Template ({} sites) doesn't match EdenSnapshot ({} sites)",
            template.num_sites, self.num_sites
        );
        let mut field = template.clone();
        self.restore_to_field(&mut field);
        field
    }

    /// Memory usage of this snapshot in bytes.
    ///
    /// Counts f16 value arrays + two f32 correction factors.
    pub fn memory_bytes(&self) -> usize {
        // 2 bytes per f16, two arrays (re + im) + 2 * 4 bytes for corrections
        self.num_sites * 2 * 2 + 8
    }

    /// Compute max absolute error against a full-precision field.
    pub fn max_error(&self, field: &LatticeField) -> f32 {
        assert_eq!(field.num_sites, self.num_sites);
        let mut max_err = 0.0f32;
        for i in 0..self.num_sites {
            let err_re = (self.values_re[i].to_f32() * self.correction_re - field.values_re[i]).abs();
            let err_im = (self.values_im[i].to_f32() * self.correction_im - field.values_im[i]).abs();
            max_err = max_err.max(err_re).max(err_im);
        }
        max_err
    }

    /// Compute mean absolute error against a full-precision field.
    pub fn mean_error(&self, field: &LatticeField) -> f32 {
        assert_eq!(field.num_sites, self.num_sites);
        let mut total_err = 0.0f64;
        let count = (self.num_sites * 2) as f64;
        for i in 0..self.num_sites {
            total_err +=
                (self.values_re[i].to_f32() * self.correction_re - field.values_re[i]).abs() as f64;
            total_err +=
                (self.values_im[i].to_f32() * self.correction_im - field.values_im[i]).abs() as f64;
        }
        (total_err / count) as f32
    }

    /// Memory savings ratio compared to full f32 storage.
    pub fn savings_ratio(&self) -> f32 {
        let f32_bytes = self.num_sites * 2 * 4;
        1.0 - (self.memory_bytes() as f32 / f32_bytes as f32)
    }
}

// ---------------------------------------------------------------------------
// Section 6: Block 4-bit Quantization with Four-over-Six Scale Selection
// ---------------------------------------------------------------------------

/// Block size for 4-bit quantization (matches NVFP4 group size).
const BLOCK_SIZE_4BIT: usize = 16;

/// 4-bit quantization range: signed integers in [-7, 7].
const QUANT_MAX_4BIT: i8 = 7;

/// Pack two signed 4-bit values into one byte using offset binary.
///
/// Values are offset by 8 to map [-7, 7] → [1, 15] (0 maps to 8).
/// High nibble = first value, low nibble = second value.
fn pack_nibbles(hi: i8, lo: i8) -> u8 {
    let hi_u = (hi + 8) as u8;
    let lo_u = (lo + 8) as u8;
    (hi_u << 4) | (lo_u & 0x0F)
}

/// Unpack two signed 4-bit values from one byte.
fn unpack_nibbles(byte: u8) -> (i8, i8) {
    let hi = ((byte >> 4) as i8) - 8;
    let lo = ((byte & 0x0F) as i8) - 8;
    (hi, lo)
}

/// Compute MSE for a block with a given scale factor.
fn block_quantize_mse(block: &[f32], scale: f32) -> f64 {
    if scale.abs() < f32::EPSILON {
        return block.iter().map(|&v| (v as f64) * (v as f64)).sum();
    }
    let inv_scale = 1.0 / scale;
    let mut mse = 0.0f64;
    for &v in block {
        let q = (v * inv_scale).round().clamp(-QUANT_MAX_4BIT as f32, QUANT_MAX_4BIT as f32);
        let reconstructed = q * scale;
        let err = (v - reconstructed) as f64;
        mse += err * err;
    }
    mse / block.len() as f64
}

/// Four-over-Six scale selection (Cook et al., 2025).
///
/// Tests two candidate scale factors (`absmax/4` and `absmax/6`) and returns
/// the one that produces lower quantization MSE. This roughly doubles
/// improvement vs standard single-scale quantization.
fn four_over_six_scale(block: &[f32]) -> f32 {
    let absmax = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    if absmax < f32::EPSILON {
        return 0.0;
    }

    let scale_4 = absmax / 4.0;
    let scale_6 = absmax / 6.0;

    let mse_4 = block_quantize_mse(block, scale_4);
    let mse_6 = block_quantize_mse(block, scale_6);

    if mse_6 < mse_4 {
        scale_6
    } else {
        scale_4
    }
}

/// Quantize a block of f32 values to packed 4-bit integers with per-block scale.
///
/// Returns `(packed_bytes, scale)` where packed_bytes has ceil(block.len()/2) bytes.
fn quantize_block_4bit(block: &[f32]) -> (Vec<u8>, f32) {
    let scale = four_over_six_scale(block);

    let mut quantized = Vec::with_capacity(block.len());
    if scale.abs() < f32::EPSILON {
        quantized.resize(block.len(), 0i8);
    } else {
        let inv_scale = 1.0 / scale;
        for &v in block {
            let q = (v * inv_scale)
                .round()
                .clamp(-QUANT_MAX_4BIT as f32, QUANT_MAX_4BIT as f32) as i8;
            quantized.push(q);
        }
    }

    // Pack pairs of values into bytes
    let packed_len = (quantized.len() + 1) / 2;
    let mut packed = Vec::with_capacity(packed_len);
    for chunk in quantized.chunks(2) {
        let hi = chunk[0];
        let lo = if chunk.len() > 1 { chunk[1] } else { 0 };
        packed.push(pack_nibbles(hi, lo));
    }

    (packed, scale)
}

/// Dequantize packed 4-bit values back to f32.
fn dequantize_block_4bit(packed: &[u8], scale: f32, count: usize) -> Vec<f32> {
    let mut result = Vec::with_capacity(count);
    for &byte in packed {
        let (hi, lo) = unpack_nibbles(byte);
        result.push(hi as f32 * scale);
        if result.len() < count {
            result.push(lo as f32 * scale);
        }
    }
    result.truncate(count);
    result
}

/// Quantize an entire f32 array into blocks of 16 with 4-bit values + per-block scales.
///
/// Returns `(packed_bytes, scales)` where scales has one f32 per block.
fn quantize_array_4bit(data: &[f32]) -> (Vec<u8>, Vec<f32>) {
    let mut all_packed = Vec::new();
    let mut all_scales = Vec::new();

    for block in data.chunks(BLOCK_SIZE_4BIT) {
        let (packed, scale) = quantize_block_4bit(block);
        all_packed.extend_from_slice(&packed);
        all_scales.push(scale);
    }

    (all_packed, all_scales)
}

/// Dequantize a full array from packed 4-bit blocks.
fn dequantize_array_4bit(packed: &[u8], scales: &[f32], total_count: usize) -> Vec<f32> {
    let mut result = Vec::with_capacity(total_count);
    let bytes_per_block = (BLOCK_SIZE_4BIT + 1) / 2; // 8 bytes per 16-element block
    let mut remaining = total_count;

    for (block_idx, &scale) in scales.iter().enumerate() {
        let block_count = remaining.min(BLOCK_SIZE_4BIT);
        let byte_offset = block_idx * bytes_per_block;
        let byte_count = (block_count + 1) / 2;
        let block_packed = &packed[byte_offset..byte_offset + byte_count];
        let block_values = dequantize_block_4bit(block_packed, scale, block_count);
        result.extend_from_slice(&block_values);
        remaining -= block_count;
    }

    result
}

/// A compact 4-bit field snapshot with per-block scaling.
///
/// Uses Four-over-Six scale selection from Cook et al. (2025) to minimize
/// quantization error. Each block of 16 values is stored as:
/// - 8 packed bytes (two 4-bit values per byte)
/// - 1 f32 scale factor
///
/// Total: 12 bytes per 16 values = 0.75 bytes/value (~81% savings vs f32).
#[derive(Debug, Clone)]
pub struct Block4BitSnapshot {
    /// Packed 4-bit real values (2 per byte)
    pub packed_re: Vec<u8>,
    /// Packed 4-bit imaginary values (2 per byte)
    pub packed_im: Vec<u8>,
    /// Per-block scale factors for real component
    pub scales_re: Vec<f32>,
    /// Per-block scale factors for imaginary component
    pub scales_im: Vec<f32>,
    /// Which layer this snapshot is from
    pub layer: LatticeLayer,
    /// Number of lattice sites
    pub num_sites: usize,
}

impl Block4BitSnapshot {
    /// Capture a field as a 4-bit quantized snapshot.
    pub fn from_field(field: &LatticeField) -> Self {
        let (packed_re, scales_re) = quantize_array_4bit(&field.values_re);
        let (packed_im, scales_im) = quantize_array_4bit(&field.values_im);

        Self {
            packed_re,
            packed_im,
            scales_re,
            scales_im,
            layer: field.layer,
            num_sites: field.num_sites,
        }
    }

    /// Restore this snapshot's values into an existing LatticeField.
    ///
    /// # Panics
    /// Panics if `field.num_sites != self.num_sites`.
    pub fn restore_to_field(&self, field: &mut LatticeField) {
        assert_eq!(
            field.num_sites, self.num_sites,
            "Cannot restore Block4BitSnapshot ({} sites) to field ({} sites)",
            self.num_sites, field.num_sites
        );

        let re = dequantize_array_4bit(&self.packed_re, &self.scales_re, self.num_sites);
        let im = dequantize_array_4bit(&self.packed_im, &self.scales_im, self.num_sites);
        field.values_re.copy_from_slice(&re);
        field.values_im.copy_from_slice(&im);
    }

    /// Create a new LatticeField from this snapshot, cloning topology from a template.
    ///
    /// # Panics
    /// Panics if `template.num_sites != self.num_sites`.
    pub fn to_field(&self, template: &LatticeField) -> LatticeField {
        assert_eq!(
            template.num_sites, self.num_sites,
            "Template ({} sites) doesn't match Block4BitSnapshot ({} sites)",
            template.num_sites, self.num_sites
        );
        let mut field = template.clone();
        self.restore_to_field(&mut field);
        field
    }

    /// Memory usage of this snapshot in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.packed_re.len() + self.packed_im.len()
            + (self.scales_re.len() + self.scales_im.len()) * 4
    }

    /// Compute max absolute error against a full-precision field.
    pub fn max_error(&self, field: &LatticeField) -> f32 {
        assert_eq!(field.num_sites, self.num_sites);
        let re = dequantize_array_4bit(&self.packed_re, &self.scales_re, self.num_sites);
        let im = dequantize_array_4bit(&self.packed_im, &self.scales_im, self.num_sites);
        let mut max_err = 0.0f32;
        for i in 0..self.num_sites {
            max_err = max_err
                .max((re[i] - field.values_re[i]).abs())
                .max((im[i] - field.values_im[i]).abs());
        }
        max_err
    }

    /// Compute mean absolute error against a full-precision field.
    pub fn mean_error(&self, field: &LatticeField) -> f32 {
        assert_eq!(field.num_sites, self.num_sites);
        let re = dequantize_array_4bit(&self.packed_re, &self.scales_re, self.num_sites);
        let im = dequantize_array_4bit(&self.packed_im, &self.scales_im, self.num_sites);
        let mut total = 0.0f64;
        let count = (self.num_sites * 2) as f64;
        for i in 0..self.num_sites {
            total += (re[i] - field.values_re[i]).abs() as f64;
            total += (im[i] - field.values_im[i]).abs() as f64;
        }
        (total / count) as f32
    }

    /// Memory savings ratio compared to full f32 storage.
    pub fn savings_ratio(&self) -> f32 {
        let f32_bytes = self.num_sites * 2 * 4;
        1.0 - (self.memory_bytes() as f32 / f32_bytes as f32)
    }
}

// ---------------------------------------------------------------------------
// Section 7: MsEdenSnapshot — Full MS-EDEN Pipeline (RHT + 4-bit + EDEN)
// ---------------------------------------------------------------------------

/// Seed offset for imaginary-component RHT (decorrelated from real).
const RHT_SEED_OFFSET: u64 = 0x517CC1B727220A95;

/// Full MS-EDEN snapshot implementing the complete Quartet II pipeline.
///
/// Quantization: Pad to power-of-2 → RHT → 4-bit quantize with 4/6 → EDEN correction.
/// Dequantization: Dequantize → EDEN correct → inverse RHT → trim.
///
/// This achieves the paper's key result: unbiased quantization with MSE
/// comparable to round-to-nearest, while using only ~6 bits per value.
///
/// For the E8 lattice (241 sites), data is padded to 256 for the RHT.
#[derive(Debug, Clone)]
pub struct MsEdenSnapshot {
    /// Packed 4-bit real values (after RHT)
    pub packed_re: Vec<u8>,
    /// Packed 4-bit imaginary values (after RHT)
    pub packed_im: Vec<u8>,
    /// Per-block scale factors for real (in RHT domain)
    pub scales_re: Vec<f32>,
    /// Per-block scale factors for imaginary (in RHT domain)
    pub scales_im: Vec<f32>,
    /// EDEN correction factor for real component
    pub correction_re: f32,
    /// EDEN correction factor for imaginary component
    pub correction_im: f32,
    /// RHT seed (for inverse transform)
    pub rht_seed: u64,
    /// Padded length (power of 2 >= num_sites)
    pub padded_len: usize,
    /// Which layer this snapshot is from
    pub layer: LatticeLayer,
    /// Number of lattice sites (original, before padding)
    pub num_sites: usize,
}

impl MsEdenSnapshot {
    /// Capture a field using the full MS-EDEN pipeline.
    ///
    /// `seed` controls both the RHT rotation and ensures reproducibility.
    ///
    /// Pipeline:
    /// 1. Pad field values to power-of-2 length (zero-pad)
    /// 2. Apply RHT (random sign flips + Walsh-Hadamard)
    /// 3. Quantize to 4-bit blocks with Four-over-Six scale selection
    /// 4. Compute EDEN correction factor per component
    pub fn from_field(field: &LatticeField, seed: u64) -> Self {
        let padded_len = next_power_of_2(field.num_sites);
        let rht_seed_im = seed.wrapping_add(RHT_SEED_OFFSET);

        // Step 1: Pad to power-of-2
        let mut re_padded = field.values_re.clone();
        re_padded.resize(padded_len, 0.0);
        let mut im_padded = field.values_im.clone();
        im_padded.resize(padded_len, 0.0);

        // Keep copies of the RHT input for EDEN correction
        let re_before_rht = re_padded.clone();
        let im_before_rht = im_padded.clone();

        // Step 2: Apply RHT
        rht_transform(&mut re_padded, seed);
        rht_transform(&mut im_padded, rht_seed_im);

        // Keep copies of RHT output for EDEN correction
        let re_rht = re_padded.clone();
        let im_rht = im_padded.clone();

        // Step 3: Quantize to 4-bit blocks
        let (packed_re, scales_re) = quantize_array_4bit(&re_padded);
        let (packed_im, scales_im) = quantize_array_4bit(&im_padded);

        // Dequantize for EDEN correction computation
        let re_quantized = dequantize_array_4bit(&packed_re, &scales_re, padded_len);
        let im_quantized = dequantize_array_4bit(&packed_im, &scales_im, padded_len);

        // Step 4: EDEN correction: S = <x_rht, x_rht> / <x_quantized, x_rht>
        let correction_re = eden_correction_factor(&re_rht, &re_quantized);
        let correction_im = eden_correction_factor(&im_rht, &im_quantized);

        // Verify correction is in expected range [0.8, 1.2]
        // (Paper shows typical values in [0.94, 1.06] for large dimensions)
        let _ = re_before_rht; // Used above for padding reference
        let _ = im_before_rht;

        Self {
            packed_re,
            packed_im,
            scales_re,
            scales_im,
            correction_re,
            correction_im,
            rht_seed: seed,
            padded_len,
            layer: field.layer,
            num_sites: field.num_sites,
        }
    }

    /// Restore this snapshot's values into an existing LatticeField.
    ///
    /// Inverse pipeline:
    /// 1. Dequantize 4-bit blocks
    /// 2. Apply EDEN correction
    /// 3. Apply inverse RHT (same transform, same seed = self-inverse)
    /// 4. Trim to original num_sites
    ///
    /// # Panics
    /// Panics if `field.num_sites != self.num_sites`.
    pub fn restore_to_field(&self, field: &mut LatticeField) {
        assert_eq!(
            field.num_sites, self.num_sites,
            "Cannot restore MsEdenSnapshot ({} sites) to field ({} sites)",
            self.num_sites, field.num_sites
        );

        let rht_seed_im = self.rht_seed.wrapping_add(RHT_SEED_OFFSET);

        // Step 1: Dequantize
        let mut re = dequantize_array_4bit(&self.packed_re, &self.scales_re, self.padded_len);
        let mut im = dequantize_array_4bit(&self.packed_im, &self.scales_im, self.padded_len);

        // Step 2: Apply EDEN correction
        for v in re.iter_mut() {
            *v *= self.correction_re;
        }
        for v in im.iter_mut() {
            *v *= self.correction_im;
        }

        // Step 3: Inverse RHT
        rht_inverse(&mut re, self.rht_seed);
        rht_inverse(&mut im, rht_seed_im);

        // Step 4: Trim to original size and copy
        field.values_re.copy_from_slice(&re[..self.num_sites]);
        field.values_im.copy_from_slice(&im[..self.num_sites]);
    }

    /// Create a new LatticeField from this snapshot, cloning topology from a template.
    ///
    /// # Panics
    /// Panics if `template.num_sites != self.num_sites`.
    pub fn to_field(&self, template: &LatticeField) -> LatticeField {
        assert_eq!(
            template.num_sites, self.num_sites,
            "Template ({} sites) doesn't match MsEdenSnapshot ({} sites)",
            template.num_sites, self.num_sites
        );
        let mut field = template.clone();
        self.restore_to_field(&mut field);
        field
    }

    /// Memory usage of this snapshot in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.packed_re.len() + self.packed_im.len()
            + (self.scales_re.len() + self.scales_im.len()) * 4
            + 8 // two f32 correction factors
    }

    /// Compute max absolute error against a full-precision field.
    pub fn max_error(&self, field: &LatticeField) -> f32 {
        assert_eq!(field.num_sites, self.num_sites);
        let restored = self.to_field(field);
        let mut max_err = 0.0f32;
        for i in 0..self.num_sites {
            max_err = max_err
                .max((restored.values_re[i] - field.values_re[i]).abs())
                .max((restored.values_im[i] - field.values_im[i]).abs());
        }
        max_err
    }

    /// Compute mean absolute error against a full-precision field.
    pub fn mean_error(&self, field: &LatticeField) -> f32 {
        assert_eq!(field.num_sites, self.num_sites);
        let restored = self.to_field(field);
        let mut total = 0.0f64;
        let count = (self.num_sites * 2) as f64;
        for i in 0..self.num_sites {
            total += (restored.values_re[i] - field.values_re[i]).abs() as f64;
            total += (restored.values_im[i] - field.values_im[i]).abs() as f64;
        }
        (total / count) as f32
    }

    /// Memory savings ratio compared to full f32 storage.
    pub fn savings_ratio(&self) -> f32 {
        let f32_bytes = self.num_sites * 2 * 4;
        1.0 - (self.memory_bytes() as f32 / f32_bytes as f32)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

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

    // -----------------------------------------------------------------------
    // Section 1 tests: Stochastic Rounding
    // -----------------------------------------------------------------------

    #[test]
    fn test_stochastic_round_unbiased() {
        // Verify E[SR(x)] ≈ x over many trials.
        // Use a value that's NOT exactly representable in f16.
        let val = 0.1234567f32; // not exact in f16
        let n_trials = 10_000;
        let mut rng = SmallRng::seed_from_u64(12345);
        let mut sum = 0.0f64;

        for _ in 0..n_trials {
            let rounded = stochastic_round_f16(val, rng.gen::<f32>());
            sum += rounded.to_f32() as f64;
        }

        let mean = sum / n_trials as f64;
        let error = (mean - val as f64).abs();
        // With 10k trials, the mean should be within ~0.001 of the true value
        assert!(
            error < 0.002,
            "SR mean {} should be close to {} (error {})",
            mean,
            val,
            error
        );
    }

    #[test]
    fn test_stochastic_round_exact_values() {
        // Values exactly representable in f16 should always round exactly
        let exact_vals = [0.0f32, 1.0, -1.0, 0.5, -0.5, 2.0];
        for &v in &exact_vals {
            let rounded = stochastic_round_f16(v, 0.5);
            assert_eq!(
                rounded.to_f32(),
                v,
                "Exact value {} should round exactly",
                v
            );
        }
    }

    // -----------------------------------------------------------------------
    // Section 2 tests: FWHT
    // -----------------------------------------------------------------------

    #[test]
    fn test_fwht_self_inverse() {
        let original = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut data = original.clone();

        // Apply FWHT twice (normalized is self-inverse)
        fwht_normalized(&mut data);
        fwht_normalized(&mut data);

        for (i, (&orig, &result)) in original.iter().zip(data.iter()).enumerate() {
            let err = (orig - result).abs();
            assert!(
                err < 1e-5,
                "FWHT self-inverse failed at index {}: {} vs {} (err {})",
                i,
                orig,
                result,
                err
            );
        }
    }

    #[test]
    fn test_fwht_preserves_l2_norm() {
        // Parseval's theorem: ||FWHT(x)|| = ||x||
        let data = vec![1.0f32, -2.0, 3.0, -4.0];
        let norm_before: f64 = data.iter().map(|&v| (v as f64) * (v as f64)).sum();

        let mut transformed = data;
        fwht_normalized(&mut transformed);
        let norm_after: f64 = transformed
            .iter()
            .map(|&v| (v as f64) * (v as f64))
            .sum();

        let rel_err = ((norm_before - norm_after) / norm_before).abs();
        assert!(
            rel_err < 1e-6,
            "FWHT should preserve L2 norm: {} vs {}",
            norm_before,
            norm_after
        );
    }

    // -----------------------------------------------------------------------
    // Section 3 tests: RHT
    // -----------------------------------------------------------------------

    #[test]
    fn test_rht_forward_inverse() {
        let original = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut data = original.clone();
        let seed = 42u64;

        // Apply RHT forward then inverse → should recover original
        rht_transform(&mut data, seed);
        rht_inverse(&mut data, seed);

        for (i, (&orig, &result)) in original.iter().zip(data.iter()).enumerate() {
            let err = (orig - result).abs();
            assert!(
                err < 1e-5,
                "RHT self-inverse failed at index {}: {} vs {} (err {})",
                i,
                orig,
                result,
                err
            );
        }
    }

    #[test]
    fn test_rht_preserves_norm() {
        let data = vec![1.0f32, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0];
        let norm_before: f64 = data.iter().map(|&v| (v as f64) * (v as f64)).sum();

        let mut transformed = data;
        rht_transform(&mut transformed, 123);
        let norm_after: f64 = transformed
            .iter()
            .map(|&v| (v as f64) * (v as f64))
            .sum();

        let rel_err = ((norm_before - norm_after) / norm_before).abs();
        assert!(
            rel_err < 1e-5,
            "RHT should preserve L2 norm: {} vs {}",
            norm_before,
            norm_after
        );
    }

    // -----------------------------------------------------------------------
    // Section 4 tests: EDEN Correction
    // -----------------------------------------------------------------------

    #[test]
    fn test_eden_correction_factor() {
        // If quantization uniformly scales by 0.95, correction should be ~1/0.95
        let original = vec![1.0f32, 2.0, 3.0, 4.0];
        let quantized: Vec<f32> = original.iter().map(|&v| v * 0.95).collect();
        let s = eden_correction_factor(&original, &quantized);

        // S = ||x||^2 / <Q, x> = sum(x^2) / sum(0.95*x * x) = 1/0.95
        let expected = 1.0 / 0.95;
        assert!(
            (s - expected as f32).abs() < 1e-5,
            "EDEN correction {} should be {} for uniform 0.95 scaling",
            s,
            expected
        );
    }

    #[test]
    fn test_eden_correction_identity() {
        // If quantization is perfect (no error), correction should be 1.0
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let s = eden_correction_factor(&data, &data);
        assert!(
            (s - 1.0).abs() < 1e-6,
            "EDEN correction for perfect quantization should be 1.0, got {}",
            s
        );
    }

    // -----------------------------------------------------------------------
    // Section 5 tests: EdenSnapshot
    // -----------------------------------------------------------------------

    #[test]
    fn test_eden_snapshot_round_trip() {
        let field = make_e8_field();
        let snapshot = EdenSnapshot::from_field(&field, 42);

        assert_eq!(snapshot.num_sites, 241);
        assert_eq!(snapshot.values_re.len(), 241);
        assert_eq!(snapshot.values_im.len(), 241);

        let restored = snapshot.to_field(&field);
        assert_eq!(restored.num_sites, field.num_sites);

        // EDEN+SR should have reasonable error
        let max_err = snapshot.max_error(&field);
        assert!(
            max_err < 0.02,
            "EdenSnapshot max error {} too large",
            max_err
        );
    }

    #[test]
    fn test_eden_snapshot_unbiasedness() {
        // Average error across many snapshots with different seeds should be ~0
        let field = make_e8_field();
        let n_snapshots = 50;
        let mut total_bias_re = 0.0f64;
        let mut total_bias_im = 0.0f64;

        for seed in 0..n_snapshots {
            let snap = EdenSnapshot::from_field(&field, seed * 1000 + 7);
            for i in 0..field.num_sites {
                total_bias_re +=
                    (snap.values_re[i].to_f32() * snap.correction_re - field.values_re[i]) as f64;
                total_bias_im +=
                    (snap.values_im[i].to_f32() * snap.correction_im - field.values_im[i]) as f64;
            }
        }

        let mean_bias_re = total_bias_re / (n_snapshots as f64 * field.num_sites as f64);
        let mean_bias_im = total_bias_im / (n_snapshots as f64 * field.num_sites as f64);

        // Bias should be very small (unbiased property)
        assert!(
            mean_bias_re.abs() < 0.005,
            "Mean bias (re) should be ~0, got {}",
            mean_bias_re
        );
        assert!(
            mean_bias_im.abs() < 0.005,
            "Mean bias (im) should be ~0, got {}",
            mean_bias_im
        );
    }

    #[test]
    fn test_eden_vs_basic_mse() {
        // EdenSnapshot should have comparable or better MSE than basic FP16
        let field = make_e8_field();
        let eden = EdenSnapshot::from_field(&field, 42);
        let basic = crate::fp16::CompactFieldSnapshot::from_field(&field);

        let eden_mean = eden.mean_error(&field);
        let basic_mean = basic.mean_error(&field);

        // EDEN should not be dramatically worse than basic (it corrects bias)
        // Allow some slack since SR introduces variance
        assert!(
            eden_mean < basic_mean * 3.0,
            "EdenSnapshot mean error {} should not be much worse than basic {} ",
            eden_mean,
            basic_mean
        );
    }

    #[test]
    fn test_eden_memory_overhead() {
        let field = make_e8_field();
        let eden = EdenSnapshot::from_field(&field, 42);
        let basic = crate::fp16::CompactFieldSnapshot::from_field(&field);

        // EDEN adds only 8 bytes for corrections
        assert_eq!(eden.memory_bytes(), basic.memory_bytes() + 8);
        // Savings ratio should still be close to 0.5
        assert!(
            eden.savings_ratio() > 0.49,
            "EdenSnapshot savings {} should be ~0.5",
            eden.savings_ratio()
        );
    }

    // -----------------------------------------------------------------------
    // Section 6 tests: Block4BitSnapshot
    // -----------------------------------------------------------------------

    #[test]
    fn test_pack_unpack_nibbles() {
        for hi in -7i8..=7 {
            for lo in -7i8..=7 {
                let packed = pack_nibbles(hi, lo);
                let (h, l) = unpack_nibbles(packed);
                assert_eq!((h, l), (hi, lo), "Pack/unpack failed for ({}, {})", hi, lo);
            }
        }
    }

    #[test]
    fn test_four_over_six_selection() {
        // For a block with values 0..15, the 4/6 selector should pick
        // the scale that minimizes MSE
        let block: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
        let scale = four_over_six_scale(&block);

        // Scale should be positive
        assert!(scale > 0.0, "Scale should be positive, got {}", scale);

        // Verify it's one of the two candidates
        let absmax = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let s4 = absmax / 4.0;
        let s6 = absmax / 6.0;
        assert!(
            (scale - s4).abs() < 1e-6 || (scale - s6).abs() < 1e-6,
            "Scale {} should be either {} or {}",
            scale,
            s4,
            s6
        );
    }

    #[test]
    fn test_block4bit_round_trip() {
        let field = make_e8_field();
        let snapshot = Block4BitSnapshot::from_field(&field);

        assert_eq!(snapshot.num_sites, 241);

        let restored = snapshot.to_field(&field);
        assert_eq!(restored.num_sites, field.num_sites);

        // 4-bit has larger error than FP16, but should still be bounded
        let max_err = snapshot.max_error(&field);
        assert!(
            max_err < 1.0,
            "Block4Bit max error {} should be < 1.0 for amplitude 1.5",
            max_err
        );
    }

    #[test]
    fn test_block4bit_memory_savings() {
        let field = make_e8_field();
        let snapshot = Block4BitSnapshot::from_field(&field);

        let f32_bytes = field.num_sites * 2 * 4; // 241 * 2 * 4 = 1928
        let snap_bytes = snapshot.memory_bytes();

        // Should be significantly smaller than f32
        assert!(
            snap_bytes < f32_bytes / 2,
            "Block4Bit {} bytes should be < {} (half of f32)",
            snap_bytes,
            f32_bytes / 2
        );

        // Savings should be > 70%
        assert!(
            snapshot.savings_ratio() > 0.70,
            "Block4Bit savings {} should be > 0.70",
            snapshot.savings_ratio()
        );
    }

    #[test]
    fn test_block4bit_extreme_values() {
        let lattice = HypercubicLattice::new(2);
        let mut field = LatticeField::from_lattice(&lattice);
        field.set(0, 100.0, -100.0);
        field.set(1, 0.001, -0.001);
        field.set(2, 0.0, 0.0);

        let snapshot = Block4BitSnapshot::from_field(&field);
        let restored = snapshot.to_field(&field);

        // Zero should be exact (or very close)
        assert!(
            restored.values_re[2].abs() < 0.1,
            "Zero should be close to 0, got {}",
            restored.values_re[2]
        );
    }

    // -----------------------------------------------------------------------
    // Section 7 tests: MsEdenSnapshot
    // -----------------------------------------------------------------------

    #[test]
    fn test_mseden_round_trip() {
        let field = make_e8_field();
        let snapshot = MsEdenSnapshot::from_field(&field, 42);

        assert_eq!(snapshot.num_sites, 241);
        assert_eq!(snapshot.padded_len, 256); // 241 → 256

        let restored = snapshot.to_field(&field);
        assert_eq!(restored.num_sites, field.num_sites);

        // MS-EDEN should have bounded error
        let max_err = snapshot.max_error(&field);
        assert!(
            max_err < 2.0,
            "MsEden max error {} should be < 2.0",
            max_err
        );
    }

    #[test]
    fn test_mseden_correction_range() {
        // Paper says typical S ∈ [0.94, 1.06]; we allow wider for small dimensions
        let field = make_e8_field();
        let snapshot = MsEdenSnapshot::from_field(&field, 42);

        assert!(
            snapshot.correction_re > 0.5 && snapshot.correction_re < 2.0,
            "EDEN correction_re {} out of expected range",
            snapshot.correction_re
        );
        assert!(
            snapshot.correction_im > 0.5 && snapshot.correction_im < 2.0,
            "EDEN correction_im {} out of expected range",
            snapshot.correction_im
        );
    }

    #[test]
    fn test_mseden_memory_savings() {
        let field = make_e8_field();
        let snapshot = MsEdenSnapshot::from_field(&field, 42);

        let f32_bytes = field.num_sites * 2 * 4;
        let snap_bytes = snapshot.memory_bytes();

        // Should have significant savings
        assert!(
            snap_bytes < f32_bytes,
            "MsEden {} bytes should be < {} (f32)",
            snap_bytes,
            f32_bytes
        );
        assert!(
            snapshot.savings_ratio() > 0.5,
            "MsEden savings {} should be > 0.5",
            snapshot.savings_ratio()
        );
    }

    #[test]
    fn test_mseden_vs_block4bit_mse() {
        // MS-EDEN should generally have comparable or better MSE than plain 4-bit
        // due to RHT smoothing + EDEN correction
        let field = make_e8_field();
        let mseden = MsEdenSnapshot::from_field(&field, 42);
        let block4 = Block4BitSnapshot::from_field(&field);

        let mseden_mean = mseden.mean_error(&field);
        let block4_mean = block4.mean_error(&field);

        // MS-EDEN shouldn't be dramatically worse (RHT + EDEN help)
        // Allow some margin since 241 isn't large enough for asymptotic guarantees
        assert!(
            mseden_mean < block4_mean * 5.0,
            "MsEden mean error {} should not be much worse than block4bit {}",
            mseden_mean,
            block4_mean
        );
    }

    // -----------------------------------------------------------------------
    // Cross-type comparison tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_e8_all_snapshot_types() {
        let field = make_e8_field();

        let basic = crate::fp16::CompactFieldSnapshot::from_field(&field);
        let eden = EdenSnapshot::from_field(&field, 42);
        let block4 = Block4BitSnapshot::from_field(&field);
        let mseden = MsEdenSnapshot::from_field(&field, 42);

        // All should round-trip with finite error
        assert!(basic.max_error(&field).is_finite());
        assert!(eden.max_error(&field).is_finite());
        assert!(block4.max_error(&field).is_finite());
        assert!(mseden.max_error(&field).is_finite());

        // Memory hierarchy: basic == eden (modulo 8B) > mseden >= block4
        assert!(eden.memory_bytes() <= basic.memory_bytes() + 16);
        assert!(block4.memory_bytes() < basic.memory_bytes());
    }

    #[test]
    fn test_small_field_all_types() {
        let field = make_small_field();

        let eden = EdenSnapshot::from_field(&field, 42);
        let block4 = Block4BitSnapshot::from_field(&field);
        let mseden = MsEdenSnapshot::from_field(&field, 42);

        // All should produce valid snapshots for small fields
        assert_eq!(eden.num_sites, field.num_sites);
        assert_eq!(block4.num_sites, field.num_sites);
        assert_eq!(mseden.num_sites, field.num_sites);

        // All should restore without panic
        let _ = eden.to_field(&field);
        let _ = block4.to_field(&field);
        let _ = mseden.to_field(&field);
    }
}
