// Copyright (c) 2025-2026 brdigetrlol. All rights reserved.
// SPDX-License-Identifier: LicenseRef-Icarus-Proprietary
// See LICENSE in the repository root for full license terms.

//! Extended Binary Golay Code \[24, 12, 8\]
//!
//! The extended Golay code is a perfect, self-dual binary code and the
//! algebraic foundation of the Leech lattice. Every Leech lattice minimal
//! vector can be described in terms of Golay codewords or octads.
//!
//! # Construction
//!
//! Built from the (23, 12, 7) perfect binary Golay code whose generator
//! polynomial is g(x) = x^11 + x^9 + x^7 + x^6 + x^5 + x + 1, which
//! divides x^23 + 1 over GF(2). Extended to \[24, 12, 8\] by appending
//! an overall parity check bit.
//!
//! # Weight Distribution
//!
//! | Weight | Count | Name |
//! |--------|-------|------|
//! | 0      | 1     | zero word |
//! | 8      | 759   | octads |
//! | 12     | 2576  | dodecads |
//! | 16     | 759   | complements of octads |
//! | 24     | 1     | all-ones word |
//! | **Total** | **4096** | |
//!
//! # Properties
//!
//! - Self-dual: C = C^perp
//! - Automorphism group: Mathieu group M_24 (order 244,823,040)
//! - The 759 octads form a Steiner system S(5, 8, 24)

/// Generator polynomial for the (23, 12, 7) binary Golay code.
///
/// g(x) = x^11 + x^9 + x^7 + x^6 + x^5 + x + 1
///
/// Bit layout: bit i = coefficient of x^i.
const GOLAY23_GEN: u32 = 0xAE3;

/// The Extended Binary Golay Code \[24, 12, 8\].
///
/// Stores precomputed codewords, octad indices, and a syndrome lookup
/// table for efficient decoding (up to 3-error correction).
#[derive(Debug, Clone)]
pub struct GolayCode {
    /// All 4096 codewords as 24-bit packed integers (bit i = coordinate i).
    codewords: Vec<u32>,
    /// Indices into `codewords` for the 759 octads (weight-8 codewords).
    octad_indices: Vec<u16>,
    /// Syndrome lookup table for the (23, 12, 7) inner code.
    /// Maps 11-bit syndrome → error pattern (weight ≤ 3).
    /// Entry = u32::MAX means "no single pattern found" (shouldn't happen for perfect code).
    syndrome_table: [u32; 2048],
}

impl GolayCode {
    /// Construct the Extended Golay Code.
    ///
    /// Generates all 4096 codewords, identifies the 759 octads,
    /// and builds the syndrome lookup table for 3-error correction.
    pub fn new() -> Self {
        let mut codewords = Vec::with_capacity(4096);
        let mut octad_indices = Vec::with_capacity(759);

        for msg in 0u16..4096 {
            let cw = Self::encode_raw(msg);
            if cw.count_ones() == 8 {
                octad_indices.push(msg);
            }
            codewords.push(cw);
        }

        let syndrome_table = Self::build_syndrome_table();

        Self {
            codewords,
            octad_indices,
            syndrome_table,
        }
    }

    /// Encode a 12-bit message into a 24-bit extended Golay codeword.
    ///
    /// Uses systematic encoding: bits 0-10 are parity (from the generator
    /// polynomial), bits 11-22 are the message, bit 23 is overall parity.
    pub fn encode(msg: u16) -> u32 {
        assert!(msg < 4096, "message must be 12 bits (0..4096)");
        Self::encode_raw(msg)
    }

    /// Encode without bounds check (internal use).
    fn encode_raw(msg: u16) -> u32 {
        // Systematic encoding for (23,12,7):
        // c(x) = x^11 * m(x) + (x^11 * m(x) mod g(x))
        let shifted = (msg as u32) << 11;
        let remainder = gf2_mod(shifted, GOLAY23_GEN);
        let c23 = shifted | remainder;

        // Extend to [24,12,8] with overall parity check (bit 23)
        let parity = c23.count_ones() & 1;
        c23 | (parity << 23)
    }

    /// All 4096 codewords as packed 24-bit integers.
    pub fn codewords(&self) -> &[u32] {
        &self.codewords
    }

    /// Number of codewords (always 4096 = 2^12).
    pub fn num_codewords(&self) -> usize {
        self.codewords.len()
    }

    /// Number of octads (always 759).
    pub fn num_octads(&self) -> usize {
        self.octad_indices.len()
    }

    /// Get the 759 octads as sets of 8 coordinate indices.
    ///
    /// Each octad is a weight-8 codeword. The octads form a Steiner system
    /// S(5, 8, 24): any 5 of the 24 coordinates are contained in exactly
    /// one octad.
    pub fn octads(&self) -> Vec<[usize; 8]> {
        self.octad_indices
            .iter()
            .map(|&idx| {
                let cw = self.codewords[idx as usize];
                Self::codeword_support(cw)
            })
            .collect()
    }

    /// Get the support (set of nonzero positions) of a weight-8 codeword.
    fn codeword_support(cw: u32) -> [usize; 8] {
        debug_assert_eq!(cw.count_ones(), 8);
        let mut positions = [0usize; 8];
        let mut pos = 0;
        for bit in 0..24 {
            if cw & (1 << bit) != 0 {
                positions[pos] = bit;
                pos += 1;
            }
        }
        positions
    }

    /// Check if a 24-bit word is a valid Golay codeword.
    pub fn is_codeword(&self, word: u32) -> bool {
        // Must have even weight (overall parity check)
        if word.count_ones() % 2 != 0 {
            return false;
        }
        // The 23-bit part must have zero syndrome
        let r23 = word & 0x7F_FFFF;
        gf2_mod(r23, GOLAY23_GEN) == 0
    }

    /// Decode a received 24-bit word to the nearest codeword.
    ///
    /// The extended Golay code can correct up to 3 errors (since d=8, t=3).
    /// Returns `(nearest_codeword, hamming_distance)`.
    pub fn decode(&self, received: u32) -> (u32, u32) {
        let received = received & 0xFF_FFFF; // mask to 24 bits
        let overall_parity = received.count_ones() & 1; // 0=even, 1=odd
        let r23 = received & 0x7F_FFFF; // bits 0-22
        let syndrome = gf2_mod(r23, GOLAY23_GEN) as usize;

        if syndrome == 0 {
            if overall_parity == 0 {
                // No errors
                (received, 0)
            } else {
                // Single error in parity bit (position 23)
                (received ^ (1 << 23), 1)
            }
        } else if overall_parity == 1 {
            // Odd number of total errors, syndrome nonzero →
            // odd number of errors in positions 0-22.
            // For ≤3 total errors with odd parity and nonzero syndrome,
            // all errors are in positions 0-22 (1 or 3 errors there).
            if let Some(error_pattern) = self.syndrome_lookup(syndrome) {
                let corrected23 = r23 ^ error_pattern;
                let p = (corrected23.count_ones() & 1) as u32;
                let corrected = corrected23 | (p << 23);
                let dist = (received ^ corrected).count_ones();
                (corrected, dist)
            } else {
                self.nearest_brute_force(received)
            }
        } else {
            // Even number of total errors, syndrome nonzero →
            // bit 23 is in error + some errors in 0-22.
            // Flip bit 23 first, then decode the inner 23-bit code.
            let flipped = received ^ (1 << 23);
            // Now we have an odd number of errors in the 23-bit part
            if let Some(error_pattern) = self.syndrome_lookup(syndrome) {
                let corrected23 = r23 ^ error_pattern;
                let p = (corrected23.count_ones() & 1) as u32;
                let corrected = corrected23 | (p << 23);
                let dist = (received ^ corrected).count_ones();
                if dist <= 3 {
                    (corrected, dist)
                } else {
                    // Syndrome table gave wrong coset leader; fall back
                    self.nearest_brute_force(received)
                }
            } else {
                self.nearest_brute_force(flipped)
            }
        }
    }

    /// Convert a packed 24-bit codeword to a coordinate array.
    pub fn to_coords(word: u32) -> [u8; 24] {
        let mut coords = [0u8; 24];
        for i in 0..24 {
            coords[i] = ((word >> i) & 1) as u8;
        }
        coords
    }

    /// Convert a coordinate array to a packed 24-bit word.
    pub fn from_coords(coords: &[u8]) -> u32 {
        assert!(coords.len() >= 24);
        let mut word = 0u32;
        for i in 0..24 {
            if coords[i] != 0 {
                word |= 1 << i;
            }
        }
        word
    }

    /// Get the Hamming weight of a codeword.
    pub fn weight(cw: u32) -> u32 {
        cw.count_ones()
    }

    /// Build the syndrome lookup table for the (23, 12, 7) inner code.
    ///
    /// Since the Golay code is perfect, every non-zero 11-bit syndrome
    /// corresponds to exactly one error pattern of weight ≤ 3.
    /// Total: 1 (weight 0) + 23 (weight 1) + 253 (weight 2) + 1771 (weight 3) = 2048.
    fn build_syndrome_table() -> [u32; 2048] {
        let mut table = [u32::MAX; 2048];
        table[0] = 0; // syndrome 0 → no error

        // Weight-1 error patterns: 23 patterns
        for i in 0..23u32 {
            let error = 1u32 << i;
            let syn = gf2_mod(error, GOLAY23_GEN) as usize;
            table[syn] = error;
        }

        // Weight-2 error patterns: C(23,2) = 253 patterns
        for i in 0..23u32 {
            for j in (i + 1)..23 {
                let error = (1u32 << i) | (1u32 << j);
                let syn = gf2_mod(error, GOLAY23_GEN) as usize;
                if table[syn] == u32::MAX {
                    table[syn] = error;
                }
            }
        }

        // Weight-3 error patterns: C(23,3) = 1771 patterns
        for i in 0..23u32 {
            for j in (i + 1)..23 {
                for k in (j + 1)..23 {
                    let error = (1u32 << i) | (1u32 << j) | (1u32 << k);
                    let syn = gf2_mod(error, GOLAY23_GEN) as usize;
                    if table[syn] == u32::MAX {
                        table[syn] = error;
                    }
                }
            }
        }

        // Verify: perfect code → all 2048 entries filled
        debug_assert!(
            table.iter().all(|&x| x != u32::MAX),
            "Syndrome table incomplete — generator polynomial may be wrong"
        );

        table
    }

    /// Look up an error pattern from the syndrome table.
    fn syndrome_lookup(&self, syndrome: usize) -> Option<u32> {
        if syndrome < 2048 {
            let val = self.syndrome_table[syndrome];
            if val != u32::MAX {
                Some(val)
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Find nearest codeword by brute force (fallback for edge cases).
    fn nearest_brute_force(&self, word: u32) -> (u32, u32) {
        let mut best_cw = 0u32;
        let mut best_dist = 25u32;
        for &cw in &self.codewords {
            let dist = (word ^ cw).count_ones();
            if dist < best_dist {
                best_dist = dist;
                best_cw = cw;
                if dist == 0 {
                    break;
                }
            }
        }
        (best_cw, best_dist)
    }
}

impl Default for GolayCode {
    fn default() -> Self {
        Self::new()
    }
}

/// GF(2) polynomial modular reduction: computes a(x) mod b(x).
///
/// Polynomials are represented as u32 where bit i = coefficient of x^i.
fn gf2_mod(mut a: u32, b: u32) -> u32 {
    assert!(b != 0, "division by zero polynomial");
    let deg_b = 31 - b.leading_zeros();
    loop {
        if a == 0 {
            return 0;
        }
        let deg_a = 31 - a.leading_zeros();
        if deg_a < deg_b {
            return a;
        }
        a ^= b << (deg_a - deg_b);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generator_polynomial_divides_x23_plus_1() {
        // g(x) must divide x^23 + 1 in GF(2)[x]
        let x23_plus_1: u32 = (1 << 23) | 1;
        let remainder = gf2_mod(x23_plus_1, GOLAY23_GEN);
        assert_eq!(remainder, 0, "g(x) does not divide x^23 + 1");
    }

    #[test]
    fn test_generator_polynomial_degree() {
        // g(x) should have degree 11 (for a (23,12) code: n-k = 23-12 = 11)
        let degree = 31 - GOLAY23_GEN.leading_zeros();
        assert_eq!(degree, 11);
    }

    #[test]
    fn test_codeword_count() {
        let code = GolayCode::new();
        assert_eq!(code.num_codewords(), 4096, "should have 2^12 = 4096 codewords");
    }

    #[test]
    fn test_octad_count() {
        let code = GolayCode::new();
        assert_eq!(code.num_octads(), 759, "should have exactly 759 octads");
    }

    #[test]
    fn test_weight_distribution() {
        let code = GolayCode::new();
        let mut weight_counts = [0u32; 25];
        for &cw in code.codewords() {
            weight_counts[cw.count_ones() as usize] += 1;
        }

        assert_eq!(weight_counts[0], 1, "w0 should be 1 (zero word)");
        assert_eq!(weight_counts[8], 759, "w8 should be 759 (octads)");
        assert_eq!(weight_counts[12], 2576, "w12 should be 2576 (dodecads)");
        assert_eq!(weight_counts[16], 759, "w16 should be 759 (complement octads)");
        assert_eq!(weight_counts[24], 1, "w24 should be 1 (all-ones)");

        // All other weights should be zero
        for w in 0..25 {
            if w != 0 && w != 8 && w != 12 && w != 16 && w != 24 {
                assert_eq!(
                    weight_counts[w], 0,
                    "weight {} should have count 0, got {}",
                    w, weight_counts[w]
                );
            }
        }
    }

    #[test]
    fn test_minimum_distance() {
        let code = GolayCode::new();
        let mut min_dist = 25u32;
        // Check distance between zero codeword and all others
        for &cw in code.codewords() {
            if cw != 0 {
                let dist = cw.count_ones();
                if dist < min_dist {
                    min_dist = dist;
                }
            }
        }
        assert_eq!(min_dist, 8, "minimum distance should be 8");
    }

    #[test]
    fn test_self_dual() {
        // The extended Golay code is self-dual: C = C^perp.
        // For any two codewords c1, c2: <c1, c2> = 0 mod 2
        // (i.e., they have even-sized intersection).
        let code = GolayCode::new();
        // Check a sample of pairs (all 4096^2 would be slow)
        for i in (0..4096).step_by(17) {
            for j in (0..4096).step_by(19) {
                let c1 = code.codewords()[i];
                let c2 = code.codewords()[j];
                let inner = (c1 & c2).count_ones();
                assert_eq!(
                    inner % 2,
                    0,
                    "codewords {} and {} are not orthogonal (inner product = {})",
                    i,
                    j,
                    inner
                );
            }
        }
    }

    #[test]
    fn test_all_codewords_even_weight() {
        // Extended code: all codewords have even weight
        let code = GolayCode::new();
        for (i, &cw) in code.codewords().iter().enumerate() {
            assert_eq!(
                cw.count_ones() % 2,
                0,
                "codeword {} (msg={}) has odd weight {}",
                i,
                i,
                cw.count_ones()
            );
        }
    }

    #[test]
    fn test_syndrome_table_complete() {
        let code = GolayCode::new();
        // Every syndrome should have a valid entry (perfect code)
        for syn in 0..2048 {
            assert_ne!(
                code.syndrome_table[syn],
                u32::MAX,
                "syndrome {} has no entry",
                syn
            );
        }
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let code = GolayCode::new();
        for msg in 0u16..4096 {
            let cw = GolayCode::encode(msg);
            assert!(code.is_codeword(cw), "encode({}) produced invalid codeword", msg);
            let (decoded, dist) = code.decode(cw);
            assert_eq!(dist, 0, "decode of valid codeword should give distance 0");
            assert_eq!(decoded, cw, "decode should return the same codeword");
        }
    }

    #[test]
    fn test_single_error_correction() {
        let code = GolayCode::new();
        // Test single-bit errors on a sample of codewords
        for msg in (0u16..4096).step_by(64) {
            let cw = GolayCode::encode(msg);
            for bit in 0..24 {
                let corrupted = cw ^ (1 << bit);
                let (decoded, dist) = code.decode(corrupted);
                assert_eq!(decoded, cw, "failed to correct 1-bit error at position {}", bit);
                assert_eq!(dist, 1);
            }
        }
    }

    #[test]
    fn test_double_error_correction() {
        let code = GolayCode::new();
        // Test double-bit errors on msg=0
        let cw = GolayCode::encode(0);
        for i in 0..24 {
            for j in (i + 1)..24 {
                let corrupted = cw ^ (1 << i) ^ (1 << j);
                let (decoded, dist) = code.decode(corrupted);
                assert_eq!(decoded, cw, "failed to correct 2-bit error at ({}, {})", i, j);
                assert_eq!(dist, 2);
            }
        }
    }

    #[test]
    fn test_triple_error_correction() {
        let code = GolayCode::new();
        // Test triple-bit errors on msg=0
        let cw = GolayCode::encode(0);
        for i in (0..24).step_by(4) {
            for j in ((i + 1)..24).step_by(3) {
                for k in ((j + 1)..24).step_by(2) {
                    let corrupted = cw ^ (1 << i) ^ (1 << j) ^ (1 << k);
                    let (decoded, dist) = code.decode(corrupted);
                    assert_eq!(
                        decoded, cw,
                        "failed to correct 3-bit error at ({}, {}, {})",
                        i, j, k
                    );
                    assert_eq!(dist, 3);
                }
            }
        }
    }

    #[test]
    fn test_is_codeword() {
        let code = GolayCode::new();
        for msg in 0u16..4096 {
            let cw = GolayCode::encode(msg);
            assert!(code.is_codeword(cw));
        }
        // Non-codewords should fail
        let cw = GolayCode::encode(0);
        assert!(!code.is_codeword(cw ^ 1)); // flip one bit
    }

    #[test]
    fn test_to_from_coords() {
        for msg in (0u16..4096).step_by(100) {
            let cw = GolayCode::encode(msg);
            let coords = GolayCode::to_coords(cw);
            let back = GolayCode::from_coords(&coords);
            assert_eq!(back, cw);
        }
    }

    #[test]
    fn test_octad_is_steiner_system() {
        // S(5, 8, 24): any 5 coordinates belong to exactly one octad.
        // Test a sample of 5-subsets.
        let code = GolayCode::new();
        let octads = code.octads();

        // Check a few specific 5-subsets
        let test_subsets: Vec<[usize; 5]> = vec![
            [0, 1, 2, 3, 4],
            [0, 5, 10, 15, 20],
            [1, 3, 7, 11, 23],
            [2, 6, 14, 18, 22],
        ];

        for subset in &test_subsets {
            let mut containing_octads = 0;
            for octad in &octads {
                if subset.iter().all(|&s| octad.contains(&s)) {
                    containing_octads += 1;
                }
            }
            assert_eq!(
                containing_octads, 1,
                "5-subset {:?} is in {} octads, expected exactly 1",
                subset, containing_octads
            );
        }
    }

    #[test]
    fn test_gf2_mod() {
        // x^3 mod (x^2 + 1) = x (since x^3 = x*(x^2+1) + x)
        // x^3 = 0b1000, x^2 + 1 = 0b101
        assert_eq!(gf2_mod(0b1000, 0b101), 0b10);

        // x^2 mod (x^2 + 1) = 1
        assert_eq!(gf2_mod(0b100, 0b101), 1);

        // 0 mod anything = 0
        assert_eq!(gf2_mod(0, 0b101), 0);
    }
}
