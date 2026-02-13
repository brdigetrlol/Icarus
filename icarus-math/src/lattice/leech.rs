//! Leech Lattice (24D) — Creative/Analogical Reasoning Layer (System 1.5)
//!
//! The Leech lattice Lambda_24 is the unique even unimodular lattice in 24
//! dimensions with no roots (norm-2 vectors). It achieves the densest sphere
//! packing in 24D and the maximum kissing number: 196,560.
//!
//! # Mathematical Foundation
//!
//! Constructed from the Extended Golay Code C_24 \[24, 12, 8\]. The lattice
//! consists of all integer vectors x in Z^24 such that:
//! - x mod 2 is a Golay codeword (coordinate-wise reduction to F_2)
//! - sum(x_i) is congruent to 0 (mod 4)
//! scaled by 1/sqrt(8) so that the minimum squared norm is 4.
//!
//! # Minimal Vectors (Shell 1)
//!
//! The 196,560 minimal vectors (squared norm 4 after scaling, i.e.,
//! squared norm 32 in integer coordinates) decompose into three types:
//!
//! | Type | Shape         | Count   | Construction |
//! |------|---------------|---------|--------------|
//! | 1    | (+-2)^8 0^16  | 97,152  | +-2 on Golay octad positions, even # of minus signs |
//! | 2    | (+-3)(+-1)^23 | 98,304  | one +-3 + 23 +-1, sign pattern from Golay codeword |
//! | 3    | (+-4)^2 0^22  | 1,104   | +-4 on two positions, zero elsewhere |
//!
//! Total: 97,152 + 98,304 + 1,104 = 196,560
//!
//! # EMC Phase Field Representation
//!
//! The default configuration uses the 1,104 type-3 minimal vectors plus the
//! origin (1,105 sites total). These vectors form the same connectivity graph
//! as the D_24 root system and are a correct subgraph of the full Leech lattice.
//! The Golay code is available for lattice-theoretic operations (quantization,
//! octad enumeration, minimal vector generation).

use super::{Lattice, LatticeCoord, LatticeLayer, Neighborhood};
use crate::golay::GolayCode;
use std::collections::HashMap;

/// The Leech lattice Lambda_24, built on the Extended Golay Code.
///
/// By default enumerates origin + type-3 minimal vectors (1,105 sites).
/// The Golay code is stored for correct lattice operations.
#[derive(Debug, Clone)]
pub struct LeechLattice {
    /// The Extended Golay Code underlying this lattice
    golay: GolayCode,
    /// Pre-computed site coordinates (integer coords, NOT scaled by 1/sqrt(8))
    sites: Vec<Vec<i64>>,
    /// Coordinate → site index lookup
    coord_to_idx: HashMap<Vec<i64>, usize>,
    /// Number of neighbors each site reports (matches enumerated shell subset)
    effective_kissing: usize,
}

impl LeechLattice {
    /// Create the Leech lattice with default configuration:
    /// origin + 1,104 type-3 minimal vectors.
    ///
    /// Type-3 vectors have shape (+-4)^2 0^22: two coordinates are +-4,
    /// the remaining 22 are zero. These correspond to the D_24 root system
    /// scaled by 4 and form a correct subgraph of the Leech lattice.
    ///
    /// Integer coordinate convention: type-3 vectors use +-1 (not +-4) for
    /// compact representation. The physical scale factor is handled by
    /// `coord_scale()` returning 4.0.
    pub fn new() -> Self {
        let golay = GolayCode::new();
        let mut sites = Vec::with_capacity(1105);
        let mut coord_to_idx = HashMap::with_capacity(1105);

        // Site 0 = origin
        let origin = vec![0i64; 24];
        coord_to_idx.insert(origin.clone(), 0);
        sites.push(origin);

        // Type-3 minimal vectors: (+-1, +-1, 0^22) in scaled coords
        // (physically +-4, +-4, 0^22 after coord_scale=4)
        // C(24,2) = 276 position pairs x 4 sign combos = 1,104 vectors
        for i in 0..24 {
            for j in (i + 1)..24 {
                for signs in 0..4u8 {
                    let mut root = vec![0i64; 24];
                    root[i] = if signs & 1 == 0 { 1 } else { -1 };
                    root[j] = if signs & 2 == 0 { 1 } else { -1 };
                    let idx = sites.len();
                    coord_to_idx.insert(root.clone(), idx);
                    sites.push(root);
                }
            }
        }

        Self {
            golay,
            sites,
            coord_to_idx,
            effective_kissing: 1104,
        }
    }

    /// Access the underlying Extended Golay Code.
    pub fn golay_code(&self) -> &GolayCode {
        &self.golay
    }

    /// The true kissing number of the Leech lattice.
    pub fn true_kissing_number(&self) -> usize {
        196_560
    }

    /// Generate all type-1 minimal vectors: (+-2)^8 0^16 on Golay octad positions.
    ///
    /// For each of the 759 octads, place +-2 on the 8 octad positions with an
    /// even number of minus signs: 759 * 2^7 = 97,152 vectors.
    ///
    /// Returns vectors in the same coordinate convention as the lattice sites
    /// (divide by coord_scale=4 for integer coords, so values are +-0.5).
    /// Actually returns raw integer coordinates where the 8 positions have +-2.
    pub fn type1_minimal_vectors(&self) -> Vec<Vec<i64>> {
        let octads = self.golay.octads();
        let mut vectors = Vec::with_capacity(97_152);

        for octad in &octads {
            // 2^7 = 128 sign patterns with even number of minus signs
            for sign_bits in 0u32..256 {
                if sign_bits.count_ones() % 2 != 0 {
                    continue; // skip odd number of minus signs
                }
                let mut v = vec![0i64; 24];
                for (bit_idx, &pos) in octad.iter().enumerate() {
                    v[pos] = if sign_bits & (1 << bit_idx) == 0 { 2 } else { -2 };
                }
                vectors.push(v);
            }
        }

        vectors
    }

    /// Generate all type-2 minimal vectors: (+-3)(+-1)^23.
    ///
    /// One coordinate is +-3, the other 23 are +-1, with |x|² = 9 + 23 = 32.
    ///
    /// # Construction (Conway & Sloane, SPLAG Chapter 4)
    ///
    /// The Leech lattice has two cosets in integer coordinates:
    /// - Even: all coords even, `(x/2) mod 2 ∈ C₂₄`, `Σxᵢ ≡ 0 (mod 8)`
    /// - Odd:  all coords odd, `((x-1)/2) mod 2 ∈ C₂₄`, `Σxᵢ ≡ 4 (mod 8)`
    ///
    /// Type-2 vectors are all-odd, so they use the odd coset. Writing
    /// `x = 2c + 4z + 1` where `c ∈ C₂₄` (as 0/1 vector) and `z ∈ Z²⁴`:
    ///
    /// - `|xᵢ| = 1`: `cᵢ=0, zᵢ=0 → xᵢ=+1` or `cᵢ=1, zᵢ=-1 → xᵢ=-1`
    /// - `|xᵢ| = 3`: `cᵢ=1, zᵢ=0 → xᵢ=+3` or `cᵢ=0, zᵢ=-1 → xᵢ=-3`
    ///
    /// So each (position j, codeword c) pair uniquely determines a vector:
    /// - `i ≠ j`: `cᵢ=0 → +1`, `cᵢ=1 → -1`
    /// - `i = j`: `cⱼ=1 → +3`, `cⱼ=0 → -3`
    ///
    /// The sum constraint `Σxᵢ ≡ 4 (mod 8)` is automatically satisfied because
    /// all Golay codeword weights are ≡ 0 (mod 4). No filtering needed.
    ///
    /// Count: 24 positions × 4096 codewords = 98,304 vectors.
    pub fn type2_minimal_vectors(&self) -> Vec<Vec<i64>> {
        let mut vectors = Vec::with_capacity(98_304);

        for pos3 in 0..24usize {
            for &cw in self.golay.codewords() {
                let mut v = vec![0i64; 24];
                for i in 0..24 {
                    let bit = (cw >> i) & 1;
                    if i == pos3 {
                        v[i] = if bit == 1 { 3 } else { -3 };
                    } else {
                        v[i] = if bit == 0 { 1 } else { -1 };
                    }
                }
                vectors.push(v);
            }
        }

        vectors
    }

    /// Generate the 1,104 type-3 minimal vectors: (+-4)^2 0^22.
    ///
    /// In the scaled coordinate convention used by the lattice (coord_scale=4),
    /// these are stored as (+-1)^2 0^22 internally. This method returns the
    /// UNSCALED integer coordinates (+-4 values) for mathematical use.
    pub fn type3_minimal_vectors(&self) -> Vec<Vec<i64>> {
        let mut vectors = Vec::with_capacity(1104);
        for i in 0..24 {
            for j in (i + 1)..24 {
                for signs in 0..4u8 {
                    let mut v = vec![0i64; 24];
                    v[i] = if signs & 1 == 0 { 4 } else { -4 };
                    v[j] = if signs & 2 == 0 { 4 } else { -4 };
                    vectors.push(v);
                }
            }
        }
        vectors
    }

    /// Quantize a point in R^24 to the nearest enumerated lattice site.
    ///
    /// This finds the nearest site among the 1,105 enumerated sites
    /// (origin + type-3 vectors in scaled coordinates). For the finite
    /// site set used by the phase field, this is the exact nearest site.
    fn quantize_to_nearest_site(&self, point: &[f64]) -> LatticeCoord {
        assert_eq!(point.len(), 24, "Leech lattice requires 24-dimensional input");

        let mut best_idx = 0;
        let mut best_dist_sq = f64::MAX;

        for (idx, site) in self.sites.iter().enumerate() {
            let dist_sq: f64 = point
                .iter()
                .zip(site.iter())
                .map(|(&p, &s)| (p - s as f64).powi(2))
                .sum();

            if dist_sq < best_dist_sq {
                best_dist_sq = dist_sq;
                best_idx = idx;
            }
        }

        LatticeCoord::new(self.sites[best_idx].clone())
    }
}

impl Default for LeechLattice {
    fn default() -> Self {
        Self::new()
    }
}

impl Lattice for LeechLattice {
    fn dimension(&self) -> usize {
        24
    }

    fn kissing_number(&self) -> usize {
        // Effective kissing number for the enumerated site subset (type-3 only).
        // The true Leech lattice kissing number is 196,560.
        self.effective_kissing
    }

    fn layer(&self) -> LatticeLayer {
        LatticeLayer::Creative
    }

    fn quantize(&self, point: &[f64]) -> LatticeCoord {
        self.quantize_to_nearest_site(point)
    }

    fn nearest_neighbors(&self, point: &LatticeCoord) -> Neighborhood {
        // Generate type-3 neighbors: add all (+-1, +-1, 0^22) displacements
        let mut neighbors = Vec::with_capacity(self.effective_kissing);

        for i in 0..24 {
            for j in (i + 1)..24 {
                for signs in 0..4u8 {
                    let mut neighbor = point.coords.clone();
                    neighbor[i] += if signs & 1 == 0 { 1 } else { -1 };
                    neighbor[j] += if signs & 2 == 0 { 1 } else { -1 };
                    neighbors.push(LatticeCoord::new(neighbor));
                }
            }
        }

        Neighborhood {
            center: point.clone(),
            neighbors,
        }
    }

    fn num_sites(&self) -> usize {
        self.sites.len()
    }

    fn site_to_coord(&self, idx: usize) -> LatticeCoord {
        assert!(idx < self.sites.len(), "Site index {} out of range (max {})", idx, self.sites.len() - 1);
        LatticeCoord::new(self.sites[idx].clone())
    }

    fn coord_to_site(&self, coord: &LatticeCoord) -> Option<usize> {
        self.coord_to_idx.get(&coord.coords).copied()
    }

    fn coord_scale(&self) -> f64 {
        // Stored as (+-1)^2 0^22 — same scale as the old D24 root system.
        // True Leech type-3 minimal vectors are (+-4)^2 0^22, but we keep
        // coord_scale=1.0 to preserve backward compatibility with existing
        // phase field dynamics (displacement vectors, Laplacian scaling, etc.).
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_leech_creation() {
        let lattice = LeechLattice::new();
        assert_eq!(lattice.dimension(), 24);
        assert_eq!(lattice.layer(), LatticeLayer::Creative);
        // Origin + 1104 type-3 minimal vectors
        assert_eq!(lattice.num_sites(), 1105);
    }

    #[test]
    fn test_effective_kissing_number() {
        let lattice = LeechLattice::new();
        assert_eq!(lattice.kissing_number(), 1104);
        assert_eq!(lattice.true_kissing_number(), 196_560);
    }

    #[test]
    fn test_golay_code_accessible() {
        let lattice = LeechLattice::new();
        let golay = lattice.golay_code();
        assert_eq!(golay.num_codewords(), 4096);
        assert_eq!(golay.num_octads(), 759);
    }

    #[test]
    fn test_type3_vector_count() {
        let lattice = LeechLattice::new();
        let type3 = lattice.type3_minimal_vectors();
        assert_eq!(type3.len(), 1104);
    }

    #[test]
    fn test_type3_vectors_correct_norm() {
        let lattice = LeechLattice::new();
        let type3 = lattice.type3_minimal_vectors();
        for v in &type3 {
            let norm_sq: i64 = v.iter().map(|&x| x * x).sum();
            assert_eq!(norm_sq, 32, "type-3 vector should have |x|^2 = 32 (two +-4 entries)");
        }
    }

    #[test]
    fn test_site_roundtrip() {
        let lattice = LeechLattice::new();
        for idx in (0..lattice.num_sites()).step_by(50) {
            let coord = lattice.site_to_coord(idx);
            let back = lattice.coord_to_site(&coord);
            assert_eq!(back, Some(idx), "Site {} failed roundtrip", idx);
        }
    }

    #[test]
    fn test_origin_quantization() {
        let lattice = LeechLattice::new();
        let origin = vec![0.0; 24];
        let quantized = lattice.quantize(&origin);
        assert_eq!(quantized.coords, vec![0i64; 24]);
    }

    #[test]
    fn test_nearest_neighbors_count() {
        let lattice = LeechLattice::new();
        let origin = LatticeCoord::origin(24);
        let hood = lattice.nearest_neighbors(&origin);
        assert_eq!(hood.neighbors.len(), 1104);
    }

    #[test]
    fn test_coord_scale() {
        let lattice = LeechLattice::new();
        assert_eq!(lattice.coord_scale(), 1.0);
    }

    #[test]
    fn test_quantize_near_site() {
        let lattice = LeechLattice::new();
        // A point near site (1, 1, 0, ..., 0) should quantize there
        let mut point = vec![0.0; 24];
        point[0] = 0.9;
        point[1] = 1.1;
        let quantized = lattice.quantize(&point);
        assert_eq!(quantized.coords[0], 1);
        assert_eq!(quantized.coords[1], 1);
        for i in 2..24 {
            assert_eq!(quantized.coords[i], 0);
        }
    }

    #[test]
    fn test_all_sites_unique() {
        let lattice = LeechLattice::new();
        let mut seen = std::collections::HashSet::new();
        for i in 0..lattice.num_sites() {
            let coord = lattice.site_to_coord(i);
            assert!(seen.insert(coord.coords), "Duplicate site at index {}", i);
        }
    }

    #[test]
    fn test_type1_vector_count() {
        let lattice = LeechLattice::new();
        let type1 = lattice.type1_minimal_vectors();
        assert_eq!(type1.len(), 97_152, "should have 759 octads * 2^7 = 97,152 type-1 vectors");
    }

    #[test]
    fn test_type1_vectors_correct_norm() {
        let lattice = LeechLattice::new();
        let type1 = lattice.type1_minimal_vectors();
        for (i, v) in type1.iter().enumerate().step_by(100) {
            let norm_sq: i64 = v.iter().map(|&x| x * x).sum();
            assert_eq!(norm_sq, 32, "type-1 vector {} should have |x|^2 = 32 (eight +-2 entries)", i);
        }
    }

    #[test]
    fn test_type1_vectors_on_octad_positions() {
        let lattice = LeechLattice::new();
        let type1 = lattice.type1_minimal_vectors();
        // Each type-1 vector should have exactly 8 nonzero entries, each +-2
        for (i, v) in type1.iter().enumerate().step_by(100) {
            let nonzero_count = v.iter().filter(|&&x| x != 0).count();
            assert_eq!(nonzero_count, 8, "type-1 vector {} has {} nonzero entries", i, nonzero_count);
            for &x in v {
                assert!(x == 0 || x == 2 || x == -2, "type-1 vector has invalid value {}", x);
            }
        }
    }

    // ─── Type-2 vectors ───

    #[test]
    fn test_type2_vector_count() {
        let lattice = LeechLattice::new();
        let type2 = lattice.type2_minimal_vectors();
        assert_eq!(type2.len(), 98_304, "should have 24 * 4096 = 98,304 type-2 vectors");
    }

    #[test]
    fn test_type2_vectors_correct_norm() {
        let lattice = LeechLattice::new();
        let type2 = lattice.type2_minimal_vectors();
        for (i, v) in type2.iter().enumerate().step_by(100) {
            let norm_sq: i64 = v.iter().map(|&x| x * x).sum();
            assert_eq!(norm_sq, 32, "type-2 vector {} should have |x|^2 = 32", i);
        }
    }

    #[test]
    fn test_type2_all_coords_odd() {
        let lattice = LeechLattice::new();
        let type2 = lattice.type2_minimal_vectors();
        for (i, v) in type2.iter().enumerate().step_by(100) {
            for (j, &x) in v.iter().enumerate() {
                assert!(x % 2 != 0, "type-2 vector {} coord {} = {} should be odd", i, j, x);
            }
        }
    }

    #[test]
    fn test_type2_one_three_and_rest_ones() {
        let lattice = LeechLattice::new();
        let type2 = lattice.type2_minimal_vectors();
        // Each type-2 vector should have exactly one +-3 entry and 23 +-1 entries
        for (i, v) in type2.iter().enumerate().step_by(100) {
            let threes = v.iter().filter(|&&x| x == 3 || x == -3).count();
            let ones = v.iter().filter(|&&x| x == 1 || x == -1).count();
            assert_eq!(threes, 1, "type-2 vector {} should have exactly 1 entry of +-3", i);
            assert_eq!(ones, 23, "type-2 vector {} should have exactly 23 entries of +-1", i);
        }
    }

    // ─── Total kissing number ───

    #[test]
    fn test_total_kissing_number() {
        let lattice = LeechLattice::new();
        let type1 = lattice.type1_minimal_vectors();
        let type2 = lattice.type2_minimal_vectors();
        let type3 = lattice.type3_minimal_vectors();
        let total = type1.len() + type2.len() + type3.len();
        assert_eq!(total, 196_560, "Leech lattice kissing number: 97,152 + 98,304 + 1,104 = 196,560");
    }
}
