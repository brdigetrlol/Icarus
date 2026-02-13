// Copyright (c) 2025-2026 brdigetrlol. All rights reserved.
// SPDX-License-Identifier: LicenseRef-Icarus-Proprietary
// See LICENSE in the repository root for full license terms.

//! E8 Lattice Implementation
//!
//! The E8 lattice is the densest sphere packing in 8 dimensions (Viazovska, 2016).
//! Kissing number: 240
//! Used for analytical/logical reasoning (System 2) in Icarus EMC.
//!
//! Construction: E8 = D8 ∪ (D8 + (1/2, 1/2, ..., 1/2))
//! Where D8 = {(x₁, ..., x₈) ∈ Z⁸ | Σxᵢ ∈ 2Z}
//!
//! For Icarus MVP, the E8 lattice uses 241 sites:
//! site 0 = origin, sites 1..=240 = the 240 nearest neighbors (root vectors).

use super::{Lattice, LatticeCoord, LatticeLayer, Neighborhood};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct E8Lattice {
    /// Pre-computed root vectors (the 240 nearest neighbors of the origin)
    roots: Vec<Vec<i64>>,
    /// Map from coordinate tuple to site index for O(1) lookup
    coord_to_idx: HashMap<Vec<i64>, usize>,
    /// Number of active sites (origin + shells)
    num_active_sites: usize,
}

impl E8Lattice {
    /// Create an E8 lattice with the first shell (origin + 240 nearest neighbors = 241 sites)
    pub fn new() -> Self {
        let roots = Self::generate_root_vectors();
        let mut coord_to_idx = HashMap::with_capacity(241);

        // Site 0 = origin
        coord_to_idx.insert(vec![0i64; 8], 0);

        // Sites 1..=240 = root vectors
        for (i, root) in roots.iter().enumerate() {
            coord_to_idx.insert(root.clone(), i + 1);
        }

        Self {
            roots,
            coord_to_idx,
            num_active_sites: 241,
        }
    }

    /// Create an E8 lattice with multiple shells
    /// Shell 0 = origin (1 point)
    /// Shell 1 = 240 nearest neighbors (norm² = 2)
    /// Shell 2 = next shell at norm² = 4 (2160 points)
    pub fn with_shells(num_shells: usize) -> Self {
        let roots = Self::generate_root_vectors();
        let mut all_sites: Vec<Vec<i64>> = Vec::new();
        let mut coord_to_idx = HashMap::new();

        // Shell 0: origin
        all_sites.push(vec![0i64; 8]);
        coord_to_idx.insert(vec![0i64; 8], 0);

        if num_shells >= 1 {
            // Shell 1: root vectors (norm² = 2)
            for root in &roots {
                let idx = all_sites.len();
                all_sites.push(root.clone());
                coord_to_idx.insert(root.clone(), idx);
            }
        }

        // Additional shells would be computed here for num_shells >= 2
        // Shell 2 has 2160 vectors at norm² = 4, etc.

        let num_active_sites = all_sites.len();

        Self {
            roots,
            coord_to_idx,
            num_active_sites,
        }
    }

    /// Round a point to the nearest D8 lattice point
    fn round_to_d8(point: &[f64]) -> Vec<i64> {
        assert_eq!(point.len(), 8, "E8 requires 8-dimensional input");

        let mut rounded: Vec<i64> = point.iter().map(|&x| x.round() as i64).collect();

        // D8 constraint: Σxᵢ ∈ 2Z (sum must be even)
        let sum: i64 = rounded.iter().sum();
        if sum % 2 != 0 {
            // Find coordinate with largest rounding error and adjust
            let mut max_error = 0.0f64;
            let mut flip_idx = 0;

            for i in 0..8 {
                let error = (point[i] - rounded[i] as f64).abs();
                if error > max_error {
                    max_error = error;
                    flip_idx = i;
                }
            }

            rounded[flip_idx] += if point[flip_idx] > rounded[flip_idx] as f64 {
                1
            } else {
                -1
            };
        }

        rounded
    }

    /// Round a point to the D8 + (1/2,...,1/2) coset
    fn round_to_d8_coset(point: &[f64]) -> Vec<f64> {
        let shifted: Vec<f64> = point.iter().map(|&x| x - 0.5).collect();
        let d8_point = Self::round_to_d8(&shifted);
        d8_point.iter().map(|&x| x as f64 + 0.5).collect()
    }

    fn distance_sq_f64(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y).powi(2))
            .sum()
    }

    /// Generate all 240 root vectors of E8 in **doubled coordinates**.
    ///
    /// In standard E8 coordinates:
    ///   Type 1: 112 vectors (±1, ±1, 0, 0, 0, 0, 0, 0), norm² = 2
    ///   Type 2: 128 vectors (±½, ±½, ..., ±½) with even parity, norm² = 2
    ///
    /// Since LatticeCoord uses i64, we multiply all coordinates by 2 to keep integers:
    ///   Type 1: (±2, ±2, 0, 0, 0, 0, 0, 0), norm²_doubled = 8
    ///   Type 2: (±1, ±1, ..., ±1), norm²_doubled = 8
    ///
    /// Physical distances use coord_scale() = 0.5 to recover the true norm √2.
    fn generate_root_vectors() -> Vec<Vec<i64>> {
        let mut roots = Vec::with_capacity(240);

        // Type 1: C(8,2)=28 position pairs × 4 sign combos = 112
        // Doubled coordinates: (±2, ±2, 0, 0, 0, 0, 0, 0)
        for i in 0..8 {
            for j in (i + 1)..8 {
                for signs in 0..4u8 {
                    let mut root = vec![0i64; 8];
                    root[i] = if signs & 1 == 0 { 2 } else { -2 };
                    root[j] = if signs & 2 == 0 { 2 } else { -2 };
                    roots.push(root);
                }
            }
        }

        // Type 2: (±½)^8 with even number of minus signs = 128
        // Doubled coordinates: (±1)^8
        for pattern in 0..256u16 {
            if pattern.count_ones() % 2 == 0 {
                let root: Vec<i64> = (0..8)
                    .map(|i| if pattern & (1 << i) == 0 { 1 } else { -1 })
                    .collect();
                roots.push(root);
            }
        }

        debug_assert_eq!(roots.len(), 240, "E8 must have exactly 240 roots");
        roots
    }

    /// Get the pre-computed root vectors
    pub fn root_vectors(&self) -> &[Vec<i64>] {
        &self.roots
    }
}

impl Default for E8Lattice {
    fn default() -> Self {
        Self::new()
    }
}

impl Lattice for E8Lattice {
    fn dimension(&self) -> usize {
        8
    }

    fn kissing_number(&self) -> usize {
        240
    }

    fn layer(&self) -> LatticeLayer {
        LatticeLayer::Analytical
    }

    fn quantize(&self, point: &[f64]) -> LatticeCoord {
        assert_eq!(point.len(), 8, "E8 requires 8-dimensional input");

        // Candidate 1: nearest D8 point (standard integer coords)
        let d8_candidate = Self::round_to_d8(point);
        let d8_dist = Self::distance_sq_f64(
            point,
            &d8_candidate.iter().map(|&x| x as f64).collect::<Vec<_>>(),
        );

        // Candidate 2: nearest D8 + (1/2,...,1/2) coset point (standard half-integer coords)
        let coset_candidate = Self::round_to_d8_coset(point);
        let coset_dist = Self::distance_sq_f64(point, &coset_candidate);

        if d8_dist <= coset_dist {
            // D8 point: double the coordinates for our 2× representation
            LatticeCoord::new(d8_candidate.iter().map(|&x| x * 2).collect())
        } else {
            // Coset point: (half-integer × 2) gives odd integers — already doubled
            LatticeCoord::new(
                coset_candidate
                    .iter()
                    .map(|&x| (x * 2.0).round() as i64)
                    .collect(),
            )
        }
    }

    fn distance_sq(&self, a: &LatticeCoord, b: &LatticeCoord) -> f64 {
        assert_eq!(a.dimension(), b.dimension());
        let raw: f64 = a
            .coords
            .iter()
            .zip(b.coords.iter())
            .map(|(x, y)| ((*x - *y) as f64).powi(2))
            .sum();
        // Coordinates are 2×, so physical distance² = raw × (0.5)² = raw × 0.25
        raw * 0.25
    }

    fn coord_scale(&self) -> f64 {
        0.5
    }

    fn nearest_neighbors(&self, point: &LatticeCoord) -> Neighborhood {
        let neighbors = self
            .roots
            .iter()
            .map(|root| {
                LatticeCoord::new(
                    point
                        .coords
                        .iter()
                        .zip(root.iter())
                        .map(|(p, r)| p + r)
                        .collect(),
                )
            })
            .collect();

        Neighborhood {
            center: point.clone(),
            neighbors,
        }
    }

    fn num_sites(&self) -> usize {
        self.num_active_sites
    }

    fn site_to_coord(&self, idx: usize) -> LatticeCoord {
        assert!(idx < self.num_active_sites, "Site index out of range");
        if idx == 0 {
            LatticeCoord::origin(8)
        } else {
            LatticeCoord::new(self.roots[idx - 1].clone())
        }
    }

    fn coord_to_site(&self, coord: &LatticeCoord) -> Option<usize> {
        self.coord_to_idx.get(&coord.coords).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_e8_creation() {
        let lattice = E8Lattice::new();
        assert_eq!(lattice.dimension(), 8);
        assert_eq!(lattice.kissing_number(), 240);
        assert_eq!(lattice.num_sites(), 241);
        assert_eq!(lattice.layer(), LatticeLayer::Analytical);
    }

    #[test]
    fn test_root_vector_count() {
        let lattice = E8Lattice::new();
        assert_eq!(lattice.root_vectors().len(), 240);
    }

    #[test]
    fn test_root_vector_norms() {
        let lattice = E8Lattice::new();
        for root in lattice.root_vectors() {
            let norm_sq: i64 = root.iter().map(|x| x * x).sum();
            // All roots have norm² = 8 in doubled coordinates
            // (physical norm² = 8 × 0.25 = 2)
            assert_eq!(
                norm_sq, 8,
                "Root norm² in doubled coords should be 8, got {}",
                norm_sq
            );
        }
    }

    #[test]
    fn test_root_type_counts() {
        let lattice = E8Lattice::new();
        // Type 1: exactly 2 nonzero coords with |value| = 2 (doubled from ±1)
        let type1_count = lattice
            .root_vectors()
            .iter()
            .filter(|r| {
                r.iter().filter(|&&x| x != 0).count() == 2
                    && r.iter().all(|&x| x.abs() == 2 || x == 0)
            })
            .count();
        // Type 2: all 8 coords have |value| = 1 (doubled from ±½)
        let type2_count = lattice
            .root_vectors()
            .iter()
            .filter(|r| r.iter().all(|&x| x.abs() == 1))
            .count();
        assert_eq!(type1_count, 112);
        assert_eq!(type2_count, 128);
        assert_eq!(type1_count + type2_count, 240);
    }

    #[test]
    fn test_site_coord_roundtrip() {
        let lattice = E8Lattice::new();
        for idx in 0..lattice.num_sites() {
            let coord = lattice.site_to_coord(idx);
            let back = lattice.coord_to_site(&coord);
            assert_eq!(back, Some(idx), "Site {} failed roundtrip", idx);
        }
    }

    #[test]
    fn test_origin_quantization() {
        let lattice = E8Lattice::new();
        let origin = vec![0.0; 8];
        let quantized = lattice.quantize(&origin);
        assert_eq!(quantized.coords, vec![0i64; 8]);
    }

    #[test]
    fn test_d8_quantization_doubled_coords() {
        let lattice = E8Lattice::new();
        let point = vec![1.1, 2.0, 3.1, 4.0, 5.1, 6.0, 7.1, 8.0];
        let quantized = lattice.quantize(&point);
        // In doubled coordinates: D8 points have all-even coords, coset points all-odd
        let all_even = quantized.coords.iter().all(|&x| x % 2 == 0);
        let all_odd = quantized.coords.iter().all(|&x| x.abs() % 2 == 1);
        assert!(
            all_even || all_odd,
            "E8 doubled coords must be all-even (D8) or all-odd (coset), got {:?}",
            quantized.coords
        );
    }

    #[test]
    fn test_nearest_neighbors_count() {
        let lattice = E8Lattice::new();
        let origin = LatticeCoord::origin(8);
        let hood = lattice.nearest_neighbors(&origin);
        assert_eq!(hood.neighbors.len(), 240);
    }

    #[test]
    fn test_geometric_product_scalars() {
        let lattice = E8Lattice::new();
        let a = LatticeCoord::new(vec![0; 8]);
        // Type 1 root vector in doubled coordinates
        let b = LatticeCoord::new(vec![2, 2, 0, 0, 0, 0, 0, 0]);
        let dist = lattice.distance(&a, &b);
        // Physical distance = sqrt(8) * 0.5 = sqrt(2)
        assert!((dist - 2.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_performance_quantization() {
        use std::time::Instant;

        let lattice = E8Lattice::new();
        let test_points: Vec<Vec<f64>> = (0..10_000)
            .map(|i| {
                (0..8)
                    .map(|j| (i as f64) * 0.1 + (j as f64) * 0.1)
                    .collect()
            })
            .collect();

        let start = Instant::now();
        for point in &test_points {
            let _ = lattice.quantize(point);
        }
        let elapsed = start.elapsed();

        let ops_per_sec = (test_points.len() as f64) / elapsed.as_secs_f64();
        assert!(
            ops_per_sec > 100_000.0,
            "E8 quantization perf: {:.0} ops/sec (target >100K)",
            ops_per_sec
        );
    }
}
