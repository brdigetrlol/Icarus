// Copyright (c) 2025-2026 brdigetrlol. All rights reserved.
// SPDX-License-Identifier: LicenseRef-Icarus-Proprietary
// See LICENSE in the repository root for full license terms.

//! HCP (Hexagonal Close-Packed) Lattice — implemented as A_n lattice
//!
//! Construction: A_n = {(x₁, ..., x_{n+1}) ∈ Z^{n+1} | Σxᵢ = 0}
//! Projected to n dimensions by dropping the last coordinate.
//!
//! Used for 64D associative/intuitive processing (System 1) in Icarus EMC.
//! Kissing number for A_n: n(n+1) (root system of type A)

use super::{Lattice, LatticeCoord, LatticeLayer, Neighborhood};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct HCPLattice {
    dimension: usize,
    /// Pre-computed site coordinates (origin + neighbors)
    sites: Vec<Vec<i64>>,
    coord_to_idx: HashMap<Vec<i64>, usize>,
}

impl HCPLattice {
    /// Create an HCP lattice with origin + first shell of nearest neighbors
    pub fn new(dimension: usize) -> Self {
        assert!(dimension > 0, "Dimension must be positive");

        let mut sites = Vec::new();
        let mut coord_to_idx = HashMap::new();

        // Site 0 = origin
        let origin = vec![0i64; dimension];
        coord_to_idx.insert(origin.clone(), 0);
        sites.push(origin);

        // First shell: root vectors of A_n are e_i - e_j for i ≠ j in n dimensions
        for i in 0..dimension {
            for j in 0..dimension {
                if i != j {
                    let mut root = vec![0i64; dimension];
                    root[i] = 1;
                    root[j] = -1;
                    let idx = sites.len();
                    coord_to_idx.insert(root.clone(), idx);
                    sites.push(root);
                }
            }
        }

        Self {
            dimension,
            sites,
            coord_to_idx,
        }
    }

    /// Quantize a point to the A_n lattice
    fn quantize_a_n(point: &[f64]) -> Vec<i64> {
        let n = point.len();

        // Extend to (n+1) dimensions with last coord making sum zero
        let sum: f64 = point.iter().sum();
        let mut extended = point.to_vec();
        extended.push(-sum);

        // Round to integers
        let mut quantized: Vec<i64> = extended.iter().map(|&x| x.round() as i64).collect();

        // Enforce sum = 0 constraint (A_n defining property)
        let actual_sum: i64 = quantized.iter().sum();
        if actual_sum != 0 {
            let mut errors: Vec<(usize, f64)> = extended
                .iter()
                .zip(quantized.iter())
                .enumerate()
                .map(|(i, (&orig, &quant))| (i, (orig - quant as f64).abs()))
                .collect();

            errors.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let correction_needed = actual_sum.unsigned_abs() as usize;
            for item in errors.iter().take(correction_needed.min(errors.len())) {
                quantized[item.0] += if actual_sum > 0 { -1 } else { 1 };
            }
        }

        // Project back to n dimensions (drop last coordinate)
        quantized[..n].to_vec()
    }
}

impl Default for HCPLattice {
    fn default() -> Self {
        Self::new(64)
    }
}

impl Lattice for HCPLattice {
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn kissing_number(&self) -> usize {
        // A_n root system has n(n+1) roots → kissing number
        // But for projected lattice, it's n(n-1) neighbors in n dims
        self.dimension * (self.dimension - 1)
    }

    fn layer(&self) -> LatticeLayer {
        LatticeLayer::Associative
    }

    fn quantize(&self, point: &[f64]) -> LatticeCoord {
        assert_eq!(
            point.len(),
            self.dimension,
            "Point dimension must match lattice dimension"
        );
        LatticeCoord::new(Self::quantize_a_n(point))
    }

    fn nearest_neighbors(&self, point: &LatticeCoord) -> Neighborhood {
        let n = self.dimension;
        let mut neighbors = Vec::with_capacity(n * (n - 1));

        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let mut neighbor = point.coords.clone();
                    neighbor[i] += 1;
                    neighbor[j] -= 1;
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
        assert!(idx < self.sites.len(), "Site index out of range");
        LatticeCoord::new(self.sites[idx].clone())
    }

    fn coord_to_site(&self, coord: &LatticeCoord) -> Option<usize> {
        self.coord_to_idx.get(&coord.coords).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hcp_creation() {
        let lattice = HCPLattice::new(64);
        assert_eq!(lattice.dimension(), 64);
        assert_eq!(lattice.layer(), LatticeLayer::Associative);
    }

    #[test]
    fn test_kissing_number() {
        let lattice = HCPLattice::new(8);
        // A_8: 8*7 = 56 nearest neighbors
        assert_eq!(lattice.kissing_number(), 56);
    }

    #[test]
    fn test_num_sites() {
        let lattice = HCPLattice::new(8);
        // origin + 56 neighbors = 57
        assert_eq!(lattice.num_sites(), 57);
    }

    #[test]
    fn test_site_roundtrip() {
        let lattice = HCPLattice::new(8);
        for idx in 0..lattice.num_sites() {
            let coord = lattice.site_to_coord(idx);
            let back = lattice.coord_to_site(&coord);
            assert_eq!(back, Some(idx), "Site {} failed roundtrip", idx);
        }
    }

    #[test]
    fn test_origin_quantization() {
        let lattice = HCPLattice::new(8);
        let origin = vec![0.0; 8];
        let quantized = lattice.quantize(&origin);
        assert_eq!(quantized.coords, vec![0i64; 8]);
    }

    #[test]
    fn test_nearest_neighbors_count() {
        let lattice = HCPLattice::new(8);
        let origin = LatticeCoord::origin(8);
        let hood = lattice.nearest_neighbors(&origin);
        assert_eq!(hood.neighbors.len(), 56);
    }

    #[test]
    fn test_quantization_3d() {
        let lattice = HCPLattice::new(3);
        let point = vec![1.2, 2.3, 3.4];
        let quantized = lattice.quantize(&point);
        assert_eq!(quantized.coords.len(), 3);
    }

    #[test]
    fn test_distance() {
        let lattice = HCPLattice::new(3);
        let a = LatticeCoord::new(vec![1, 0, -1]);
        let b = LatticeCoord::new(vec![0, 1, -1]);
        let dist = lattice.distance(&a, &b);
        assert!((dist - 2.0_f64.sqrt()).abs() < 1e-10);
    }
}
