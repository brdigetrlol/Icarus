//! Hypercubic Lattice (Z^n) Implementation
//!
//! The simplest lattice: integer coordinates in n dimensions.
//! Used for 1024D sensory manifold buffer in Icarus EMC.
//!
//! Properties:
//! - Kissing number: 2n
//! - Orthogonal axes, high bandwidth
//! - Fast quantization (just rounding)

use super::{Lattice, LatticeCoord, LatticeLayer, Neighborhood};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct HypercubicLattice {
    dimension: usize,
    /// Pre-computed sites: origin + 2n nearest neighbors
    sites: Vec<Vec<i64>>,
    coord_to_idx: HashMap<Vec<i64>, usize>,
}

impl HypercubicLattice {
    pub fn new(dimension: usize) -> Self {
        assert!(dimension > 0, "Dimension must be positive");

        let mut sites = Vec::with_capacity(1 + 2 * dimension);
        let mut coord_to_idx = HashMap::with_capacity(1 + 2 * dimension);

        // Site 0 = origin
        let origin = vec![0i64; dimension];
        coord_to_idx.insert(origin.clone(), 0);
        sites.push(origin);

        // Sites 1..=2n: ±e_i unit vectors
        for i in 0..dimension {
            let mut plus = vec![0i64; dimension];
            plus[i] = 1;
            coord_to_idx.insert(plus.clone(), sites.len());
            sites.push(plus);

            let mut minus = vec![0i64; dimension];
            minus[i] = -1;
            coord_to_idx.insert(minus.clone(), sites.len());
            sites.push(minus);
        }

        Self {
            dimension,
            sites,
            coord_to_idx,
        }
    }
}

impl Default for HypercubicLattice {
    fn default() -> Self {
        Self::new(1024)
    }
}

impl Lattice for HypercubicLattice {
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn kissing_number(&self) -> usize {
        2 * self.dimension
    }

    fn layer(&self) -> LatticeLayer {
        LatticeLayer::Sensory
    }

    fn quantize(&self, point: &[f64]) -> LatticeCoord {
        assert_eq!(
            point.len(),
            self.dimension,
            "Point dimension must match lattice dimension"
        );
        LatticeCoord::new(point.iter().map(|&x| x.round() as i64).collect())
    }

    fn nearest_neighbors(&self, point: &LatticeCoord) -> Neighborhood {
        let mut neighbors = Vec::with_capacity(2 * self.dimension);

        for i in 0..self.dimension {
            let mut plus = point.coords.clone();
            plus[i] += 1;
            neighbors.push(LatticeCoord::new(plus));

            let mut minus = point.coords.clone();
            minus[i] -= 1;
            neighbors.push(LatticeCoord::new(minus));
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
    fn test_hypercubic_creation() {
        let lattice = HypercubicLattice::new(1024);
        assert_eq!(lattice.dimension(), 1024);
        assert_eq!(lattice.kissing_number(), 2048);
        assert_eq!(lattice.num_sites(), 2049); // 1 + 2*1024
        assert_eq!(lattice.layer(), LatticeLayer::Sensory);
    }

    #[test]
    fn test_quantization() {
        let lattice = HypercubicLattice::new(8);
        let point = vec![1.4, 2.6, 3.1, 4.9, 5.5, 6.2, 7.8, 8.3];
        let quantized = lattice.quantize(&point);
        assert_eq!(quantized.coords, vec![1, 3, 3, 5, 6, 6, 8, 8]);
    }

    #[test]
    fn test_origin_quantization() {
        let lattice = HypercubicLattice::new(8);
        let origin = vec![0.0; 8];
        let quantized = lattice.quantize(&origin);
        assert_eq!(quantized.coords, vec![0i64; 8]);
    }

    #[test]
    fn test_nearest_neighbors() {
        let lattice = HypercubicLattice::new(8);
        let origin = LatticeCoord::origin(8);
        let hood = lattice.nearest_neighbors(&origin);
        assert_eq!(hood.neighbors.len(), 16);

        for neighbor in &hood.neighbors {
            let diff: i64 = neighbor
                .coords
                .iter()
                .zip(origin.coords.iter())
                .map(|(a, b)| (a - b).abs())
                .sum();
            assert_eq!(diff, 1);
        }
    }

    #[test]
    fn test_site_roundtrip() {
        let lattice = HypercubicLattice::new(8);
        for idx in 0..lattice.num_sites() {
            let coord = lattice.site_to_coord(idx);
            let back = lattice.coord_to_site(&coord);
            assert_eq!(back, Some(idx), "Site {} failed roundtrip", idx);
        }
    }

    #[test]
    fn test_distance() {
        let lattice = HypercubicLattice::new(3);
        let a = LatticeCoord::new(vec![0, 0, 0]);
        let b = LatticeCoord::new(vec![3, 4, 0]);
        let dist = lattice.distance(&a, &b);
        assert!((dist - 5.0).abs() < 1e-10);
    }
}
