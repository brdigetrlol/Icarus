//! Lattice module for the Icarus EMC architecture
//!
//! Multi-resolution crystallographic lattice hierarchy:
//! - E8: 8D, analytical/logical core (System 2), kissing=240
//! - Leech: 24D, creative/analogical reasoning (System 1.5), kissing=196560
//! - HCP: 64D, associative/intuitive processing (System 1)
//! - Hypercubic: 1024D, sensory manifold buffer

pub mod e8;
pub mod leech;
pub mod hcp;
pub mod hypercubic;

pub use e8::E8Lattice;
pub use leech::LeechLattice;
pub use hcp::HCPLattice;
pub use hypercubic::HypercubicLattice;

use serde::{Deserialize, Serialize};

/// Identifies which layer of the multi-resolution hierarchy a lattice belongs to
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LatticeLayer {
    /// E8: 8D analytical core (System 2)
    Analytical,
    /// Leech 24D: creative/analogical (System 1.5)
    Creative,
    /// HCP 64D: associative/intuitive (System 1)
    Associative,
    /// Hypercubic 1024D: sensory manifold buffer
    Sensory,
}

/// Integer lattice coordinates
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LatticeCoord {
    pub coords: Vec<i64>,
}

impl LatticeCoord {
    pub fn new(coords: Vec<i64>) -> Self {
        Self { coords }
    }

    pub fn origin(dim: usize) -> Self {
        Self {
            coords: vec![0; dim],
        }
    }

    pub fn dimension(&self) -> usize {
        self.coords.len()
    }
}

/// A neighborhood of lattice points (nearest neighbors of a site)
#[derive(Debug, Clone)]
pub struct Neighborhood {
    pub center: LatticeCoord,
    pub neighbors: Vec<LatticeCoord>,
}

/// Core trait for all lattice implementations in the EMC hierarchy
pub trait Lattice: Send + Sync {
    /// Dimension of the underlying vector space
    fn dimension(&self) -> usize;

    /// Number of nearest neighbors (kissing number)
    fn kissing_number(&self) -> usize;

    /// Which layer of the hierarchy this lattice belongs to
    fn layer(&self) -> LatticeLayer;

    /// Quantize a continuous point to the nearest lattice coordinate
    fn quantize(&self, point: &[f64]) -> LatticeCoord;

    /// Squared distance between two lattice coordinates
    fn distance_sq(&self, a: &LatticeCoord, b: &LatticeCoord) -> f64 {
        assert_eq!(a.dimension(), b.dimension());
        a.coords
            .iter()
            .zip(b.coords.iter())
            .map(|(x, y)| ((*x - *y) as f64).powi(2))
            .sum()
    }

    /// Euclidean distance between two lattice coordinates
    fn distance(&self, a: &LatticeCoord, b: &LatticeCoord) -> f64 {
        self.distance_sq(a, b).sqrt()
    }

    /// Get all nearest neighbors of a lattice point
    fn nearest_neighbors(&self, point: &LatticeCoord) -> Neighborhood;

    /// Total number of active sites in this lattice instance.
    /// For bounded lattices (e.g. E8 with origin + 240 neighbors), returns the
    /// number of allocated sites. For unbounded usage, returns the configured size.
    fn num_sites(&self) -> usize;

    /// Map a site index (0..num_sites) to its lattice coordinate
    fn site_to_coord(&self, idx: usize) -> LatticeCoord;

    /// Map a lattice coordinate back to its site index, if it exists
    fn coord_to_site(&self, coord: &LatticeCoord) -> Option<usize>;

    /// Coordinate scale factor for converting integer coordinates to physical space.
    ///
    /// Some lattices use scaled integer coordinates to represent non-integer points.
    /// For example, E8 uses 2× coordinates to represent half-integer vectors as integers.
    /// Physical displacement = integer_displacement × coord_scale().
    /// Default: 1.0 (integer coordinates = physical coordinates).
    fn coord_scale(&self) -> f64 {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lattice_coord_origin() {
        let origin = LatticeCoord::origin(8);
        assert_eq!(origin.dimension(), 8);
        assert!(origin.coords.iter().all(|&x| x == 0));
    }
}
