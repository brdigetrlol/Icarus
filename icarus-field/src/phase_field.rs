//! Lattice Phase Field with CSR Neighbor Topology
//!
//! The LatticeField stores a complex phase field z(x) = A·e^{iθ} at every lattice site,
//! using Structure-of-Arrays (SoA) layout for GPU-friendly access.
//!
//! Neighbor topology is stored in Compressed Sparse Row (CSR) format for efficient
//! graph Laplacian computation.

use icarus_math::lattice::{Lattice, LatticeLayer};

/// A complex phase field defined on a lattice with CSR neighbor topology.
///
/// The CSR format stores neighbor relationships:
/// - `neighbor_offsets[i]..neighbor_offsets[i+1]` gives the range of neighbors for site `i`
/// - `neighbor_indices[offset]` gives the site index of each neighbor
/// - `neighbor_weights[offset]` gives the metric-weighted connection strength
#[derive(Debug, Clone)]
pub struct LatticeField {
    /// Real parts of the complex field (SoA layout)
    pub values_re: Vec<f32>,
    /// Imaginary parts of the complex field (SoA layout)
    pub values_im: Vec<f32>,
    /// Which layer this field lives on
    pub layer: LatticeLayer,
    /// Number of lattice sites
    pub num_sites: usize,
    /// CSR column indices: neighbor site indices
    pub neighbor_indices: Vec<u32>,
    /// CSR row offsets: neighbor_offsets[i]..neighbor_offsets[i+1] are neighbors of site i
    pub neighbor_offsets: Vec<u32>,
    /// Per-edge weights (metric-dependent), same length as neighbor_indices
    pub neighbor_weights: Vec<f32>,
    /// Lattice dimension
    pub dim: usize,
    /// Displacement vectors from site i to its neighbor j (dim floats per edge)
    /// Layout: displacement_vectors[edge_idx * dim + d] for dimension d
    pub displacement_vectors: Vec<f32>,
}

impl LatticeField {
    /// Build a LatticeField from any lattice implementation.
    /// Constructs the CSR neighbor topology and initializes the field to zero.
    pub fn from_lattice(lattice: &dyn Lattice) -> Self {
        let num_sites = lattice.num_sites();
        let dim = lattice.dimension();
        let layer = lattice.layer();
        let scale = lattice.coord_scale() as f32;

        let mut neighbor_indices = Vec::new();
        let mut neighbor_offsets = vec![0u32];
        let mut displacement_vectors = Vec::new();

        for site_idx in 0..num_sites {
            let coord = lattice.site_to_coord(site_idx);
            let neighborhood = lattice.nearest_neighbors(&coord);

            for neighbor_coord in &neighborhood.neighbors {
                if let Some(neighbor_idx) = lattice.coord_to_site(neighbor_coord) {
                    neighbor_indices.push(neighbor_idx as u32);

                    // Compute displacement vector: neighbor - center, scaled to physical coords
                    for d in 0..dim {
                        let disp = (neighbor_coord.coords[d] - coord.coords[d]) as f32 * scale;
                        displacement_vectors.push(disp);
                    }
                }
            }
            neighbor_offsets.push(neighbor_indices.len() as u32);
        }

        // Initialize weights to 1.0 (flat metric)
        let num_edges = neighbor_indices.len();
        let neighbor_weights = vec![1.0f32; num_edges];

        Self {
            values_re: vec![0.0; num_sites],
            values_im: vec![0.0; num_sites],
            layer,
            num_sites,
            neighbor_indices,
            neighbor_offsets,
            neighbor_weights,
            dim,
            displacement_vectors,
        }
    }

    /// Number of neighbors for site `i`
    pub fn num_neighbors(&self, site: usize) -> usize {
        let start = self.neighbor_offsets[site] as usize;
        let end = self.neighbor_offsets[site + 1] as usize;
        end - start
    }

    /// Iterator over (neighbor_idx, weight) for site `i`
    pub fn neighbors_of(&self, site: usize) -> impl Iterator<Item = (usize, f32)> + '_ {
        let start = self.neighbor_offsets[site] as usize;
        let end = self.neighbor_offsets[site + 1] as usize;
        self.neighbor_indices[start..end]
            .iter()
            .zip(self.neighbor_weights[start..end].iter())
            .map(|(&idx, &w)| (idx as usize, w))
    }

    /// Displacement vector from site i to its k-th neighbor
    pub fn displacement(&self, site: usize, k: usize) -> &[f32] {
        let start = self.neighbor_offsets[site] as usize;
        let edge_idx = start + k;
        let base = edge_idx * self.dim;
        &self.displacement_vectors[base..base + self.dim]
    }

    /// Get the complex value at a site: (re, im)
    pub fn get(&self, site: usize) -> (f32, f32) {
        (self.values_re[site], self.values_im[site])
    }

    /// Set the complex value at a site
    pub fn set(&mut self, site: usize, re: f32, im: f32) {
        self.values_re[site] = re;
        self.values_im[site] = im;
    }

    /// |z|² at site
    pub fn norm_sq(&self, site: usize) -> f32 {
        let r = self.values_re[site];
        let i = self.values_im[site];
        r * r + i * i
    }

    /// Total energy: Σ |z_i|²
    pub fn total_energy(&self) -> f32 {
        self.values_re
            .iter()
            .zip(self.values_im.iter())
            .map(|(&r, &i)| r * r + i * i)
            .sum()
    }

    /// Initialize field from random values using a simple LCG
    pub fn init_random(&mut self, seed: u64, amplitude: f32) {
        let mut state = seed;
        for i in 0..self.num_sites {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let re = ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0;
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let im = ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0;
            self.values_re[i] = re * amplitude;
            self.values_im[i] = im * amplitude;
        }
    }

    /// Update neighbor weights from a metric field.
    /// Weight w_ij = g^{μν} e^μ_{ij} e^ν_{ij} / |e_{ij}|²
    /// where e_{ij} is the displacement vector from i to j.
    pub fn update_weights_from_metric(&mut self, metric_data: &[f32], metric_dim: usize) {
        let packed_size = metric_dim * (metric_dim + 1) / 2;

        for site in 0..self.num_sites {
            let start = self.neighbor_offsets[site] as usize;
            let end = self.neighbor_offsets[site + 1] as usize;
            let metric_base = site * packed_size;

            for edge in start..end {
                let disp_base = edge * self.dim;
                let disp = &self.displacement_vectors[disp_base..disp_base + self.dim];

                // Compute g^{μν} e_μ e_ν (using identity inverse for now)
                // For general metric, need the inverse metric tensor
                let mut disp_sq = 0.0f32;
                let mut g_disp = 0.0f32;

                for mu in 0..metric_dim.min(self.dim) {
                    for nu in 0..metric_dim.min(self.dim) {
                        // Read g_μν from packed storage (upper triangle)
                        let (row, col) = if mu <= nu { (mu, nu) } else { (nu, mu) };
                        let packed_idx = row * (2 * metric_dim - row - 1) / 2 + col;
                        let g_mu_nu = if metric_base + packed_idx < metric_data.len() {
                            metric_data[metric_base + packed_idx]
                        } else {
                            if mu == nu { 1.0 } else { 0.0 }
                        };
                        g_disp += g_mu_nu * disp[mu] * disp[nu];
                    }
                    disp_sq += disp[mu] * disp[mu];
                }

                self.neighbor_weights[edge] = if disp_sq > 1e-12 {
                    g_disp / disp_sq
                } else {
                    1.0
                };
            }
        }
    }

    /// Memory usage in bytes (approximate)
    pub fn memory_bytes(&self) -> usize {
        (self.values_re.len() + self.values_im.len() + self.neighbor_weights.len()) * 4
            + (self.neighbor_indices.len() + self.neighbor_offsets.len()) * 4
            + self.displacement_vectors.len() * 4
    }

    /// Total number of edges (directed) in the CSR graph
    pub fn num_edges(&self) -> usize {
        self.neighbor_indices.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use icarus_math::lattice::e8::E8Lattice;
    use icarus_math::lattice::hypercubic::HypercubicLattice;

    #[test]
    fn test_from_e8_lattice() {
        let lattice = E8Lattice::new();
        let field = LatticeField::from_lattice(&lattice);

        assert_eq!(field.num_sites, 241);
        assert_eq!(field.dim, 8);
        assert_eq!(field.layer, LatticeLayer::Analytical);
        assert_eq!(field.values_re.len(), 241);
        assert_eq!(field.values_im.len(), 241);
        // CSR offsets should have num_sites + 1 entries
        assert_eq!(field.neighbor_offsets.len(), 242);
        // Origin has 240 neighbors
        assert_eq!(field.num_neighbors(0), 240);
    }

    #[test]
    fn test_from_hypercubic_lattice() {
        let lattice = HypercubicLattice::new(4);
        let field = LatticeField::from_lattice(&lattice);

        assert_eq!(field.num_sites, 9); // origin + 2*4 = 9
        assert_eq!(field.dim, 4);
        // Origin has 8 neighbors
        assert_eq!(field.num_neighbors(0), 8);
    }

    #[test]
    fn test_get_set() {
        let lattice = E8Lattice::new();
        let mut field = LatticeField::from_lattice(&lattice);

        field.set(0, 1.0, 2.0);
        let (re, im) = field.get(0);
        assert!((re - 1.0).abs() < 1e-6);
        assert!((im - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_norm_sq() {
        let lattice = E8Lattice::new();
        let mut field = LatticeField::from_lattice(&lattice);

        field.set(0, 3.0, 4.0);
        assert!((field.norm_sq(0) - 25.0).abs() < 1e-6);
    }

    #[test]
    fn test_random_init() {
        let lattice = E8Lattice::new();
        let mut field = LatticeField::from_lattice(&lattice);
        field.init_random(42, 1.0);

        // Should have non-zero values
        let energy = field.total_energy();
        assert!(energy > 0.0);

        // Deterministic
        let mut field2 = LatticeField::from_lattice(&lattice);
        field2.init_random(42, 1.0);
        for i in 0..field.num_sites {
            assert!((field.values_re[i] - field2.values_re[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_neighbors_iterator() {
        let lattice = E8Lattice::new();
        let field = LatticeField::from_lattice(&lattice);

        let neighbors: Vec<_> = field.neighbors_of(0).collect();
        assert_eq!(neighbors.len(), 240);
        // All weights should be 1.0 (flat metric)
        for &(_, w) in &neighbors {
            assert!((w - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_displacement_vectors() {
        let lattice = HypercubicLattice::new(3);
        let field = LatticeField::from_lattice(&lattice);

        // Origin (site 0) has 6 neighbors in 3D hypercubic
        assert_eq!(field.num_neighbors(0), 6);

        // Each displacement should have norm 1
        for k in 0..field.num_neighbors(0) {
            let disp = field.displacement(0, k);
            let norm_sq: f32 = disp.iter().map(|d| d * d).sum();
            assert!((norm_sq - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_memory_bytes() {
        let lattice = E8Lattice::new();
        let field = LatticeField::from_lattice(&lattice);
        assert!(field.memory_bytes() > 0);
    }

    #[test]
    fn test_e8_topology_stats() {
        let lattice = E8Lattice::new();
        let field = LatticeField::from_lattice(&lattice);

        eprintln!("E8 topology: {} sites, {} directed edges", field.num_sites, field.num_edges());
        for i in 0..field.num_sites.min(5) {
            eprintln!("  Site {}: {} neighbors", i, field.num_neighbors(i));
        }
        eprintln!("  Site 113: {} neighbors", field.num_neighbors(113));

        // Origin always has 240 neighbors
        assert_eq!(field.num_neighbors(0), 240);
        // Non-origin sites should have >1 neighbor in correct E8 topology
        assert!(field.num_neighbors(1) > 1, "Site 1 should have >1 neighbor, got {}", field.num_neighbors(1));
    }
}
