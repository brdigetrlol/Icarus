//! Geodesic path-finding on the crystal manifold
//!
//! Dijkstra's algorithm on the CSR lattice graph with metric-tensor-derived
//! edge weights: d(i,j) = sqrt(g_{μν}(i) · e^μ_{ij} · e^ν_{ij})
//!
//! This gives the proper geodesic distance element ds = sqrt(g_{μν} dx^μ dx^ν)
//! along lattice edges, enabling shortest-path queries on curved geometry.

use crate::phase_field::LatticeField;
use icarus_math::metric::{MetricField, SiteMetric};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Result of a single-source geodesic distance computation.
#[derive(Debug, Clone)]
pub struct GeodesicResult {
    /// Geodesic distance from source to each site (f32::INFINITY if unreachable)
    pub distances: Vec<f32>,
    /// Predecessor on the shortest path (usize::MAX = source or unreachable)
    pub predecessors: Vec<usize>,
    /// Source site index
    pub source: usize,
}

impl GeodesicResult {
    /// Extract the shortest geodesic path from source to target.
    /// Returns None if target is unreachable.
    pub fn path_to(&self, target: usize) -> Option<Vec<usize>> {
        if self.distances[target].is_infinite() {
            return None;
        }
        if target == self.source {
            return Some(vec![self.source]);
        }

        let mut path = Vec::new();
        let mut current = target;
        while current != self.source {
            path.push(current);
            let pred = self.predecessors[current];
            if pred == usize::MAX {
                return None; // Broken chain
            }
            current = pred;
        }
        path.push(self.source);
        path.reverse();
        Some(path)
    }

    /// Geodesic distance from source to target. Returns None if unreachable.
    pub fn distance_to(&self, target: usize) -> Option<f32> {
        let d = self.distances[target];
        if d.is_infinite() {
            None
        } else {
            Some(d)
        }
    }

    /// Find the farthest reachable site (geodesic eccentricity of source).
    pub fn farthest_site(&self) -> (usize, f32) {
        let mut best_site = self.source;
        let mut best_dist = 0.0f32;
        for (i, &d) in self.distances.iter().enumerate() {
            if d.is_finite() && d > best_dist {
                best_dist = d;
                best_site = i;
            }
        }
        (best_site, best_dist)
    }
}

/// Dijkstra priority queue entry (min-heap via Reverse ordering).
struct DijkstraEntry {
    site: usize,
    dist: f32,
}

impl PartialEq for DijkstraEntry {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist
    }
}

impl Eq for DijkstraEntry {}

impl PartialOrd for DijkstraEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DijkstraEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap (BinaryHeap is max-heap)
        other
            .dist
            .partial_cmp(&self.dist)
            .unwrap_or(Ordering::Equal)
    }
}

/// Compute the metric-weighted edge distance between adjacent sites.
///
/// d(i,j) = sqrt(g_{μν}(i) · e^μ · e^ν)
///
/// where e is the displacement vector from site i to neighbor j.
/// For identity metric, this reduces to the Euclidean distance.
fn edge_distance(metric: &SiteMetric, displacement: &[f32]) -> f32 {
    let dim = metric.dim.min(displacement.len());
    let mut d_sq = 0.0f32;
    for mu in 0..dim {
        for nu in 0..dim {
            d_sq += metric.get(mu, nu) * displacement[mu] * displacement[nu];
        }
    }
    // Clamp to zero (numerical noise can produce tiny negative values)
    d_sq.max(0.0).sqrt()
}

/// Compute single-source geodesic distances from `source` to all reachable sites.
///
/// Uses Dijkstra's algorithm on the CSR neighbor graph with metric-weighted
/// edge distances. Complexity: O((V + E) log V) where V = num_sites, E = num_edges.
pub fn geodesic_distances(
    field: &LatticeField,
    metric: &MetricField,
    source: usize,
) -> GeodesicResult {
    let n = field.num_sites;
    let mut distances = vec![f32::INFINITY; n];
    let mut predecessors = vec![usize::MAX; n];
    let mut visited = vec![false; n];
    let mut heap = BinaryHeap::new();

    distances[source] = 0.0;
    heap.push(DijkstraEntry {
        site: source,
        dist: 0.0,
    });

    while let Some(DijkstraEntry { site, dist }) = heap.pop() {
        if visited[site] {
            continue;
        }
        visited[site] = true;

        // Early exit: if popped distance exceeds recorded (stale entry), skip
        if dist > distances[site] {
            continue;
        }

        let site_metric = metric.get_site(site);
        let start = field.neighbor_offsets[site] as usize;
        let end = field.neighbor_offsets[site + 1] as usize;

        for (k, edge) in (start..end).enumerate() {
            let neighbor = field.neighbor_indices[edge] as usize;
            if visited[neighbor] {
                continue;
            }

            let disp = field.displacement(site, k);
            let edge_dist = edge_distance(&site_metric, disp);
            let new_dist = dist + edge_dist;

            if new_dist < distances[neighbor] {
                distances[neighbor] = new_dist;
                predecessors[neighbor] = site;
                heap.push(DijkstraEntry {
                    site: neighbor,
                    dist: new_dist,
                });
            }
        }
    }

    GeodesicResult {
        distances,
        predecessors,
        source,
    }
}

/// Compute the geodesic distance between two specific sites.
///
/// This runs full Dijkstra from source (no early termination, since the graph
/// is typically small — hundreds to low thousands of sites). For repeated
/// queries from the same source, cache the `GeodesicResult` instead.
pub fn geodesic_distance(
    field: &LatticeField,
    metric: &MetricField,
    source: usize,
    target: usize,
) -> Option<f32> {
    let result = geodesic_distances(field, metric, source);
    result.distance_to(target)
}

/// Compute the geodesic diameter of the manifold (maximum pairwise geodesic distance).
///
/// Uses the double-sweep heuristic: pick an arbitrary site, find the farthest
/// site from it, then find the farthest site from *that*. The distance of the
/// second sweep is a tight lower bound on the true diameter (exact for trees,
/// near-exact for lattice graphs).
pub fn geodesic_diameter(field: &LatticeField, metric: &MetricField) -> f32 {
    if field.num_sites == 0 {
        return 0.0;
    }

    // First sweep: from site 0
    let first = geodesic_distances(field, metric, 0);
    let (far1, _) = first.farthest_site();

    // Second sweep: from the farthest site found
    let second = geodesic_distances(field, metric, far1);
    let (_, diameter) = second.farthest_site();

    diameter
}

/// Find the site with minimum energy (|z|²) and compute geodesic distances from it.
///
/// Useful for the planning agent to understand the energy landscape geometry:
/// how far each site is (in curved space) from the current energy minimum.
pub fn distances_from_energy_minimum(
    field: &LatticeField,
    metric: &MetricField,
) -> GeodesicResult {
    let min_site = (0..field.num_sites)
        .min_by(|&a, &b| {
            field
                .norm_sq(a)
                .partial_cmp(&field.norm_sq(b))
                .unwrap_or(Ordering::Equal)
        })
        .unwrap_or(0);

    geodesic_distances(field, metric, min_site)
}

/// Find the site with maximum amplitude (|z|) — the dominant attractor site.
pub fn find_max_amplitude_site(field: &LatticeField) -> (usize, f32) {
    let mut best_site = 0;
    let mut best_amp = 0.0f32;
    for i in 0..field.num_sites {
        let amp = field.norm_sq(i).sqrt();
        if amp > best_amp {
            best_amp = amp;
            best_site = i;
        }
    }
    (best_site, best_amp)
}

/// Find the site with minimum free energy contribution.
///
/// Approximates the local free energy at each site as:
/// f(i) = potential_weight * (|z_i|² - target²)² / 4
/// (ignoring kinetic term which requires neighbor access)
pub fn find_min_energy_site(field: &LatticeField, target_amplitude: f32) -> (usize, f32) {
    let target_sq = target_amplitude * target_amplitude;
    let mut best_site = 0;
    let mut best_energy = f32::INFINITY;
    for i in 0..field.num_sites {
        let ns = field.norm_sq(i);
        let diff = ns - target_sq;
        let potential = diff * diff * 0.25;
        if potential < best_energy {
            best_energy = potential;
            best_site = i;
        }
    }
    (best_site, best_energy)
}

#[cfg(test)]
mod tests {
    use super::*;
    use icarus_math::lattice::e8::E8Lattice;
    use icarus_math::lattice::hypercubic::HypercubicLattice;
    use icarus_math::metric::MetricField;

    #[test]
    fn test_geodesic_flat_e8() {
        let lattice = E8Lattice::new();
        let field = LatticeField::from_lattice(&lattice);
        let metric = MetricField::identity(8, field.num_sites);

        let result = geodesic_distances(&field, &metric, 0);

        // Source distance is zero
        assert_eq!(result.distances[0], 0.0);

        // All 240 nearest neighbors should have the same distance
        // (E8 root vectors all have norm sqrt(2))
        let start = field.neighbor_offsets[0] as usize;
        let end = field.neighbor_offsets[1] as usize;
        let first_neighbor = field.neighbor_indices[start] as usize;
        let expected_dist = result.distances[first_neighbor];
        assert!(expected_dist > 0.0, "neighbor distance should be positive");

        // All neighbors of origin in E8 have norm sqrt(2)
        assert!(
            (expected_dist - std::f32::consts::SQRT_2).abs() < 0.01,
            "E8 root vector norm should be sqrt(2), got {}",
            expected_dist
        );

        for edge in start..end {
            let nb = field.neighbor_indices[edge] as usize;
            assert!(
                (result.distances[nb] - expected_dist).abs() < 1e-5,
                "all E8 neighbors should have equal geodesic distance, site {} dist={} expected={}",
                nb, result.distances[nb], expected_dist
            );
        }
    }

    #[test]
    fn test_geodesic_path_recovery() {
        let lattice = HypercubicLattice::new(3);
        let field = LatticeField::from_lattice(&lattice);
        let metric = MetricField::identity(3, field.num_sites);

        let result = geodesic_distances(&field, &metric, 0);

        // Path from source to self
        let path = result.path_to(0).unwrap();
        assert_eq!(path, vec![0]);

        // Path to any neighbor is direct
        let first_nb = field.neighbor_indices[0] as usize;
        let path = result.path_to(first_nb).unwrap();
        assert_eq!(path.len(), 2);
        assert_eq!(path[0], 0);
        assert_eq!(path[1], first_nb);
    }

    #[test]
    fn test_geodesic_distance_function() {
        let lattice = HypercubicLattice::new(4);
        let field = LatticeField::from_lattice(&lattice);
        let metric = MetricField::identity(4, field.num_sites);

        // Distance from 0 to 0 is 0
        let d = geodesic_distance(&field, &metric, 0, 0).unwrap();
        assert!(d.abs() < 1e-6);

        // Distance from 0 to a neighbor is 1.0 (unit vectors in Z^4)
        let first_nb = field.neighbor_indices[0] as usize;
        let d = geodesic_distance(&field, &metric, 0, first_nb).unwrap();
        assert!((d - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_geodesic_diameter() {
        let lattice = HypercubicLattice::new(3);
        let field = LatticeField::from_lattice(&lattice);
        let metric = MetricField::identity(3, field.num_sites);

        let diam = geodesic_diameter(&field, &metric);
        // Z^3 with origin + 6 neighbors: diameter = 2.0 (through origin)
        assert!(diam >= 1.0, "diameter should be at least 1.0");
        assert!(diam <= 3.0, "diameter should be at most 3.0 for Z^3 BFS-1 shell");
    }

    #[test]
    fn test_geodesic_curved_metric() {
        let lattice = HypercubicLattice::new(3);
        let field = LatticeField::from_lattice(&lattice);
        let mut metric = MetricField::identity(3, field.num_sites);

        // Stretch metric at origin: g_{00} = 4.0 (doubles distance in x-direction)
        let mut m = metric.get_site(0);
        m.set(0, 0, 4.0);
        metric.set_site(0, &m);

        let result_flat = geodesic_distances(
            &field,
            &MetricField::identity(3, field.num_sites),
            0,
        );
        let result_curved = geodesic_distances(&field, &metric, 0);

        // In the curved metric, the +x neighbor should be farther than in flat
        // Find a +x neighbor (displacement = [1,0,0])
        let start = field.neighbor_offsets[0] as usize;
        let end = field.neighbor_offsets[1] as usize;
        let mut x_neighbor = None;
        for k in 0..(end - start) {
            let disp = field.displacement(0, k);
            if (disp[0] - 1.0).abs() < 1e-6
                && disp[1].abs() < 1e-6
                && disp[2].abs() < 1e-6
            {
                x_neighbor = Some(field.neighbor_indices[start + k] as usize);
                break;
            }
        }

        if let Some(xn) = x_neighbor {
            // sqrt(4.0 * 1^2) = 2.0 in curved metric, 1.0 in flat
            let flat_d = result_flat.distances[xn];
            let curved_d = result_curved.distances[xn];
            assert!(
                curved_d > flat_d * 1.5,
                "curved distance ({}) should be > 1.5 * flat distance ({})",
                curved_d,
                flat_d
            );
            assert!(
                (curved_d - 2.0).abs() < 0.01,
                "stretched metric should give distance 2.0, got {}",
                curved_d
            );
        }
    }

    #[test]
    fn test_find_max_amplitude_site() {
        let lattice = E8Lattice::new();
        let mut field = LatticeField::from_lattice(&lattice);
        field.set(42, 3.0, 4.0); // |z| = 5
        field.set(100, 1.0, 0.0); // |z| = 1

        let (site, amp) = find_max_amplitude_site(&field);
        assert_eq!(site, 42);
        assert!((amp - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_find_min_energy_site() {
        let lattice = E8Lattice::new();
        let mut field = LatticeField::from_lattice(&lattice);

        // Target amplitude = 1.0, so |z|=1 minimizes potential
        field.set(10, 1.0, 0.0); // |z|=1 → V=0
        field.set(20, 0.0, 0.0); // |z|=0 → V=(0-1)^2/4 = 0.25
        field.set(30, 2.0, 0.0); // |z|=2 → V=(4-1)^2/4 = 2.25

        let (site, energy) = find_min_energy_site(&field, 1.0);
        assert_eq!(site, 10);
        assert!(energy < 1e-6);
    }

    #[test]
    fn test_distances_from_energy_minimum() {
        let lattice = HypercubicLattice::new(3);
        let mut field = LatticeField::from_lattice(&lattice);
        field.set(0, 0.5, 0.5); // non-minimal
        // Site with smallest |z|^2 starts as 0 (all zeroes after init) but site 0 is now non-zero
        // The minimum-energy site will be one of the sites that are still zero

        let metric = MetricField::identity(3, field.num_sites);
        let result = distances_from_energy_minimum(&field, &metric);

        // The source should be a zero-energy site (not site 0)
        assert_ne!(result.source, 0);
        assert_eq!(result.distances[result.source], 0.0);
    }

    #[test]
    fn test_farthest_site() {
        let lattice = HypercubicLattice::new(4);
        let field = LatticeField::from_lattice(&lattice);
        let metric = MetricField::identity(4, field.num_sites);

        let result = geodesic_distances(&field, &metric, 0);
        let (farthest, dist) = result.farthest_site();

        assert_ne!(farthest, 0);
        assert!(dist > 0.0);
    }
}
