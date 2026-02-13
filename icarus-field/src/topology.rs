//! Persistent homology for topological regularization of the phase field.
//!
//! Computes β₀ (connected components) via sublevel set filtration on |z|²,
//! using Union-Find with the elder rule. The topological energy penalizes
//! unwanted persistence, and the gradient (via Wirtinger derivatives) pushes
//! the field toward the desired topological structure.

use serde::{Deserialize, Serialize};

use crate::phase_field::LatticeField;

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

/// A persistence pair recording the birth and death of a connected component.
#[derive(Debug, Clone)]
pub struct PersistencePair {
    /// Filtration value at which the component was born.
    pub birth: f32,
    /// Filtration value at which the component merged into an older one.
    pub death: f32,
    /// Site index whose activation created the component.
    pub birth_site: usize,
    /// Site index whose edge caused the merge (death event).
    pub death_site: usize,
}

impl PersistencePair {
    /// Lifetime of this topological feature.
    pub fn persistence(&self) -> f32 {
        self.death - self.birth
    }

    /// Essential features survive the entire filtration (infinite death).
    pub fn is_essential(&self) -> bool {
        self.death.is_infinite()
    }
}

/// Summary of the β₀ persistence diagram.
#[derive(Debug, Clone)]
pub struct TopologicalSummary {
    /// All finite persistence pairs.
    pub pairs: Vec<PersistencePair>,
    /// Current number of connected components (Betti-0).
    pub betti_0: usize,
    /// Sum of persistence values for finite pairs.
    pub total_persistence: f32,
    /// Maximum persistence among finite pairs.
    pub max_persistence: f32,
    /// Number of features with persistence above threshold.
    pub significant_features: usize,
    /// Threshold used for significance filtering.
    pub threshold: f32,
}

/// Penalty mode for topological regularization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PenaltyMode {
    /// F = weight * Σ (death - birth)² over significant pairs.
    TotalPersistence,
    /// F = weight * (significant_features - target_betti_0)².
    TargetBetti,
    /// F = weight * max(death - birth)².
    MaxPersistence,
}

/// Parameters for topological regularization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyParams {
    /// Weight of the topological energy term.
    pub weight: f32,
    /// Pairs with persistence below this are ignored.
    pub persistence_threshold: f32,
    /// Target number of significant components (for TargetBetti mode).
    pub target_betti_0: usize,
    /// Which penalty function to use.
    pub penalty_mode: PenaltyMode,
    /// Step size for the topological gradient descent.
    pub gradient_step: f32,
}

impl Default for TopologyParams {
    fn default() -> Self {
        Self {
            weight: 0.01,
            persistence_threshold: 0.01,
            target_betti_0: 1,
            penalty_mode: PenaltyMode::TotalPersistence,
            gradient_step: 0.001,
        }
    }
}

// ---------------------------------------------------------------------------
// Union-Find with elder rule (path halving, no rank)
// ---------------------------------------------------------------------------

struct UnionFind {
    parent: Vec<usize>,
    birth: Vec<f32>,
    num_components: usize,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            birth: vec![f32::INFINITY; n],
            num_components: 0,
        }
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]]; // path halving
            x = self.parent[x];
        }
        x
    }

    fn activate(&mut self, x: usize, birth_time: f32) {
        self.birth[x] = birth_time;
        self.num_components += 1;
    }

    #[cfg(test)]
    fn is_active(&self, x: usize) -> bool {
        self.birth[x].is_finite()
    }

    /// Elder rule union: the component with the earlier birth survives.
    /// Returns `Some((dying_root, dying_birth))` if a merge occurred.
    fn union(&mut self, a: usize, b: usize) -> Option<(usize, f32)> {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return None;
        }

        // Elder rule: lower birth value survives
        let (older, younger) = if self.birth[ra] <= self.birth[rb] {
            (ra, rb)
        } else {
            (rb, ra)
        };

        self.parent[younger] = older;
        self.num_components -= 1;
        Some((younger, self.birth[younger]))
    }
}

// ---------------------------------------------------------------------------
// Sublevel set persistence (β₀)
// ---------------------------------------------------------------------------

/// Compute β₀ persistence of the sublevel set filtration f(i) = |z_i|².
///
/// Sites are added in order of increasing |z|². Edges to already-added
/// neighbors trigger Union-Find merges. The elder rule ensures correct
/// persistence pairing: when two components merge, the younger one dies.
///
/// Complexity: O(N log N) for the sort + O(N α(N)) for Union-Find operations.
pub fn compute_sublevel_persistence(field: &LatticeField, threshold: f32) -> TopologicalSummary {
    let n = field.num_sites;

    // Filtration function: f(i) = |z_i|²
    let filtration: Vec<f32> = (0..n).map(|i| field.norm_sq(i)).collect();

    // Sort sites by ascending filtration value
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        filtration[a]
            .partial_cmp(&filtration[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut added = vec![false; n];
    let mut uf = UnionFind::new(n);
    let mut pairs = Vec::new();

    for &site in &order {
        let f_val = filtration[site];
        added[site] = true;
        uf.activate(site, f_val);

        // Check edges to already-added neighbors via CSR adjacency
        let start = field.neighbor_offsets[site] as usize;
        let end = field.neighbor_offsets[site + 1] as usize;

        for idx in start..end {
            let neighbor = field.neighbor_indices[idx] as usize;
            if added[neighbor] {
                if let Some((dying_root, dying_birth)) = uf.union(site, neighbor) {
                    let persistence = f_val - dying_birth;
                    if persistence > 0.0 {
                        pairs.push(PersistencePair {
                            birth: dying_birth,
                            death: f_val,
                            birth_site: dying_root,
                            death_site: site,
                        });
                    }
                }
            }
        }
    }

    let betti_0 = uf.num_components;
    let total_persistence: f32 = pairs.iter().map(|p| p.persistence()).sum();
    let max_persistence = pairs
        .iter()
        .map(|p| p.persistence())
        .fold(0.0f32, f32::max);
    let significant_features = pairs
        .iter()
        .filter(|p| p.persistence() > threshold)
        .count();

    TopologicalSummary {
        pairs,
        betti_0,
        total_persistence,
        max_persistence,
        significant_features,
        threshold,
    }
}

// ---------------------------------------------------------------------------
// Topological energy
// ---------------------------------------------------------------------------

/// Compute the scalar topological energy from a persistence summary.
pub fn topological_energy(summary: &TopologicalSummary, params: &TopologyParams) -> f32 {
    match params.penalty_mode {
        PenaltyMode::TotalPersistence => {
            let sum_sq: f32 = summary
                .pairs
                .iter()
                .filter(|p| p.persistence() > params.persistence_threshold)
                .map(|p| {
                    let pers = p.persistence();
                    pers * pers
                })
                .sum();
            params.weight * sum_sq
        }
        PenaltyMode::TargetBetti => {
            let diff = summary.significant_features as f32 - params.target_betti_0 as f32;
            params.weight * diff * diff
        }
        PenaltyMode::MaxPersistence => {
            params.weight * summary.max_persistence * summary.max_persistence
        }
    }
}

// ---------------------------------------------------------------------------
// Topological gradient (Wirtinger derivatives)
// ---------------------------------------------------------------------------

/// Compute the topological gradient ∂F_topo/∂z* for each site.
///
/// Uses the chain rule: ∂F/∂z*_i = (∂F/∂f_i) · (∂f_i/∂z*_i) where f_i = |z_i|²
/// and ∂(|z|²)/∂z* = z (Wirtinger derivative).
///
/// Returns (grad_re, grad_im) vectors.
pub fn topological_gradient(
    field: &LatticeField,
    summary: &TopologicalSummary,
    params: &TopologyParams,
) -> (Vec<f32>, Vec<f32>) {
    let n = field.num_sites;
    let mut accum = vec![0.0f32; n];

    match params.penalty_mode {
        PenaltyMode::TotalPersistence => {
            for pair in &summary.pairs {
                if pair.persistence() < params.persistence_threshold {
                    continue;
                }
                let p = pair.persistence();
                // ∂F/∂f(birth_site) = -2 * weight * persistence
                accum[pair.birth_site] += -2.0 * params.weight * p;
                // ∂F/∂f(death_site) = +2 * weight * persistence
                if pair.death_site < n {
                    accum[pair.death_site] += 2.0 * params.weight * p;
                }
            }
        }
        PenaltyMode::MaxPersistence => {
            if let Some(max_pair) = summary
                .pairs
                .iter()
                .max_by(|a, b| {
                    a.persistence()
                        .partial_cmp(&b.persistence())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
            {
                let p = max_pair.persistence();
                accum[max_pair.birth_site] += -2.0 * params.weight * p;
                if max_pair.death_site < n {
                    accum[max_pair.death_site] += 2.0 * params.weight * p;
                }
            }
        }
        PenaltyMode::TargetBetti => {
            let diff = summary.significant_features as f32 - params.target_betti_0 as f32;
            if diff.abs() > 0.5 {
                let scale = 2.0 * params.weight * diff;
                for pair in &summary.pairs {
                    if pair.persistence() < params.persistence_threshold {
                        continue;
                    }
                    let p = pair.persistence();
                    accum[pair.birth_site] += -scale * p;
                    if pair.death_site < n {
                        accum[pair.death_site] += scale * p;
                    }
                }
            }
        }
    }

    // Wirtinger chain rule: ∂F/∂z*_i = accum[i] * z_i
    let mut grad_re = vec![0.0f32; n];
    let mut grad_im = vec![0.0f32; n];
    for i in 0..n {
        grad_re[i] = accum[i] * field.values_re[i];
        grad_im[i] = accum[i] * field.values_im[i];
    }

    (grad_re, grad_im)
}

// ---------------------------------------------------------------------------
// Convenience: one-shot topology step
// ---------------------------------------------------------------------------

/// Compute persistence, apply one gradient step, return the summary.
///
/// This is the entry point called by the manifold tick Phase 1e.
pub fn apply_topology_step(
    field: &mut LatticeField,
    params: &TopologyParams,
) -> TopologicalSummary {
    let summary = compute_sublevel_persistence(field, params.persistence_threshold);
    let (grad_re, grad_im) = topological_gradient(field, &summary, params);
    let step = params.gradient_step;
    for i in 0..field.num_sites {
        field.values_re[i] -= step * grad_re[i];
        field.values_im[i] -= step * grad_im[i];
    }
    summary
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use icarus_math::lattice::e8::E8Lattice;

    fn make_e8_field() -> LatticeField {
        let lattice = E8Lattice::new();
        LatticeField::from_lattice(&lattice)
    }

    fn make_e8_field_random(seed: u64, amplitude: f32) -> LatticeField {
        let mut field = make_e8_field();
        field.init_random(seed, amplitude);
        field
    }

    // -- Union-Find tests --

    #[test]
    fn test_union_find_basic() {
        let mut uf = UnionFind::new(5);
        assert!(!uf.is_active(0));
        assert_eq!(uf.num_components, 0);

        uf.activate(0, 0.1);
        uf.activate(1, 0.2);
        uf.activate(2, 0.3);
        assert_eq!(uf.num_components, 3);
        assert!(uf.is_active(0));

        // Union 0 and 1: 0 is older
        let result = uf.union(0, 1);
        assert!(result.is_some());
        let (dying, birth) = result.unwrap();
        assert_eq!(dying, 1);
        assert!((birth - 0.2).abs() < 1e-6);
        assert_eq!(uf.num_components, 2);

        // Same component — no merge
        assert!(uf.union(0, 1).is_none());
        assert_eq!(uf.num_components, 2);
    }

    #[test]
    fn test_union_find_elder_rule() {
        let mut uf = UnionFind::new(4);
        // Activate in reverse order: 3 is oldest, 0 is youngest
        uf.activate(3, 0.1);
        uf.activate(2, 0.2);
        uf.activate(1, 0.3);
        uf.activate(0, 0.4);

        // Union 0 (young) and 3 (old): 0 should die
        let result = uf.union(0, 3);
        assert!(result.is_some());
        let (dying, _birth) = result.unwrap();
        assert_eq!(dying, 0); // younger dies

        // Root of both should be 3 (the elder)
        assert_eq!(uf.find(0), 3);
        assert_eq!(uf.find(3), 3);
    }

    // -- Persistence tests --

    #[test]
    fn test_persistence_pair_properties() {
        let pair = PersistencePair {
            birth: 0.1,
            death: 0.5,
            birth_site: 0,
            death_site: 10,
        };
        assert!((pair.persistence() - 0.4).abs() < 1e-6);
        assert!(!pair.is_essential());

        let essential = PersistencePair {
            birth: 0.1,
            death: f32::INFINITY,
            birth_site: 0,
            death_site: 0,
        };
        assert!(essential.is_essential());
    }

    #[test]
    fn test_persistence_uniform_field() {
        // All sites have the same |z|² → near-zero persistence
        let mut field = make_e8_field();
        let amp = 0.5f32;
        for i in 0..field.num_sites {
            field.values_re[i] = amp;
            field.values_im[i] = 0.0;
        }

        let summary = compute_sublevel_persistence(&field, 0.001);
        // All filtration values equal → all merges have zero persistence
        assert!(
            summary.total_persistence < 1e-6,
            "Uniform field should have near-zero persistence, got {}",
            summary.total_persistence
        );
        // Connected graph → β₀ = 1
        assert_eq!(summary.betti_0, 1);
    }

    #[test]
    fn test_persistence_e8_random() {
        let field = make_e8_field_random(42, 0.5);
        let summary = compute_sublevel_persistence(&field, 0.01);

        // E8 graph is connected → after full filtration, β₀ = 1
        assert_eq!(
            summary.betti_0, 1,
            "E8 lattice should be connected, got β₀={}",
            summary.betti_0
        );
        // Random field should produce some finite persistence pairs
        assert!(
            !summary.pairs.is_empty(),
            "Random field should produce persistence pairs"
        );
        // Total persistence should be positive
        assert!(summary.total_persistence > 0.0);
        // Non-zero persistence pairs <= N-1 (zero-persistence merges are filtered)
        assert!(summary.pairs.len() <= field.num_sites - 1);
    }

    #[test]
    fn test_persistence_pair_count() {
        // For a connected graph, N-1 merges occur total, but zero-persistence
        // pairs (new site immediately merges into existing component) are filtered.
        // In highly-connected E8, most merges are zero-persistence.
        let field = make_e8_field_random(123, 1.0);
        let summary = compute_sublevel_persistence(&field, 0.0);
        assert!(summary.pairs.len() > 0);
        assert!(summary.pairs.len() <= field.num_sites - 1);
        assert_eq!(summary.betti_0, 1); // graph is connected
    }

    // -- Energy tests --

    #[test]
    fn test_topological_energy_total_persistence() {
        let field = make_e8_field_random(42, 0.5);
        let summary = compute_sublevel_persistence(&field, 0.01);
        let params = TopologyParams {
            weight: 1.0,
            persistence_threshold: 0.0,
            penalty_mode: PenaltyMode::TotalPersistence,
            ..Default::default()
        };
        let energy = topological_energy(&summary, &params);
        // Energy = Σ persistence² — should be positive for random field
        assert!(energy > 0.0, "Energy should be positive, got {}", energy);
    }

    #[test]
    fn test_topological_energy_max_persistence() {
        let field = make_e8_field_random(42, 0.5);
        let summary = compute_sublevel_persistence(&field, 0.01);
        let params = TopologyParams {
            weight: 1.0,
            persistence_threshold: 0.0,
            penalty_mode: PenaltyMode::MaxPersistence,
            ..Default::default()
        };
        let energy = topological_energy(&summary, &params);
        let expected = summary.max_persistence * summary.max_persistence;
        assert!(
            (energy - expected).abs() < 1e-6,
            "MaxPersistence energy mismatch: {} vs {}",
            energy,
            expected
        );
    }

    #[test]
    fn test_topological_energy_target_betti() {
        let field = make_e8_field_random(42, 0.5);
        let summary = compute_sublevel_persistence(&field, 0.01);
        let params = TopologyParams {
            weight: 1.0,
            target_betti_0: summary.significant_features,
            penalty_mode: PenaltyMode::TargetBetti,
            ..Default::default()
        };
        // When target matches, energy should be zero
        let energy = topological_energy(&summary, &params);
        assert!(
            energy < 1e-6,
            "Energy should be ~0 when at target, got {}",
            energy
        );
    }

    // -- Gradient tests --

    #[test]
    fn test_gradient_reduces_total_persistence() {
        let mut field = make_e8_field_random(42, 0.8);
        let params = TopologyParams {
            weight: 0.1,
            persistence_threshold: 0.001,
            target_betti_0: 1,
            penalty_mode: PenaltyMode::TotalPersistence,
            gradient_step: 0.01,
        };

        // Compute energy before
        let summary_before = compute_sublevel_persistence(&field, params.persistence_threshold);
        let energy_before = topological_energy(&summary_before, &params);

        // Apply gradient step
        let (grad_re, grad_im) = topological_gradient(&field, &summary_before, &params);
        for i in 0..field.num_sites {
            field.values_re[i] -= params.gradient_step * grad_re[i];
            field.values_im[i] -= params.gradient_step * grad_im[i];
        }

        // Compute energy after
        let summary_after = compute_sublevel_persistence(&field, params.persistence_threshold);
        let energy_after = topological_energy(&summary_after, &params);

        assert!(
            energy_after < energy_before,
            "Gradient should reduce energy: {} -> {}",
            energy_before,
            energy_after
        );
    }

    #[test]
    fn test_gradient_threshold_filtering() {
        let field = make_e8_field_random(42, 0.5);
        let params = TopologyParams {
            weight: 1.0,
            persistence_threshold: 1000.0, // absurdly high — no pairs qualify
            penalty_mode: PenaltyMode::TotalPersistence,
            ..Default::default()
        };
        let summary = compute_sublevel_persistence(&field, params.persistence_threshold);
        let (grad_re, grad_im) = topological_gradient(&field, &summary, &params);

        // All gradients should be zero
        let total_grad: f32 = grad_re.iter().map(|x| x.abs()).sum::<f32>()
            + grad_im.iter().map(|x| x.abs()).sum::<f32>();
        assert!(
            total_grad < 1e-10,
            "With high threshold, gradient should be zero, got {}",
            total_grad
        );
    }

    #[test]
    fn test_apply_topology_step_roundtrip() {
        let mut field = make_e8_field_random(42, 0.5);
        let params = TopologyParams::default();

        let summary = apply_topology_step(&mut field, &params);
        assert_eq!(summary.betti_0, 1);
        assert!(!summary.pairs.is_empty());
    }

    #[test]
    fn test_default_topology_params() {
        let params = TopologyParams::default();
        assert!((params.weight - 0.01).abs() < 1e-6);
        assert!((params.persistence_threshold - 0.01).abs() < 1e-6);
        assert_eq!(params.target_betti_0, 1);
        assert_eq!(params.penalty_mode, PenaltyMode::TotalPersistence);
        assert!((params.gradient_step - 0.001).abs() < 1e-6);
    }
}
