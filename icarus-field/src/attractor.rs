//! Attractor Basin Detection for the Icarus EMC
//!
//! The RAE dynamics drive the phase field toward discrete attractors.
//! This module identifies and classifies those attractors by:
//! 1. Running RAE from many random initial conditions
//! 2. Clustering the converged states
//! 3. Reporting the attractor basins
//!
//! Validation criterion: from 100 random ICs → at least 2 distinct attractors

use crate::phase_field::LatticeField;
use crate::rae::{RAEParams, RAESolver};

/// A detected attractor in the phase field
#[derive(Debug, Clone)]
pub struct Attractor {
    /// Representative field configuration (re values)
    pub values_re: Vec<f32>,
    /// Representative field configuration (im values)
    pub values_im: Vec<f32>,
    /// Number of initial conditions that converged to this attractor
    pub basin_size: usize,
    /// Final RMS rate of change (convergence quality)
    pub final_rms_rate: f32,
    /// Final free energy
    pub final_energy: f32,
}

/// Result of attractor basin analysis
#[derive(Debug, Clone)]
pub struct AttractorAnalysis {
    /// Detected attractors
    pub attractors: Vec<Attractor>,
    /// Which attractor each IC converged to (index into attractors)
    pub assignments: Vec<usize>,
    /// Total number of ICs that converged
    pub num_converged: usize,
    /// Total number of ICs that did NOT converge within max_steps
    pub num_diverged: usize,
}

/// Parameters for attractor detection
#[derive(Debug, Clone)]
pub struct AttractorSearchParams {
    /// Number of random initial conditions to try
    pub num_initial_conditions: usize,
    /// Maximum RAE steps per IC
    pub max_steps: u64,
    /// Convergence threshold: max |dz/dt| < epsilon
    pub convergence_epsilon: f32,
    /// Distance threshold for clustering attractors
    pub attractor_distance_threshold: f32,
    /// Initial amplitude for random ICs
    pub initial_amplitude: f32,
    /// RAE parameters
    pub rae_params: RAEParams,
}

impl Default for AttractorSearchParams {
    fn default() -> Self {
        Self {
            num_initial_conditions: 100,
            max_steps: 10000,
            convergence_epsilon: 1e-4,
            attractor_distance_threshold: 0.5,
            initial_amplitude: 0.8,
            rae_params: RAEParams::default_e8(),
        }
    }
}

/// Compute the L2 distance between two field configurations
fn field_distance(re_a: &[f32], im_a: &[f32], re_b: &[f32], im_b: &[f32]) -> f32 {
    let sum_sq: f32 = re_a
        .iter()
        .zip(im_a.iter())
        .zip(re_b.iter().zip(im_b.iter()))
        .map(|((&ra, &ia), (&rb, &ib))| {
            let dr = ra - rb;
            let di = ia - ib;
            dr * dr + di * di
        })
        .sum();
    sum_sq.sqrt()
}

/// Run attractor search: evolve from many ICs and cluster the results.
pub fn find_attractors(
    template_field: &LatticeField,
    params: &AttractorSearchParams,
) -> AttractorAnalysis {
    let n = template_field.num_sites;
    let mut attractors: Vec<Attractor> = Vec::new();
    let mut assignments = Vec::with_capacity(params.num_initial_conditions);
    let mut num_converged = 0usize;
    let mut num_diverged = 0usize;

    for ic_idx in 0..params.num_initial_conditions {
        // Create field with random IC
        let mut field = template_field.clone();
        field.init_random(ic_idx as u64 * 7919 + 31, params.initial_amplitude);

        // Run RAE until convergence or max steps
        let mut solver = RAESolver::new(params.rae_params.clone(), n);
        let mut converged = false;

        // Run in chunks to check convergence periodically
        let check_interval = 100;
        let mut remaining = params.max_steps;

        while remaining > 0 {
            let chunk = remaining.min(check_interval);
            solver.run(&mut field, chunk);
            remaining -= chunk;

            if solver.max_rate() < params.convergence_epsilon {
                converged = true;
                break;
            }
        }

        if converged {
            num_converged += 1;
        } else {
            num_diverged += 1;
        }

        // Compute final energy
        let energy_params = &params.rae_params.energy_params;
        let (energy, _, _) = crate::free_energy::free_energy(&field, energy_params);

        // Try to match to an existing attractor
        let mut matched = None;
        for (attr_idx, attr) in attractors.iter().enumerate() {
            let dist = field_distance(
                &field.values_re,
                &field.values_im,
                &attr.values_re,
                &attr.values_im,
            );
            if dist < params.attractor_distance_threshold {
                matched = Some(attr_idx);
                break;
            }
        }

        match matched {
            Some(attr_idx) => {
                attractors[attr_idx].basin_size += 1;
                assignments.push(attr_idx);
            }
            None => {
                let attr_idx = attractors.len();
                attractors.push(Attractor {
                    values_re: field.values_re.clone(),
                    values_im: field.values_im.clone(),
                    basin_size: 1,
                    final_rms_rate: solver.rms_rate(),
                    final_energy: energy,
                });
                assignments.push(attr_idx);
            }
        }
    }

    AttractorAnalysis {
        attractors,
        assignments,
        num_converged,
        num_diverged,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use icarus_math::lattice::e8::E8Lattice;
    use icarus_math::lattice::hypercubic::HypercubicLattice;

    #[test]
    fn test_field_distance_identical() {
        let a_re = vec![1.0, 2.0, 3.0];
        let a_im = vec![4.0, 5.0, 6.0];
        let dist = field_distance(&a_re, &a_im, &a_re, &a_im);
        assert!(dist < 1e-10);
    }

    #[test]
    fn test_field_distance_orthogonal() {
        let a_re = vec![1.0, 0.0];
        let a_im = vec![0.0, 0.0];
        let b_re = vec![0.0, 1.0];
        let b_im = vec![0.0, 0.0];
        let dist = field_distance(&a_re, &a_im, &b_re, &b_im);
        assert!((dist - 2.0f32.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn test_attractor_search_small() {
        // Use a small hypercubic lattice for speed
        let lattice = HypercubicLattice::new(3);
        let template = LatticeField::from_lattice(&lattice);

        let mut params = AttractorSearchParams {
            num_initial_conditions: 10,
            max_steps: 2000,
            convergence_epsilon: 1e-3,
            attractor_distance_threshold: 1.0,
            initial_amplitude: 0.5,
            rae_params: RAEParams::default_hypercubic(3),
        };
        params.rae_params.gamma = 0.3; // Strong damping for fast convergence

        let analysis = find_attractors(&template, &params);

        assert_eq!(analysis.assignments.len(), 10);
        assert!(
            analysis.attractors.len() >= 1,
            "Should find at least 1 attractor, found {}",
            analysis.attractors.len()
        );
    }

    #[test]
    fn test_attractor_determinism() {
        let lattice = HypercubicLattice::new(3);
        let template = LatticeField::from_lattice(&lattice);

        let mut params = AttractorSearchParams {
            num_initial_conditions: 5,
            max_steps: 1000,
            convergence_epsilon: 1e-3,
            attractor_distance_threshold: 1.0,
            initial_amplitude: 0.5,
            rae_params: RAEParams::default_hypercubic(3),
        };
        params.rae_params.gamma = 0.3;

        let analysis1 = find_attractors(&template, &params);
        let analysis2 = find_attractors(&template, &params);

        assert_eq!(analysis1.attractors.len(), analysis2.attractors.len());
        assert_eq!(analysis1.assignments, analysis2.assignments);
    }

    #[test]
    fn test_attractor_search_e8_quick() {
        // Quick E8 test with few ICs
        let lattice = E8Lattice::new();
        let template = LatticeField::from_lattice(&lattice);

        let mut params = AttractorSearchParams {
            num_initial_conditions: 5,
            max_steps: 500,
            convergence_epsilon: 1e-2, // Loose convergence for speed
            attractor_distance_threshold: 2.0,
            initial_amplitude: 0.5,
            rae_params: RAEParams::default_e8(),
        };
        params.rae_params.gamma = 0.5;
        params.rae_params.omega = 0.0;

        let analysis = find_attractors(&template, &params);

        assert_eq!(analysis.assignments.len(), 5);
        // Should have found at least 1 attractor
        assert!(!analysis.attractors.is_empty());
    }
}
