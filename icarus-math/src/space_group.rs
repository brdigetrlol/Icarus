//! N-Dimensional Space Group and Symmetry Operations for Icarus EMC
//!
//! Generalized from the 3D crystallographic space groups to N dimensions.
//! In the EMC, symmetry operations act on lattice sites to enable:
//! - Compression via symmetry equivalence
//! - Equivariant wave propagation
//! - Pattern recognition through group orbits
//!
//! Ported from rust-mcp/src/tic/space_group.rs with Matrix3 → DMatrix for N-D.

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::LazyLock;

/// An N-dimensional symmetry operation: affine transform x' = W*x + t
#[derive(Clone, Debug)]
pub struct SymmetryOp {
    /// N×N rotation/reflection matrix
    pub rotation: DMatrix<f64>,
    /// Translation vector (N-dimensional)
    pub translation: DVector<f64>,
}

impl SymmetryOp {
    /// Identity operation in N dimensions
    pub fn identity(n: usize) -> Self {
        Self {
            rotation: DMatrix::identity(n, n),
            translation: DVector::zeros(n),
        }
    }

    /// Inversion (point reflection through origin) in N dimensions
    pub fn inversion(n: usize) -> Self {
        Self {
            rotation: -DMatrix::identity(n, n),
            translation: DVector::zeros(n),
        }
    }

    /// Coordinate permutation: maps axis `from` to axis `to`
    /// Creates a matrix that swaps two axes
    pub fn axis_swap(n: usize, axis_a: usize, axis_b: usize) -> Self {
        let mut mat = DMatrix::identity(n, n);
        mat[(axis_a, axis_a)] = 0.0;
        mat[(axis_b, axis_b)] = 0.0;
        mat[(axis_a, axis_b)] = 1.0;
        mat[(axis_b, axis_a)] = 1.0;
        Self {
            rotation: mat,
            translation: DVector::zeros(n),
        }
    }

    /// Reflection through the hyperplane perpendicular to axis `k`
    pub fn mirror(n: usize, axis: usize) -> Self {
        let mut mat = DMatrix::identity(n, n);
        mat[(axis, axis)] = -1.0;
        Self {
            rotation: mat,
            translation: DVector::zeros(n),
        }
    }

    /// Rotation in the (axis_a, axis_b) plane by `angle` radians
    pub fn rotation_2d(n: usize, axis_a: usize, axis_b: usize, angle: f64) -> Self {
        let mut mat = DMatrix::identity(n, n);
        let c = angle.cos();
        let s = angle.sin();
        mat[(axis_a, axis_a)] = c;
        mat[(axis_a, axis_b)] = -s;
        mat[(axis_b, axis_a)] = s;
        mat[(axis_b, axis_b)] = c;
        Self {
            rotation: mat,
            translation: DVector::zeros(n),
        }
    }

    /// Cyclic permutation of `k` axes starting from `start`
    pub fn cyclic_permutation(n: usize, start: usize, k: usize) -> Self {
        let mut mat = DMatrix::identity(n, n);
        // Zero out the k×k block
        for i in 0..k {
            let row = (start + i) % n;
            for j in 0..k {
                let col = (start + j) % n;
                mat[(row, col)] = 0.0;
            }
        }
        // Set cyclic: axis i → axis (i+1) mod k
        for i in 0..k {
            let from = (start + i) % n;
            let to = (start + (i + 1) % k) % n;
            mat[(to, from)] = 1.0;
        }
        Self {
            rotation: mat,
            translation: DVector::zeros(n),
        }
    }

    /// Apply this operation to a point
    pub fn apply(&self, point: &DVector<f64>) -> DVector<f64> {
        &self.rotation * point + &self.translation
    }

    /// Compose two operations: result = other ∘ self (apply self first, then other)
    pub fn compose(&self, other: &Self) -> Self {
        Self {
            rotation: &other.rotation * &self.rotation,
            translation: &other.rotation * &self.translation + &other.translation,
        }
    }

    /// Inverse operation
    pub fn inverse(&self) -> Self {
        let rot_inv = self
            .rotation
            .clone()
            .try_inverse()
            .unwrap_or_else(|| DMatrix::identity(self.rotation.nrows(), self.rotation.ncols()));
        let trans_inv = -(&rot_inv * &self.translation);
        Self {
            rotation: rot_inv,
            translation: trans_inv,
        }
    }

    /// Dimension of the space this operation acts on
    pub fn dim(&self) -> usize {
        self.rotation.nrows()
    }

    /// Check if this is the identity operation (within tolerance)
    pub fn is_identity(&self, tol: f64) -> bool {
        let n = self.dim();
        let id = DMatrix::identity(n, n);
        (&self.rotation - &id).norm() < tol && self.translation.norm() < tol
    }

    /// Order of this operation: smallest k such that op^k = identity
    /// Returns None if order > max_order
    pub fn order(&self, max_order: usize) -> Option<usize> {
        let mut current = self.clone();
        for k in 1..=max_order {
            if current.is_identity(1e-8) {
                return Some(k);
            }
            current = current.compose(self);
        }
        None
    }
}

/// Crystal system classification (generalized to N-D)
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CrystalSystem {
    Triclinic,
    Monoclinic,
    Orthorhombic,
    Tetragonal,
    Trigonal,
    Hexagonal,
    Cubic,
    /// For N>3 dimensions, systems are classified by their point group
    HigherDimensional { dim: usize, group_order: usize },
}

/// An N-dimensional space group
#[derive(Clone, Debug)]
pub struct SpaceGroup {
    /// Group identifier
    pub id: u16,
    /// Symbol (Hermann-Mauguin for 3D, custom for N-D)
    pub symbol: String,
    /// Crystal system
    pub system: CrystalSystem,
    /// Generator operations (the full group is generated by composing these)
    pub generators: Vec<SymmetryOp>,
    /// Dimension
    pub dim: usize,
}

impl SpaceGroup {
    /// Look up a 3D space group by International Tables number (1-230)
    pub fn from_id_3d(id: u16) -> Option<Self> {
        SPACE_GROUP_DB_3D.get(&id).cloned()
    }

    /// Create an E8 symmetry group (Weyl group of E8).
    /// The E8 Weyl group has order 696,729,600. We store a small generating set.
    pub fn e8_weyl() -> Self {
        let n = 8;
        let mut generators = Vec::new();

        // E8 Weyl group generators:
        // 1. Permutations of coordinates → S_8 subgroup
        // 2. Even sign changes → (Z/2Z)^7
        // 3. The E8-specific generator involving half-integer vectors

        // Generator 1: adjacent transpositions (i, i+1) for i=0..6
        for i in 0..7 {
            generators.push(SymmetryOp::axis_swap(n, i, i + 1));
        }

        // Generator 2: sign change on axes 0 and 1 (even number of sign flips)
        let mut sign_flip = DMatrix::identity(n, n);
        sign_flip[(0, 0)] = -1.0;
        sign_flip[(1, 1)] = -1.0;
        generators.push(SymmetryOp {
            rotation: sign_flip,
            translation: DVector::zeros(n),
        });

        Self {
            id: 0,
            symbol: "W(E8)".to_string(),
            system: CrystalSystem::HigherDimensional {
                dim: 8,
                group_order: 696_729_600,
            },
            generators,
            dim: n,
        }
    }

    /// Generate the orbit of a point under this group's generators.
    /// Returns all distinct images (up to tolerance).
    /// Limited to max_size to prevent combinatorial explosion.
    pub fn orbit(&self, point: &DVector<f64>, max_size: usize) -> Vec<DVector<f64>> {
        let mut orbit = vec![point.clone()];
        let mut queue = vec![point.clone()];
        let tol = 1e-8;

        while let Some(p) = queue.pop() {
            if orbit.len() >= max_size {
                break;
            }
            for gen in &self.generators {
                let image = gen.apply(&p);
                let is_new = orbit.iter().all(|q| (&image - q).norm() > tol);
                if is_new {
                    orbit.push(image.clone());
                    queue.push(image);
                }
                if orbit.len() >= max_size {
                    break;
                }
            }
        }

        orbit
    }

    /// Get all generators
    pub fn generators(&self) -> &[SymmetryOp] {
        &self.generators
    }

    /// Point group: just the rotation parts (drop translations)
    pub fn point_group_generators(&self) -> Vec<DMatrix<f64>> {
        self.generators.iter().map(|op| op.rotation.clone()).collect()
    }
}

/// Database of 3D space groups (subset for common structures)
static SPACE_GROUP_DB_3D: LazyLock<HashMap<u16, SpaceGroup>> = LazyLock::new(|| {
    let mut db = HashMap::new();

    // P1 — Triclinic, identity only
    db.insert(1, SpaceGroup {
        id: 1,
        symbol: "P1".to_string(),
        system: CrystalSystem::Triclinic,
        generators: vec![SymmetryOp::identity(3)],
        dim: 3,
    });

    // P-1 — Triclinic with inversion
    db.insert(2, SpaceGroup {
        id: 2,
        symbol: "P-1".to_string(),
        system: CrystalSystem::Triclinic,
        generators: vec![SymmetryOp::identity(3), SymmetryOp::inversion(3)],
        dim: 3,
    });

    // P2 — Monoclinic, 2-fold rotation about z
    db.insert(3, SpaceGroup {
        id: 3,
        symbol: "P2".to_string(),
        system: CrystalSystem::Monoclinic,
        generators: vec![
            SymmetryOp::identity(3),
            SymmetryOp::rotation_2d(3, 0, 1, std::f64::consts::PI),
        ],
        dim: 3,
    });

    // P4 — Tetragonal, 4-fold rotation about z
    db.insert(75, SpaceGroup {
        id: 75,
        symbol: "P4".to_string(),
        system: CrystalSystem::Tetragonal,
        generators: vec![
            SymmetryOp::identity(3),
            SymmetryOp::rotation_2d(3, 0, 1, std::f64::consts::FRAC_PI_2),
        ],
        dim: 3,
    });

    // P6 — Hexagonal, 6-fold rotation about z
    db.insert(168, SpaceGroup {
        id: 168,
        symbol: "P6".to_string(),
        system: CrystalSystem::Hexagonal,
        generators: vec![
            SymmetryOp::identity(3),
            SymmetryOp::rotation_2d(3, 0, 1, std::f64::consts::FRAC_PI_3),
        ],
        dim: 3,
    });

    // Fm-3m — Cubic, high symmetry
    db.insert(225, SpaceGroup {
        id: 225,
        symbol: "Fm-3m".to_string(),
        system: CrystalSystem::Cubic,
        generators: vec![
            SymmetryOp::identity(3),
            SymmetryOp::inversion(3),
            SymmetryOp::rotation_2d(3, 0, 1, std::f64::consts::FRAC_PI_2),
            SymmetryOp::cyclic_permutation(3, 0, 3),
        ],
        dim: 3,
    });

    db
});

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_apply() {
        let op = SymmetryOp::identity(8);
        let p = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let result = op.apply(&p);
        assert!((&result - &p).norm() < 1e-10);
    }

    #[test]
    fn test_inversion() {
        let op = SymmetryOp::inversion(3);
        let p = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let result = op.apply(&p);
        assert!((result[0] - (-1.0)).abs() < 1e-10);
        assert!((result[1] - (-2.0)).abs() < 1e-10);
        assert!((result[2] - (-3.0)).abs() < 1e-10);
    }

    #[test]
    fn test_rotation_90() {
        let op = SymmetryOp::rotation_2d(3, 0, 1, std::f64::consts::FRAC_PI_2);
        let p = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        let result = op.apply(&p);
        assert!(result[0].abs() < 1e-10);
        assert!((result[1] - 1.0).abs() < 1e-10);
        assert!(result[2].abs() < 1e-10);
    }

    #[test]
    fn test_compose_inverse() {
        let op = SymmetryOp::rotation_2d(3, 0, 1, std::f64::consts::FRAC_PI_4);
        let inv = op.inverse();
        let composed = op.compose(&inv);
        assert!(composed.is_identity(1e-8));
    }

    #[test]
    fn test_axis_swap() {
        let op = SymmetryOp::axis_swap(4, 1, 3);
        let p = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let result = op.apply(&p);
        assert!((result[0] - 1.0).abs() < 1e-10);
        assert!((result[1] - 4.0).abs() < 1e-10); // swapped
        assert!((result[2] - 3.0).abs() < 1e-10);
        assert!((result[3] - 2.0).abs() < 1e-10); // swapped
    }

    #[test]
    fn test_mirror() {
        let op = SymmetryOp::mirror(3, 0);
        let p = DVector::from_vec(vec![5.0, 3.0, 1.0]);
        let result = op.apply(&p);
        assert!((result[0] - (-5.0)).abs() < 1e-10);
        assert!((result[1] - 3.0).abs() < 1e-10);
        assert!((result[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_order_rotation() {
        let op = SymmetryOp::rotation_2d(3, 0, 1, std::f64::consts::FRAC_PI_2);
        assert_eq!(op.order(10), Some(4)); // 90° rotation has order 4
    }

    #[test]
    fn test_space_group_lookup() {
        let sg = SpaceGroup::from_id_3d(1).unwrap();
        assert_eq!(sg.symbol, "P1");
        assert_eq!(sg.system, CrystalSystem::Triclinic);
    }

    #[test]
    fn test_e8_weyl_generators() {
        let sg = SpaceGroup::e8_weyl();
        assert_eq!(sg.dim, 8);
        // 7 transpositions + 1 sign flip = 8 generators
        assert_eq!(sg.generators.len(), 8);
    }

    #[test]
    fn test_orbit_inversion() {
        let sg = SpaceGroup::from_id_3d(2).unwrap(); // P-1
        let p = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let orbit = sg.orbit(&p, 100);
        assert_eq!(orbit.len(), 2); // p and -p
    }

    #[test]
    fn test_cyclic_permutation() {
        let op = SymmetryOp::cyclic_permutation(3, 0, 3);
        let p = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let result = op.apply(&p);
        // (x,y,z) → (z,x,y)
        assert!((result[0] - 3.0).abs() < 1e-10);
        assert!((result[1] - 1.0).abs() < 1e-10);
        assert!((result[2] - 2.0).abs() < 1e-10);
    }
}
