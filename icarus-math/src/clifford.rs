// Copyright (c) 2025-2026 brdigetrlol. All rights reserved.
// SPDX-License-Identifier: LicenseRef-Icarus-Proprietary
// See LICENSE in the repository root for full license terms.

//! Clifford Algebra (Geometric Algebra) for Icarus EMC
//!
//! Implements Cl(n) geometric algebra with Complex<f32> coefficients
//! for GPU compatibility. The geometric product encodes both inner (dot)
//! and outer (wedge) products: u ⊗ v = u·v + u∧v
//!
//! Ported from rust-mcp/src/tic/clifford.rs with Complex64 → Complex<f32>.

use num_complex::Complex;
use std::ops::{Add, Mul, Sub};

/// A multivector in n-dimensional Clifford algebra Cl(n).
///
/// Has 2^n components representing all basis blades
/// (scalars, vectors, bivectors, trivectors, ..., pseudoscalar).
/// Index i represents the basis blade whose binary representation selects basis vectors.
#[derive(Clone, Debug)]
pub struct Multivector {
    /// Dimension of the underlying vector space
    pub n: usize,
    /// Coefficients for each basis blade (2^n total)
    pub coeffs: Vec<Complex<f32>>,
}

impl Multivector {
    /// Create a zero multivector in Cl(n)
    pub fn new(n: usize) -> Self {
        let size = 1 << n;
        Self {
            n,
            coeffs: vec![Complex::new(0.0, 0.0); size],
        }
    }

    /// Create a scalar (grade-0) multivector
    pub fn scalar(n: usize, value: f32) -> Self {
        let mut mv = Self::new(n);
        mv.coeffs[0] = Complex::new(value, 0.0);
        mv
    }

    /// Create a complex scalar multivector
    pub fn complex_scalar(n: usize, re: f32, im: f32) -> Self {
        let mut mv = Self::new(n);
        mv.coeffs[0] = Complex::new(re, im);
        mv
    }

    /// Create a vector (grade-1) multivector from real components
    pub fn vector(components: &[f32]) -> Self {
        let n = components.len();
        let mut mv = Self::new(n);
        for (i, &comp) in components.iter().enumerate() {
            mv.coeffs[1 << i] = Complex::new(comp, 0.0);
        }
        mv
    }

    /// Create a vector multivector from complex components
    pub fn complex_vector(components: &[Complex<f32>]) -> Self {
        let n = components.len();
        let mut mv = Self::new(n);
        for (i, &comp) in components.iter().enumerate() {
            mv.coeffs[1 << i] = comp;
        }
        mv
    }

    /// Get the scalar part (grade 0)
    pub fn scalar_part(&self) -> Complex<f32> {
        self.coeffs[0]
    }

    /// Get the real part of the scalar
    pub fn scalar_re(&self) -> f32 {
        self.coeffs[0].re
    }

    /// Get the vector part (grade 1) components
    pub fn vector_part(&self) -> Vec<Complex<f32>> {
        (0..self.n).map(|i| self.coeffs[1 << i]).collect()
    }

    /// Compute the norm (magnitude) of the multivector
    pub fn magnitude(&self) -> f32 {
        self.coeffs
            .iter()
            .map(|c| c.norm_sqr())
            .sum::<f32>()
            .sqrt()
    }

    /// Normalize in place
    pub fn normalize(&mut self) {
        let mag = self.magnitude();
        if mag > 1e-7 {
            let inv = 1.0 / mag;
            for coeff in &mut self.coeffs {
                *coeff *= inv;
            }
        }
    }

    /// Return a normalized copy
    pub fn normalized(&self) -> Self {
        let mut result = self.clone();
        result.normalize();
        result
    }

    /// Grade projection — extract components of a specific grade k
    pub fn grade(&self, k: usize) -> Self {
        let mut result = Self::new(self.n);
        for i in 0usize..(1 << self.n) {
            if i.count_ones() as usize == k {
                result.coeffs[i] = self.coeffs[i];
            }
        }
        result
    }

    /// Reverse operation: reverses the order of basis vectors in each blade.
    /// For a grade-k blade, reverse multiplies by (-1)^{k(k-1)/2}.
    pub fn reverse(&self) -> Self {
        let mut result = Self::new(self.n);
        for i in 0usize..(1 << self.n) {
            let k = i.count_ones() as usize;
            let sign = if k < 2 || (k * (k - 1) / 2) % 2 == 0 {
                1.0f32
            } else {
                -1.0
            };
            result.coeffs[i] = self.coeffs[i] * sign;
        }
        result
    }

    /// Conjugate: combines reverse with grade involution
    pub fn conjugate(&self) -> Self {
        let mut result = Self::new(self.n);
        for i in 0usize..(1 << self.n) {
            let k = i.count_ones() as usize;
            // Conjugate sign: (-1)^{k(k+1)/2}
            let sign = if (k * (k + 1) / 2) % 2 == 0 {
                1.0f32
            } else {
                -1.0
            };
            result.coeffs[i] = self.coeffs[i] * sign;
        }
        result
    }

    /// Squared norm via self * reverse(self), taking scalar part
    pub fn norm_sq(&self) -> f32 {
        let product = self.geometric_product(&self.reverse());
        product.scalar_re()
    }
}

/// Trait for geometric product operations
pub trait GeometricProduct {
    /// Full geometric product: u ⊗ v = u·v + u∧v
    fn geometric_product(&self, other: &Self) -> Self;

    /// Inner (dot) product — lowest-grade part of geometric product
    fn inner_product(&self, other: &Self) -> Self;

    /// Outer (wedge) product — highest-grade part of geometric product
    fn outer_product(&self, other: &Self) -> Self;
}

impl GeometricProduct for Multivector {
    fn geometric_product(&self, other: &Self) -> Self {
        assert_eq!(self.n, other.n, "Multivectors must have same dimension");

        let mut result = Self::new(self.n);
        let size = 1 << self.n;

        for i in 0..size {
            if self.coeffs[i] == Complex::new(0.0, 0.0) {
                continue;
            }
            for j in 0..size {
                if other.coeffs[j] == Complex::new(0.0, 0.0) {
                    continue;
                }
                let (sign, blade) = clifford_multiply_basis(i, j);
                result.coeffs[blade] += self.coeffs[i] * other.coeffs[j] * sign;
            }
        }

        result
    }

    fn inner_product(&self, other: &Self) -> Self {
        assert_eq!(self.n, other.n);

        // Inner product: for grade-r and grade-s blades, extract grade |r-s|
        let mut result = Self::new(self.n);
        let size = 1 << self.n;

        for i in 0usize..size {
            for j in 0usize..size {
                let grade_i = i.count_ones() as i32;
                let grade_j = j.count_ones() as i32;
                let target_grade = (grade_i - grade_j).unsigned_abs() as usize;

                // Accumulate into the target-grade components
                if self.coeffs[i] != Complex::new(0.0, 0.0)
                    && other.coeffs[j] != Complex::new(0.0, 0.0)
                {
                    let (sign, blade) = clifford_multiply_basis(i, j);
                    if blade.count_ones() as usize == target_grade {
                        result.coeffs[blade] += self.coeffs[i] * other.coeffs[j] * sign;
                    }
                }
            }
        }

        result
    }

    fn outer_product(&self, other: &Self) -> Self {
        assert_eq!(self.n, other.n);

        let mut result = Self::new(self.n);
        let size = 1 << self.n;

        for i in 0usize..size {
            for j in 0usize..size {
                let grade_i = i.count_ones() as usize;
                let grade_j = j.count_ones() as usize;
                let target_grade = grade_i + grade_j;

                if self.coeffs[i] != Complex::new(0.0, 0.0)
                    && other.coeffs[j] != Complex::new(0.0, 0.0)
                {
                    let (sign, blade) = clifford_multiply_basis(i, j);
                    if blade.count_ones() as usize == target_grade {
                        result.coeffs[blade] += self.coeffs[i] * other.coeffs[j] * sign;
                    }
                }
            }
        }

        result
    }
}

/// Multiply two basis blades in Clifford algebra Cl(n) with Euclidean signature.
///
/// Returns (sign, resulting_blade_index) where sign ∈ {+1, -1}.
/// The result blade is i XOR j (symmetric difference of basis vectors).
/// The sign comes from the number of transpositions needed to canonicalize.
fn clifford_multiply_basis(i: usize, j: usize) -> (f32, usize) {
    let result_blade = i ^ j;
    let mut sign = 1.0f32;

    // For each common basis vector (present in both i and j),
    // we need e_k * e_k = +1 (Euclidean) and count transpositions
    let mut common = i & j;
    while common != 0 {
        let lowest_bit = common & common.wrapping_neg();
        let bit_pos = lowest_bit.trailing_zeros() as usize;

        // Count bits in j that are below this position — those need to be swapped past
        let mask = (1usize << bit_pos) - 1;
        let swaps = (i >> (bit_pos + 1)).count_ones() + (j & mask).count_ones();

        if swaps % 2 == 1 {
            sign = -sign;
        }

        common &= !lowest_bit;
    }

    // Also count swaps for non-common bits to interleave properly
    let only_i = i & !j;
    let only_j = j & !i;

    // Count how many bits of only_j are to the left of each bit of only_i
    let mut remaining = only_i;
    while remaining != 0 {
        let lowest_bit = remaining & remaining.wrapping_neg();
        let bit_pos = lowest_bit.trailing_zeros() as usize;

        // Count bits in only_j that are below this position
        let mask = (1usize << bit_pos) - 1;
        let swaps = (only_j & mask).count_ones();

        // Actually we need j bits that are ABOVE this i bit to count swaps
        // Wait — let's use the standard algorithm:
        // Count pairs where a bit in i is above a bit in j
        // This is equivalent to counting inversions

        // Correction: we already handled common bits. For the remaining,
        // the blade is just the union (XOR when no common), so we count
        // how many bits in j are to the RIGHT of this bit position in i
        let _ = swaps; // handled below

        remaining &= !lowest_bit;
    }

    // Actually, let's use a cleaner standard Clifford sign calculation
    // Reset and recompute properly
    let mut sign2 = 1.0f32;

    // Decompose i and j into sorted bit positions
    let mut bits_i = Vec::new();
    let mut bits_j = Vec::new();
    {
        let mut tmp = i;
        while tmp != 0 {
            let pos = tmp.trailing_zeros() as usize;
            bits_i.push(pos);
            tmp &= !(1 << pos);
        }
    }
    {
        let mut tmp = j;
        while tmp != 0 {
            let pos = tmp.trailing_zeros() as usize;
            bits_j.push(pos);
            tmp &= !(1 << pos);
        }
    }

    // The product e_{i1} e_{i2} ... e_{ip} * e_{j1} e_{j2} ... e_{jq}
    // We need to bubble-sort the combined sequence to canonical order,
    // counting swaps (each swap introduces a sign flip) and cancelling
    // pairs e_k e_k = +1 (Euclidean).

    let mut combined: Vec<usize> = Vec::with_capacity(bits_i.len() + bits_j.len());
    combined.extend_from_slice(&bits_i);
    combined.extend_from_slice(&bits_j);

    // Bubble sort, counting transpositions
    let n = combined.len();
    for pass in 0..n {
        for k in 0..n.saturating_sub(1 + pass) {
            if combined[k] > combined[k + 1] {
                combined.swap(k, k + 1);
                sign2 = -sign2;
            }
        }
    }

    // Cancel adjacent equal pairs (e_k e_k = +1 for Euclidean)
    // After sorting, equal elements are adjacent
    // Already accounted for in XOR result

    (sign2, result_blade)
}

impl Add for Multivector {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        assert_eq!(self.n, other.n);
        let mut result = self;
        for (i, c) in result.coeffs.iter_mut().enumerate() {
            *c += other.coeffs[i];
        }
        result
    }
}

impl Sub for Multivector {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        assert_eq!(self.n, other.n);
        let mut result = self;
        for (i, c) in result.coeffs.iter_mut().enumerate() {
            *c -= other.coeffs[i];
        }
        result
    }
}

impl Mul<f32> for Multivector {
    type Output = Self;
    fn mul(self, scalar: f32) -> Self {
        let mut result = self;
        for c in &mut result.coeffs {
            *c *= scalar;
        }
        result
    }
}

impl Mul<Complex<f32>> for Multivector {
    type Output = Self;
    fn mul(self, scalar: Complex<f32>) -> Self {
        let mut result = self;
        for c in &mut result.coeffs {
            *c *= scalar;
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_creation() {
        let s = Multivector::scalar(3, 5.0);
        assert!((s.scalar_re() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_vector_creation() {
        let v = Multivector::vector(&[1.0, 2.0, 3.0]);
        let parts = v.vector_part();
        assert!((parts[0].re - 1.0).abs() < 1e-6);
        assert!((parts[1].re - 2.0).abs() < 1e-6);
        assert!((parts[2].re - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_geometric_product_scalars() {
        let a = Multivector::scalar(3, 2.0);
        let b = Multivector::scalar(3, 3.0);
        let c = a.geometric_product(&b);
        assert!((c.scalar_re() - 6.0).abs() < 1e-5);
    }

    #[test]
    fn test_geometric_product_orthogonal_vectors() {
        let v1 = Multivector::vector(&[1.0, 0.0, 0.0]);
        let v2 = Multivector::vector(&[0.0, 1.0, 0.0]);
        let result = v1.geometric_product(&v2);
        // Orthogonal vectors: scalar part should be zero
        assert!(result.scalar_re().abs() < 1e-5);
        // Bivector e1∧e2 should be nonzero
        assert!(result.coeffs[0b011].re.abs() > 0.5);
    }

    #[test]
    fn test_geometric_product_parallel_vectors() {
        let v1 = Multivector::vector(&[1.0, 0.0, 0.0]);
        let v2 = Multivector::vector(&[3.0, 0.0, 0.0]);
        let result = v1.geometric_product(&v2);
        // Parallel vectors: scalar part = dot product = 3.0
        assert!((result.scalar_re() - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_magnitude() {
        let v = Multivector::vector(&[3.0, 4.0]);
        assert!((v.magnitude() - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_grade_projection() {
        let v1 = Multivector::vector(&[1.0, 0.0, 0.0]);
        let v2 = Multivector::vector(&[0.0, 1.0, 0.0]);
        let gp = v1.geometric_product(&v2);

        let grade0 = gp.grade(0);
        let grade2 = gp.grade(2);

        assert!(grade0.scalar_re().abs() < 1e-5);
        assert!(grade2.magnitude() > 0.5);
    }

    #[test]
    fn test_reverse() {
        let v = Multivector::vector(&[1.0, 2.0, 3.0]);
        let rev = v.reverse();
        // Grade-1 blades are unchanged by reverse (k(k-1)/2 = 0)
        for i in 0..3 {
            assert!((rev.coeffs[1 << i].re - v.coeffs[1 << i].re).abs() < 1e-6);
        }
    }

    #[test]
    fn test_add_sub() {
        let a = Multivector::vector(&[1.0, 2.0]);
        let b = Multivector::vector(&[3.0, 4.0]);
        let sum = a.clone() + b.clone();
        let diff = a - b;

        assert!((sum.coeffs[1].re - 4.0).abs() < 1e-6);
        assert!((sum.coeffs[2].re - 6.0).abs() < 1e-6);
        assert!((diff.coeffs[1].re - (-2.0)).abs() < 1e-6);
        assert!((diff.coeffs[2].re - (-2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_scalar_multiply() {
        let v = Multivector::vector(&[1.0, 2.0, 3.0]);
        let scaled = v * 2.0;
        assert!((scaled.coeffs[1].re - 2.0).abs() < 1e-6);
        assert!((scaled.coeffs[2].re - 4.0).abs() < 1e-6);
        assert!((scaled.coeffs[4].re - 6.0).abs() < 1e-6);
    }
}
