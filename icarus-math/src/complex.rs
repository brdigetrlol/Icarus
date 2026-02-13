// Copyright (c) 2025-2026 brdigetrlol. All rights reserved.
// SPDX-License-Identifier: LicenseRef-Icarus-Proprietary
// See LICENSE in the repository root for full license terms.

//! Complex<f32> Structure-of-Arrays (SoA) Helpers for GPU-friendly layout
//!
//! GPU kernels operate on contiguous f32 arrays (values_re[], values_im[]),
//! not interleaved Complex<f32>. This module provides conversion utilities
//! and SoA operations that mirror what the CUDA kernels will do.

use num_complex::Complex;

/// Structure-of-Arrays representation of a complex field.
/// GPU-friendly: re[] and im[] are contiguous for coalesced memory access.
#[derive(Debug, Clone)]
pub struct ComplexSoA {
    pub re: Vec<f32>,
    pub im: Vec<f32>,
}

impl ComplexSoA {
    /// Create a zero-initialized SoA field with `n` elements
    pub fn zeros(n: usize) -> Self {
        Self {
            re: vec![0.0; n],
            im: vec![0.0; n],
        }
    }

    /// Create from separate real and imaginary arrays
    pub fn from_parts(re: Vec<f32>, im: Vec<f32>) -> Self {
        assert_eq!(re.len(), im.len(), "Real and imaginary arrays must match");
        Self { re, im }
    }

    /// Create from a slice of Complex<f32> (AoS → SoA conversion)
    pub fn from_aos(values: &[Complex<f32>]) -> Self {
        let n = values.len();
        let mut re = Vec::with_capacity(n);
        let mut im = Vec::with_capacity(n);
        for v in values {
            re.push(v.re);
            im.push(v.im);
        }
        Self { re, im }
    }

    /// Convert back to Array-of-Structures (SoA → AoS)
    pub fn to_aos(&self) -> Vec<Complex<f32>> {
        self.re
            .iter()
            .zip(self.im.iter())
            .map(|(&r, &i)| Complex::new(r, i))
            .collect()
    }

    /// Number of complex values
    pub fn len(&self) -> usize {
        self.re.len()
    }

    /// Whether empty
    pub fn is_empty(&self) -> bool {
        self.re.is_empty()
    }

    /// Get element at index as Complex<f32>
    pub fn get(&self, idx: usize) -> Complex<f32> {
        Complex::new(self.re[idx], self.im[idx])
    }

    /// Set element at index from Complex<f32>
    pub fn set(&mut self, idx: usize, val: Complex<f32>) {
        self.re[idx] = val.re;
        self.im[idx] = val.im;
    }

    /// Compute |z|² = re² + im² for each element
    pub fn norm_sq(&self) -> Vec<f32> {
        self.re
            .iter()
            .zip(self.im.iter())
            .map(|(&r, &i)| r * r + i * i)
            .collect()
    }

    /// Compute |z| for each element
    pub fn magnitudes(&self) -> Vec<f32> {
        self.norm_sq().iter().map(|&ns| ns.sqrt()).collect()
    }

    /// Element-wise addition: self + other
    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.len(), other.len());
        Self {
            re: self
                .re
                .iter()
                .zip(other.re.iter())
                .map(|(&a, &b)| a + b)
                .collect(),
            im: self
                .im
                .iter()
                .zip(other.im.iter())
                .map(|(&a, &b)| a + b)
                .collect(),
        }
    }

    /// Element-wise subtraction: self - other
    pub fn sub(&self, other: &Self) -> Self {
        assert_eq!(self.len(), other.len());
        Self {
            re: self
                .re
                .iter()
                .zip(other.re.iter())
                .map(|(&a, &b)| a - b)
                .collect(),
            im: self
                .im
                .iter()
                .zip(other.im.iter())
                .map(|(&a, &b)| a - b)
                .collect(),
        }
    }

    /// Element-wise complex multiplication: self * other
    /// (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    pub fn mul(&self, other: &Self) -> Self {
        assert_eq!(self.len(), other.len());
        let n = self.len();
        let mut re = Vec::with_capacity(n);
        let mut im = Vec::with_capacity(n);
        for idx in 0..n {
            let a = self.re[idx];
            let b = self.im[idx];
            let c = other.re[idx];
            let d = other.im[idx];
            re.push(a * c - b * d);
            im.push(a * d + b * c);
        }
        Self { re, im }
    }

    /// Scale all elements by a real scalar
    pub fn scale(&self, factor: f32) -> Self {
        Self {
            re: self.re.iter().map(|&r| r * factor).collect(),
            im: self.im.iter().map(|&i| i * factor).collect(),
        }
    }

    /// Scale by a complex scalar: z * (a + bi)
    pub fn scale_complex(&self, scalar: Complex<f32>) -> Self {
        let n = self.len();
        let mut re = Vec::with_capacity(n);
        let mut im = Vec::with_capacity(n);
        for idx in 0..n {
            let r = self.re[idx];
            let i = self.im[idx];
            re.push(r * scalar.re - i * scalar.im);
            im.push(r * scalar.im + i * scalar.re);
        }
        Self { re, im }
    }

    /// Fused multiply-add: self += scale * other (in-place)
    /// Common in PDE solvers: z_new = z_old + dt * dz_dt
    pub fn fma_inplace(&mut self, scale: f32, other: &Self) {
        assert_eq!(self.len(), other.len());
        for idx in 0..self.len() {
            self.re[idx] += scale * other.re[idx];
            self.im[idx] += scale * other.im[idx];
        }
    }

    /// Compute total energy: Σ |z_i|²
    pub fn total_energy(&self) -> f32 {
        self.norm_sq().iter().sum()
    }

    /// In-place element-wise multiply by i (rotate 90°): z → iz = -im + i*re
    pub fn mul_i_inplace(&mut self) {
        for idx in 0..self.len() {
            let r = self.re[idx];
            let i = self.im[idx];
            self.re[idx] = -i;
            self.im[idx] = r;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros() {
        let soa = ComplexSoA::zeros(10);
        assert_eq!(soa.len(), 10);
        assert!(soa.re.iter().all(|&x| x == 0.0));
        assert!(soa.im.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_roundtrip_aos_soa() {
        let aos = vec![
            Complex::new(1.0, 2.0),
            Complex::new(3.0, 4.0),
            Complex::new(5.0, 6.0),
        ];
        let soa = ComplexSoA::from_aos(&aos);
        let back = soa.to_aos();
        for (a, b) in aos.iter().zip(back.iter()) {
            assert!((a.re - b.re).abs() < 1e-6);
            assert!((a.im - b.im).abs() < 1e-6);
        }
    }

    #[test]
    fn test_norm_sq() {
        let soa = ComplexSoA::from_parts(vec![3.0, 0.0], vec![4.0, 1.0]);
        let ns = soa.norm_sq();
        assert!((ns[0] - 25.0).abs() < 1e-6);
        assert!((ns[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_complex_multiply() {
        // (1+2i)(3+4i) = (3-8) + (4+6)i = -5 + 10i
        let a = ComplexSoA::from_parts(vec![1.0], vec![2.0]);
        let b = ComplexSoA::from_parts(vec![3.0], vec![4.0]);
        let c = a.mul(&b);
        assert!((c.re[0] - (-5.0)).abs() < 1e-6);
        assert!((c.im[0] - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_fma_inplace() {
        let mut z = ComplexSoA::from_parts(vec![1.0, 2.0], vec![3.0, 4.0]);
        let dz = ComplexSoA::from_parts(vec![0.5, 0.5], vec![0.5, 0.5]);
        z.fma_inplace(2.0, &dz);
        assert!((z.re[0] - 2.0).abs() < 1e-6);
        assert!((z.re[1] - 3.0).abs() < 1e-6);
        assert!((z.im[0] - 4.0).abs() < 1e-6);
        assert!((z.im[1] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_total_energy() {
        let soa = ComplexSoA::from_parts(vec![1.0, 0.0], vec![0.0, 1.0]);
        assert!((soa.total_energy() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_mul_i() {
        let mut soa = ComplexSoA::from_parts(vec![3.0], vec![4.0]);
        soa.mul_i_inplace();
        // i*(3+4i) = -4 + 3i
        assert!((soa.re[0] - (-4.0)).abs() < 1e-6);
        assert!((soa.im[0] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_scale_complex() {
        let soa = ComplexSoA::from_parts(vec![1.0], vec![0.0]);
        let scaled = soa.scale_complex(Complex::new(0.0, 1.0));
        // 1.0 * i = i → re=0, im=1
        assert!(scaled.re[0].abs() < 1e-6);
        assert!((scaled.im[0] - 1.0).abs() < 1e-6);
    }
}
