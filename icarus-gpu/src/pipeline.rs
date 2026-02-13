// Copyright (c) 2025-2026 brdigetrlol. All rights reserved.
// SPDX-License-Identifier: LicenseRef-Icarus-Proprietary
// See LICENSE in the repository root for full license terms.

//! Compute pipeline with GPU/CPU backend abstraction
//!
//! The ComputeBackend trait allows icarus-engine to work transparently
//! with either GPU or CPU execution.

use anyhow::Result;
use icarus_field::phase_field::LatticeField;
use icarus_field::rae::RAEParams;

/// Abstraction over GPU and CPU compute backends.
pub trait ComputeBackend: Send {
    /// Perform N steps of the RAE PDE on the given field (in-place).
    fn rae_step(&mut self, field: &mut LatticeField, params: &RAEParams, num_steps: u64) -> Result<()>;

    /// Compute the free energy (total, kinetic, potential).
    fn free_energy(
        &mut self,
        field: &LatticeField,
        params: &icarus_field::free_energy::FreeEnergyParams,
    ) -> Result<(f32, f32, f32)>;

    /// Update the metric tensor in-place.
    ///
    /// Applies `g[i] += -alpha * grad[i] + beta * ricci[i]` with clamping to
    /// `[eps_min, eps_max]`. All arrays are packed symmetric (upper-triangle),
    /// `components_per_site` elements per site.
    fn metric_update(
        &mut self,
        metric_data: &mut [f32],
        grad_data: &[f32],
        ricci_data: &[f32],
        num_sites: usize,
        components_per_site: usize,
        alpha: f32,
        beta: f32,
        eps_min: f32,
        eps_max: f32,
    ) -> Result<()>;

    /// Dense matrix-vector multiply for inter-layer transfer.
    ///
    /// Computes `y = W * x` for both real and imaginary components.
    /// `weights` is row-major `(dst_n × src_n)`.
    /// Returns `(y_re, y_im)` each of length `dst_n`.
    fn transfer_matvec(
        &mut self,
        weights: &[f32],
        source_re: &[f32],
        source_im: &[f32],
        dst_n: usize,
        src_n: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)>;

    /// Name of this backend (for logging).
    fn name(&self) -> &str;
}

/// CPU backend — delegates directly to icarus-field reference implementations.
pub struct CpuBackend;

impl ComputeBackend for CpuBackend {
    fn rae_step(&mut self, field: &mut LatticeField, params: &RAEParams, num_steps: u64) -> Result<()> {
        let mut solver = icarus_field::rae::RAESolver::new(params.clone(), field.num_sites);
        solver.run(field, num_steps);
        Ok(())
    }

    fn free_energy(
        &mut self,
        field: &LatticeField,
        params: &icarus_field::free_energy::FreeEnergyParams,
    ) -> Result<(f32, f32, f32)> {
        Ok(icarus_field::free_energy::free_energy(field, params))
    }

    fn metric_update(
        &mut self,
        metric_data: &mut [f32],
        grad_data: &[f32],
        ricci_data: &[f32],
        num_sites: usize,
        components_per_site: usize,
        alpha: f32,
        beta: f32,
        eps_min: f32,
        eps_max: f32,
    ) -> Result<()> {
        for i in 0..num_sites {
            let base = i * components_per_site;
            for c in 0..components_per_site {
                let idx = base + c;
                let g = metric_data[idx];
                let dg = -alpha * grad_data[idx] + beta * ricci_data[idx];
                let mut g_new = g + dg;
                if g_new < eps_min {
                    g_new = eps_min;
                }
                if g_new > eps_max {
                    g_new = eps_max;
                }
                metric_data[idx] = g_new;
            }
        }
        Ok(())
    }

    fn transfer_matvec(
        &mut self,
        weights: &[f32],
        source_re: &[f32],
        source_im: &[f32],
        dst_n: usize,
        src_n: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let mut y_re = vec![0.0f32; dst_n];
        let mut y_im = vec![0.0f32; dst_n];
        for i in 0..dst_n {
            let row_base = i * src_n;
            let mut sum_re = 0.0f32;
            let mut sum_im = 0.0f32;
            for j in 0..src_n {
                let w = weights[row_base + j];
                sum_re += w * source_re[j];
                sum_im += w * source_im[j];
            }
            y_re[i] = sum_re;
            y_im[i] = sum_im;
        }
        Ok((y_re, y_im))
    }

    fn name(&self) -> &str {
        "CPU"
    }
}

/// GPU backend — uses CUDA kernels for all compute.
pub struct GpuBackend {
    pub kernels: crate::kernels::IcarusKernels,
}

impl GpuBackend {
    pub fn new(device_id: usize) -> Result<Self> {
        let kernels = crate::kernels::IcarusKernels::new(device_id)?;
        Ok(Self { kernels })
    }
}

impl ComputeBackend for GpuBackend {
    fn rae_step(&mut self, field: &mut LatticeField, params: &RAEParams, num_steps: u64) -> Result<()> {
        self.kernels.rae_step(field, params, num_steps)
    }

    fn free_energy(
        &mut self,
        field: &LatticeField,
        params: &icarus_field::free_energy::FreeEnergyParams,
    ) -> Result<(f32, f32, f32)> {
        self.kernels.free_energy(field, params)
    }

    fn metric_update(
        &mut self,
        metric_data: &mut [f32],
        grad_data: &[f32],
        ricci_data: &[f32],
        num_sites: usize,
        components_per_site: usize,
        alpha: f32,
        beta: f32,
        eps_min: f32,
        eps_max: f32,
    ) -> Result<()> {
        self.kernels.metric_update(
            metric_data,
            grad_data,
            ricci_data,
            num_sites,
            components_per_site,
            alpha,
            beta,
            eps_min,
            eps_max,
        )
    }

    fn transfer_matvec(
        &mut self,
        weights: &[f32],
        source_re: &[f32],
        source_im: &[f32],
        dst_n: usize,
        src_n: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        self.kernels.transfer_matvec(weights, source_re, source_im, dst_n, src_n)
    }

    fn name(&self) -> &str {
        "CUDA"
    }
}
