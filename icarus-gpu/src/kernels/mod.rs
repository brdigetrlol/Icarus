//! CUDA kernel manager for the Icarus EMC
//!
//! Compiles and loads all PTX kernels, provides high-level wrappers
//! that handle upload/download and kernel dispatch.

pub mod rae;
pub mod metric;
pub mod transfer;
pub mod reduction;

use anyhow::{Context, Result};
use cudarc::driver::{CudaContext, CudaModule, CudaStream};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use std::sync::Arc;

use crate::buffers::GpuFieldBuffers;
use crate::device::{find_cuda_include, IcarusDevice};
use icarus_field::free_energy::FreeEnergyParams;
use icarus_field::phase_field::LatticeField;
use icarus_field::rae::{IntegratorMethod, RAEParams};

/// All compiled CUDA kernels for Icarus.
pub struct IcarusKernels {
    device: IcarusDevice,
    rae_module: Arc<CudaModule>,
    rae_si_module: Arc<CudaModule>,
    reduction_module: Arc<CudaModule>,
    metric_module: Arc<CudaModule>,
    transfer_module: Arc<CudaModule>,
}

impl IcarusKernels {
    /// Compile all PTX kernels and load them.
    pub fn new(device_id: usize) -> Result<Self> {
        let device = IcarusDevice::new(device_id)?;
        let ctx = device.ctx().clone();

        let cuda_include = find_cuda_include();
        let opts = CompileOptions {
            arch: Some("sm_120"),
            include_paths: cuda_include.into_iter().collect(),
            ..Default::default()
        };

        let rae_ptx = compile_ptx_with_opts(rae::RAE_KERNEL_SRC, opts.clone())
            .map_err(|e| anyhow::anyhow!("Failed to compile RAE kernel: {:?}", e))?;
        let rae_module = ctx
            .load_module(rae_ptx)
            .context("Failed to load RAE module")?;

        let rae_si_ptx =
            compile_ptx_with_opts(rae::RAE_SEMI_IMPLICIT_KERNEL_SRC, opts.clone())
                .map_err(|e| {
                    anyhow::anyhow!("Failed to compile RAE semi-implicit kernel: {:?}", e)
                })?;
        let rae_si_module = ctx
            .load_module(rae_si_ptx)
            .context("Failed to load RAE semi-implicit module")?;

        let reduction_ptx = compile_ptx_with_opts(reduction::REDUCTION_KERNEL_SRC, opts.clone())
            .map_err(|e| anyhow::anyhow!("Failed to compile reduction kernel: {:?}", e))?;
        let reduction_module = ctx
            .load_module(reduction_ptx)
            .context("Failed to load reduction module")?;

        let metric_ptx = compile_ptx_with_opts(metric::METRIC_KERNEL_SRC, opts.clone())
            .map_err(|e| anyhow::anyhow!("Failed to compile metric kernel: {:?}", e))?;
        let metric_module = ctx
            .load_module(metric_ptx)
            .context("Failed to load metric module")?;

        let transfer_ptx = compile_ptx_with_opts(transfer::TRANSFER_KERNEL_SRC, opts.clone())
            .map_err(|e| anyhow::anyhow!("Failed to compile transfer kernel: {:?}", e))?;
        let transfer_module = ctx
            .load_module(transfer_ptx)
            .context("Failed to load transfer module")?;

        Ok(Self {
            device,
            rae_module,
            rae_si_module,
            reduction_module,
            metric_module,
            transfer_module,
        })
    }

    pub fn ctx(&self) -> &Arc<CudaContext> {
        self.device.ctx()
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        self.device.stream()
    }

    /// Run N RAE steps on the GPU, uploading/downloading as needed.
    pub fn rae_step(
        &self,
        field: &mut LatticeField,
        params: &RAEParams,
        num_steps: u64,
    ) -> Result<()> {
        let stream = self.stream().clone();
        let bufs = GpuFieldBuffers::from_field(&stream, field)?;

        // Allocate scratch buffers for dz/dt
        let mut dz_re = crate::buffers::GpuBuffer::<f32>::zeros(&stream, field.num_sites)?;
        let mut dz_im = crate::buffers::GpuBuffer::<f32>::zeros(&stream, field.num_sites)?;

        let n = field.num_sites as i32;
        let target_sq = params.energy_params.target_amplitude * params.energy_params.target_amplitude;
        let module = match params.method {
            IntegratorMethod::Euler => &self.rae_module,
            IntegratorMethod::SemiImplicit => &self.rae_si_module,
        };

        for _ in 0..num_steps {
            rae::launch_rae_step(
                &stream,
                module,
                &bufs,
                &mut dz_re,
                &mut dz_im,
                n,
                params.dt,
                params.omega,
                params.gamma,
                params.energy_params.kinetic_weight,
                params.energy_params.potential_weight,
                target_sq,
                params.method,
            )?;
        }

        bufs.download_to_field(&stream, field)?;
        crate::memory::sync(&stream)?;
        Ok(())
    }

    /// Compute free energy on the GPU.
    pub fn free_energy(
        &self,
        field: &LatticeField,
        params: &FreeEnergyParams,
    ) -> Result<(f32, f32, f32)> {
        let stream = self.stream().clone();
        let bufs = GpuFieldBuffers::from_field(&stream, field)?;
        let target_sq = params.target_amplitude * params.target_amplitude;

        let kinetic = reduction::launch_kinetic_energy(
            &stream,
            &self.reduction_module,
            &bufs,
            params.kinetic_weight,
        )?;

        let potential = reduction::launch_potential_energy(
            &stream,
            &self.reduction_module,
            &bufs,
            params.potential_weight,
            target_sq,
        )?;

        Ok((kinetic + potential, kinetic, potential))
    }

    /// Run metric tensor update on the GPU.
    ///
    /// Applies `g += -alpha * grad + beta * ricci` with eigenvalue clamping.
    /// `metric_data` is modified in-place.
    pub fn metric_update(
        &self,
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
        let stream = self.stream().clone();

        let mut metric_buf = crate::buffers::GpuBuffer::from_host(&stream, metric_data)?;
        let grad_buf = crate::buffers::GpuBuffer::from_host(&stream, grad_data)?;
        let ricci_buf = crate::buffers::GpuBuffer::from_host(&stream, ricci_data)?;

        metric::launch_metric_update(
            &stream,
            &self.metric_module,
            &mut metric_buf,
            &grad_buf,
            &ricci_buf,
            num_sites as i32,
            components_per_site as i32,
            alpha,
            beta,
            eps_min,
            eps_max,
        )?;

        let result = metric_buf.to_host(&stream)?;
        metric_data.copy_from_slice(&result);
        crate::memory::sync(&stream)?;
        Ok(())
    }

    /// Run inter-layer transfer (dense matvec) on the GPU.
    ///
    /// Computes y = W * x for both real and imaginary components.
    /// Returns (y_re, y_im) each of length `dst_n`.
    pub fn transfer_matvec(
        &self,
        weights: &[f32],
        source_re: &[f32],
        source_im: &[f32],
        dst_n: usize,
        src_n: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let stream = self.stream().clone();

        let w_buf = crate::buffers::GpuBuffer::from_host(&stream, weights)?;
        let x_re_buf = crate::buffers::GpuBuffer::from_host(&stream, source_re)?;
        let x_im_buf = crate::buffers::GpuBuffer::from_host(&stream, source_im)?;
        let mut y_re_buf = crate::buffers::GpuBuffer::<f32>::zeros(&stream, dst_n)?;
        let mut y_im_buf = crate::buffers::GpuBuffer::<f32>::zeros(&stream, dst_n)?;

        transfer::launch_transfer_matvec(
            &stream,
            &self.transfer_module,
            &w_buf,
            &x_re_buf,
            &x_im_buf,
            &mut y_re_buf,
            &mut y_im_buf,
            dst_n as i32,
            src_n as i32,
        )?;

        let y_re = y_re_buf.to_host(&stream)?;
        let y_im = y_im_buf.to_host(&stream)?;
        crate::memory::sync(&stream)?;
        Ok((y_re, y_im))
    }
}
