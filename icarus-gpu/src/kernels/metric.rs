//! Metric tensor update CUDA kernel
//!
//! Implements geometrodynamic metric learning on the GPU:
//!   dg_μν/dt = -α · δL/δg^μν + β · R_μν

use anyhow::{Context, Result};
use cudarc::driver::{CudaModule, CudaStream, LaunchConfig, PushKernelArg};
use std::sync::Arc;

use crate::buffers::GpuBuffer;

/// CUDA C source for the metric tensor update kernel.
///
/// Updates packed upper-triangle metric components per site based on
/// the loss gradient and Ricci curvature regularization.
pub const METRIC_KERNEL_SRC: &str = r#"
// Metric tensor update: g_new = g - alpha * dL_dg + beta * ricci
// Each site stores dim*(dim+1)/2 packed upper-triangle components.
extern "C" __global__ void metric_update(
    float* __restrict__ metric_data,
    const float* __restrict__ grad_data,
    const float* __restrict__ ricci_data,
    int num_sites,
    int components_per_site,
    float alpha,
    float beta,
    float eps_min,
    float eps_max)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_sites) return;

    int base = i * components_per_site;
    for (int c = 0; c < components_per_site; c++) {
        int idx = base + c;
        float g = metric_data[idx];
        float dg = -alpha * grad_data[idx] + beta * ricci_data[idx];
        g += dg;

        // Eigenvalue pinching for diagonal elements
        // (For packed symmetric, diagonal elements are at indices
        //  0, dim, 2*dim-1, ... but we apply clamping to all for safety)
        if (g < eps_min) g = eps_min;
        if (g > eps_max) g = eps_max;

        metric_data[idx] = g;
    }
}
"#;

const BLOCK_SIZE: u32 = 256;

/// Launch the metric tensor update kernel.
///
/// Applies `g += -alpha * grad + beta * ricci` with clamping to [eps_min, eps_max].
/// `metric_buf` is modified in-place on the GPU.
pub fn launch_metric_update(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    metric_buf: &mut GpuBuffer<f32>,
    grad_buf: &GpuBuffer<f32>,
    ricci_buf: &GpuBuffer<f32>,
    num_sites: i32,
    components_per_site: i32,
    alpha: f32,
    beta: f32,
    eps_min: f32,
    eps_max: f32,
) -> Result<()> {
    let grid = ((num_sites as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };

    let func = module
        .load_function("metric_update")
        .context("Failed to load metric_update")?;
    let mut builder = stream.launch_builder(&func);
    builder.arg(&metric_buf.slice);
    builder.arg(&grad_buf.slice);
    builder.arg(&ricci_buf.slice);
    builder.arg(&num_sites);
    builder.arg(&components_per_site);
    builder.arg(&alpha);
    builder.arg(&beta);
    builder.arg(&eps_min);
    builder.arg(&eps_max);
    unsafe { builder.launch(cfg) }.context("metric_update launch failed")?;

    Ok(())
}
