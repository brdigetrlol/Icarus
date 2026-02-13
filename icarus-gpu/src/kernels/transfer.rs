//! Inter-lattice transfer CUDA kernel
//!
//! Implements dense matrix-vector multiply for transferring phase field
//! values between lattice layers (e.g. E8 → Leech embedding, Leech → E8
//! distillation). The transfer matrix W is dense because the layers have
//! different topologies.

use anyhow::{Context, Result};
use cudarc::driver::{CudaModule, CudaStream, LaunchConfig, PushKernelArg};
use std::sync::Arc;

use crate::buffers::GpuBuffer;

/// CUDA C source for the inter-layer transfer kernel.
///
/// y_re[i] = Σ_j W[i*src_n + j] * x_re[j]
/// y_im[i] = Σ_j W[i*src_n + j] * x_im[j]
pub const TRANSFER_KERNEL_SRC: &str = r#"
// Dense matrix-vector multiply: y = W * x (applied to both re and im)
// W is (dst_n x src_n), x is (src_n), y is (dst_n)
extern "C" __global__ void transfer_matvec(
    const float* __restrict__ W,
    const float* __restrict__ x_re,
    const float* __restrict__ x_im,
    float* __restrict__ y_re,
    float* __restrict__ y_im,
    int dst_n,
    int src_n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dst_n) return;

    float sum_re = 0.0f;
    float sum_im = 0.0f;
    int row_base = i * src_n;

    for (int j = 0; j < src_n; j++) {
        float w = W[row_base + j];
        sum_re += w * x_re[j];
        sum_im += w * x_im[j];
    }

    y_re[i] = sum_re;
    y_im[i] = sum_im;
}
"#;

const BLOCK_SIZE: u32 = 256;

/// Launch the inter-layer transfer kernel (dense matvec).
///
/// Computes y = W * x for both real and imaginary components.
/// W is (dst_n × src_n), x is (src_n), y is (dst_n).
pub fn launch_transfer_matvec(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    w_buf: &GpuBuffer<f32>,
    x_re_buf: &GpuBuffer<f32>,
    x_im_buf: &GpuBuffer<f32>,
    y_re_buf: &mut GpuBuffer<f32>,
    y_im_buf: &mut GpuBuffer<f32>,
    dst_n: i32,
    src_n: i32,
) -> Result<()> {
    let grid = ((dst_n as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };

    let func = module
        .load_function("transfer_matvec")
        .context("Failed to load transfer_matvec")?;
    let mut builder = stream.launch_builder(&func);
    builder.arg(&w_buf.slice);
    builder.arg(&x_re_buf.slice);
    builder.arg(&x_im_buf.slice);
    builder.arg(&y_re_buf.slice);
    builder.arg(&y_im_buf.slice);
    builder.arg(&dst_n);
    builder.arg(&src_n);
    unsafe { builder.launch(cfg) }.context("transfer_matvec launch failed")?;

    Ok(())
}
