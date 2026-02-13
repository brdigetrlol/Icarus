//! Free energy reduction CUDA kernels
//!
//! Computes kinetic and potential energy terms via parallel tree reduction
//! in shared memory. Each block reduces its portion, then the host sums
//! the per-block partial results (the lattice is small enough that this
//! single-pass approach is sufficient).

use anyhow::{Context, Result};
use cudarc::driver::{CudaModule, CudaStream, LaunchConfig, PushKernelArg};
use std::sync::Arc;

use crate::buffers::GpuFieldBuffers;

/// CUDA C source for the reduction kernels.
pub const REDUCTION_KERNEL_SRC: &str = r#"
// Kinetic energy: 0.5 * sum_edges w_ij * |z_j - z_i|^2, weighted by kinetic_weight
// We compute per-site contribution: (1/K_i) * sum_{j in N(i)} w_ij * |z_j - z_i|^2
// Then sum over all sites. Factor of 0.5 avoids double-counting edges.
extern "C" __global__ void kinetic_energy_reduce(
    const float* __restrict__ values_re,
    const float* __restrict__ values_im,
    const unsigned int* __restrict__ neighbor_offsets,
    const unsigned int* __restrict__ neighbor_indices,
    const float* __restrict__ neighbor_weights,
    float* __restrict__ block_sums,
    int n,
    float kinetic_weight)
{
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float local_sum = 0.0f;
    if (i < n) {
        float zi_re = values_re[i];
        float zi_im = values_im[i];
        unsigned int start = neighbor_offsets[i];
        unsigned int end   = neighbor_offsets[i + 1];

        for (unsigned int edge = start; edge < end; edge++) {
            unsigned int j = neighbor_indices[edge];
            float w = neighbor_weights[edge];
            float dr = values_re[j] - zi_re;
            float di = values_im[j] - zi_im;
            local_sum += w * (dr * dr + di * di);
        }
        // Factor 0.5 to avoid double-counting + kinetic_weight
        local_sum *= 0.5f * kinetic_weight;
    }

    sdata[tid] = local_sum;
    __syncthreads();

    // Tree reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_sums[blockIdx.x] = sdata[0];
    }
}

// Potential energy: sum_i V(|z_i|^2) where V(s) = (s - target_sq)^2 / 4
extern "C" __global__ void potential_energy_reduce(
    const float* __restrict__ values_re,
    const float* __restrict__ values_im,
    float* __restrict__ block_sums,
    int n,
    float potential_weight,
    float target_sq)
{
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float local_sum = 0.0f;
    if (i < n) {
        float re = values_re[i];
        float im = values_im[i];
        float ns = re * re + im * im;
        float diff = ns - target_sq;
        // V(|z|^2) = (|z|^2 - target)^2 / 4
        local_sum = potential_weight * diff * diff * 0.25f;
    }

    sdata[tid] = local_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_sums[blockIdx.x] = sdata[0];
    }
}
"#;

const BLOCK_SIZE: u32 = 256;

/// Launch kinetic energy reduction, returning the scalar result.
pub fn launch_kinetic_energy(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    bufs: &GpuFieldBuffers,
    kinetic_weight: f32,
) -> Result<f32> {
    let n = bufs.num_sites as i32;
    let grid = ((n as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Allocate output buffer for per-block partial sums
    let block_sums: cudarc::driver::CudaSlice<f32> = stream
        .clone_htod(&vec![0.0f32; grid as usize])
        .context("alloc block_sums")?;

    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: BLOCK_SIZE * std::mem::size_of::<f32>() as u32,
    };

    let func = module
        .load_function("kinetic_energy_reduce")
        .context("Failed to load kinetic_energy_reduce")?;
    let mut builder = stream.launch_builder(&func);
    builder.arg(&bufs.values_re.slice);
    builder.arg(&bufs.values_im.slice);
    builder.arg(&bufs.neighbor_offsets.slice);
    builder.arg(&bufs.neighbor_indices.slice);
    builder.arg(&bufs.neighbor_weights.slice);
    builder.arg(&block_sums);
    builder.arg(&n);
    builder.arg(&kinetic_weight);
    unsafe { builder.launch(cfg) }.context("kinetic_energy_reduce launch failed")?;

    // Download partial sums and finish reduction on CPU
    let mut host_sums = vec![0.0f32; grid as usize];
    stream
        .memcpy_dtoh(&block_sums, &mut host_sums)
        .context("download block_sums")?;
    stream.synchronize().context("sync after kinetic")?;

    Ok(host_sums.iter().sum())
}

/// Launch potential energy reduction, returning the scalar result.
pub fn launch_potential_energy(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    bufs: &GpuFieldBuffers,
    potential_weight: f32,
    target_sq: f32,
) -> Result<f32> {
    let n = bufs.num_sites as i32;
    let grid = ((n as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;

    let block_sums: cudarc::driver::CudaSlice<f32> = stream
        .clone_htod(&vec![0.0f32; grid as usize])
        .context("alloc block_sums")?;

    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: BLOCK_SIZE * std::mem::size_of::<f32>() as u32,
    };

    let func = module
        .load_function("potential_energy_reduce")
        .context("Failed to load potential_energy_reduce")?;
    let mut builder = stream.launch_builder(&func);
    builder.arg(&bufs.values_re.slice);
    builder.arg(&bufs.values_im.slice);
    builder.arg(&block_sums);
    builder.arg(&n);
    builder.arg(&potential_weight);
    builder.arg(&target_sq);
    unsafe { builder.launch(cfg) }.context("potential_energy_reduce launch failed")?;

    let mut host_sums = vec![0.0f32; grid as usize];
    stream
        .memcpy_dtoh(&block_sums, &mut host_sums)
        .context("download block_sums")?;
    stream.synchronize().context("sync after potential")?;

    Ok(host_sums.iter().sum())
}
