//! RAE PDE solver CUDA kernel
//!
//! Implements the Resonant Attractor Equation on the GPU:
//!   dz/dt = kw * Δz - dV·z + iω·z - γ·z
//!
//! Two-phase approach:
//!   1. `rae_compute_dz` — computes dz/dt using CSR Laplacian + potential + resonance
//!   2. `rae_euler_update` — applies forward Euler: z += dt * dz

use anyhow::{Context, Result};
use cudarc::driver::{CudaModule, CudaStream, LaunchConfig, PushKernelArg};
use std::sync::Arc;

use crate::buffers::{GpuBuffer, GpuFieldBuffers};

/// CUDA C source for the RAE kernels.
use icarus_field::rae::IntegratorMethod;

pub const RAE_KERNEL_SRC: &str = r#"
extern "C" __global__ void rae_compute_dz(
    const float* __restrict__ values_re,
    const float* __restrict__ values_im,
    const unsigned int* __restrict__ neighbor_offsets,
    const unsigned int* __restrict__ neighbor_indices,
    const float* __restrict__ neighbor_weights,
    float* __restrict__ dz_re,
    float* __restrict__ dz_im,
    int n,
    float omega,
    float gamma,
    float kinetic_weight,
    float potential_weight,
    float target_sq)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float zi_re = values_re[i];
    float zi_im = values_im[i];
    float ns = zi_re * zi_re + zi_im * zi_im;

    // Graph Laplacian: (2/K) * sum w_ij * (z_j - z_i)
    unsigned int start = neighbor_offsets[i];
    unsigned int end   = neighbor_offsets[i + 1];
    float k = (float)(end - start);

    float lap_re = 0.0f;
    float lap_im = 0.0f;

    if (k > 0.0f) {
        for (unsigned int edge = start; edge < end; edge++) {
            unsigned int j = neighbor_indices[edge];
            float w = neighbor_weights[edge];
            lap_re += w * (values_re[j] - zi_re);
            lap_im += w * (values_im[j] - zi_im);
        }
        float lap_scale = 2.0f / k;
        lap_re *= lap_scale;
        lap_im *= lap_scale;
    }

    // Double-well potential derivative: (|z|^2 - target^2) / 2
    float dv = (ns - target_sq) * 0.5f * potential_weight;

    // dz/dt = kw * laplacian - dV*z + iw*z - gamma*z
    // i*omega*z = i*omega*(re + i*im) = -omega*im + i*omega*re
    dz_re[i] = kinetic_weight * lap_re - dv * zi_re - omega * zi_im - gamma * zi_re;
    dz_im[i] = kinetic_weight * lap_im - dv * zi_im + omega * zi_re - gamma * zi_im;
}

extern "C" __global__ void rae_euler_update(
    float* __restrict__ values_re,
    float* __restrict__ values_im,
    const float* __restrict__ dz_re,
    const float* __restrict__ dz_im,
    int n,
    float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    values_re[i] += dt * dz_re[i];
    values_im[i] += dt * dz_im[i];
}
"#;

/// CUDA C source for the semi-implicit RAE kernel.
///
/// Two-phase approach (same as Euler, to avoid read-write hazards):
///   1. `rae_semi_implicit_solve` — reads old values, computes explicit RHS,
///      solves 2×2 system per site, writes new values to dz_re/dz_im (temp storage).
///   2. `rae_semi_implicit_finalize` — copies new values from dz_re/dz_im back to
///      values_re/values_im, and stores the actual dz/dt for diagnostics.
pub const RAE_SEMI_IMPLICIT_KERNEL_SRC: &str = r#"
extern "C" __global__ void rae_semi_implicit_solve(
    const float* __restrict__ values_re,
    const float* __restrict__ values_im,
    const unsigned int* __restrict__ neighbor_offsets,
    const unsigned int* __restrict__ neighbor_indices,
    const float* __restrict__ neighbor_weights,
    float* __restrict__ out_re,
    float* __restrict__ out_im,
    int n,
    float dt,
    float omega,
    float gamma,
    float kinetic_weight,
    float potential_weight,
    float target_sq)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float zi_re = values_re[i];
    float zi_im = values_im[i];
    float ns = zi_re * zi_re + zi_im * zi_im;

    // Explicit neighbor accumulation
    unsigned int start = neighbor_offsets[i];
    unsigned int end   = neighbor_offsets[i + 1];
    float k = (float)(end - start);

    float nb_sum_re = 0.0f;
    float nb_sum_im = 0.0f;
    float w_sum = 0.0f;

    if (k > 0.0f) {
        for (unsigned int edge = start; edge < end; edge++) {
            unsigned int j = neighbor_indices[edge];
            float w = neighbor_weights[edge];
            nb_sum_re += w * values_re[j];
            nb_sum_im += w * values_im[j];
            w_sum += w;
        }
        float lap_scale = 2.0f / k;
        nb_sum_re *= lap_scale;
        nb_sum_im *= lap_scale;
        w_sum *= lap_scale;
    }

    // Double-well potential derivative (explicit)
    float dv = (ns - target_sq) * 0.5f * potential_weight;

    // Explicit RHS: z + dt * (kw * nb_sum - dV * z)
    float rhs_re = zi_re + dt * (kinetic_weight * nb_sum_re - dv * zi_re);
    float rhs_im = zi_im + dt * (kinetic_weight * nb_sum_im - dv * zi_im);

    // Implicit diagonal: s_i = kw * (2/K) * sum w_ij
    float s_i = kinetic_weight * w_sum;

    // 2x2 system: (I - dt*L_diag) z_new = rhs
    // Matrix: [[d, c], [-c, d]]  where d = 1 + dt*(s_i + gamma), c = dt*omega
    // Inverse: (1/(d^2+c^2)) * [[d, -c], [c, d]]
    float d = 1.0f + dt * (s_i + gamma);
    float c = dt * omega;
    float inv_det = 1.0f / (d * d + c * c);

    float new_re = (d * rhs_re - c * rhs_im) * inv_det;
    float new_im = (c * rhs_re + d * rhs_im) * inv_det;

    // Store new values in output buffers (dz_re/dz_im repurposed as temp)
    out_re[i] = new_re;
    out_im[i] = new_im;
}

extern "C" __global__ void rae_semi_implicit_finalize(
    float* __restrict__ values_re,
    float* __restrict__ values_im,
    float* __restrict__ dz_re,
    float* __restrict__ dz_im,
    int n,
    float inv_dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float old_re = values_re[i];
    float old_im = values_im[i];
    float new_re = dz_re[i];
    float new_im = dz_im[i];

    // Write new field values
    values_re[i] = new_re;
    values_im[i] = new_im;

    // Write dz/dt for diagnostics
    dz_re[i] = (new_re - old_re) * inv_dt;
    dz_im[i] = (new_im - old_im) * inv_dt;
}
"#;

const BLOCK_SIZE: u32 = 256;

/// Launch one Euler RAE step: compute dz/dt then z += dt * dz.
fn launch_rae_euler_step(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    bufs: &GpuFieldBuffers,
    dz_re: &mut GpuBuffer<f32>,
    dz_im: &mut GpuBuffer<f32>,
    n: i32,
    dt: f32,
    omega: f32,
    gamma: f32,
    kinetic_weight: f32,
    potential_weight: f32,
    target_sq: f32,
) -> Result<()> {
    let grid = ((n as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };

    // Phase 1: compute dz/dt
    {
        let func = module
            .load_function("rae_compute_dz")
            .context("Failed to load rae_compute_dz")?;
        let mut builder = stream.launch_builder(&func);
        builder.arg(&bufs.values_re.slice);
        builder.arg(&bufs.values_im.slice);
        builder.arg(&bufs.neighbor_offsets.slice);
        builder.arg(&bufs.neighbor_indices.slice);
        builder.arg(&bufs.neighbor_weights.slice);
        builder.arg(&mut dz_re.slice);
        builder.arg(&mut dz_im.slice);
        builder.arg(&n);
        builder.arg(&omega);
        builder.arg(&gamma);
        builder.arg(&kinetic_weight);
        builder.arg(&potential_weight);
        builder.arg(&target_sq);
        unsafe { builder.launch(cfg) }.context("rae_compute_dz launch failed")?;
    }

    // Phase 2: Euler update z += dt * dz
    {
        let func = module
            .load_function("rae_euler_update")
            .context("Failed to load rae_euler_update")?;
        let mut builder = stream.launch_builder(&func);
        builder.arg(&bufs.values_re.slice);
        builder.arg(&bufs.values_im.slice);
        builder.arg(&dz_re.slice);
        builder.arg(&dz_im.slice);
        builder.arg(&n);
        builder.arg(&dt);
        unsafe { builder.launch(cfg) }.context("rae_euler_update launch failed")?;
    }

    Ok(())
}

/// Launch one semi-implicit RAE step: compute explicit RHS, solve 2×2, finalize.
fn launch_rae_semi_implicit_step(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    bufs: &GpuFieldBuffers,
    dz_re: &mut GpuBuffer<f32>,
    dz_im: &mut GpuBuffer<f32>,
    n: i32,
    dt: f32,
    omega: f32,
    gamma: f32,
    kinetic_weight: f32,
    potential_weight: f32,
    target_sq: f32,
) -> Result<()> {
    let grid = ((n as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };

    // Phase 1: compute explicit RHS + implicit solve → store new values in dz_re/dz_im
    {
        let func = module
            .load_function("rae_semi_implicit_solve")
            .context("Failed to load rae_semi_implicit_solve")?;
        let mut builder = stream.launch_builder(&func);
        builder.arg(&bufs.values_re.slice);
        builder.arg(&bufs.values_im.slice);
        builder.arg(&bufs.neighbor_offsets.slice);
        builder.arg(&bufs.neighbor_indices.slice);
        builder.arg(&bufs.neighbor_weights.slice);
        builder.arg(&mut dz_re.slice);
        builder.arg(&mut dz_im.slice);
        builder.arg(&n);
        builder.arg(&dt);
        builder.arg(&omega);
        builder.arg(&gamma);
        builder.arg(&kinetic_weight);
        builder.arg(&potential_weight);
        builder.arg(&target_sq);
        unsafe { builder.launch(cfg) }.context("rae_semi_implicit_solve launch failed")?;
    }

    // Phase 2: copy new values to field, compute dz/dt for diagnostics
    {
        let inv_dt = 1.0f32 / dt;
        let func = module
            .load_function("rae_semi_implicit_finalize")
            .context("Failed to load rae_semi_implicit_finalize")?;
        let mut builder = stream.launch_builder(&func);
        builder.arg(&bufs.values_re.slice);
        builder.arg(&bufs.values_im.slice);
        builder.arg(&mut dz_re.slice);
        builder.arg(&mut dz_im.slice);
        builder.arg(&n);
        builder.arg(&inv_dt);
        unsafe { builder.launch(cfg) }.context("rae_semi_implicit_finalize launch failed")?;
    }

    Ok(())
}

/// Launch one RAE step, dispatching to Euler or semi-implicit.
pub fn launch_rae_step(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    bufs: &GpuFieldBuffers,
    dz_re: &mut GpuBuffer<f32>,
    dz_im: &mut GpuBuffer<f32>,
    n: i32,
    dt: f32,
    omega: f32,
    gamma: f32,
    kinetic_weight: f32,
    potential_weight: f32,
    target_sq: f32,
    method: IntegratorMethod,
) -> Result<()> {
    match method {
        IntegratorMethod::Euler => launch_rae_euler_step(
            stream, module, bufs, dz_re, dz_im, n, dt, omega, gamma,
            kinetic_weight, potential_weight, target_sq,
        ),
        IntegratorMethod::SemiImplicit => launch_rae_semi_implicit_step(
            stream, module, bufs, dz_re, dz_im, n, dt, omega, gamma,
            kinetic_weight, potential_weight, target_sq,
        ),
    }
}
