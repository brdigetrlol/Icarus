// Copyright (c) 2025-2026 brdigetrlol. All rights reserved.
// SPDX-License-Identifier: LicenseRef-Icarus-Proprietary
// See LICENSE in the repository root for full license terms.

//! GPU memory helpers for the Icarus EMC
//!
//! Convenience wrappers around cudarc alloc/copy operations.
//! The E8 lattice (241 sites) is small enough that simple alloc/free
//! is perfectly adequate — no pooling needed at this scale.

use anyhow::{Context, Result};
use cudarc::driver::{CudaSlice, CudaStream, DeviceRepr, ValidAsZeroBits};
use half::f16;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::sync::Arc;

/// Allocate a zeroed device buffer.
pub fn alloc_zeros<T: DeviceRepr + ValidAsZeroBits>(
    stream: &Arc<CudaStream>,
    len: usize,
) -> Result<CudaSlice<T>> {
    stream
        .alloc_zeros(len)
        .context("Failed to allocate zeroed device memory")
}

/// Copy a host slice to a new device buffer.
pub fn htod<T: DeviceRepr>(
    stream: &Arc<CudaStream>,
    host: &[T],
) -> Result<CudaSlice<T>> {
    stream
        .clone_htod(host)
        .context("Failed to copy host->device")
}

/// Copy device buffer to a host Vec.
pub fn dtoh<T: DeviceRepr>(
    stream: &Arc<CudaStream>,
    device: &CudaSlice<T>,
) -> Result<Vec<T>> {
    stream
        .clone_dtoh(device)
        .context("Failed to copy device->host")
}

/// Synchronize the stream.
pub fn sync(stream: &Arc<CudaStream>) -> Result<()> {
    stream.synchronize().context("Stream synchronize failed")
}

// ---------------------------------------------------------------------------
// FP16 GPU Transport Helpers
// ---------------------------------------------------------------------------

/// Convert an f32 slice to f16 bits (u16) for compact GPU upload.
///
/// Useful when a CUDA kernel accepts half-precision inputs — convert on the
/// host side, upload as u16, reinterpret as __half on device.
pub fn f32_to_f16_vec(data: &[f32]) -> Vec<u16> {
    data.iter()
        .map(|&v| half::f16::from_f32(v).to_bits())
        .collect()
}

/// Convert f16 bits (u16) back to f32 after GPU download.
pub fn f16_to_f32_vec(data: &[u16]) -> Vec<f32> {
    data.iter()
        .map(|&bits| half::f16::from_bits(bits).to_f32())
        .collect()
}

/// Upload f32 data to GPU as f16 (packed into u16).
///
/// Converts host f32 → f16 on CPU, then uploads the u16 buffer.
/// Returns a CudaSlice<u16> with half the memory footprint.
pub fn htod_f16(
    stream: &Arc<CudaStream>,
    host_f32: &[f32],
) -> Result<CudaSlice<u16>> {
    let f16_bits = f32_to_f16_vec(host_f32);
    stream
        .clone_htod(&f16_bits)
        .context("Failed to upload f16 data to device")
}

/// Download f16 data from GPU and convert back to f32.
///
/// Downloads u16 buffer from device, then converts f16 → f32 on CPU.
pub fn dtoh_f16(
    stream: &Arc<CudaStream>,
    device: &CudaSlice<u16>,
) -> Result<Vec<f32>> {
    let bits: Vec<u16> = stream
        .clone_dtoh(device)
        .context("Failed to download f16 data from device")?;
    Ok(f16_to_f32_vec(&bits))
}

// ---------------------------------------------------------------------------
// Stochastic Rounding GPU Transport (Quartet II / MS-EDEN)
// ---------------------------------------------------------------------------

/// Convert an f32 slice to f16 bits using stochastic rounding.
///
/// Unlike `f32_to_f16_vec` which uses deterministic round-to-nearest (biased),
/// stochastic rounding satisfies `E[SR(x)] = x` — the expected value equals
/// the original. This eliminates systematic quantization bias per Quartet II
/// (arXiv:2601.22813).
///
/// For a value x between two representable f16 values a < b:
///   - Rounds to b with probability (x - a) / (b - a)
///   - Rounds to a with probability (b - x) / (b - a)
pub fn f32_to_f16_stochastic_vec(data: &[f32], seed: u64) -> Vec<u16> {
    let mut rng = SmallRng::seed_from_u64(seed);
    data.iter()
        .map(|&v| {
            let down = f16::from_f32(v);
            let back = down.to_f32();

            if back == v || v.is_nan() || v.is_infinite() {
                return down.to_bits();
            }

            let up = if v > back {
                next_f16_repr(down)
            } else {
                let tmp = down;
                let d = prev_f16_repr(down);
                // down should be the lower bound, up the upper
                // If back > v, then down is actually above v
                return stochastic_pick(d, tmp, v, &mut rng);
            };
            stochastic_pick(down, up, v, &mut rng)
        })
        .collect()
}

/// Stochastic pick between two adjacent f16 values for a target f32 value.
fn stochastic_pick(lo: f16, hi: f16, target: f32, rng: &mut SmallRng) -> u16 {
    let lo_f32 = lo.to_f32();
    let hi_f32 = hi.to_f32();
    let span = hi_f32 - lo_f32;
    if span.abs() < f32::EPSILON {
        return lo.to_bits();
    }
    let prob_hi = (target - lo_f32) / span;
    if rng.gen::<f32>() < prob_hi {
        hi.to_bits()
    } else {
        lo.to_bits()
    }
}

/// Next representable f16 above `v` (increment by 1 ULP).
fn next_f16_repr(v: f16) -> f16 {
    let bits = v.to_bits();
    if v.is_nan() || v == f16::INFINITY {
        return v;
    }
    if bits == 0 {
        // +0 → smallest positive subnormal
        return f16::from_bits(1);
    }
    if v.is_sign_positive() {
        f16::from_bits(bits + 1)
    } else {
        f16::from_bits(bits - 1)
    }
}

/// Previous representable f16 below `v` (decrement by 1 ULP).
fn prev_f16_repr(v: f16) -> f16 {
    let bits = v.to_bits();
    if v.is_nan() || v == f16::NEG_INFINITY {
        return v;
    }
    if bits == 0x8000 {
        // -0 → smallest negative subnormal
        return f16::from_bits(0x8001);
    }
    if bits == 0 {
        // +0 → -0 → smallest negative subnormal
        return f16::from_bits(0x8001);
    }
    if v.is_sign_positive() {
        f16::from_bits(bits - 1)
    } else {
        f16::from_bits(bits + 1)
    }
}

/// Upload f32 data to GPU as f16 with stochastic rounding.
///
/// Same as `htod_f16` but uses unbiased stochastic rounding instead of
/// deterministic round-to-nearest. Produces `E[Q(x)] = x` over repeated
/// quantizations, eliminating systematic drift in field snapshots.
pub fn htod_f16_stochastic(
    stream: &Arc<CudaStream>,
    host_f32: &[f32],
    seed: u64,
) -> Result<CudaSlice<u16>> {
    let f16_bits = f32_to_f16_stochastic_vec(host_f32, seed);
    stream
        .clone_htod(&f16_bits)
        .context("Failed to upload stochastic f16 data to device")
}

// ---------------------------------------------------------------------------
// VRAM Budget Planner
// ---------------------------------------------------------------------------

/// VRAM placement decision for a manifold layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerPlacement {
    /// Run this layer on GPU.
    Gpu,
    /// Fall back to CPU (exceeds VRAM budget or GPU disabled).
    Cpu,
}

/// Input specification for VRAM estimation of a single layer.
///
/// Build from lattice config before field construction — uses the same
/// site/edge formulas as `ManifoldConfig::estimated_memory_bytes()`.
#[derive(Debug, Clone)]
pub struct LayerSpec {
    /// Number of lattice sites
    pub num_sites: usize,
    /// Total number of directed edges (sum of neighbors across all sites)
    pub num_edges: usize,
    /// Lattice dimension
    pub dim: usize,
    /// Whether metric learning is enabled on this layer
    pub enable_metric_learning: bool,
}

/// VRAM estimate breakdown for a single layer.
#[derive(Debug, Clone)]
pub struct LayerVramEstimate {
    /// Bytes for GPU field buffers (values_re, values_im, offsets, indices, weights)
    pub field_bytes: usize,
    /// Bytes for RAE scratch buffers (dz_re, dz_im)
    pub scratch_bytes: usize,
    /// Bytes for metric update buffers (metric + grad + ricci); 0 if disabled
    pub metric_bytes: usize,
    /// Peak VRAM: max(field+scratch, metric) since operations are sequential
    pub peak_bytes: usize,
}

/// Plans VRAM allocation across manifold layers.
///
/// Since `tick()` processes layers sequentially, only one layer's GPU buffers
/// exist at any given time. The VRAM constraint is therefore per-layer peak,
/// not the sum across layers.
pub struct VramBudgetPlanner;

impl VramBudgetPlanner {
    /// Estimate peak VRAM usage for a single layer.
    pub fn estimate_layer(spec: &LayerSpec) -> LayerVramEstimate {
        // Field buffers allocated during rae_step / free_energy
        let field_bytes = (2 * spec.num_sites       // values_re + values_im  (f32)
            + (spec.num_sites + 1)                   // neighbor_offsets       (u32)
            + 2 * spec.num_edges)                    // neighbor_indices (u32) + weights (f32)
            * 4; // sizeof(f32) == sizeof(u32) == 4

        // RAE scratch (dz_re, dz_im)
        let scratch_bytes = 2 * spec.num_sites * 4;

        let rae_peak = field_bytes + scratch_bytes;

        // Metric update buffers (only when metric learning is active)
        let metric_bytes = if spec.enable_metric_learning {
            let packed_size = spec.dim * (spec.dim + 1) / 2;
            3 * spec.num_sites * packed_size * 4 // metric + grad + ricci
        } else {
            0
        };

        // Peak is the max of sequential phases (rae_step then metric_update)
        let peak_bytes = rae_peak.max(metric_bytes);

        LayerVramEstimate {
            field_bytes,
            scratch_bytes,
            metric_bytes,
            peak_bytes,
        }
    }

    /// Plan layer placements given a VRAM budget (bytes).
    ///
    /// A layer goes on GPU if its peak VRAM fits within the budget.
    /// If budget is 0, all layers fall back to CPU.
    pub fn plan(
        specs: &[LayerSpec],
        vram_budget: usize,
    ) -> Vec<(LayerVramEstimate, LayerPlacement)> {
        specs
            .iter()
            .map(|spec| {
                let estimate = Self::estimate_layer(spec);
                let placement = if vram_budget > 0 && estimate.peak_bytes <= vram_budget {
                    LayerPlacement::Gpu
                } else {
                    LayerPlacement::Cpu
                };
                (estimate, placement)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn e8_spec(metric: bool) -> LayerSpec {
        LayerSpec {
            num_sites: 241,
            num_edges: 241 * 240,
            dim: 8,
            enable_metric_learning: metric,
        }
    }

    fn leech_spec(metric: bool) -> LayerSpec {
        LayerSpec {
            num_sites: 1105,
            num_edges: 1105 * 1104,
            dim: 24,
            enable_metric_learning: metric,
        }
    }

    fn hcp16_spec(metric: bool) -> LayerSpec {
        LayerSpec {
            num_sites: 241,      // 1 + 16*15
            num_edges: 241 * 240, // 16*15 = 240 kissing
            dim: 16,
            enable_metric_learning: metric,
        }
    }

    #[test]
    fn test_estimate_e8_no_metric() {
        let est = VramBudgetPlanner::estimate_layer(&e8_spec(false));

        // field: (2*241 + 242 + 2*57840) * 4 = 116404 * 4 = 465616
        assert_eq!(est.field_bytes, (2 * 241 + 242 + 2 * 57840) * 4);
        // scratch: 2*241*4 = 1928
        assert_eq!(est.scratch_bytes, 2 * 241 * 4);
        // no metric
        assert_eq!(est.metric_bytes, 0);
        // peak = rae_peak
        assert_eq!(est.peak_bytes, est.field_bytes + est.scratch_bytes);
        // Sanity: E8 should be well under 1MB
        assert!(est.peak_bytes < 1_000_000);
    }

    #[test]
    fn test_estimate_e8_with_metric() {
        let est = VramBudgetPlanner::estimate_layer(&e8_spec(true));

        let packed_size = 8 * 9 / 2; // 36
        let expected_metric = 3 * 241 * packed_size * 4;
        assert_eq!(est.metric_bytes, expected_metric);
        // RAE peak should still dominate for E8 (field+scratch > metric)
        let rae_peak = est.field_bytes + est.scratch_bytes;
        assert_eq!(est.peak_bytes, rae_peak.max(est.metric_bytes));
    }

    #[test]
    fn test_estimate_leech() {
        let est = VramBudgetPlanner::estimate_layer(&leech_spec(false));

        // Leech has 1105 sites and 1105*1104 edges — substantial VRAM
        assert!(est.peak_bytes > 5_000_000, "Leech should need >5MB, got {}", est.peak_bytes);
        assert!(est.peak_bytes < 20_000_000, "Leech should need <20MB, got {}", est.peak_bytes);
    }

    #[test]
    fn test_plan_all_gpu() {
        let specs = vec![e8_spec(true), leech_spec(false), hcp16_spec(false)];
        let budget = 100 * 1024 * 1024; // 100MB — plenty
        let plan = VramBudgetPlanner::plan(&specs, budget);

        assert_eq!(plan.len(), 3);
        for (_, placement) in &plan {
            assert_eq!(*placement, LayerPlacement::Gpu);
        }
    }

    #[test]
    fn test_plan_all_cpu() {
        let specs = vec![e8_spec(true), leech_spec(false)];
        let plan = VramBudgetPlanner::plan(&specs, 0);

        for (_, placement) in &plan {
            assert_eq!(*placement, LayerPlacement::Cpu);
        }
    }

    #[test]
    fn test_plan_mixed() {
        let specs = vec![e8_spec(false), leech_spec(false)];
        // E8 peak is ~467KB, Leech peak is ~9.8MB
        // Budget that fits E8 but not Leech
        let budget = 1_000_000; // 1MB
        let plan = VramBudgetPlanner::plan(&specs, budget);

        assert_eq!(plan[0].1, LayerPlacement::Gpu, "E8 should fit in 1MB");
        assert_eq!(plan[1].1, LayerPlacement::Cpu, "Leech should NOT fit in 1MB");
    }

    #[test]
    fn test_plan_empty() {
        let plan = VramBudgetPlanner::plan(&[], 1_000_000);
        assert!(plan.is_empty());
    }

    #[test]
    fn test_f32_to_f16_stochastic_exact_values() {
        // Values exactly representable in f16 should round-trip perfectly
        let data = vec![0.0f32, 1.0, -1.0, 0.5, -0.5, 2.0];
        let bits = f32_to_f16_stochastic_vec(&data, 42);
        for (i, (&orig, &b)) in data.iter().zip(bits.iter()).enumerate() {
            let restored = half::f16::from_bits(b).to_f32();
            assert_eq!(restored, orig, "exact value {} at index {} should survive SR", orig, i);
        }
    }

    #[test]
    fn test_f32_to_f16_stochastic_unbiased() {
        // Over many trials, the mean of SR(x) should approximate x
        let val = 0.3333f32; // not exactly representable in f16
        let n = 10_000;
        let mut sum = 0.0f64;
        for seed in 0..n {
            let bits = f32_to_f16_stochastic_vec(&[val], seed);
            sum += half::f16::from_bits(bits[0]).to_f32() as f64;
        }
        let mean = (sum / n as f64) as f32;
        let err = (mean - val).abs();
        assert!(
            err < 0.002,
            "SR mean {} should be close to {} (err {})",
            mean, val, err
        );
    }

    #[test]
    fn test_f32_to_f16_stochastic_deterministic_same_seed() {
        let data = vec![0.1f32, 0.2, 0.3, 0.4, 0.5];
        let a = f32_to_f16_stochastic_vec(&data, 123);
        let b = f32_to_f16_stochastic_vec(&data, 123);
        assert_eq!(a, b, "Same seed should produce identical results");
    }

    #[test]
    fn test_stochastic_vs_rtn_different() {
        // Stochastic rounding should sometimes differ from RTN
        let data: Vec<f32> = (0..100).map(|i| 0.001 * i as f32 + 0.0001).collect();
        let rtn_bits: Vec<u16> = data.iter().map(|&v| half::f16::from_f32(v).to_bits()).collect();
        let sr_bits = f32_to_f16_stochastic_vec(&data, 77);
        let diffs = rtn_bits.iter().zip(sr_bits.iter()).filter(|(a, b)| a != b).count();
        assert!(diffs > 0, "SR should differ from RTN for at least some non-exact values");
    }
}
