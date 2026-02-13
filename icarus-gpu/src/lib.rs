// Copyright (c) 2025-2026 brdigetrlol. All rights reserved.
// SPDX-License-Identifier: LicenseRef-Icarus-Proprietary
// See LICENSE in the repository root for full license terms.

pub mod device;
pub mod memory;
pub mod buffers;
pub mod pipeline;
pub mod kernels;
pub mod npu_client;
pub mod npu_backend;

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use icarus_field::free_energy::{free_energy, FreeEnergyParams};
    use icarus_field::phase_field::LatticeField;
    use icarus_field::rae::RAEParams;
    use icarus_math::lattice::e8::E8Lattice;

    use crate::kernels::IcarusKernels;

    /// Helper: create an E8 field initialized with deterministic random values.
    fn make_e8_field(seed: u64, amplitude: f32) -> LatticeField {
        let lattice = E8Lattice::new();
        let mut field = LatticeField::from_lattice(&lattice);
        field.init_random(seed, amplitude);
        field
    }

    #[test]
    fn test_gpu_kernel_compilation() -> Result<()> {
        let kernels = IcarusKernels::new(0)?;
        assert_eq!(kernels.ctx().ordinal(), 0);
        Ok(())
    }

    #[test]
    fn test_gpu_free_energy_parity() -> Result<()> {
        let field = make_e8_field(42, 0.5);
        let params = FreeEnergyParams::default();

        // CPU reference
        let (cpu_total, cpu_kinetic, cpu_potential) = free_energy(&field, &params);

        // GPU
        let kernels = IcarusKernels::new(0)?;
        let (gpu_total, gpu_kinetic, gpu_potential) = kernels.free_energy(&field, &params)?;

        // Compare — allow some f32 accumulation tolerance
        let tol = 1e-2;
        assert!(
            (cpu_kinetic - gpu_kinetic).abs() < tol,
            "Kinetic mismatch: CPU={} GPU={}",
            cpu_kinetic, gpu_kinetic
        );
        assert!(
            (cpu_potential - gpu_potential).abs() < tol,
            "Potential mismatch: CPU={} GPU={}",
            cpu_potential, gpu_potential
        );
        assert!(
            (cpu_total - gpu_total).abs() < tol,
            "Total mismatch: CPU={} GPU={}",
            cpu_total, gpu_total
        );
        Ok(())
    }

    #[test]
    fn test_gpu_free_energy_zero_field() -> Result<()> {
        // Zero field should have potential energy = N * pw * target_sq^2 / 4
        let lattice = E8Lattice::new();
        let field = LatticeField::from_lattice(&lattice);
        let params = FreeEnergyParams::default();

        let (cpu_total, _, _) = free_energy(&field, &params);
        let kernels = IcarusKernels::new(0)?;
        let (gpu_total, _, _) = kernels.free_energy(&field, &params)?;

        assert!(
            (cpu_total - gpu_total).abs() < 1e-3,
            "Zero field energy mismatch: CPU={} GPU={}",
            cpu_total, gpu_total
        );
        Ok(())
    }

    #[test]
    fn test_gpu_rae_single_step_parity() -> Result<()> {
        let params = RAEParams::default_e8();

        // CPU path
        let mut cpu_field = make_e8_field(42, 0.5);
        let mut solver = icarus_field::rae::RAESolver::new(params.clone(), cpu_field.num_sites);
        solver.step(&mut cpu_field);

        // GPU path
        let mut gpu_field = make_e8_field(42, 0.5);
        let kernels = IcarusKernels::new(0)?;
        kernels.rae_step(&mut gpu_field, &params, 1)?;

        // Compare all sites
        let tol = 1e-4;
        let mut max_diff_re = 0.0f32;
        let mut max_diff_im = 0.0f32;
        for i in 0..cpu_field.num_sites {
            let diff_re = (cpu_field.values_re[i] - gpu_field.values_re[i]).abs();
            let diff_im = (cpu_field.values_im[i] - gpu_field.values_im[i]).abs();
            max_diff_re = max_diff_re.max(diff_re);
            max_diff_im = max_diff_im.max(diff_im);
            assert!(
                diff_re < tol,
                "RE mismatch at site {}: CPU={} GPU={} diff={}",
                i, cpu_field.values_re[i], gpu_field.values_re[i], diff_re
            );
            assert!(
                diff_im < tol,
                "IM mismatch at site {}: CPU={} GPU={} diff={}",
                i, cpu_field.values_im[i], gpu_field.values_im[i], diff_im
            );
        }
        eprintln!(
            "Single step parity: max_diff_re={:.2e} max_diff_im={:.2e}",
            max_diff_re, max_diff_im
        );
        Ok(())
    }

    #[test]
    fn test_gpu_rae_multi_step_parity() -> Result<()> {
        let params = RAEParams::default_e8();
        let num_steps = 10u64;

        // CPU path
        let mut cpu_field = make_e8_field(42, 0.5);
        let mut solver = icarus_field::rae::RAESolver::new(params.clone(), cpu_field.num_sites);
        solver.run(&mut cpu_field, num_steps);

        // GPU path — runs all steps on-device then downloads once
        let mut gpu_field = make_e8_field(42, 0.5);
        let kernels = IcarusKernels::new(0)?;
        kernels.rae_step(&mut gpu_field, &params, num_steps)?;

        // After 10 steps, f32 drift accumulates — use slightly wider tolerance
        let tol = 1e-3;
        let mut max_diff = 0.0f32;
        for i in 0..cpu_field.num_sites {
            let diff_re = (cpu_field.values_re[i] - gpu_field.values_re[i]).abs();
            let diff_im = (cpu_field.values_im[i] - gpu_field.values_im[i]).abs();
            let diff = diff_re.max(diff_im);
            max_diff = max_diff.max(diff);
            assert!(
                diff < tol,
                "Mismatch at site {} after {} steps: CPU=({}, {}) GPU=({}, {}) diff=({:.2e}, {:.2e})",
                i, num_steps,
                cpu_field.values_re[i], cpu_field.values_im[i],
                gpu_field.values_re[i], gpu_field.values_im[i],
                diff_re, diff_im
            );
        }
        eprintln!(
            "{} step parity: max_diff={:.2e}",
            num_steps, max_diff
        );
        Ok(())
    }

    #[test]
    fn test_gpu_rae_energy_decreases() -> Result<()> {
        let mut params = RAEParams::default_e8();
        params.omega = 0.0;      // no resonance — pure dissipation
        params.gamma = 0.5;      // strong damping
        let energy_params = params.energy_params.clone();

        let mut field = make_e8_field(42, 0.8);
        let kernels = IcarusKernels::new(0)?;

        let e0 = kernels.free_energy(&field, &energy_params)?.0;
        kernels.rae_step(&mut field, &params, 100)?;
        let e1 = kernels.free_energy(&field, &energy_params)?.0;
        kernels.rae_step(&mut field, &params, 100)?;
        let e2 = kernels.free_energy(&field, &energy_params)?.0;

        eprintln!("GPU energy: e0={:.4} e1={:.4} e2={:.4}", e0, e1, e2);
        assert!(
            e1 <= e0 + 0.1,
            "Energy should decrease: {} -> {}",
            e0, e1
        );
        assert!(
            e2 <= e1 + 0.1,
            "Energy should decrease: {} -> {}",
            e1, e2
        );
        Ok(())
    }

    #[test]
    fn test_gpu_rae_deterministic() -> Result<()> {
        let params = RAEParams::default_e8();
        let kernels = IcarusKernels::new(0)?;

        // Run 1
        let mut field1 = make_e8_field(42, 0.5);
        kernels.rae_step(&mut field1, &params, 5)?;

        // Run 2 — same seed, same params
        let mut field2 = make_e8_field(42, 0.5);
        kernels.rae_step(&mut field2, &params, 5)?;

        // Must be bit-exact (same device, same stream, same data)
        for i in 0..field1.num_sites {
            assert_eq!(
                field1.values_re[i], field2.values_re[i],
                "RE not deterministic at site {}",
                i
            );
            assert_eq!(
                field1.values_im[i], field2.values_im[i],
                "IM not deterministic at site {}",
                i
            );
        }
        Ok(())
    }

    #[test]
    fn test_gpu_metric_update() -> Result<()> {
        let kernels = IcarusKernels::new(0)?;

        // Small test: 4 sites, dim=3, packed_size=6 components per site
        let num_sites = 4;
        let dim = 3;
        let cps = dim * (dim + 1) / 2; // 6
        let total = num_sites * cps;

        // Identity metric at all sites: diagonal entries = 1.0
        let mut metric_data = vec![0.0f32; total];
        for s in 0..num_sites {
            // For dim=3, packed indices of diagonals: (0,0)=0, (1,1)=3, (2,2)=5
            metric_data[s * cps + 0] = 1.0; // g00
            metric_data[s * cps + 3] = 1.0; // g11
            metric_data[s * cps + 5] = 1.0; // g22
        }

        // Gradient: uniform 0.1 everywhere
        let grad_data = vec![0.1f32; total];
        // Ricci: uniform 0.05 everywhere
        let ricci_data = vec![0.05f32; total];

        let alpha = 0.01f32;
        let beta = 0.02f32;
        let eps_min = 0.01f32;
        let eps_max = 100.0f32;

        // CPU reference: g += -alpha * grad + beta * ricci, then clamp
        let mut expected = metric_data.clone();
        for v in expected.iter_mut() {
            *v += -alpha * 0.1 + beta * 0.05;
            *v = v.max(eps_min).min(eps_max);
        }

        // GPU
        kernels.metric_update(
            &mut metric_data,
            &grad_data,
            &ricci_data,
            num_sites,
            cps,
            alpha,
            beta,
            eps_min,
            eps_max,
        )?;

        let tol = 1e-6;
        for i in 0..total {
            let diff = (metric_data[i] - expected[i]).abs();
            assert!(
                diff < tol,
                "Metric mismatch at index {}: GPU={} expected={} diff={}",
                i, metric_data[i], expected[i], diff
            );
        }
        Ok(())
    }

    #[test]
    fn test_gpu_metric_update_clamping() -> Result<()> {
        let kernels = IcarusKernels::new(0)?;

        // 2 sites, dim=2, packed_size=3
        let num_sites = 2;
        let cps = 3;
        let total = num_sites * cps;

        // Metric very close to zero — gradient will push it below eps_min
        let mut metric_data = vec![0.02f32; total];
        let grad_data = vec![10.0f32; total]; // large gradient
        let ricci_data = vec![0.0f32; total];

        let alpha = 0.01f32;
        let beta = 0.0f32;
        let eps_min = 0.01f32;
        let eps_max = 100.0f32;

        // Without clamping: 0.02 + (-0.01 * 10.0) = 0.02 - 0.1 = -0.08
        // With clamping: max(-0.08, 0.01) = 0.01

        kernels.metric_update(
            &mut metric_data,
            &grad_data,
            &ricci_data,
            num_sites,
            cps,
            alpha,
            beta,
            eps_min,
            eps_max,
        )?;

        for i in 0..total {
            assert!(
                metric_data[i] >= eps_min - 1e-7,
                "Clamping failed at index {}: {} < {}",
                i, metric_data[i], eps_min
            );
        }
        Ok(())
    }

    #[test]
    fn test_gpu_transfer_parity() -> Result<()> {
        use icarus_math::lattice::LatticeLayer;
        use icarus_math::transfer::TransferOperator;

        let kernels = IcarusKernels::new(0)?;

        // E8 (8D) → Leech (24D) embedding with random weights
        let src_n = 8;
        let dst_n = 24;
        let op = TransferOperator::random_init(
            LatticeLayer::Analytical,
            LatticeLayer::Creative,
            src_n,
            dst_n,
            42,
        );

        // Create deterministic source data
        let source_re: Vec<f32> = (0..src_n).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let source_im: Vec<f32> = (0..src_n).map(|i| (i as f32 + 1.0) * -0.05).collect();

        // CPU reference (bias is zero by default, so matches GPU kernel)
        let (cpu_re, cpu_im) = op.apply(&source_re, &source_im);

        // GPU
        let (gpu_re, gpu_im) = kernels.transfer_matvec(
            &op.weights,
            &source_re,
            &source_im,
            dst_n,
            src_n,
        )?;

        assert_eq!(gpu_re.len(), dst_n);
        assert_eq!(gpu_im.len(), dst_n);

        let tol = 1e-5;
        let mut max_diff = 0.0f32;
        for i in 0..dst_n {
            let diff_re = (cpu_re[i] - gpu_re[i]).abs();
            let diff_im = (cpu_im[i] - gpu_im[i]).abs();
            max_diff = max_diff.max(diff_re).max(diff_im);
            assert!(
                diff_re < tol,
                "Transfer RE mismatch at {}: CPU={} GPU={} diff={}",
                i, cpu_re[i], gpu_re[i], diff_re
            );
            assert!(
                diff_im < tol,
                "Transfer IM mismatch at {}: CPU={} GPU={} diff={}",
                i, cpu_im[i], gpu_im[i], diff_im
            );
        }
        eprintln!("Transfer parity: max_diff={:.2e}", max_diff);
        Ok(())
    }

    #[test]
    fn test_gpu_transfer_identity() -> Result<()> {
        use icarus_math::lattice::LatticeLayer;
        use icarus_math::transfer::TransferOperator;

        let kernels = IcarusKernels::new(0)?;

        // Identity embedding: 8D → 24D
        let src_n = 8;
        let dst_n = 24;
        let op = TransferOperator::identity_init(
            LatticeLayer::Analytical,
            LatticeLayer::Creative,
            src_n,
            dst_n,
        );

        let source_re: Vec<f32> = (0..src_n).map(|i| (i + 1) as f32).collect();
        let source_im: Vec<f32> = (0..src_n).map(|i| (i + 1) as f32 * 0.5).collect();

        let (gpu_re, gpu_im) = kernels.transfer_matvec(
            &op.weights,
            &source_re,
            &source_im,
            dst_n,
            src_n,
        )?;

        // First 8 elements should match source exactly
        let tol = 1e-6;
        for i in 0..src_n {
            assert!(
                (gpu_re[i] - source_re[i]).abs() < tol,
                "Identity embedding RE mismatch at {}: {} vs {}",
                i, gpu_re[i], source_re[i]
            );
            assert!(
                (gpu_im[i] - source_im[i]).abs() < tol,
                "Identity embedding IM mismatch at {}: {} vs {}",
                i, gpu_im[i], source_im[i]
            );
        }
        // Remaining 16 elements should be zero
        for i in src_n..dst_n {
            assert!(gpu_re[i].abs() < tol, "Padding RE not zero at {}: {}", i, gpu_re[i]);
            assert!(gpu_im[i].abs() < tol, "Padding IM not zero at {}: {}", i, gpu_im[i]);
        }
        Ok(())
    }

    #[test]
    fn test_gpu_backend_trait() -> Result<()> {
        use crate::pipeline::{ComputeBackend, GpuBackend};

        let mut backend = GpuBackend::new(0)?;
        assert_eq!(backend.name(), "CUDA");

        let mut field = make_e8_field(42, 0.5);
        let params = RAEParams::default_e8();

        backend.rae_step(&mut field, &params, 5)?;

        let (total, kinetic, potential) = backend.free_energy(&field, &params.energy_params)?;
        assert!(total > 0.0);
        assert!(kinetic >= 0.0);
        assert!(potential >= 0.0);
        Ok(())
    }
}
