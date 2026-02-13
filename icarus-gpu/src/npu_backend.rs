//! NPU compute backend — offloads matrix operations to Intel NPU via TCP bridge.
//!
//! Uses the NpuBridgeClient to delegate transfer_matvec to the NPU hardware,
//! while falling back to CPU for PDE-based operations (rae_step, free_energy,
//! metric_update) that are not structured as simple matmul/matvec.

use anyhow::Result;
use icarus_field::phase_field::LatticeField;
use icarus_field::rae::RAEParams;

use crate::npu_client::NpuBridgeClient;
use crate::pipeline::{ComputeBackend, CpuBackend};

/// NPU compute backend that delegates matrix operations to the Intel NPU
/// via the Windows bridge, with CPU fallback for non-matmul operations.
pub struct NpuBackend {
    client: NpuBridgeClient,
    cpu: CpuBackend,
}

impl NpuBackend {
    /// Connect using env vars or WSL2 gateway auto-detection.
    pub fn new() -> Result<Self> {
        let client = NpuBridgeClient::connect()
            .map_err(|e| anyhow::anyhow!("NPU bridge connection failed: {}", e))?;
        Ok(Self {
            client,
            cpu: CpuBackend,
        })
    }

    /// Connect to a specific host:port.
    pub fn connect_to(host: &str, port: u16) -> Result<Self> {
        let client = NpuBridgeClient::connect_to(host, port)
            .map_err(|e| anyhow::anyhow!("NPU bridge connection failed: {}", e))?;
        Ok(Self {
            client,
            cpu: CpuBackend,
        })
    }

    /// Get a mutable reference to the underlying bridge client.
    pub fn client_mut(&mut self) -> &mut NpuBridgeClient {
        &mut self.client
    }
}

impl ComputeBackend for NpuBackend {
    fn rae_step(
        &mut self,
        field: &mut LatticeField,
        params: &RAEParams,
        num_steps: u64,
    ) -> Result<()> {
        self.cpu.rae_step(field, params, num_steps)
    }

    fn free_energy(
        &mut self,
        field: &LatticeField,
        params: &icarus_field::free_energy::FreeEnergyParams,
    ) -> Result<(f32, f32, f32)> {
        self.cpu.free_energy(field, params)
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
        self.cpu.metric_update(
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
        let (y_re, _dur_re) = self
            .client
            .matvec(dst_n as u32, src_n as u32, weights, source_re)
            .map_err(|e| anyhow::anyhow!("NPU matvec (re) failed: {}", e))?;
        let (y_im, _dur_im) = self
            .client
            .matvec(dst_n as u32, src_n as u32, weights, source_im)
            .map_err(|e| anyhow::anyhow!("NPU matvec (im) failed: {}", e))?;
        Ok((y_re, y_im))
    }

    fn name(&self) -> &str {
        "NPU"
    }
}
