// Copyright (c) 2025-2026 brdigetrlol. All rights reserved.
// SPDX-License-Identifier: LicenseRef-Icarus-Proprietary
// See LICENSE in the repository root for full license terms.

//! CUDA device wrapper for the Icarus EMC
//!
//! Thin wrapper around cudarc's CudaContext and CudaStream.

use anyhow::{Context, Result};
use cudarc::driver::{CudaContext, CudaStream};
use std::sync::Arc;

/// CUDA device wrapper with context and default stream.
pub struct IcarusDevice {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    device_id: usize,
}

impl IcarusDevice {
    /// Initialize CUDA and create a device context.
    pub fn new(device_id: usize) -> Result<Self> {
        let ctx = CudaContext::new(device_id)
            .with_context(|| format!("Failed to initialize CUDA device {}", device_id))?;
        let stream = ctx.default_stream();
        Ok(Self {
            ctx,
            stream,
            device_id,
        })
    }

    pub fn ctx(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Create an additional stream for concurrent kernel execution.
    pub fn create_stream(&self) -> Result<Arc<CudaStream>> {
        self.ctx
            .new_stream()
            .context("Failed to create CUDA stream")
    }

    pub fn device_id(&self) -> usize {
        self.device_id
    }
}

/// Find the CUDA include path (needed for FP16 headers etc.).
pub fn find_cuda_include() -> Option<String> {
    let paths = [
        "/usr/local/cuda/targets/x86_64-linux/include",
        "/usr/local/cuda-13.1/targets/x86_64-linux/include",
        "/usr/local/cuda/include",
    ];
    for p in paths {
        if std::path::Path::new(p).join("cuda_fp16.h").exists() {
            return Some(p.to_string());
        }
    }
    None
}
