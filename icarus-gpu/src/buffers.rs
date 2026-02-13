//! Typed GPU buffer wrappers for the Icarus EMC
//!
//! Bundles a CudaSlice with its logical length and provides upload/download.

use anyhow::{Context, Result};
use cudarc::driver::{CudaSlice, CudaStream, DeviceRepr, ValidAsZeroBits};
use std::sync::Arc;

/// A typed GPU buffer with known length.
pub struct GpuBuffer<T: DeviceRepr> {
    pub slice: CudaSlice<T>,
    pub len: usize,
}

impl<T: DeviceRepr + ValidAsZeroBits> GpuBuffer<T> {
    /// Allocate a zeroed buffer.
    pub fn zeros(stream: &Arc<CudaStream>, len: usize) -> Result<Self> {
        let slice = stream.alloc_zeros(len).context("GpuBuffer::zeros")?;
        Ok(Self { slice, len })
    }
}

impl<T: DeviceRepr> GpuBuffer<T> {
    /// Upload from host slice.
    pub fn from_host(stream: &Arc<CudaStream>, data: &[T]) -> Result<Self> {
        let slice = stream
            .clone_htod(data)
            .context("GpuBuffer::from_host")?;
        Ok(Self {
            len: data.len(),
            slice,
        })
    }

    /// Download to host Vec.
    pub fn to_host(&self, stream: &Arc<CudaStream>) -> Result<Vec<T>> {
        stream
            .clone_dtoh(&self.slice)
            .context("GpuBuffer::to_host")
    }

    /// Upload new data in-place (must be same length).
    pub fn upload(&mut self, stream: &Arc<CudaStream>, data: &[T]) -> Result<()> {
        assert_eq!(data.len(), self.len, "upload length mismatch");
        stream
            .memcpy_htod(data, &mut self.slice)
            .context("GpuBuffer::upload")?;
        Ok(())
    }
}

/// All the field buffers needed on GPU for the RAE solver.
pub struct GpuFieldBuffers {
    pub values_re: GpuBuffer<f32>,
    pub values_im: GpuBuffer<f32>,
    pub neighbor_offsets: GpuBuffer<u32>,
    pub neighbor_indices: GpuBuffer<u32>,
    pub neighbor_weights: GpuBuffer<f32>,
    pub num_sites: usize,
}

impl GpuFieldBuffers {
    /// Upload a LatticeField to the GPU.
    pub fn from_field(
        stream: &Arc<CudaStream>,
        field: &icarus_field::phase_field::LatticeField,
    ) -> Result<Self> {
        Ok(Self {
            values_re: GpuBuffer::from_host(stream, &field.values_re)?,
            values_im: GpuBuffer::from_host(stream, &field.values_im)?,
            neighbor_offsets: GpuBuffer::from_host(stream, &field.neighbor_offsets)?,
            neighbor_indices: GpuBuffer::from_host(stream, &field.neighbor_indices)?,
            neighbor_weights: GpuBuffer::from_host(stream, &field.neighbor_weights)?,
            num_sites: field.num_sites,
        })
    }

    /// Download phase field values back to a CPU LatticeField.
    pub fn download_to_field(
        &self,
        stream: &Arc<CudaStream>,
        field: &mut icarus_field::phase_field::LatticeField,
    ) -> Result<()> {
        let re = self.values_re.to_host(stream)?;
        let im = self.values_im.to_host(stream)?;
        field.values_re.copy_from_slice(&re);
        field.values_im.copy_from_slice(&im);
        Ok(())
    }
}
