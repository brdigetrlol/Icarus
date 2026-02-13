// Copyright (c) 2025-2026 brdigetrlol. All rights reserved.
// SPDX-License-Identifier: LicenseRef-Icarus-Proprietary
// See LICENSE in the repository root for full license terms.

//! TCP client for the Icarus NPU Bridge running on Windows.
//!
//! Connects to the bridge server over TCP and sends binary protocol
//! requests for matrix operations on the Intel NPU.

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{BufWriter, Read, Write};
use std::net::TcpStream;
use std::time::{Duration, Instant};

// Operation codes (must match protocol.rs on Windows side)
const OP_MATMUL: u32 = 1;
const OP_MATVEC: u32 = 2;
#[allow(dead_code)]
const OP_BATCH_MATMUL: u32 = 3;
const OP_PING: u32 = 100;
const OP_DEVICE_INFO: u32 = 101;
const OP_BENCHMARK: u32 = 102;

// Response status
const STATUS_OK: u32 = 0;

const DEFAULT_PORT: u16 = 9876;
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
const READ_TIMEOUT: Duration = Duration::from_secs(120);

pub struct NpuBridgeClient {
    stream: TcpStream,
}

impl NpuBridgeClient {
    /// Connect to the NPU bridge. Uses ICARUS_NPU_HOST and ICARUS_NPU_PORT
    /// environment variables, falling back to WSL2 gateway auto-detection.
    pub fn connect() -> Result<Self, String> {
        let host = std::env::var("ICARUS_NPU_HOST").unwrap_or_else(|_| detect_wsl2_gateway());
        let port: u16 = std::env::var("ICARUS_NPU_PORT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_PORT);

        Self::connect_to(&host, port)
    }

    pub fn connect_to(host: &str, port: u16) -> Result<Self, String> {
        let addr = format!("{}:{}", host, port);
        let stream = TcpStream::connect_timeout(
            &addr.parse().map_err(|e| format!("Invalid address '{}': {}", addr, e))?,
            CONNECT_TIMEOUT,
        )
        .map_err(|e| format!("Failed to connect to NPU bridge at {}: {}", addr, e))?;

        stream
            .set_read_timeout(Some(READ_TIMEOUT))
            .map_err(|e| format!("Failed to set read timeout: {}", e))?;
        stream
            .set_nodelay(true)
            .map_err(|e| format!("Failed to set TCP_NODELAY: {}", e))?;

        Ok(Self { stream })
    }

    /// Send a request and read the response.
    fn request(&mut self, op_code: u32, payload: &[u8]) -> Result<Vec<u8>, String> {
        let mut writer = BufWriter::new(&self.stream);

        writer
            .write_u32::<LittleEndian>(op_code)
            .map_err(|e| format!("Write op_code failed: {}", e))?;
        writer
            .write_u32::<LittleEndian>(payload.len() as u32)
            .map_err(|e| format!("Write payload_len failed: {}", e))?;
        if !payload.is_empty() {
            writer
                .write_all(payload)
                .map_err(|e| format!("Write payload failed: {}", e))?;
        }
        writer
            .flush()
            .map_err(|e| format!("Flush failed: {}", e))?;

        // Read response
        let mut reader = &self.stream;
        let status = reader
            .read_u32::<LittleEndian>()
            .map_err(|e| format!("Read status failed: {}", e))?;
        let resp_len = reader
            .read_u32::<LittleEndian>()
            .map_err(|e| format!("Read response length failed: {}", e))? as usize;

        let mut resp_payload = vec![0u8; resp_len];
        if resp_len > 0 {
            reader
                .read_exact(&mut resp_payload)
                .map_err(|e| format!("Read response payload failed: {}", e))?;
        }

        if status != STATUS_OK {
            let msg = String::from_utf8_lossy(&resp_payload).to_string();
            return Err(format!("NPU bridge error: {}", msg));
        }

        Ok(resp_payload)
    }

    /// Ping the bridge server.
    pub fn ping(&mut self) -> Result<String, String> {
        let t0 = Instant::now();
        let resp = self.request(OP_PING, &[])?;
        let elapsed = t0.elapsed();
        let msg = String::from_utf8_lossy(&resp).to_string();
        Ok(format!("{} ({}ms)", msg, elapsed.as_millis()))
    }

    /// Get device info from the bridge.
    pub fn device_info(&mut self) -> Result<String, String> {
        let resp = self.request(OP_DEVICE_INFO, &[])?;
        Ok(String::from_utf8_lossy(&resp).to_string())
    }

    /// Run benchmark on the NPU.
    pub fn benchmark(&mut self) -> Result<String, String> {
        let resp = self.request(OP_BENCHMARK, &[])?;
        Ok(String::from_utf8_lossy(&resp).to_string())
    }

    /// Matrix multiplication: C[M,N] = A[M,K] * B[K,N]
    pub fn matmul(
        &mut self,
        m: u32,
        k: u32,
        n: u32,
        a: &[f32],
        b: &[f32],
    ) -> Result<(Vec<f32>, Duration), String> {
        if a.len() != (m * k) as usize {
            return Err(format!(
                "A has {} elements, expected {}x{}={}",
                a.len(),
                m,
                k,
                m * k
            ));
        }
        if b.len() != (k * n) as usize {
            return Err(format!(
                "B has {} elements, expected {}x{}={}",
                b.len(),
                k,
                n,
                k * n
            ));
        }

        let mut payload = Vec::with_capacity(12 + (a.len() + b.len()) * 4);
        payload.extend_from_slice(&m.to_le_bytes());
        payload.extend_from_slice(&k.to_le_bytes());
        payload.extend_from_slice(&n.to_le_bytes());
        for &v in a {
            payload.extend_from_slice(&v.to_le_bytes());
        }
        for &v in b {
            payload.extend_from_slice(&v.to_le_bytes());
        }

        let t0 = Instant::now();
        let resp = self.request(OP_MATMUL, &payload)?;
        let elapsed = t0.elapsed();

        let result = decode_f32_slice(&resp);
        if result.len() != (m * n) as usize {
            return Err(format!(
                "Expected {} output elements, got {}",
                m * n,
                result.len()
            ));
        }

        Ok((result, elapsed))
    }

    /// Matrix-vector multiplication: y[M] = W[M,N] * x[N]
    pub fn matvec(
        &mut self,
        m: u32,
        n: u32,
        w: &[f32],
        x: &[f32],
    ) -> Result<(Vec<f32>, Duration), String> {
        if w.len() != (m * n) as usize {
            return Err(format!(
                "W has {} elements, expected {}x{}={}",
                w.len(),
                m,
                n,
                m * n
            ));
        }
        if x.len() != n as usize {
            return Err(format!("x has {} elements, expected {}", x.len(), n));
        }

        let mut payload = Vec::with_capacity(8 + (w.len() + x.len()) * 4);
        payload.extend_from_slice(&m.to_le_bytes());
        payload.extend_from_slice(&n.to_le_bytes());
        for &v in w {
            payload.extend_from_slice(&v.to_le_bytes());
        }
        for &v in x {
            payload.extend_from_slice(&v.to_le_bytes());
        }

        let t0 = Instant::now();
        let resp = self.request(OP_MATVEC, &payload)?;
        let elapsed = t0.elapsed();

        let result = decode_f32_slice(&resp);
        if result.len() != m as usize {
            return Err(format!(
                "Expected {} output elements, got {}",
                m,
                result.len()
            ));
        }

        Ok((result, elapsed))
    }

    /// Batch matrix multiplication: multiple matmuls in one round-trip.
    #[allow(dead_code)]
    pub fn batch_matmul(
        &mut self,
        ops: &[(u32, u32, u32, Vec<f32>, Vec<f32>)],
    ) -> Result<(Vec<Vec<f32>>, Duration), String> {
        let mut payload = Vec::new();
        payload.extend_from_slice(&(ops.len() as u32).to_le_bytes());

        for (m, k, n, a, b) in ops {
            payload.extend_from_slice(&m.to_le_bytes());
            payload.extend_from_slice(&k.to_le_bytes());
            payload.extend_from_slice(&n.to_le_bytes());
            for &v in a {
                payload.extend_from_slice(&v.to_le_bytes());
            }
            for &v in b {
                payload.extend_from_slice(&v.to_le_bytes());
            }
        }

        let t0 = Instant::now();
        let resp = self.request(OP_BATCH_MATMUL, &payload)?;
        let elapsed = t0.elapsed();

        // Parse response: [4B count] then count * [4B len] [len bytes data]
        if resp.len() < 4 {
            return Err("Batch response too short".into());
        }
        let count = u32::from_le_bytes([resp[0], resp[1], resp[2], resp[3]]) as usize;
        let mut offset = 4;
        let mut results = Vec::with_capacity(count);

        for _ in 0..count {
            if offset + 4 > resp.len() {
                return Err("Batch response truncated".into());
            }
            let len = u32::from_le_bytes([
                resp[offset],
                resp[offset + 1],
                resp[offset + 2],
                resp[offset + 3],
            ]) as usize;
            offset += 4;

            if offset + len > resp.len() {
                return Err("Batch response item truncated".into());
            }
            let data = decode_f32_slice(&resp[offset..offset + len]);
            offset += len;
            results.push(data);
        }

        Ok((results, elapsed))
    }
}

/// Auto-detect the Windows host IP from WSL2 by reading the default gateway.
/// Falls back to 127.0.0.1 if detection fails (works with mirrored networking).
fn detect_wsl2_gateway() -> String {
    // Try reading the default route — the gateway is the Windows host
    if let Ok(output) = std::process::Command::new("ip")
        .args(["route", "show", "default"])
        .output()
    {
        let stdout = String::from_utf8_lossy(&output.stdout);
        // Format: "default via <IP> dev eth0 ..."
        if let Some(ip) = stdout.split_whitespace().nth(2) {
            if ip.parse::<std::net::Ipv4Addr>().is_ok() {
                return ip.to_string();
            }
        }
    }
    "127.0.0.1".to_string()
}

fn decode_f32_slice(bytes: &[u8]) -> Vec<f32> {
    let mut result = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        result.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    result
}
