// Copyright (c) 2025-2026 brdigetrlol. All rights reserved.
// SPDX-License-Identifier: LicenseRef-Icarus-Proprietary
// See LICENSE in the repository root for full license terms.

//! Memory Agent — Field state snapshots for pattern recall
//!
//! Periodically captures snapshots of the analytical layer's field state
//! into a ring buffer. Supports nearest-neighbor retrieval for recognizing
//! previously visited attractor basins.

use anyhow::Result;
use icarus_gpu::pipeline::ComputeBackend;

use crate::agents::{Agent, TickPhase};
use crate::manifold::CausalCrystalManifold;

/// A stored snapshot of the field state.
#[derive(Debug, Clone)]
pub struct FieldSnapshot {
    /// Tick at which this snapshot was taken
    pub tick: u64,
    /// Real parts of the field values
    pub values_re: Vec<f32>,
    /// Imaginary parts of the field values
    pub values_im: Vec<f32>,
    /// Free energy at snapshot time
    pub energy: f32,
}

/// Memory agent: stores and retrieves field state snapshots.
pub struct MemoryAgent {
    /// Ring buffer of snapshots
    snapshots: Vec<FieldSnapshot>,
    /// Maximum number of snapshots to retain
    capacity: usize,
    /// Take a snapshot every N ticks
    pub snapshot_interval: u64,
    /// Internal tick counter
    tick_counter: u64,
}

impl MemoryAgent {
    pub fn new(capacity: usize) -> Self {
        Self {
            snapshots: Vec::with_capacity(capacity.min(1024)),
            capacity,
            snapshot_interval: 10,
            tick_counter: 0,
        }
    }

    /// Number of stored snapshots.
    pub fn snapshot_count(&self) -> usize {
        self.snapshots.len()
    }

    /// Get a specific snapshot by index.
    pub fn get_snapshot(&self, idx: usize) -> Option<&FieldSnapshot> {
        self.snapshots.get(idx)
    }

    /// Find the snapshot most similar to the given field state.
    ///
    /// Returns (snapshot_index, L2_distance) or None if no snapshots exist.
    pub fn find_nearest(
        &self,
        values_re: &[f32],
        values_im: &[f32],
    ) -> Option<(usize, f32)> {
        if self.snapshots.is_empty() {
            return None;
        }

        let mut best_idx = 0;
        let mut best_dist = f32::MAX;

        for (idx, snap) in self.snapshots.iter().enumerate() {
            if snap.values_re.len() != values_re.len() {
                continue;
            }
            let dist: f32 = snap
                .values_re
                .iter()
                .zip(snap.values_im.iter())
                .zip(values_re.iter().zip(values_im.iter()))
                .map(|((&sr, &si), (&cr, &ci))| {
                    let dr = sr - cr;
                    let di = si - ci;
                    dr * dr + di * di
                })
                .sum::<f32>()
                .sqrt();

            if dist < best_dist {
                best_dist = dist;
                best_idx = idx;
            }
        }

        Some((best_idx, best_dist))
    }
}

impl Agent for MemoryAgent {
    fn name(&self) -> &str {
        "memory"
    }

    fn phases(&self) -> &[TickPhase] {
        &[TickPhase::Post]
    }

    fn tick(
        &mut self,
        _phase: TickPhase,
        manifold: &mut CausalCrystalManifold,
        _backend: &mut dyn ComputeBackend,
    ) -> Result<()> {
        self.tick_counter += 1;

        if self.tick_counter % self.snapshot_interval != 0 {
            return Ok(());
        }

        // Snapshot the first (analytical) layer
        if let Some(layer) = manifold.layers.first() {
            let params = &layer.solver.params.energy_params;
            let (energy, _, _) =
                icarus_field::free_energy::free_energy(&layer.field, params);

            let snapshot = FieldSnapshot {
                tick: manifold.tick,
                values_re: layer.field.values_re.clone(),
                values_im: layer.field.values_im.clone(),
                energy,
            };

            if self.snapshots.len() >= self.capacity {
                self.snapshots.remove(0);
            }
            self.snapshots.push(snapshot);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ManifoldConfig;
    use crate::manifold::CausalCrystalManifold;
    use icarus_gpu::pipeline::CpuBackend;

    fn setup() -> (CausalCrystalManifold, CpuBackend) {
        let config = ManifoldConfig::e8_only();
        let mut manifold = CausalCrystalManifold::new(config);
        manifold.init_random(42, 0.5);
        (manifold, CpuBackend)
    }

    #[test]
    fn test_memory_new() {
        let agent = MemoryAgent::new(10);
        assert_eq!(agent.snapshot_count(), 0);
        assert_eq!(agent.snapshot_interval, 10);
        assert_eq!(agent.name(), "memory");
        assert_eq!(agent.phases(), &[TickPhase::Post]);
    }

    #[test]
    fn test_snapshot_interval() {
        let (mut manifold, mut backend) = setup();
        let mut agent = MemoryAgent::new(100);
        agent.snapshot_interval = 5;

        // Ticks 1-4: no snapshot
        for _ in 0..4 {
            agent.tick(TickPhase::Post, &mut manifold, &mut backend).unwrap();
        }
        assert_eq!(agent.snapshot_count(), 0);

        // Tick 5: snapshot taken
        agent.tick(TickPhase::Post, &mut manifold, &mut backend).unwrap();
        assert_eq!(agent.snapshot_count(), 1);

        // Ticks 6-9: no new snapshot
        for _ in 0..4 {
            agent.tick(TickPhase::Post, &mut manifold, &mut backend).unwrap();
        }
        assert_eq!(agent.snapshot_count(), 1);

        // Tick 10: second snapshot
        agent.tick(TickPhase::Post, &mut manifold, &mut backend).unwrap();
        assert_eq!(agent.snapshot_count(), 2);
    }

    #[test]
    fn test_ring_buffer_eviction() {
        let (mut manifold, mut backend) = setup();
        let mut agent = MemoryAgent::new(3); // capacity 3
        agent.snapshot_interval = 1;

        // Take 5 snapshots (only 3 should remain)
        for _ in 0..5 {
            manifold.tick += 1;
            agent.tick(TickPhase::Post, &mut manifold, &mut backend).unwrap();
        }

        assert_eq!(agent.snapshot_count(), 3);
        // Oldest snapshots should have been evicted; newest tick should be present
        let last = agent.get_snapshot(2).unwrap();
        assert_eq!(last.tick, 5);
    }

    #[test]
    fn test_get_snapshot() {
        let (mut manifold, mut backend) = setup();
        let mut agent = MemoryAgent::new(10);
        agent.snapshot_interval = 1;

        manifold.tick = 42;
        agent.tick(TickPhase::Post, &mut manifold, &mut backend).unwrap();

        let snap = agent.get_snapshot(0).unwrap();
        assert_eq!(snap.tick, 42);
        assert!(!snap.values_re.is_empty());
        assert_eq!(snap.values_re.len(), snap.values_im.len());
        assert!(snap.energy.is_finite());

        assert!(agent.get_snapshot(1).is_none());
    }

    #[test]
    fn test_find_nearest_empty() {
        let agent = MemoryAgent::new(10);
        assert!(agent.find_nearest(&[1.0], &[0.0]).is_none());
    }

    #[test]
    fn test_find_nearest_exact_match() {
        let (mut manifold, mut backend) = setup();
        let mut agent = MemoryAgent::new(10);
        agent.snapshot_interval = 1;

        agent.tick(TickPhase::Post, &mut manifold, &mut backend).unwrap();
        let snap = agent.get_snapshot(0).unwrap();
        let re = snap.values_re.clone();
        let im = snap.values_im.clone();

        let (idx, dist) = agent.find_nearest(&re, &im).unwrap();
        assert_eq!(idx, 0);
        assert!(dist < 1e-6, "Exact match should have ~zero distance, got {dist}");
    }

    #[test]
    fn test_find_nearest_selects_closest() {
        let (mut manifold, mut backend) = setup();
        let mut agent = MemoryAgent::new(10);
        agent.snapshot_interval = 1;

        // Snapshot 0: initial state
        manifold.tick = 1;
        agent.tick(TickPhase::Post, &mut manifold, &mut backend).unwrap();
        let snap0_re = agent.get_snapshot(0).unwrap().values_re.clone();
        let snap0_im = agent.get_snapshot(0).unwrap().values_im.clone();

        // Perturb the field significantly
        let n = manifold.layers[0].field.num_sites;
        for i in 0..n {
            manifold.layers[0].field.values_re[i] += 10.0;
        }

        // Snapshot 1: perturbed state
        manifold.tick = 2;
        agent.tick(TickPhase::Post, &mut manifold, &mut backend).unwrap();

        // Query with original state — should match snapshot 0
        let (idx, _) = agent.find_nearest(&snap0_re, &snap0_im).unwrap();
        assert_eq!(idx, 0);
    }
}
