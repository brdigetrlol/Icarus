// Copyright (c) 2025-2026 brdigetrlol. All rights reserved.
// SPDX-License-Identifier: LicenseRef-Icarus-Proprietary
// See LICENSE in the repository root for full license terms.

//! Icarus MCP server implementation — actor model with command channel.
//!
//! The EMC runs in a background tokio task. MCP tools communicate via
//! mpsc commands with oneshot replies. Read-only snapshots are published
//! to shared state for zero-cost status queries.

use std::collections::VecDeque;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use serde_json::{json, Value};
use tokio::sync::{mpsc, oneshot, Mutex};

use icarus_engine::agents::action::ActionOutput;
use icarus_engine::agents::planning::ConvergenceTrend;
use icarus_engine::autonomous::{
    AutoEvent, AutoEventType, AutonomousConfig, AutonomousState, EmcSnapshot, StopCondition,
};
use icarus_engine::config::{BackendSelection, ManifoldConfig};
use icarus_engine::encoding::{InputEncoder, PhaseEncoder, SpatialEncoder, SpectralEncoder};
use icarus_engine::readout::{DirectReadout, FeatureMode, LinearReadout, Readout};
use icarus_engine::training::{nmse, ReservoirTrainer};
use icarus_engine::ensemble::EnsembleTrainer;
use icarus_engine::EmergentManifoldComputer;
use icarus_gpu::npu_client::NpuBridgeClient;
use mcp_core::error::McpError;
use mcp_core::protocol::{CallToolResult, Tool};
use mcp_core::server::{McpServer, ServerInfo};

use crate::tools;

// ---------------------------------------------------------------------------
// Command channel types
// ---------------------------------------------------------------------------

enum EmcCommand {
    Init {
        config: ManifoldConfig,
        seed: u64,
        amplitude: f32,
        preset: String,
        reply: oneshot::Sender<Result<Value, String>>,
    },
    Step {
        num_ticks: u64,
        reply: oneshot::Sender<Result<Value, String>>,
    },
    Observe {
        layer_index: usize,
        max_sites: usize,
        reply: oneshot::Sender<Result<Value, String>>,
    },
    Inject {
        layer_index: usize,
        strength: f32,
        sites: Vec<(usize, f32, f32)>,
        reply: oneshot::Sender<Result<Value, String>>,
    },
    Stats {
        reply: oneshot::Sender<Result<Value, String>>,
    },
    Encode {
        encoder_name: String,
        layer_index: usize,
        offset: usize,
        scale: f32,
        ticks_after: u64,
        input: Vec<f32>,
        reply: oneshot::Sender<Result<Value, String>>,
    },
    ReadoutCmd {
        layer_index: usize,
        max_values: usize,
        format: String,
        reply: oneshot::Sender<Result<Value, String>>,
    },
    Train {
        encoder_name: String,
        layer_index: usize,
        lambda: f32,
        auto_lambda: bool,
        k_folds: usize,
        leak_rate: f32,
        feature_mode: FeatureMode,
        warmup_ticks: u64,
        ticks_per_input: u64,
        inputs: Vec<Vec<f32>>,
        targets: Vec<Vec<f32>>,
        reply: oneshot::Sender<Result<(Value, LinearReadout), String>>,
    },
    Predict {
        encoder_name: String,
        layer_index: usize,
        ticks_per_input: u64,
        leak_rate: f32,
        inputs: Vec<Vec<f32>>,
        readout: LinearReadout,
        reply: oneshot::Sender<Result<Value, String>>,
    },
    AutoStart {
        config: AutonomousConfig,
        reply: oneshot::Sender<Result<(), String>>,
    },
    AutoStop {
        reply: oneshot::Sender<Result<Value, String>>,
    },
}

enum LoopAction {
    None,
    StartTicking { tps: Option<f64> },
    StopTicking,
}

// ---------------------------------------------------------------------------
// Actor state — owns the EMC, runs in the background task
// ---------------------------------------------------------------------------

struct ActorState {
    emc: Option<EmergentManifoldComputer>,
    autonomous: bool,
    auto_cfg: AutonomousConfig,
    auto_start_time: Option<Instant>,
    energy_history: Vec<f32>,
    prev_trend: ConvergenceTrend,
    stable_count: u32,
    prev_attractor_sites: Vec<Option<usize>>,
    snapshot: Arc<RwLock<Option<EmcSnapshot>>>,
    auto_state: Arc<RwLock<AutonomousState>>,
    events: Arc<RwLock<VecDeque<AutoEvent>>>,
    event_capacity: usize,
}

impl ActorState {
    fn new(
        snapshot: Arc<RwLock<Option<EmcSnapshot>>>,
        auto_state: Arc<RwLock<AutonomousState>>,
        events: Arc<RwLock<VecDeque<AutoEvent>>>,
    ) -> Self {
        Self {
            emc: None,
            autonomous: false,
            auto_cfg: AutonomousConfig::default(),
            auto_start_time: None,
            energy_history: Vec::new(),
            prev_trend: ConvergenceTrend::Unknown,
            stable_count: 0,
            prev_attractor_sites: Vec::new(),
            snapshot,
            auto_state,
            events,
            event_capacity: 256,
        }
    }

    fn handle_command(&mut self, cmd: EmcCommand) -> LoopAction {
        match cmd {
            EmcCommand::Init { config, seed, amplitude, preset, reply } => {
                let result = self.do_init(config, seed, amplitude, &preset);
                let _ = reply.send(result);
                LoopAction::None
            }
            EmcCommand::Step { num_ticks, reply } => {
                let result = self.do_step(num_ticks);
                let _ = reply.send(result);
                LoopAction::None
            }
            EmcCommand::Observe { layer_index, max_sites, reply } => {
                let result = self.do_observe(layer_index, max_sites);
                let _ = reply.send(result);
                LoopAction::None
            }
            EmcCommand::Inject { layer_index, strength, sites, reply } => {
                let result = self.do_inject(layer_index, strength, &sites);
                let _ = reply.send(result);
                LoopAction::None
            }
            EmcCommand::Stats { reply } => {
                let result = self.do_stats();
                let _ = reply.send(result);
                LoopAction::None
            }
            EmcCommand::Encode { encoder_name, layer_index, offset, scale, ticks_after, input, reply } => {
                let result = self.do_encode(&encoder_name, layer_index, offset, scale, ticks_after, &input);
                let _ = reply.send(result);
                LoopAction::None
            }
            EmcCommand::ReadoutCmd { layer_index, max_values, format, reply } => {
                let result = self.do_readout(layer_index, max_values, &format);
                let _ = reply.send(result);
                LoopAction::None
            }
            EmcCommand::Train { encoder_name, layer_index, lambda, auto_lambda, k_folds, leak_rate, feature_mode, warmup_ticks, ticks_per_input, inputs, targets, reply } => {
                let result = self.do_train(&encoder_name, layer_index, lambda, auto_lambda, k_folds, leak_rate, feature_mode, warmup_ticks, ticks_per_input, &inputs, &targets);
                let _ = reply.send(result);
                LoopAction::None
            }
            EmcCommand::Predict { encoder_name, layer_index, ticks_per_input, leak_rate, inputs, readout, reply } => {
                let result = self.do_predict(&encoder_name, layer_index, ticks_per_input, leak_rate, &inputs, &readout);
                let _ = reply.send(result);
                LoopAction::None
            }
            EmcCommand::AutoStart { config, reply } => {
                let tps = config.max_ticks_per_second;
                match self.do_auto_start(config) {
                    Ok(()) => {
                        let _ = reply.send(Ok(()));
                        LoopAction::StartTicking { tps }
                    }
                    Err(e) => {
                        let _ = reply.send(Err(e));
                        LoopAction::None
                    }
                }
            }
            EmcCommand::AutoStop { reply } => {
                let result = self.do_auto_stop();
                let _ = reply.send(result);
                LoopAction::StopTicking
            }
        }
    }

    // ── Init ──

    fn do_init(&mut self, config: ManifoldConfig, seed: u64, amplitude: f32, preset: &str) -> Result<Value, String> {
        let emc_result = match config.backend {
            BackendSelection::Gpu { .. } => EmergentManifoldComputer::new(config.clone()),
            BackendSelection::Cpu => Ok(EmergentManifoldComputer::new_cpu(config.clone())),
            BackendSelection::Npu => EmergentManifoldComputer::new(config.clone()),
        };

        let mut emc = match emc_result {
            Ok(e) => e,
            Err(e) => {
                tracing::warn!("GPU init failed ({e}), falling back to CPU");
                let mut cpu_config = config;
                cpu_config.backend = BackendSelection::Cpu;
                EmergentManifoldComputer::new_cpu(cpu_config)
            }
        };

        emc.init_random(seed, amplitude);

        let stats = emc.stats();
        let info = json!({
            "status": "initialized",
            "preset": preset,
            "backend": emc.backend_name(),
            "layers": stats.layer_stats.iter().map(|ls| {
                json!({
                    "layer": format!("{:?}", ls.layer),
                    "num_sites": ls.num_sites,
                    "initial_energy": ls.total_energy,
                })
            }).collect::<Vec<_>>(),
            "total_sites": stats.total_sites,
            "memory_bytes": stats.memory_bytes,
            "seed": seed,
            "amplitude": amplitude,
        });

        // Reset autonomous state
        self.autonomous = false;
        self.energy_history.clear();
        self.prev_trend = ConvergenceTrend::Unknown;
        self.stable_count = 0;
        self.prev_attractor_sites.clear();
        self.set_auto_state(AutonomousState::Idle);

        self.emc = Some(emc);
        self.publish_snapshot();

        Ok(info)
    }

    // ── Step ──

    fn do_step(&mut self, num_ticks: u64) -> Result<Value, String> {
        if self.autonomous {
            return Err("Cannot manually step while autonomous mode is active. Call icarus_auto_stop first.".into());
        }
        let emc = self.emc.as_mut().ok_or_else(|| "EMC not initialized. Call icarus_init first.".to_string())?;

        emc.run(num_ticks).map_err(|e| format!("Step failed: {e}"))?;

        let stats = emc.stats();
        let layer_info: Vec<_> = stats.layer_stats.iter().map(|ls| {
            json!({
                "layer": format!("{:?}", ls.layer),
                "total_energy": ls.total_energy,
                "kinetic_energy": ls.kinetic_energy,
                "potential_energy": ls.potential_energy,
                "mean_amplitude": ls.mean_amplitude,
            })
        }).collect();

        let result = json!({
            "tick": stats.tick,
            "ticks_executed": num_ticks,
            "layers": layer_info,
        });

        self.publish_snapshot();
        Ok(result)
    }

    // ── Observe ──

    fn do_observe(&self, layer_index: usize, max_sites: usize) -> Result<Value, String> {
        let emc = self.emc.as_ref().ok_or_else(|| "EMC not initialized. Call icarus_init first.".to_string())?;
        let obs = emc.observe();

        if layer_index >= obs.layer_states.len() {
            return Err(format!(
                "Layer index {} out of range (have {} layers)",
                layer_index, obs.layer_states.len()
            ));
        }

        let state = &obs.layer_states[layer_index];
        let n = state.values_re.len().min(max_sites);

        let sites: Vec<_> = (0..n).map(|i| {
            let re = state.values_re[i];
            let im = state.values_im[i];
            let amp = (re * re + im * im).sqrt();
            let phase = im.atan2(re);
            json!({
                "site": i,
                "re": format!("{:.6}", re),
                "im": format!("{:.6}", im),
                "amplitude": format!("{:.6}", amp),
                "phase": format!("{:.4}", phase),
            })
        }).collect();

        Ok(json!({
            "tick": obs.tick,
            "layer": format!("{:?}", state.layer),
            "layer_index": layer_index,
            "total_sites": state.values_re.len(),
            "showing": n,
            "energy": state.energy,
            "sites": sites,
        }))
    }

    // ── Inject ──

    fn do_inject(&mut self, layer_index: usize, strength: f32, sites: &[(usize, f32, f32)]) -> Result<Value, String> {
        let emc = self.emc.as_mut().ok_or_else(|| "EMC not initialized. Call icarus_init first.".to_string())?;

        if layer_index >= emc.manifold.layers.len() {
            return Err(format!("Layer index {} out of range", layer_index));
        }

        let layer = &mut emc.manifold.layers[layer_index];
        let mut injected = 0usize;

        for &(site, re, im) in sites {
            if site < layer.field.num_sites {
                if (strength - 1.0).abs() < 1e-6 {
                    layer.field.set(site, re, im);
                } else {
                    let (old_re, old_im) = layer.field.get(site);
                    layer.field.set(
                        site,
                        old_re * (1.0 - strength) + re * strength,
                        old_im * (1.0 - strength) + im * strength,
                    );
                }
                injected += 1;
            }
        }

        self.publish_snapshot();

        Ok(json!({
            "status": "injected",
            "layer_index": layer_index,
            "sites_injected": injected,
            "strength": strength,
        }))
    }

    // ── Stats ──

    fn do_stats(&self) -> Result<Value, String> {
        let emc = self.emc.as_ref().ok_or_else(|| "EMC not initialized. Call icarus_init first.".to_string())?;
        let stats = emc.stats();

        let layers: Vec<_> = stats.layer_stats.iter().map(|ls| {
            json!({
                "layer": format!("{:?}", ls.layer),
                "num_sites": ls.num_sites,
                "total_energy": ls.total_energy,
                "kinetic_energy": ls.kinetic_energy,
                "potential_energy": ls.potential_energy,
                "mean_amplitude": ls.mean_amplitude,
            })
        }).collect();

        Ok(json!({
            "tick": stats.tick,
            "backend": stats.backend_name,
            "total_sites": stats.total_sites,
            "memory_bytes": stats.memory_bytes,
            "layers": layers,
        }))
    }

    // ── Encode ──

    fn do_encode(&mut self, encoder_name: &str, layer_index: usize, offset: usize, scale: f32, ticks_after: u64, input: &[f32]) -> Result<Value, String> {
        let emc = self.emc.as_mut().ok_or_else(|| "EMC not initialized. Call icarus_init first.".to_string())?;

        if layer_index >= emc.manifold.layers.len() {
            return Err(format!(
                "Layer index {} out of range (have {} layers)",
                layer_index, emc.manifold.layers.len()
            ));
        }

        let encoder: Box<dyn InputEncoder> = match encoder_name {
            "phase" => Box::new(PhaseEncoder { offset }),
            "spectral" => Box::new(SpectralEncoder::new()),
            _ => Box::new(SpatialEncoder { offset, scale }),
        };

        encoder.encode(input, &mut emc.manifold.layers[layer_index].field);

        if ticks_after > 0 {
            emc.run(ticks_after).map_err(|e| format!("Post-encode step failed: {e}"))?;
        }

        let stats = emc.stats();
        let layer_stats = &stats.layer_stats[layer_index];

        Ok(json!({
            "status": "encoded",
            "encoder": encoder.name(),
            "input_length": input.len(),
            "layer_index": layer_index,
            "offset": offset,
            "ticks_after": ticks_after,
            "tick": stats.tick,
            "layer_energy": layer_stats.total_energy,
            "layer_mean_amplitude": layer_stats.mean_amplitude,
        }))
    }

    // ── Readout ──

    fn do_readout(&self, layer_index: usize, max_values: usize, format: &str) -> Result<Value, String> {
        let emc = self.emc.as_ref().ok_or_else(|| "EMC not initialized. Call icarus_init first.".to_string())?;

        if layer_index >= emc.manifold.layers.len() {
            return Err(format!(
                "Layer index {} out of range (have {} layers)",
                layer_index, emc.manifold.layers.len()
            ));
        }

        let field = &emc.manifold.layers[layer_index].field;
        let readout = DirectReadout;
        let state = readout.read(field);
        let num_sites = field.num_sites;

        let result = match format {
            "flat" => {
                let n = state.len().min(max_values * 2);
                let values: Vec<String> = state[..n].iter().map(|v| format!("{:.6}", v)).collect();
                json!({
                    "format": "flat",
                    "layer_index": layer_index,
                    "num_sites": num_sites,
                    "state_dim": state.len(),
                    "showing": n,
                    "values": values,
                })
            }
            _ => {
                let n = num_sites.min(max_values);
                let sites: Vec<_> = (0..n).map(|i| {
                    let re = state[i];
                    let im = state[num_sites + i];
                    let amp = (re * re + im * im).sqrt();
                    let phase = im.atan2(re);
                    json!({
                        "site": i,
                        "re": format!("{:.6}", re),
                        "im": format!("{:.6}", im),
                        "amplitude": format!("{:.6}", amp),
                        "phase": format!("{:.4}", phase),
                    })
                }).collect();
                json!({
                    "format": "complex",
                    "layer_index": layer_index,
                    "num_sites": num_sites,
                    "showing": n,
                    "sites": sites,
                })
            }
        };

        Ok(result)
    }

    // ── Train ──

    fn do_train(
        &mut self,
        encoder_name: &str,
        layer_index: usize,
        lambda: f32,
        auto_lambda: bool,
        k_folds: usize,
        leak_rate: f32,
        feature_mode: FeatureMode,
        warmup_ticks: u64,
        ticks_per_input: u64,
        inputs: &[Vec<f32>],
        targets: &[Vec<f32>],
    ) -> Result<(Value, LinearReadout), String> {
        let emc = self.emc.as_mut().ok_or_else(|| "EMC not initialized. Call icarus_init first.".to_string())?;

        let encoder: Box<dyn InputEncoder> = match encoder_name {
            "phase" => Box::new(PhaseEncoder { offset: 0 }),
            "spectral" => Box::new(SpectralEncoder::new()),
            _ => Box::new(SpatialEncoder::default()),
        };

        let (readout, selected_lambda) = if auto_lambda {
            let trainer = ReservoirTrainer::new(0.0, warmup_ticks, ticks_per_input)
                .with_leak_rate(leak_rate)
                .with_feature_mode(feature_mode);
            let (readout, sel_lambda) = trainer
                .train_auto(emc, encoder.as_ref(), inputs, targets, layer_index, k_folds)
                .map_err(|e| format!("Training failed: {e}"))?;
            (readout, Some(sel_lambda))
        } else {
            let trainer = ReservoirTrainer::new(lambda, warmup_ticks, ticks_per_input)
                .with_leak_rate(leak_rate)
                .with_feature_mode(feature_mode);
            let readout = trainer
                .train(emc, encoder.as_ref(), inputs, targets, layer_index)
                .map_err(|e| format!("Training failed: {e}"))?;
            (readout, None)
        };

        // Compute training NMSE
        let use_leaky = leak_rate < 1.0;
        let mut train_predictions = Vec::new();
        for input in inputs {
            if use_leaky {
                encoder.encode_leaky(input, &mut emc.manifold.layers[layer_index].field, leak_rate);
            } else {
                encoder.encode(input, &mut emc.manifold.layers[layer_index].field);
            }
            emc.run(ticks_per_input).map_err(|e| format!("Prediction step failed: {e}"))?;
            let pred = readout.read(&emc.manifold.layers[layer_index].field);
            train_predictions.push(pred);
        }

        let output_dim = targets[0].len();
        let mut nmse_per_output = Vec::with_capacity(output_dim);
        for k in 0..output_dim {
            let pred_k: Vec<f32> = train_predictions.iter().map(|p| p[k]).collect();
            let actual_k: Vec<f32> = targets.iter().map(|t| t[k]).collect();
            nmse_per_output.push(nmse(&pred_k, &actual_k));
        }

        let w_norm: f32 = readout.weights.iter().map(|w| w * w).sum::<f32>().sqrt();
        let effective_lambda = selected_lambda.unwrap_or(lambda);
        let feature_mode_str = match feature_mode {
            FeatureMode::Nonlinear => "nonlinear",
            _ => "linear",
        };

        let result = json!({
            "status": "trained",
            "encoder": encoder.name(),
            "num_samples": inputs.len(),
            "state_dim": readout.state_dim,
            "output_dim": readout.output_dim,
            "lambda": effective_lambda,
            "auto_lambda": auto_lambda,
            "selected_lambda": selected_lambda,
            "k_folds": if auto_lambda { Some(k_folds) } else { None },
            "leak_rate": leak_rate,
            "feature_mode": feature_mode_str,
            "warmup_ticks": warmup_ticks,
            "ticks_per_input": ticks_per_input,
            "layer_index": layer_index,
            "weight_norm": format!("{:.6}", w_norm),
            "train_nmse": nmse_per_output.iter().map(|v| format!("{:.6}", v)).collect::<Vec<_>>(),
        });

        Ok((result, readout))
    }

    // ── Predict ──

    fn do_predict(
        &mut self,
        encoder_name: &str,
        layer_index: usize,
        ticks_per_input: u64,
        leak_rate: f32,
        inputs: &[Vec<f32>],
        readout: &LinearReadout,
    ) -> Result<Value, String> {
        let emc = self.emc.as_mut().ok_or_else(|| "EMC not initialized. Call icarus_init first.".to_string())?;

        if layer_index >= emc.manifold.layers.len() {
            return Err(format!(
                "Layer index {} out of range (have {} layers)",
                layer_index, emc.manifold.layers.len()
            ));
        }

        let encoder: Box<dyn InputEncoder> = match encoder_name {
            "phase" => Box::new(PhaseEncoder { offset: 0 }),
            "spectral" => Box::new(SpectralEncoder::new()),
            _ => Box::new(SpatialEncoder::default()),
        };

        let use_leaky = leak_rate < 1.0;
        let mut predictions = Vec::with_capacity(inputs.len());
        for input in inputs {
            if use_leaky {
                encoder.encode_leaky(input, &mut emc.manifold.layers[layer_index].field, leak_rate);
            } else {
                encoder.encode(input, &mut emc.manifold.layers[layer_index].field);
            }
            emc.run(ticks_per_input).map_err(|e| format!("Prediction step failed: {e}"))?;
            let pred = readout.read(&emc.manifold.layers[layer_index].field);
            predictions.push(pred);
        }

        Ok(json!({
            "status": "predicted",
            "num_inputs": inputs.len(),
            "output_dim": readout.output_dim,
            "predictions": predictions.iter().map(|p| {
                p.iter().map(|v| format!("{:.6}", v)).collect::<Vec<_>>()
            }).collect::<Vec<_>>(),
        }))
    }

    // ── Autonomous start ──

    fn do_auto_start(&mut self, config: AutonomousConfig) -> Result<(), String> {
        if self.emc.is_none() {
            return Err("EMC not initialized. Call icarus_init first.".into());
        }
        if self.autonomous {
            return Err("Autonomous mode is already active.".into());
        }

        self.event_capacity = config.event_buffer_size;
        self.auto_cfg = config;
        self.autonomous = true;
        self.auto_start_time = Some(Instant::now());
        self.energy_history.clear();
        self.prev_trend = ConvergenceTrend::Unknown;
        self.stable_count = 0;
        self.prev_attractor_sites.clear();

        self.set_auto_state(AutonomousState::Running);
        self.emit_event(AutoEventType::Started);

        Ok(())
    }

    // ── Autonomous stop ──

    fn do_auto_stop(&mut self) -> Result<Value, String> {
        if !self.autonomous {
            return Err("Autonomous mode is not active.".into());
        }

        self.autonomous = false;

        let emc = self.emc.as_ref().ok_or("EMC not initialized")?;
        let tick = emc.total_ticks;
        let energy: f32 = emc.manifold.layers.iter().map(|l| {
            let params = &l.solver.params.energy_params;
            let (e, _, _) = icarus_field::free_energy::free_energy(&l.field, params);
            e
        }).sum();

        let trend_str = format!("{:?}", self.prev_trend);
        let reason = "manual_stop".to_string();

        self.set_auto_state(AutonomousState::Completed { reason: reason.clone() });
        self.emit_event(AutoEventType::Stopped { reason });
        self.publish_snapshot();

        Ok(json!({
            "status": "stopped",
            "tick": tick,
            "energy": energy,
            "convergence_trend": trend_str,
            "reason": "manual_stop",
        }))
    }

    // ── Autonomous tick (called by the actor loop) ──

    fn autonomous_tick(&mut self) -> bool {
        // Phase 1: mutable borrow of emc — tick + extract all data we need
        let (tick, total_energy, attractor_info) = {
            let emc = match self.emc.as_mut() {
                Some(e) => e,
                None => return false,
            };

            if let Err(e) = emc.tick() {
                // Can't call self methods here; collect error and handle below
                // Use the shared state directly
                if let Ok(mut w) = self.auto_state.write() {
                    *w = AutonomousState::Error { message: format!("{e}") };
                }
                self.autonomous = false;
                // Emit error event inline
                let event = AutoEvent {
                    tick: emc.total_ticks,
                    timestamp: chrono::Utc::now().to_rfc3339(),
                    event_type: AutoEventType::Error { message: format!("{e}") },
                };
                if let Ok(mut w) = self.events.write() {
                    if w.len() >= self.event_capacity { w.pop_front(); }
                    w.push_back(event);
                }
                return false;
            }

            let tick = emc.total_ticks;

            // Compute total energy from all layers
            let total_energy: f32 = emc.manifold.layers.iter().map(|l| {
                let params = &l.solver.params.energy_params;
                let (e, _, _) = icarus_field::free_energy::free_energy(&l.field, params);
                e
            }).sum();

            // Extract attractor info: Vec<(layer_idx, current_site)>
            let attractor_info: Vec<(usize, usize)> = emc.manifold.layers.iter().enumerate()
                .map(|(i, layer)| {
                    let (site, _amp) = icarus_field::geodesic::find_max_amplitude_site(&layer.field);
                    (i, site)
                })
                .collect();

            (tick, total_energy, attractor_info)
        };
        // emc mutable borrow is now dropped — safe to call self methods

        // --- Convergence detection ---
        self.energy_history.push(total_energy);
        if self.energy_history.len() > 100 {
            self.energy_history.remove(0);
        }

        let new_trend = self.compute_trend();
        if new_trend != self.prev_trend && new_trend != ConvergenceTrend::Unknown {
            self.emit_event(AutoEventType::ConvergenceDetected {
                trend: format!("{:?}", new_trend),
            });
        }

        if new_trend == ConvergenceTrend::Stable {
            self.stable_count += 1;
        } else {
            self.stable_count = 0;
        }
        self.prev_trend = new_trend;

        // --- Attractor transition detection ---
        let num_layers = attractor_info.len();
        if self.prev_attractor_sites.len() != num_layers {
            self.prev_attractor_sites = vec![None; num_layers];
        }

        for (i, site) in attractor_info {
            if let Some(prev_site) = self.prev_attractor_sites[i] {
                if prev_site != site {
                    self.emit_event(AutoEventType::AttractorTransition {
                        layer: i,
                        from_site: prev_site,
                        to_site: site,
                    });
                }
            }
            self.prev_attractor_sites[i] = Some(site);
        }

        // --- Tick milestones ---
        if tick > 0 && (tick % 10000 == 0 || tick % 1000 == 0 || tick % 100 == 0) {
            let milestone = if tick % 10000 == 0 { tick }
                else if tick % 1000 == 0 { tick }
                else { tick };
            self.emit_event(AutoEventType::TickMilestone { tick: milestone });
        }

        // --- Publish snapshot ---
        if self.auto_cfg.snapshot_interval > 0
            && tick % self.auto_cfg.snapshot_interval as u64 == 0
        {
            self.publish_snapshot();
        }

        // --- Check stop conditions ---
        if let Some(reason) = self.check_stop_conditions() {
            self.autonomous = false;
            self.set_auto_state(AutonomousState::Completed { reason: reason.clone() });
            self.emit_event(AutoEventType::Stopped { reason });
            self.publish_snapshot();
            return false;
        }

        true
    }

    // ── Helpers ──

    fn compute_trend(&self) -> ConvergenceTrend {
        if self.energy_history.len() < 3 {
            return ConvergenceTrend::Unknown;
        }
        let n = self.energy_history.len();
        let recent = &self.energy_history[n.saturating_sub(10)..];
        if recent.len() < 2 {
            return ConvergenceTrend::Unknown;
        }
        let mid = recent.len() / 2;
        let first_avg: f32 = recent[..mid].iter().sum::<f32>() / mid as f32;
        let second_avg: f32 = recent[mid..].iter().sum::<f32>() / (recent.len() - mid) as f32;
        let delta = second_avg - first_avg;
        let threshold = first_avg.abs() * 0.01;

        if delta < -threshold {
            ConvergenceTrend::Converging
        } else if delta > threshold {
            ConvergenceTrend::Diverging
        } else {
            ConvergenceTrend::Stable
        }
    }

    fn compute_action_output(&self) -> Option<ActionOutput> {
        let emc = self.emc.as_ref()?;
        let layer = emc.manifold.layers.first()?;
        let n = layer.field.num_sites;
        let params = &layer.solver.params.energy_params;
        let (energy, _, _) = icarus_field::free_energy::free_energy(&layer.field, params);

        let mut sum_amp = 0.0f32;
        let mut phase_histogram = [0.0f32; 8];
        let mut max_amp = 0.0f32;
        let mut dominant_phase = 0.0f32;

        for i in 0..n {
            let re = layer.field.values_re[i];
            let im = layer.field.values_im[i];
            let amp = (re * re + im * im).sqrt();
            sum_amp += amp;

            if amp > max_amp {
                max_amp = amp;
                dominant_phase = im.atan2(re);
            }

            if amp > 1e-8 {
                let angle = im.atan2(re);
                let normalized = (angle + std::f32::consts::PI) / (2.0 * std::f32::consts::PI);
                let bin = ((normalized * 8.0) as usize).min(7);
                phase_histogram[bin] += 1.0;
            }
        }

        let hist_total: f32 = phase_histogram.iter().sum();
        if hist_total > 0.0 {
            for h in &mut phase_histogram {
                *h /= hist_total;
            }
        }

        Some(ActionOutput {
            tick: emc.total_ticks,
            mean_amplitude: if n > 0 { sum_amp / n as f32 } else { 0.0 },
            phase_histogram,
            energy,
            dominant_phase,
        })
    }

    fn publish_snapshot(&self) {
        let emc = match self.emc.as_ref() {
            Some(e) => e,
            None => return,
        };

        let stats = emc.stats();
        let obs = emc.observe();
        let action_output = self.compute_action_output();

        let snap = EmcSnapshot {
            tick: emc.total_ticks,
            timestamp: chrono::Utc::now().to_rfc3339(),
            layer_stats: stats.layer_stats,
            total_sites: stats.total_sites,
            backend_name: stats.backend_name,
            memory_bytes: stats.memory_bytes,
            affective_state: stats.affective_state,
            convergence_trend: Some(self.prev_trend),
            action_output,
            layer_states: obs.layer_states,
        };

        if let Ok(mut w) = self.snapshot.write() {
            *w = Some(snap);
        }
    }

    fn set_auto_state(&self, state: AutonomousState) {
        if let Ok(mut w) = self.auto_state.write() {
            *w = state;
        }
    }

    fn emit_event(&self, event_type: AutoEventType) {
        let tick = self.emc.as_ref().map(|e| e.total_ticks).unwrap_or(0);
        let event = AutoEvent {
            tick,
            timestamp: chrono::Utc::now().to_rfc3339(),
            event_type,
        };
        if let Ok(mut w) = self.events.write() {
            if w.len() >= self.event_capacity {
                w.pop_front();
            }
            w.push_back(event);
        }
    }

    fn check_stop_conditions(&self) -> Option<String> {
        let emc = self.emc.as_ref()?;
        let tick = emc.total_ticks;

        for cond in &self.auto_cfg.stop_conditions {
            match cond {
                StopCondition::MaxTicks(max) => {
                    if tick >= *max {
                        return Some(format!("max_ticks_{}", max));
                    }
                }
                StopCondition::ConvergenceStable(n) => {
                    if self.stable_count >= *n {
                        return Some(format!("convergence_stable_{}", n));
                    }
                }
                StopCondition::EnergyBelow(threshold) => {
                    if let Some(&energy) = self.energy_history.last() {
                        if energy < *threshold {
                            return Some(format!("energy_below_{}", threshold));
                        }
                    }
                }
                StopCondition::TimeLimitSecs(secs) => {
                    if let Some(start) = self.auto_start_time {
                        if start.elapsed() >= Duration::from_secs(*secs) {
                            return Some(format!("time_limit_{}s", secs));
                        }
                    }
                }
                StopCondition::Manual => {}
            }
        }

        None
    }
}

// ---------------------------------------------------------------------------
// Actor loop — background tokio task
// ---------------------------------------------------------------------------

async fn emc_actor_loop(
    mut cmd_rx: mpsc::Receiver<EmcCommand>,
    snapshot: Arc<RwLock<Option<EmcSnapshot>>>,
    auto_state: Arc<RwLock<AutonomousState>>,
    events: Arc<RwLock<VecDeque<AutoEvent>>>,
) {
    let mut state = ActorState::new(snapshot, auto_state, events);
    let mut tick_period: Option<Duration> = None;

    loop {
        let cmd = if let Some(period) = tick_period {
            tokio::select! {
                biased;
                cmd = cmd_rx.recv() => cmd,
                _ = tokio::time::sleep(period) => {
                    if !state.autonomous_tick() {
                        tick_period = None;
                    }
                    continue;
                }
            }
        } else {
            cmd_rx.recv().await
        };

        match cmd {
            Some(cmd) => {
                match state.handle_command(cmd) {
                    LoopAction::None => {}
                    LoopAction::StartTicking { tps } => {
                        tick_period = Some(Duration::from_secs_f64(
                            1.0 / tps.unwrap_or(10000.0),
                        ));
                    }
                    LoopAction::StopTicking => {
                        tick_period = None;
                    }
                }
            }
            None => break,
        }
    }
}

// ---------------------------------------------------------------------------
// MCP Server
// ---------------------------------------------------------------------------

/// The Icarus MCP server.
/// State for a managed game server subprocess.
struct GameProcess {
    child: tokio::process::Child,
    port: u16,
    started: Instant,
}

/// The Icarus MCP server.
pub struct IcarusMcpServer {
    cmd_tx: mpsc::Sender<EmcCommand>,
    snapshot: Arc<RwLock<Option<EmcSnapshot>>>,
    auto_state: Arc<RwLock<AutonomousState>>,
    events: Arc<RwLock<VecDeque<AutoEvent>>>,
    trained_readout: Mutex<Option<LinearReadout>>,
    npu_client: Mutex<Option<NpuBridgeClient>>,
    ensemble: Mutex<Option<EnsembleTrainer>>,
    game_child: Mutex<Option<GameProcess>>,
}

impl IcarusMcpServer {
    pub fn new() -> Self {
        let (cmd_tx, cmd_rx) = mpsc::channel(64);
        let snapshot = Arc::new(RwLock::new(None));
        let auto_state = Arc::new(RwLock::new(AutonomousState::Idle));
        let events = Arc::new(RwLock::new(VecDeque::with_capacity(256)));

        tokio::spawn(emc_actor_loop(
            cmd_rx,
            snapshot.clone(),
            auto_state.clone(),
            events.clone(),
        ));

        Self {
            cmd_tx,
            snapshot,
            auto_state,
            events,
            trained_readout: Mutex::new(None),
            npu_client: Mutex::new(None),
            ensemble: Mutex::new(None),
            game_child: Mutex::new(None),
        }
    }

    /// Send a command and await the reply.
    async fn send_cmd<T>(
        &self,
        make_cmd: impl FnOnce(oneshot::Sender<T>) -> EmcCommand,
    ) -> Result<T, McpError> {
        let (tx, rx) = oneshot::channel();
        self.cmd_tx
            .send(make_cmd(tx))
            .await
            .map_err(|_| McpError::ToolError("EMC actor task has stopped".into()))?;
        rx.await
            .map_err(|_| McpError::ToolError("EMC actor dropped reply channel".into()))
    }

    // ── Handler methods ──

    async fn handle_init(&self, args: Value) -> Result<CallToolResult, McpError> {
        let preset = args.get("preset").and_then(|v| v.as_str()).unwrap_or("e8_only").to_string();
        let backend_str = args.get("backend").and_then(|v| v.as_str()).unwrap_or("gpu");
        let seed = args.get("seed").and_then(|v| v.as_f64()).unwrap_or(42.0) as u64;
        let amplitude = args.get("amplitude").and_then(|v| v.as_f64()).unwrap_or(0.5) as f32;

        let mut config = match preset.as_str() {
            "full_hierarchy" => ManifoldConfig::full_hierarchy(),
            _ => ManifoldConfig::e8_only(),
        };

        config.backend = match backend_str {
            "cpu" => BackendSelection::Cpu,
            "npu" => BackendSelection::Npu,
            _ => BackendSelection::Gpu { device_id: 0 },
        };

        let result = self.send_cmd(|reply| EmcCommand::Init {
            config, seed, amplitude, preset, reply,
        }).await?;

        match result {
            Ok(v) => Ok(CallToolResult::json(&v)),
            Err(e) => Err(McpError::ToolError(e)),
        }
    }

    async fn handle_step(&self, args: Value) -> Result<CallToolResult, McpError> {
        let num_ticks = args.get("num_ticks").and_then(|v| v.as_u64()).unwrap_or(1);

        let result = self.send_cmd(|reply| EmcCommand::Step { num_ticks, reply }).await?;

        match result {
            Ok(v) => Ok(CallToolResult::json(&v)),
            Err(e) => Err(McpError::ToolError(e)),
        }
    }

    async fn handle_observe(&self, args: Value) -> Result<CallToolResult, McpError> {
        let layer_index = args.get("layer_index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let max_sites = args.get("max_sites").and_then(|v| v.as_u64()).unwrap_or(50) as usize;

        let result = self.send_cmd(|reply| EmcCommand::Observe {
            layer_index, max_sites, reply,
        }).await?;

        match result {
            Ok(v) => Ok(CallToolResult::json(&v)),
            Err(e) => Err(McpError::ToolError(e)),
        }
    }

    async fn handle_inject(&self, args: Value) -> Result<CallToolResult, McpError> {
        let layer_index = args.get("layer_index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let strength = args.get("strength").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;

        let sites_val = args.get("sites")
            .ok_or_else(|| McpError::InvalidParams("missing 'sites' parameter".into()))?;
        let sites_arr = sites_val.as_array()
            .ok_or_else(|| McpError::InvalidParams("'sites' must be an array".into()))?;

        let mut sites = Vec::with_capacity(sites_arr.len());
        for entry in sites_arr {
            let triple = entry.as_array()
                .ok_or_else(|| McpError::InvalidParams("each site must be [idx, re, im]".into()))?;
            if triple.len() < 3 { continue; }
            let site = triple[0].as_u64().unwrap_or(0) as usize;
            let re = triple[1].as_f64().unwrap_or(0.0) as f32;
            let im = triple[2].as_f64().unwrap_or(0.0) as f32;
            sites.push((site, re, im));
        }

        let result = self.send_cmd(|reply| EmcCommand::Inject {
            layer_index, strength, sites, reply,
        }).await?;

        match result {
            Ok(v) => Ok(CallToolResult::json(&v)),
            Err(e) => Err(McpError::ToolError(e)),
        }
    }

    async fn handle_stats(&self) -> Result<CallToolResult, McpError> {
        let result = self.send_cmd(|reply| EmcCommand::Stats { reply }).await?;

        match result {
            Ok(v) => Ok(CallToolResult::json(&v)),
            Err(e) => Err(McpError::ToolError(e)),
        }
    }

    async fn handle_encode(&self, args: Value) -> Result<CallToolResult, McpError> {
        let encoder_name = args.get("encoder").and_then(|v| v.as_str()).unwrap_or("spatial").to_string();
        let layer_index = args.get("layer_index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let offset = args.get("offset").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let scale = args.get("scale").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
        let ticks_after = args.get("ticks_after").and_then(|v| v.as_u64()).unwrap_or(0);

        let input_val = args.get("input")
            .ok_or_else(|| McpError::InvalidParams("missing 'input' parameter".into()))?;
        let input_arr = input_val.as_array()
            .ok_or_else(|| McpError::InvalidParams("'input' must be an array of numbers".into()))?;
        let input: Vec<f32> = input_arr.iter().map(|v| v.as_f64().unwrap_or(0.0) as f32).collect();

        let result = self.send_cmd(|reply| EmcCommand::Encode {
            encoder_name, layer_index, offset, scale, ticks_after, input, reply,
        }).await?;

        match result {
            Ok(v) => Ok(CallToolResult::json(&v)),
            Err(e) => Err(McpError::ToolError(e)),
        }
    }

    async fn handle_readout(&self, args: Value) -> Result<CallToolResult, McpError> {
        let layer_index = args.get("layer_index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let max_values = args.get("max_values").and_then(|v| v.as_u64()).unwrap_or(100) as usize;
        let format = args.get("format").and_then(|v| v.as_str()).unwrap_or("complex").to_string();

        let result = self.send_cmd(|reply| EmcCommand::ReadoutCmd {
            layer_index, max_values, format, reply,
        }).await?;

        match result {
            Ok(v) => Ok(CallToolResult::json(&v)),
            Err(e) => Err(McpError::ToolError(e)),
        }
    }

    async fn handle_train(&self, args: Value) -> Result<CallToolResult, McpError> {
        let encoder_name = args.get("encoder").and_then(|v| v.as_str()).unwrap_or("spatial").to_string();
        let layer_index = args.get("layer_index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let lambda = args.get("lambda").and_then(|v| v.as_f64()).unwrap_or(1e-4) as f32;
        let auto_lambda = args.get("auto_lambda").and_then(|v| v.as_bool()).unwrap_or(false);
        let k_folds = args.get("k_folds").and_then(|v| v.as_u64()).unwrap_or(5) as usize;
        let warmup_ticks = args.get("warmup_ticks").and_then(|v| v.as_u64()).unwrap_or(10);
        let ticks_per_input = args.get("ticks_per_input").and_then(|v| v.as_u64()).unwrap_or(1);
        let leak_rate = args.get("leak_rate").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
        let feature_mode = match args.get("feature_mode").and_then(|v| v.as_str()) {
            Some("nonlinear") => FeatureMode::Nonlinear,
            _ => FeatureMode::Linear,
        };

        let inputs_val = args.get("inputs")
            .ok_or_else(|| McpError::InvalidParams("missing 'inputs' parameter".into()))?;
        let targets_val = args.get("targets")
            .ok_or_else(|| McpError::InvalidParams("missing 'targets' parameter".into()))?;

        let inputs_arr = inputs_val.as_array()
            .ok_or_else(|| McpError::InvalidParams("'inputs' must be an array of arrays".into()))?;
        let targets_arr = targets_val.as_array()
            .ok_or_else(|| McpError::InvalidParams("'targets' must be an array of arrays".into()))?;

        if inputs_arr.len() != targets_arr.len() {
            return Err(McpError::InvalidParams(format!(
                "inputs ({}) and targets ({}) must have equal length",
                inputs_arr.len(), targets_arr.len()
            )));
        }

        if inputs_arr.is_empty() {
            return Err(McpError::InvalidParams("inputs must not be empty".into()));
        }

        let inputs: Vec<Vec<f32>> = inputs_arr.iter().map(|row| {
            row.as_array().unwrap_or(&vec![]).iter()
                .map(|v| v.as_f64().unwrap_or(0.0) as f32).collect()
        }).collect();

        let targets: Vec<Vec<f32>> = targets_arr.iter().map(|row| {
            row.as_array().unwrap_or(&vec![]).iter()
                .map(|v| v.as_f64().unwrap_or(0.0) as f32).collect()
        }).collect();

        let result = self.send_cmd(|reply| EmcCommand::Train {
            encoder_name, layer_index, lambda, auto_lambda, k_folds,
            leak_rate, feature_mode, warmup_ticks, ticks_per_input,
            inputs, targets, reply,
        }).await?;

        match result {
            Ok((v, readout)) => {
                *self.trained_readout.lock().await = Some(readout);
                Ok(CallToolResult::json(&v))
            }
            Err(e) => Err(McpError::ToolError(e)),
        }
    }

    async fn handle_predict(&self, args: Value) -> Result<CallToolResult, McpError> {
        let layer_index = args.get("layer_index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let encoder_name = args.get("encoder").and_then(|v| v.as_str()).unwrap_or("spatial").to_string();
        let ticks_per_input = args.get("ticks_per_input").and_then(|v| v.as_u64()).unwrap_or(1);
        let leak_rate = args.get("leak_rate").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;

        let inputs_val = args.get("inputs")
            .ok_or_else(|| McpError::InvalidParams("missing 'inputs' parameter".into()))?;
        let inputs_arr = inputs_val.as_array()
            .ok_or_else(|| McpError::InvalidParams("'inputs' must be an array of arrays".into()))?;

        let inputs: Vec<Vec<f32>> = inputs_arr.iter().map(|row| {
            row.as_array().unwrap_or(&vec![]).iter()
                .map(|v| v.as_f64().unwrap_or(0.0) as f32).collect()
        }).collect();

        let readout_guard = self.trained_readout.lock().await;
        let readout = readout_guard.as_ref()
            .ok_or_else(|| McpError::ToolError("No trained model. Call icarus_train first.".into()))?
            .clone();
        drop(readout_guard);

        let result = self.send_cmd(|reply| EmcCommand::Predict {
            encoder_name, layer_index, ticks_per_input, leak_rate, inputs, readout, reply,
        }).await?;

        match result {
            Ok(v) => Ok(CallToolResult::json(&v)),
            Err(e) => Err(McpError::ToolError(e)),
        }
    }

    // ── Autonomous mode handlers ──

    async fn handle_auto_start(&self, args: Value) -> Result<CallToolResult, McpError> {
        let max_tps = args.get("max_ticks_per_second").and_then(|v| v.as_f64());
        let snapshot_interval = args.get("snapshot_interval").and_then(|v| v.as_u64()).unwrap_or(1) as u32;

        let mut stop_conditions = Vec::new();
        if let Some(conds) = args.get("stop_conditions").and_then(|v| v.as_array()) {
            for c in conds {
                let ctype = c.get("type").and_then(|v| v.as_str()).unwrap_or("");
                let value = c.get("value");
                match ctype {
                    "max_ticks" => {
                        if let Some(v) = value.and_then(|v| v.as_u64()) {
                            stop_conditions.push(StopCondition::MaxTicks(v));
                        }
                    }
                    "convergence_stable" => {
                        if let Some(v) = value.and_then(|v| v.as_u64()) {
                            stop_conditions.push(StopCondition::ConvergenceStable(v as u32));
                        }
                    }
                    "energy_below" => {
                        if let Some(v) = value.and_then(|v| v.as_f64()) {
                            stop_conditions.push(StopCondition::EnergyBelow(v as f32));
                        }
                    }
                    "time_limit_secs" => {
                        if let Some(v) = value.and_then(|v| v.as_u64()) {
                            stop_conditions.push(StopCondition::TimeLimitSecs(v));
                        }
                    }
                    "manual" => {
                        stop_conditions.push(StopCondition::Manual);
                    }
                    _ => {}
                }
            }
        }

        if stop_conditions.is_empty() {
            stop_conditions.push(StopCondition::Manual);
        }

        let config = AutonomousConfig {
            max_ticks_per_second: max_tps.or(Some(60.0)),
            stop_conditions,
            snapshot_interval,
            event_buffer_size: 256,
        };

        let config_summary = json!({
            "max_ticks_per_second": config.max_ticks_per_second,
            "snapshot_interval": config.snapshot_interval,
            "stop_conditions": format!("{:?}", config.stop_conditions),
        });

        let result = self.send_cmd(|reply| EmcCommand::AutoStart { config, reply }).await?;

        match result {
            Ok(()) => Ok(CallToolResult::json(&json!({
                "status": "started",
                "config": config_summary,
            }))),
            Err(e) => Err(McpError::ToolError(e)),
        }
    }

    async fn handle_auto_stop(&self) -> Result<CallToolResult, McpError> {
        let result = self.send_cmd(|reply| EmcCommand::AutoStop { reply }).await?;

        match result {
            Ok(v) => Ok(CallToolResult::json(&v)),
            Err(e) => Err(McpError::ToolError(e)),
        }
    }

    async fn handle_auto_status(&self) -> Result<CallToolResult, McpError> {
        let state_label = {
            let r = self.auto_state.read().map_err(|_| McpError::ToolError("lock poisoned".into()))?;
            r.label().to_string()
        };

        let snap = {
            let r = self.snapshot.read().map_err(|_| McpError::ToolError("lock poisoned".into()))?;
            r.clone()
        };

        let events_pending = {
            let r = self.events.read().map_err(|_| McpError::ToolError("lock poisoned".into()))?;
            r.len()
        };

        let result = if let Some(snap) = snap {
            json!({
                "state": state_label,
                "tick": snap.tick,
                "total_sites": snap.total_sites,
                "backend": snap.backend_name,
                "memory_bytes": snap.memory_bytes,
                "convergence_trend": snap.convergence_trend.map(|t| format!("{:?}", t)),
                "events_pending": events_pending,
                "timestamp": snap.timestamp,
            })
        } else {
            json!({
                "state": state_label,
                "tick": null,
                "events_pending": events_pending,
            })
        };

        Ok(CallToolResult::json(&result))
    }

    async fn handle_auto_events(&self, args: Value) -> Result<CallToolResult, McpError> {
        let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(50) as usize;

        let drained: Vec<AutoEvent> = {
            let mut w = self.events.write().map_err(|_| McpError::ToolError("lock poisoned".into()))?;
            let n = w.len().min(limit);
            w.drain(..n).collect()
        };

        let events_json: Vec<Value> = drained.iter().map(|e| {
            json!({
                "tick": e.tick,
                "timestamp": e.timestamp,
                "event": serde_json::to_value(&e.event_type).unwrap_or(json!("unknown")),
            })
        }).collect();

        Ok(CallToolResult::json(&json!({
            "count": events_json.len(),
            "events": events_json,
        })))
    }

    // ── Visualization handler ──

    async fn handle_visualize(&self, args: Value) -> Result<CallToolResult, McpError> {
        let mode = args.get("mode").and_then(|v| v.as_str()).unwrap_or("dashboard");
        let theme_str = args.get("theme").and_then(|v| v.as_str()).unwrap_or("dark");
        let layer_index = args.get("layer_index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;

        let theme = match theme_str {
            "light" => icarus_viz::template::Theme::Light,
            _ => icarus_viz::template::Theme::Dark,
        };

        let default_path = format!("/tmp/icarus-viz-{}.html", mode);
        let output_path = args.get("output_path")
            .and_then(|v| v.as_str())
            .unwrap_or(&default_path);

        // Read snapshot
        let snap = {
            let r = self.snapshot.read().map_err(|_| McpError::ToolError("lock poisoned".into()))?;
            r.clone()
        };
        let snap = snap.ok_or_else(|| McpError::ToolError(
            "EMC not initialized. Call icarus_init first.".into()
        ))?;

        // Read events (non-destructive clone for viz)
        let events: Vec<AutoEvent> = {
            let r = self.events.read().map_err(|_| McpError::ToolError("lock poisoned".into()))?;
            r.iter().cloned().collect()
        };

        let html = match mode {
            "lattice_field" => {
                let color_mode_str = args.get("color_mode").and_then(|v| v.as_str()).unwrap_or("amplitude");
                let config = icarus_viz::renderers::lattice_field::LatticeFieldConfig {
                    layer_index,
                    color_mode: icarus_viz::renderers::lattice_field::ColorMode::from_str(color_mode_str),
                    theme,
                    ..Default::default()
                };
                if layer_index >= snap.layer_states.len() {
                    return Err(McpError::ToolError(format!(
                        "layer_index {} out of range (have {} layers)", layer_index, snap.layer_states.len()
                    )));
                }
                icarus_viz::render_lattice_field(&snap, &config)
            }
            "energy_landscape" => {
                if layer_index >= snap.layer_states.len() {
                    return Err(McpError::ToolError(format!(
                        "layer_index {} out of range (have {} layers)", layer_index, snap.layer_states.len()
                    )));
                }
                icarus_viz::render_energy_landscape(&snap, layer_index, theme)
            }
            "phase_portrait" => {
                if layer_index >= snap.layer_states.len() {
                    return Err(McpError::ToolError(format!(
                        "layer_index {} out of range (have {} layers)", layer_index, snap.layer_states.len()
                    )));
                }
                icarus_viz::render_phase_portrait(&snap, layer_index, theme)
            }
            "neuro_dashboard" => {
                icarus_viz::render_neuro_dashboard(&snap, theme)
            }
            "timeseries" => {
                let series_strs: Vec<String> = args.get("series")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                    .unwrap_or_default();
                let series: Vec<icarus_viz::renderers::timeseries::Series> = series_strs.iter()
                    .filter_map(|s| icarus_viz::renderers::timeseries::Series::from_str(s))
                    .collect();
                icarus_viz::render_timeseries(&snap, &events, &series, theme)
            }
            "dashboard" | _ => {
                icarus_viz::render_combined_dashboard(&snap, &events, theme)
            }
        };

        std::fs::write(output_path, &html).map_err(|e| {
            McpError::ToolError(format!("Failed to write viz output: {}", e))
        })?;

        let file_size = html.len();
        Ok(CallToolResult::json(&json!({
            "status": "ok",
            "mode": mode,
            "output_path": output_path,
            "file_size_bytes": file_size,
            "theme": theme_str,
            "tick": snap.tick,
        })))
    }

    // ── NPU handlers (unchanged) ──

    async fn require_npu(&self) -> Result<tokio::sync::MutexGuard<'_, Option<NpuBridgeClient>>, McpError> {
        let mut guard = self.npu_client.lock().await;
        if guard.is_none() {
            let client = NpuBridgeClient::connect()
                .map_err(|e| McpError::ToolError(format!("NPU bridge connection failed: {}", e)))?;
            *guard = Some(client);
        }
        Ok(guard)
    }

    async fn handle_npu_ping(&self) -> Result<CallToolResult, McpError> {
        let mut guard = self.require_npu().await?;
        let client = guard.as_mut().unwrap();
        let msg = client.ping().map_err(|e| {
            McpError::ToolError(format!("NPU ping failed: {}", e))
        })?;
        Ok(CallToolResult::json(&json!({
            "status": "ok",
            "response": msg,
        })))
    }

    async fn handle_npu_device_info(&self) -> Result<CallToolResult, McpError> {
        let mut guard = self.require_npu().await?;
        let client = guard.as_mut().unwrap();
        let info = client.device_info().map_err(|e| {
            McpError::ToolError(format!("NPU device_info failed: {}", e))
        })?;
        Ok(CallToolResult::json(&json!({
            "status": "ok",
            "devices": info,
        })))
    }

    async fn handle_npu_matmul(&self, args: Value) -> Result<CallToolResult, McpError> {
        let m = args.get("m").and_then(|v| v.as_u64())
            .ok_or_else(|| McpError::InvalidParams("missing 'm' parameter".into()))? as u32;
        let k = args.get("k").and_then(|v| v.as_u64())
            .ok_or_else(|| McpError::InvalidParams("missing 'k' parameter".into()))? as u32;
        let n = args.get("n").and_then(|v| v.as_u64())
            .ok_or_else(|| McpError::InvalidParams("missing 'n' parameter".into()))? as u32;

        let a: Vec<f32> = args.get("a")
            .and_then(|v| v.as_array())
            .ok_or_else(|| McpError::InvalidParams("missing 'a' parameter".into()))?
            .iter().map(|v| v.as_f64().unwrap_or(0.0) as f32).collect();

        let b: Vec<f32> = args.get("b")
            .and_then(|v| v.as_array())
            .ok_or_else(|| McpError::InvalidParams("missing 'b' parameter".into()))?
            .iter().map(|v| v.as_f64().unwrap_or(0.0) as f32).collect();

        let mut guard = self.require_npu().await?;
        let client = guard.as_mut().unwrap();
        let (result, elapsed) = client.matmul(m, k, n, &a, &b)
            .map_err(|e| McpError::ToolError(format!("NPU matmul failed: {}", e)))?;

        let max_display = 100;
        let display_result: Vec<String> = result.iter()
            .take(max_display)
            .map(|v| format!("{:.6}", v))
            .collect();

        Ok(CallToolResult::json(&json!({
            "status": "ok",
            "m": m, "k": k, "n": n,
            "elapsed_ms": elapsed.as_secs_f64() * 1000.0,
            "result_shape": [m, n],
            "result_elements": result.len(),
            "result_preview": display_result,
            "truncated": result.len() > max_display,
        })))
    }

    async fn handle_npu_matvec(&self, args: Value) -> Result<CallToolResult, McpError> {
        let m = args.get("m").and_then(|v| v.as_u64())
            .ok_or_else(|| McpError::InvalidParams("missing 'm' parameter".into()))? as u32;
        let n = args.get("n").and_then(|v| v.as_u64())
            .ok_or_else(|| McpError::InvalidParams("missing 'n' parameter".into()))? as u32;

        let w: Vec<f32> = args.get("w")
            .and_then(|v| v.as_array())
            .ok_or_else(|| McpError::InvalidParams("missing 'w' parameter".into()))?
            .iter().map(|v| v.as_f64().unwrap_or(0.0) as f32).collect();

        let x: Vec<f32> = args.get("x")
            .and_then(|v| v.as_array())
            .ok_or_else(|| McpError::InvalidParams("missing 'x' parameter".into()))?
            .iter().map(|v| v.as_f64().unwrap_or(0.0) as f32).collect();

        let mut guard = self.require_npu().await?;
        let client = guard.as_mut().unwrap();
        let (result, elapsed) = client.matvec(m, n, &w, &x)
            .map_err(|e| McpError::ToolError(format!("NPU matvec failed: {}", e)))?;

        let display_result: Vec<String> = result.iter()
            .map(|v| format!("{:.6}", v)).collect();

        Ok(CallToolResult::json(&json!({
            "status": "ok",
            "m": m, "n": n,
            "elapsed_ms": elapsed.as_secs_f64() * 1000.0,
            "result_length": result.len(),
            "result": display_result,
        })))
    }

    async fn handle_npu_benchmark(&self) -> Result<CallToolResult, McpError> {
        let mut guard = self.require_npu().await?;
        let client = guard.as_mut().unwrap();
        let report = client.benchmark().map_err(|e| {
            McpError::ToolError(format!("NPU benchmark failed: {}", e))
        })?;
        Ok(CallToolResult::json(&json!({
            "status": "ok",
            "report": report,
        })))
    }

    // ── Ensemble handlers ──

    async fn handle_ensemble_init(&self, args: Value) -> Result<CallToolResult, McpError> {
        let num_instances = args.get("num_instances").and_then(|v| v.as_u64()).unwrap_or(2) as usize;
        let feature_mode_str = args.get("feature_mode").and_then(|v| v.as_str()).unwrap_or("nonlinear");
        let seed = args.get("seed").and_then(|v| v.as_f64()).unwrap_or(42.0) as u64;
        let amplitude = args.get("amplitude").and_then(|v| v.as_f64()).unwrap_or(0.1) as f32;
        let warmup_ticks = args.get("warmup_ticks").and_then(|v| v.as_u64()).unwrap_or(100);

        let feature_mode = match feature_mode_str {
            "linear" => FeatureMode::Linear,
            _ => FeatureMode::Nonlinear,
        };

        let mut ensemble = EnsembleTrainer::new(feature_mode);

        for i in 0..num_instances {
            let config = ManifoldConfig::e8_only();
            ensemble.add_cpu_instance(config).map_err(|e| {
                McpError::ToolError(format!("Failed to add instance {}: {}", i, e))
            })?;
        }

        ensemble.init_random(seed, amplitude);

        if warmup_ticks > 0 {
            ensemble.warmup(warmup_ticks).map_err(|e| {
                McpError::ToolError(format!("Warmup failed: {}", e))
            })?;
        }

        let status: Vec<Value> = ensemble.instance_status().iter().map(|s| {
            json!({
                "backend": s.backend_name,
                "feature_dim": s.feature_dim,
                "ticks": s.total_ticks,
            })
        }).collect();

        let total_dim = ensemble.total_state_dim();

        *self.ensemble.lock().await = Some(ensemble);

        Ok(CallToolResult::json(&json!({
            "status": "initialized",
            "num_instances": num_instances,
            "feature_mode": feature_mode_str,
            "seed": seed,
            "amplitude": amplitude,
            "warmup_ticks": warmup_ticks,
            "total_state_dim": total_dim,
            "instances": status,
        })))
    }

    async fn handle_ensemble_status(&self) -> Result<CallToolResult, McpError> {
        let guard = self.ensemble.lock().await;
        let ensemble = guard.as_ref().ok_or_else(|| {
            McpError::ToolError("Ensemble not initialized. Call icarus_ensemble_init first.".into())
        })?;

        let instances: Vec<Value> = ensemble.instance_status().iter().map(|s| {
            json!({
                "backend": s.backend_name,
                "feature_dim": s.feature_dim,
                "ticks": s.total_ticks,
            })
        }).collect();

        Ok(CallToolResult::json(&json!({
            "status": "ok",
            "num_instances": instances.len(),
            "total_state_dim": ensemble.total_state_dim(),
            "trained": ensemble.is_trained(),
            "num_samples": ensemble.num_samples(),
            "instances": instances,
        })))
    }

    // ── Game handlers ──

    async fn handle_game_start(&self, args: Value) -> Result<CallToolResult, McpError> {
        let port = args.get("port").and_then(|v| v.as_u64()).unwrap_or(3000) as u16;
        let seed = args.get("seed").and_then(|v| v.as_u64()).unwrap_or(42);

        let mut guard = self.game_child.lock().await;
        if let Some(ref proc) = *guard {
            return Err(McpError::ToolError(format!(
                "Game server already running on port {} (started {}s ago). Stop it first.",
                proc.port,
                proc.started.elapsed().as_secs()
            )));
        }

        // Find the game binary — check release first, then debug
        let binary = {
            let release = std::path::Path::new("/root/.cargo-target/release/icarus-game");
            let debug = std::path::Path::new("/root/.cargo-target/debug/icarus-game");
            if release.exists() {
                release.to_path_buf()
            } else if debug.exists() {
                debug.to_path_buf()
            } else {
                return Err(McpError::ToolError(
                    "icarus-game binary not found. Build it with: cargo build -p icarus-game".into()
                ));
            }
        };

        let child = tokio::process::Command::new(&binary)
            .env("ICARUS_PORT", port.to_string())
            .env("ICARUS_SEED", seed.to_string())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| McpError::ToolError(format!("Failed to spawn icarus-game: {}", e)))?;

        let pid = child.id().unwrap_or(0);
        *guard = Some(GameProcess {
            child,
            port,
            started: Instant::now(),
        });

        // Give the server a moment to bind
        tokio::time::sleep(Duration::from_millis(500)).await;

        Ok(CallToolResult::json(&json!({
            "status": "started",
            "port": port,
            "seed": seed,
            "pid": pid,
            "url": format!("http://localhost:{}", port),
            "binary": binary.display().to_string(),
        })))
    }

    async fn handle_game_stop(&self) -> Result<CallToolResult, McpError> {
        let mut guard = self.game_child.lock().await;
        let proc = guard.take().ok_or_else(|| {
            McpError::ToolError("No game server is running.".into())
        })?;

        let elapsed = proc.started.elapsed();
        let port = proc.port;
        let mut child = proc.child;

        child.kill().await.map_err(|e| {
            McpError::ToolError(format!("Failed to kill game process: {}", e))
        })?;

        Ok(CallToolResult::json(&json!({
            "status": "stopped",
            "port": port,
            "uptime_secs": elapsed.as_secs(),
        })))
    }

    async fn handle_game_status(&self) -> Result<CallToolResult, McpError> {
        let guard = self.game_child.lock().await;
        let proc = guard.as_ref().ok_or_else(|| {
            McpError::ToolError("No game server is running. Call icarus_game_start first.".into())
        })?;

        let port = proc.port;
        let uptime = proc.started.elapsed().as_secs();
        drop(guard);

        // Query the game's /status endpoint via HTTP
        let response = tokio::time::timeout(
            Duration::from_secs(5),
            async {
                let stream = tokio::net::TcpStream::connect(format!("localhost:{}", port))
                    .await
                    .map_err(|e| format!("Connect failed: {}", e))?;

                let request = format!(
                    "GET /status HTTP/1.1\r\nHost: localhost:{}\r\nConnection: close\r\n\r\n",
                    port
                );

                use tokio::io::{AsyncReadExt, AsyncWriteExt};
                let mut stream = stream;
                stream.write_all(request.as_bytes()).await
                    .map_err(|e| format!("Write failed: {}", e))?;

                let mut buf = Vec::with_capacity(4096);
                stream.read_to_end(&mut buf).await
                    .map_err(|e| format!("Read failed: {}", e))?;

                let text = String::from_utf8_lossy(&buf);
                // Find JSON body after headers (blank line)
                if let Some(idx) = text.find("\r\n\r\n") {
                    let body = &text[idx + 4..];
                    serde_json::from_str::<Value>(body)
                        .map_err(|e| format!("JSON parse failed: {}", e))
                } else {
                    Err("No HTTP response body".into())
                }
            }
        ).await;

        match response {
            Ok(Ok(game_status)) => {
                Ok(CallToolResult::json(&json!({
                    "status": "ok",
                    "port": port,
                    "uptime_secs": uptime,
                    "game": game_status,
                })))
            }
            Ok(Err(e)) => {
                Ok(CallToolResult::json(&json!({
                    "status": "error",
                    "port": port,
                    "uptime_secs": uptime,
                    "error": e,
                })))
            }
            Err(_) => {
                Ok(CallToolResult::json(&json!({
                    "status": "timeout",
                    "port": port,
                    "uptime_secs": uptime,
                    "error": "Game server did not respond within 5 seconds",
                })))
            }
        }
    }
}

#[async_trait]
impl McpServer for IcarusMcpServer {
    fn server_info(&self) -> ServerInfo {
        ServerInfo::new("icarus", env!("CARGO_PKG_VERSION"))
    }

    fn tools(&self) -> Vec<Tool> {
        tools::all_tools()
    }

    async fn call_tool(&self, name: &str, args: Value) -> Result<CallToolResult, McpError> {
        match name {
            "icarus_init" => self.handle_init(args).await,
            "icarus_step" => self.handle_step(args).await,
            "icarus_observe" => self.handle_observe(args).await,
            "icarus_inject" => self.handle_inject(args).await,
            "icarus_stats" => self.handle_stats().await,
            "icarus_encode" => self.handle_encode(args).await,
            "icarus_readout" => self.handle_readout(args).await,
            "icarus_train" => self.handle_train(args).await,
            "icarus_predict" => self.handle_predict(args).await,
            "icarus_npu_ping" => self.handle_npu_ping().await,
            "icarus_npu_device_info" => self.handle_npu_device_info().await,
            "icarus_npu_matmul" => self.handle_npu_matmul(args).await,
            "icarus_npu_matvec" => self.handle_npu_matvec(args).await,
            "icarus_npu_benchmark" => self.handle_npu_benchmark().await,
            "icarus_auto_start" => self.handle_auto_start(args).await,
            "icarus_auto_stop" => self.handle_auto_stop().await,
            "icarus_auto_status" => self.handle_auto_status().await,
            "icarus_auto_events" => self.handle_auto_events(args).await,
            "icarus_visualize" => self.handle_visualize(args).await,
            "icarus_ensemble_init" => self.handle_ensemble_init(args).await,
            "icarus_ensemble_status" => self.handle_ensemble_status().await,
            "icarus_game_start" => self.handle_game_start(args).await,
            "icarus_game_stop" => self.handle_game_stop().await,
            "icarus_game_status" => self.handle_game_status().await,
            _ => Err(McpError::UnknownTool(name.into())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mcp_core::protocol::Content;
    use serde_json::json;

    /// Extract the JSON value from a CallToolResult's first text content.
    fn extract_json(result: &CallToolResult) -> Value {
        let text = match &result.content[0] {
            Content::Text { text } => text,
            _ => panic!("Expected text content"),
        };
        serde_json::from_str(text).expect("Result should be valid JSON")
    }

    /// Create an initialized server (CPU, e8_only preset).
    async fn init_server() -> IcarusMcpServer {
        let server = IcarusMcpServer::new();
        server
            .call_tool("icarus_init", json!({"preset": "e8_only", "backend": "cpu", "seed": 42.0}))
            .await
            .expect("init should succeed");
        server
    }

    // ── icarus_init ──

    #[tokio::test]
    async fn test_init_cpu_e8_only() {
        let server = IcarusMcpServer::new();
        let result = server
            .call_tool("icarus_init", json!({"preset": "e8_only", "backend": "cpu", "seed": 42.0, "amplitude": 0.5}))
            .await
            .unwrap();

        let v = extract_json(&result);
        assert_eq!(v["status"], "initialized");
        assert_eq!(v["preset"], "e8_only");
        assert_eq!(v["backend"], "CPU");
        assert_eq!(v["seed"], 42);
        assert!(v["total_sites"].as_u64().unwrap() > 0);
        assert!(v["layers"].as_array().unwrap().len() >= 1);
    }

    #[tokio::test]
    async fn test_init_full_hierarchy() {
        let server = IcarusMcpServer::new();
        let result = server
            .call_tool("icarus_init", json!({"preset": "full_hierarchy", "backend": "cpu"}))
            .await
            .unwrap();

        let v = extract_json(&result);
        assert_eq!(v["preset"], "full_hierarchy");
        assert!(v["layers"].as_array().unwrap().len() > 1);
    }

    #[tokio::test]
    async fn test_init_defaults() {
        let server = IcarusMcpServer::new();
        let result = server
            .call_tool("icarus_init", json!({"backend": "cpu"}))
            .await
            .unwrap();

        let v = extract_json(&result);
        assert_eq!(v["status"], "initialized");
    }

    // ── require_emc error ──

    #[tokio::test]
    async fn test_step_before_init_fails() {
        let server = IcarusMcpServer::new();
        let err = server.call_tool("icarus_step", json!({})).await.unwrap_err();
        match err {
            McpError::ToolError(msg) => assert!(msg.contains("not initialized")),
            other => panic!("Expected ToolError, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_observe_before_init_fails() {
        let server = IcarusMcpServer::new();
        let err = server.call_tool("icarus_observe", json!({})).await.unwrap_err();
        match err {
            McpError::ToolError(msg) => assert!(msg.contains("not initialized")),
            other => panic!("Expected ToolError, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_stats_before_init_fails() {
        let server = IcarusMcpServer::new();
        let err = server.call_tool("icarus_stats", json!({})).await.unwrap_err();
        match err {
            McpError::ToolError(msg) => assert!(msg.contains("not initialized")),
            other => panic!("Expected ToolError, got {other:?}"),
        }
    }

    // ── unknown tool ──

    #[tokio::test]
    async fn test_unknown_tool() {
        let server = IcarusMcpServer::new();
        let err = server.call_tool("nonexistent", json!({})).await.unwrap_err();
        match err {
            McpError::UnknownTool(name) => assert_eq!(name, "nonexistent"),
            other => panic!("Expected UnknownTool, got {other:?}"),
        }
    }

    // ── icarus_step ──

    #[tokio::test]
    async fn test_step_single() {
        let server = init_server().await;
        let result = server.call_tool("icarus_step", json!({})).await.unwrap();

        let v = extract_json(&result);
        assert_eq!(v["ticks_executed"], 1);
        assert_eq!(v["tick"], 1);
        assert!(v["layers"].as_array().unwrap().len() >= 1);
    }

    #[tokio::test]
    async fn test_step_multiple() {
        let server = init_server().await;
        let result = server.call_tool("icarus_step", json!({"num_ticks": 5})).await.unwrap();

        let v = extract_json(&result);
        assert_eq!(v["ticks_executed"], 5);
        assert_eq!(v["tick"], 5);
    }

    #[tokio::test]
    async fn test_step_accumulates() {
        let server = init_server().await;
        server.call_tool("icarus_step", json!({"num_ticks": 3})).await.unwrap();
        let result = server.call_tool("icarus_step", json!({"num_ticks": 2})).await.unwrap();

        let v = extract_json(&result);
        assert_eq!(v["tick"], 5);
    }

    // ── icarus_observe ──

    #[tokio::test]
    async fn test_observe_default_layer() {
        let server = init_server().await;
        let result = server.call_tool("icarus_observe", json!({})).await.unwrap();

        let v = extract_json(&result);
        assert_eq!(v["layer_index"], 0);
        assert!(v["total_sites"].as_u64().unwrap() > 0);
        let sites = v["sites"].as_array().unwrap();
        assert!(!sites.is_empty());
        let site0 = &sites[0];
        assert!(site0.get("re").is_some());
        assert!(site0.get("im").is_some());
        assert!(site0.get("amplitude").is_some());
        assert!(site0.get("phase").is_some());
    }

    #[tokio::test]
    async fn test_observe_max_sites() {
        let server = init_server().await;
        let result = server
            .call_tool("icarus_observe", json!({"max_sites": 5}))
            .await
            .unwrap();

        let v = extract_json(&result);
        assert_eq!(v["showing"], 5);
        assert_eq!(v["sites"].as_array().unwrap().len(), 5);
    }

    #[tokio::test]
    async fn test_observe_out_of_bounds_layer() {
        let server = init_server().await;
        let err = server
            .call_tool("icarus_observe", json!({"layer_index": 99}))
            .await
            .unwrap_err();
        match err {
            McpError::ToolError(msg) => assert!(msg.contains("out of range")),
            other => panic!("Expected ToolError, got {other:?}"),
        }
    }

    // ── icarus_inject ──

    #[tokio::test]
    async fn test_inject_single_site() {
        let server = init_server().await;
        let result = server
            .call_tool("icarus_inject", json!({"sites": [[0, 1.0, 0.5]], "strength": 1.0}))
            .await
            .unwrap();

        let v = extract_json(&result);
        assert_eq!(v["status"], "injected");
        assert_eq!(v["sites_injected"], 1);
    }

    #[tokio::test]
    async fn test_inject_multiple_sites() {
        let server = init_server().await;
        let result = server
            .call_tool("icarus_inject", json!({"sites": [[0, 1.0, 0.0], [1, 0.0, 1.0], [2, 0.5, 0.5]]}))
            .await
            .unwrap();

        let v = extract_json(&result);
        assert_eq!(v["sites_injected"], 3);
    }

    #[tokio::test]
    async fn test_inject_partial_strength() {
        let server = init_server().await;
        let result = server
            .call_tool("icarus_inject", json!({"sites": [[0, 1.0, 0.0]], "strength": 0.5}))
            .await
            .unwrap();

        let v = extract_json(&result);
        assert_eq!(v["sites_injected"], 1);
        assert_eq!(v["strength"], 0.5);
    }

    #[tokio::test]
    async fn test_inject_missing_sites_param() {
        let server = init_server().await;
        let err = server.call_tool("icarus_inject", json!({})).await.unwrap_err();
        match err {
            McpError::InvalidParams(msg) => assert!(msg.contains("sites")),
            other => panic!("Expected InvalidParams, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_inject_layer_out_of_range() {
        let server = init_server().await;
        let err = server
            .call_tool("icarus_inject", json!({"sites": [[0, 1.0, 0.0]], "layer_index": 99}))
            .await
            .unwrap_err();
        match err {
            McpError::ToolError(msg) => assert!(msg.contains("out of range")),
            other => panic!("Expected ToolError, got {other:?}"),
        }
    }

    // ── icarus_stats ──

    #[tokio::test]
    async fn test_stats() {
        let server = init_server().await;
        let result = server.call_tool("icarus_stats", json!({})).await.unwrap();

        let v = extract_json(&result);
        assert_eq!(v["tick"], 0);
        assert_eq!(v["backend"], "CPU");
        assert!(v["total_sites"].as_u64().unwrap() > 0);
        assert!(v["memory_bytes"].as_u64().unwrap() > 0);
        let layers = v["layers"].as_array().unwrap();
        assert!(!layers.is_empty());
        let l0 = &layers[0];
        assert!(l0.get("total_energy").is_some());
        assert!(l0.get("mean_amplitude").is_some());
    }

    // ── icarus_encode ──

    #[tokio::test]
    async fn test_encode_spatial() {
        let server = init_server().await;
        let result = server
            .call_tool("icarus_encode", json!({"encoder": "spatial", "input": [1.0, 2.0, 3.0]}))
            .await
            .unwrap();

        let v = extract_json(&result);
        assert_eq!(v["status"], "encoded");
        assert_eq!(v["encoder"], "spatial");
        assert_eq!(v["input_length"], 3);
    }

    #[tokio::test]
    async fn test_encode_phase() {
        let server = init_server().await;
        let result = server
            .call_tool("icarus_encode", json!({"encoder": "phase", "input": [0.5, 1.0]}))
            .await
            .unwrap();

        let v = extract_json(&result);
        assert_eq!(v["encoder"], "phase");
    }

    #[tokio::test]
    async fn test_encode_spectral() {
        let server = init_server().await;
        let result = server
            .call_tool("icarus_encode", json!({"encoder": "spectral", "input": [1.0, 0.0, 1.0, 0.0]}))
            .await
            .unwrap();

        let v = extract_json(&result);
        assert_eq!(v["encoder"], "spectral");
    }

    #[tokio::test]
    async fn test_encode_with_ticks_after() {
        let server = init_server().await;
        let result = server
            .call_tool("icarus_encode", json!({"input": [1.0], "ticks_after": 3}))
            .await
            .unwrap();

        let v = extract_json(&result);
        assert_eq!(v["ticks_after"], 3);
        assert_eq!(v["tick"], 3);
    }

    #[tokio::test]
    async fn test_encode_missing_input() {
        let server = init_server().await;
        let err = server.call_tool("icarus_encode", json!({})).await.unwrap_err();
        match err {
            McpError::InvalidParams(msg) => assert!(msg.contains("input")),
            other => panic!("Expected InvalidParams, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_encode_layer_out_of_range() {
        let server = init_server().await;
        let err = server
            .call_tool("icarus_encode", json!({"input": [1.0], "layer_index": 99}))
            .await
            .unwrap_err();
        match err {
            McpError::ToolError(msg) => assert!(msg.contains("out of range")),
            other => panic!("Expected ToolError, got {other:?}"),
        }
    }

    // ── icarus_readout ──

    #[tokio::test]
    async fn test_readout_complex_format() {
        let server = init_server().await;
        let result = server
            .call_tool("icarus_readout", json!({"format": "complex", "max_values": 10}))
            .await
            .unwrap();

        let v = extract_json(&result);
        assert_eq!(v["format"], "complex");
        let sites = v["sites"].as_array().unwrap();
        assert!(sites.len() <= 10);
        assert!(sites[0].get("re").is_some());
        assert!(sites[0].get("amplitude").is_some());
    }

    #[tokio::test]
    async fn test_readout_flat_format() {
        let server = init_server().await;
        let result = server
            .call_tool("icarus_readout", json!({"format": "flat", "max_values": 5}))
            .await
            .unwrap();

        let v = extract_json(&result);
        assert_eq!(v["format"], "flat");
        assert!(v["state_dim"].as_u64().unwrap() > 0);
        let values = v["values"].as_array().unwrap();
        assert!(!values.is_empty());
    }

    #[tokio::test]
    async fn test_readout_layer_out_of_range() {
        let server = init_server().await;
        let err = server
            .call_tool("icarus_readout", json!({"layer_index": 99}))
            .await
            .unwrap_err();
        match err {
            McpError::ToolError(msg) => assert!(msg.contains("out of range")),
            other => panic!("Expected ToolError, got {other:?}"),
        }
    }

    // ── icarus_train ──

    #[tokio::test]
    async fn test_train_basic() {
        let server = init_server().await;
        let result = server
            .call_tool("icarus_train", json!({
                "encoder": "spatial",
                "inputs": [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
                "targets": [[1.0], [0.0], [0.5]],
                "warmup_ticks": 2,
                "ticks_per_input": 1,
                "lambda": 0.01,
            }))
            .await
            .unwrap();

        let v = extract_json(&result);
        assert_eq!(v["status"], "trained");
        assert_eq!(v["num_samples"], 3);
        assert_eq!(v["output_dim"], 1);
        assert!(v["state_dim"].as_u64().unwrap() > 0);
        let nmse_arr = v["train_nmse"].as_array().unwrap();
        assert_eq!(nmse_arr.len(), 1);
    }

    #[tokio::test]
    async fn test_train_missing_inputs() {
        let server = init_server().await;
        let err = server
            .call_tool("icarus_train", json!({"targets": [[1.0]]}))
            .await
            .unwrap_err();
        match err {
            McpError::InvalidParams(msg) => assert!(msg.contains("inputs")),
            other => panic!("Expected InvalidParams, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_train_missing_targets() {
        let server = init_server().await;
        let err = server
            .call_tool("icarus_train", json!({"inputs": [[1.0]]}))
            .await
            .unwrap_err();
        match err {
            McpError::InvalidParams(msg) => assert!(msg.contains("targets")),
            other => panic!("Expected InvalidParams, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_train_mismatched_lengths() {
        let server = init_server().await;
        let err = server
            .call_tool("icarus_train", json!({"inputs": [[1.0], [2.0]], "targets": [[1.0]]}))
            .await
            .unwrap_err();
        match err {
            McpError::InvalidParams(msg) => assert!(msg.contains("equal length")),
            other => panic!("Expected InvalidParams, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_train_empty_inputs() {
        let server = init_server().await;
        let err = server
            .call_tool("icarus_train", json!({"inputs": [], "targets": []}))
            .await
            .unwrap_err();
        match err {
            McpError::InvalidParams(msg) => assert!(msg.contains("empty")),
            other => panic!("Expected InvalidParams, got {other:?}"),
        }
    }

    // ── icarus_predict ──

    #[tokio::test]
    async fn test_predict_after_train() {
        let server = init_server().await;

        server
            .call_tool("icarus_train", json!({
                "inputs": [[1.0, 0.0], [0.0, 1.0]],
                "targets": [[1.0], [0.0]],
                "warmup_ticks": 1,
                "ticks_per_input": 1,
            }))
            .await
            .unwrap();

        let result = server
            .call_tool("icarus_predict", json!({
                "inputs": [[1.0, 0.0], [0.5, 0.5]],
                "ticks_per_input": 1,
            }))
            .await
            .unwrap();

        let v = extract_json(&result);
        assert_eq!(v["status"], "predicted");
        assert_eq!(v["num_inputs"], 2);
        assert_eq!(v["output_dim"], 1);
        let preds = v["predictions"].as_array().unwrap();
        assert_eq!(preds.len(), 2);
        assert_eq!(preds[0].as_array().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn test_predict_before_train_fails() {
        let server = init_server().await;
        let err = server
            .call_tool("icarus_predict", json!({"inputs": [[1.0]]}))
            .await
            .unwrap_err();
        match err {
            McpError::ToolError(msg) => assert!(msg.contains("trained")),
            other => panic!("Expected ToolError, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_predict_missing_inputs() {
        let server = init_server().await;
        server
            .call_tool("icarus_train", json!({
                "inputs": [[1.0]],
                "targets": [[1.0]],
                "warmup_ticks": 1,
            }))
            .await
            .unwrap();

        let err = server.call_tool("icarus_predict", json!({})).await.unwrap_err();
        match err {
            McpError::InvalidParams(msg) => assert!(msg.contains("inputs")),
            other => panic!("Expected InvalidParams, got {other:?}"),
        }
    }

    // ── server_info and tools listing ──

    #[tokio::test]
    async fn test_server_info() {
        let server = IcarusMcpServer::new();
        let info = server.server_info();
        assert_eq!(info.name, "icarus");
    }

    #[tokio::test]
    async fn test_tools_listing() {
        let server = IcarusMcpServer::new();
        let tools = server.tools();
        assert_eq!(tools.len(), 24);
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"icarus_init"));
        assert!(names.contains(&"icarus_step"));
        assert!(names.contains(&"icarus_observe"));
        assert!(names.contains(&"icarus_inject"));
        assert!(names.contains(&"icarus_stats"));
        assert!(names.contains(&"icarus_encode"));
        assert!(names.contains(&"icarus_readout"));
        assert!(names.contains(&"icarus_train"));
        assert!(names.contains(&"icarus_predict"));
        assert!(names.contains(&"icarus_npu_ping"));
        assert!(names.contains(&"icarus_npu_device_info"));
        assert!(names.contains(&"icarus_npu_matmul"));
        assert!(names.contains(&"icarus_npu_matvec"));
        assert!(names.contains(&"icarus_npu_benchmark"));
        assert!(names.contains(&"icarus_auto_start"));
        assert!(names.contains(&"icarus_auto_stop"));
        assert!(names.contains(&"icarus_auto_status"));
        assert!(names.contains(&"icarus_auto_events"));
        assert!(names.contains(&"icarus_visualize"));
        assert!(names.contains(&"icarus_ensemble_init"));
        assert!(names.contains(&"icarus_ensemble_status"));
        assert!(names.contains(&"icarus_game_start"));
        assert!(names.contains(&"icarus_game_stop"));
        assert!(names.contains(&"icarus_game_status"));
    }

    // ── ensemble tools ──

    #[tokio::test]
    async fn test_ensemble_status_before_init_fails() {
        let server = IcarusMcpServer::new();
        let err = server.call_tool("icarus_ensemble_status", json!({})).await.unwrap_err();
        match err {
            McpError::ToolError(msg) => assert!(msg.contains("not initialized")),
            other => panic!("Expected ToolError, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_ensemble_init_default() {
        let server = IcarusMcpServer::new();
        let result = server
            .call_tool("icarus_ensemble_init", json!({}))
            .await
            .unwrap();

        let v = extract_json(&result);
        assert_eq!(v["status"], "initialized");
        assert_eq!(v["num_instances"], 2);
        assert_eq!(v["feature_mode"], "nonlinear");
        assert!(v["total_state_dim"].as_u64().unwrap() > 0);
        let instances = v["instances"].as_array().unwrap();
        assert_eq!(instances.len(), 2);
    }

    #[tokio::test]
    async fn test_ensemble_init_custom() {
        let server = IcarusMcpServer::new();
        let result = server
            .call_tool("icarus_ensemble_init", json!({
                "num_instances": 3,
                "feature_mode": "linear",
                "seed": 99.0,
                "amplitude": 0.5,
                "warmup_ticks": 10,
            }))
            .await
            .unwrap();

        let v = extract_json(&result);
        assert_eq!(v["num_instances"], 3);
        assert_eq!(v["feature_mode"], "linear");
        assert_eq!(v["seed"], 99);
        let instances = v["instances"].as_array().unwrap();
        assert_eq!(instances.len(), 3);
    }

    #[tokio::test]
    async fn test_ensemble_status_after_init() {
        let server = IcarusMcpServer::new();
        server
            .call_tool("icarus_ensemble_init", json!({"warmup_ticks": 5}))
            .await
            .unwrap();

        let result = server
            .call_tool("icarus_ensemble_status", json!({}))
            .await
            .unwrap();

        let v = extract_json(&result);
        assert_eq!(v["status"], "ok");
        assert_eq!(v["num_instances"], 2);
        assert_eq!(v["trained"], false);
        assert_eq!(v["num_samples"], 0);
        assert!(v["total_state_dim"].as_u64().unwrap() > 0);
    }

    // ── game tools ──

    #[tokio::test]
    async fn test_game_stop_before_start_fails() {
        let server = IcarusMcpServer::new();
        let err = server.call_tool("icarus_game_stop", json!({})).await.unwrap_err();
        match err {
            McpError::ToolError(msg) => assert!(msg.contains("No game server")),
            other => panic!("Expected ToolError, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_game_status_before_start_fails() {
        let server = IcarusMcpServer::new();
        let err = server.call_tool("icarus_game_status", json!({})).await.unwrap_err();
        match err {
            McpError::ToolError(msg) => assert!(msg.contains("No game server")),
            other => panic!("Expected ToolError, got {other:?}"),
        }
    }

    // ── autonomous mode ──

    #[tokio::test]
    async fn test_auto_status_idle() {
        let server = IcarusMcpServer::new();
        let result = server.call_tool("icarus_auto_status", json!({})).await.unwrap();
        let v = extract_json(&result);
        assert_eq!(v["state"], "idle");
    }

    #[tokio::test]
    async fn test_auto_start_stop() {
        let server = init_server().await;

        let result = server
            .call_tool("icarus_auto_start", json!({
                "max_ticks_per_second": 1000.0,
                "stop_conditions": [{"type": "max_ticks", "value": 50}],
            }))
            .await
            .unwrap();

        let v = extract_json(&result);
        assert_eq!(v["status"], "started");

        // Give it time to tick
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Check status
        let result = server.call_tool("icarus_auto_status", json!({})).await.unwrap();
        let v = extract_json(&result);
        // Should be completed (50 ticks at 1000 tps is 50ms)
        assert!(v["state"] == "completed" || v["state"] == "running");
    }

    #[tokio::test]
    async fn test_auto_manual_stop() {
        let server = init_server().await;

        server
            .call_tool("icarus_auto_start", json!({
                "max_ticks_per_second": 60.0,
                "stop_conditions": [{"type": "manual"}],
            }))
            .await
            .unwrap();

        tokio::time::sleep(Duration::from_millis(100)).await;

        let result = server.call_tool("icarus_auto_stop", json!({})).await.unwrap();
        let v = extract_json(&result);
        assert_eq!(v["status"], "stopped");
        assert!(v["tick"].as_u64().unwrap() > 0);
    }

    #[tokio::test]
    async fn test_auto_events() {
        let server = init_server().await;

        server
            .call_tool("icarus_auto_start", json!({
                "max_ticks_per_second": 10000.0,
                "stop_conditions": [{"type": "max_ticks", "value": 200}],
            }))
            .await
            .unwrap();

        tokio::time::sleep(Duration::from_millis(500)).await;

        let result = server.call_tool("icarus_auto_events", json!({})).await.unwrap();
        let v = extract_json(&result);
        // Should have at least a Started event
        assert!(v["count"].as_u64().unwrap() >= 1);
    }
}
