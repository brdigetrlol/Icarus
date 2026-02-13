//! Bridge between the game world and the ensemble trainer.
//!
//! Encodes player actions as training data, drives the ensemble,
//! and periodically retrains the readout for imitation learning.

use anyhow::Result;

use icarus_engine::config::ManifoldConfig;
use icarus_engine::continual::{EwcRidgeRegression, ReplayBuffer};
use icarus_engine::ensemble::{EnsembleTrainResult, EnsembleTrainer};
use icarus_engine::encoding::SpectralEncoder;
use icarus_engine::readout::{FeatureMode, LinearReadout};
use icarus_engine::training::OnlineRidgeRegression;
use icarus_engine::world::{Action, World};
use icarus_math::lattice::e8::E8Lattice;

use cortana_core::aether::ExtendedAether;

use crate::protocol::{BackendStatus, TrainingMode};

/// Configuration for the training bridge.
pub struct BridgeConfig {
    /// Number of EMC instances to create.
    pub num_instances: usize,
    /// Ticks per input step.
    pub ticks_per_input: u64,
    /// Leaky integration rate for encoding.
    pub leak_rate: f32,
    /// Feature extraction mode.
    pub feature_mode: FeatureMode,
    /// Number of warmup ticks before training begins.
    pub warmup_ticks: u64,
    /// Retrain after this many new samples.
    pub retrain_interval: u64,
    /// Base random seed.
    pub seed: u64,
    /// Maximum replay buffer capacity.
    pub replay_max_size: usize,
    /// EWC Fisher penalty strength (λ_ewc).
    pub lambda_ewc: f32,
}

impl Default for BridgeConfig {
    fn default() -> Self {
        Self {
            num_instances: 2,
            ticks_per_input: 2,
            leak_rate: 0.3,
            feature_mode: FeatureMode::Nonlinear,
            warmup_ticks: 20,
            retrain_interval: 50,
            seed: 42,
            replay_max_size: 2000,
            lambda_ewc: 50.0,
        }
    }
}

/// Connects the game World to the EnsembleTrainer for imitation learning.
///
/// Flow per game tick:
/// 1. Player acts in the world
/// 2. Bridge encodes world state + player action as (input, target)
/// 3. Drives all EMC instances with the encoded input
/// 4. Periodically retrains the readout on accumulated data
/// 5. Uses the trained readout to predict Icarus agent actions
pub struct TrainingBridge {
    ensemble: EnsembleTrainer,
    encoder: SpectralEncoder,
    config: BridgeConfig,
    samples_since_retrain: u64,
    total_samples: u64,
    last_train_result: Option<EnsembleTrainResult>,
    warmed_up: bool,
    mode: TrainingMode,
    interventions: u64,
    dagger_steps: u64,
    /// Online RLS for per-tick weight updates between batch retrains.
    online_rls: Option<OnlineRidgeRegression>,
    /// EWC-penalized ridge regression for continual learning across retrains.
    ewc: EwcRidgeRegression,
    /// Bounded replay buffer for diverse training data.
    replay: ReplayBuffer,
    /// Number of completed batch retrains (for EWC task naming).
    retrain_count: u32,
    /// NMSE values after each retrain (for graph).
    nmse_history: Vec<f32>,
    /// Last predicted action from `predict_agent_action`.
    last_predicted_action: Option<Action>,
    /// Recent event log entries.
    event_log: Vec<String>,
}

impl TrainingBridge {
    /// Create a new training bridge with the given config.
    ///
    /// Initializes `num_instances` CPU EMC instances (E8 lattice each).
    pub fn new(config: BridgeConfig) -> Result<Self> {
        let mut ensemble = EnsembleTrainer::new(config.feature_mode);

        for _ in 0..config.num_instances {
            let emc_config = ManifoldConfig::e8_only();
            ensemble.add_cpu_instance(emc_config)?;
        }

        ensemble.init_random(config.seed, 0.1);

        let encoder = SpectralEncoder::new();

        let ewc = EwcRidgeRegression::new(1.0, config.lambda_ewc);
        let replay = ReplayBuffer::new(config.replay_max_size);

        Ok(Self {
            ensemble,
            encoder,
            config,
            samples_since_retrain: 0,
            total_samples: 0,
            last_train_result: None,
            warmed_up: false,
            mode: TrainingMode::Observation,
            interventions: 0,
            dagger_steps: 0,
            online_rls: None,
            ewc,
            replay,
            retrain_count: 0,
            nmse_history: Vec::new(),
            last_predicted_action: None,
            event_log: Vec::new(),
        })
    }

    /// Ensure the ensemble is warmed up.
    fn ensure_warmup(&mut self) -> Result<()> {
        if !self.warmed_up {
            self.ensemble.warmup(self.config.warmup_ticks)?;
            self.warmed_up = true;
        }
        Ok(())
    }

    /// Record a player action and drive the ensemble.
    ///
    /// Called once per game tick when the player performs an action.
    /// Encodes the world state as input and the action as training target.
    pub fn on_player_action(&mut self, world: &World, action: &Action) -> Result<()> {
        self.ensure_warmup()?;

        let input = world.encode_state();
        let target = World::encode_action(action);

        let state = self.ensemble.drive(
            &input,
            &self.encoder,
            self.config.ticks_per_input,
            self.config.leak_rate,
            Some(&target),
        )?;

        // Online RLS: per-tick weight update for immediate adaptation
        if let Some(ref mut rls) = self.online_rls {
            rls.update(&state, &target);
        }

        // Store in replay buffer for diverse retrain data
        self.replay.add_batch(&[state], &[target]);

        self.total_samples += 1;
        self.samples_since_retrain += 1;

        Ok(())
    }

    /// Retrain the readout if enough new samples have accumulated.
    ///
    /// First retrain uses auto-lambda ridge regression to discover the optimal
    /// regularization. Subsequent retrains use EWC-penalized regression to
    /// preserve previously learned behavior while adapting to new data.
    ///
    /// Returns `Some(result)` if retraining occurred, `None` otherwise.
    pub fn retrain_if_needed(&mut self) -> Result<Option<EnsembleTrainResult>> {
        if self.samples_since_retrain < self.config.retrain_interval {
            return Ok(None);
        }
        if self.total_samples < 20 {
            return Ok(None); // Need minimum data
        }

        let result = if self.ewc.num_prior_tasks() == 0 {
            // First retrain: use auto-lambda to discover optimal ridge λ
            let k_folds = if self.total_samples < 50 { 3 } else { 5 };
            let result = self.ensemble.train_auto_lambda(k_folds)?;

            // Update EWC base lambda to match auto-lambda discovery
            self.ewc.lambda = result.lambda;

            // Register Fisher diagonal from ensemble's collected data
            let ewc_result = self.ewc.train(
                self.ensemble.collected_states(),
                self.ensemble.collected_targets(),
            );
            self.ewc.register_task(
                &format!("retrain_{}", self.retrain_count),
                ewc_result.fisher,
                ewc_result.optimal_wt,
                ewc_result.output_dim,
                ewc_result.state_dim,
            );

            result
        } else {
            // Subsequent retrains: use EWC to preserve prior knowledge
            // Sample from replay buffer for diverse training data
            let sample_count = self.replay.len().min(
                self.ensemble.collected_states().len().max(200),
            );
            let (replay_states, replay_targets) = self.replay.sample(sample_count);

            let ewc_result = self.ewc.train(&replay_states, &replay_targets);

            // Register this task's Fisher before building readout
            self.ewc.register_task(
                &format!("retrain_{}", self.retrain_count),
                ewc_result.fisher.clone(),
                ewc_result.optimal_wt.clone(),
                ewc_result.output_dim,
                ewc_result.state_dim,
            );

            // Build readout from EWC weights and install on ensemble
            let readout = LinearReadout::from_weights_with_mode(
                ewc_result.weights,
                ewc_result.bias,
                ewc_result.output_dim,
                ewc_result.state_dim,
                self.config.feature_mode,
            );
            self.ensemble.set_readout(readout);

            // Compute NMSE on replay data for reporting
            let nmse = Self::compute_nmse_from_readout(
                self.ensemble.readout().unwrap(),
                &replay_states,
                &replay_targets,
            );

            EnsembleTrainResult {
                nmse,
                lambda: self.ewc.lambda,
                num_samples: replay_states.len(),
                state_dim: ewc_result.state_dim,
                output_dim: ewc_result.output_dim,
                instance_dims: self.ensemble.instance_state_dims(),
            }
        };

        self.retrain_count += 1;
        self.samples_since_retrain = 0;
        self.nmse_history.push(result.nmse);
        self.last_train_result = Some(result.clone());

        // Warm-start online RLS from the fresh batch weights
        if let Some(readout) = self.ensemble.readout() {
            match &mut self.online_rls {
                Some(rls) => rls.warm_start(readout, result.lambda),
                None => {
                    self.online_rls =
                        Some(OnlineRidgeRegression::from_readout(readout, result.lambda));
                }
            }
        }

        Ok(Some(result))
    }

    /// Compute NMSE from a readout and data samples.
    fn compute_nmse_from_readout(
        readout: &LinearReadout,
        states: &[Vec<f32>],
        targets: &[Vec<f32>],
    ) -> f32 {
        let n = states.len() as f32;
        if n == 0.0 {
            return 1.0;
        }

        // Compute target variance
        let k = targets[0].len();
        let mut target_mean = vec![0.0f32; k];
        for t in targets {
            for (j, &v) in t.iter().enumerate() {
                target_mean[j] += v;
            }
        }
        for v in &mut target_mean {
            *v /= n;
        }

        let mut total_var = 0.0f32;
        let mut total_mse = 0.0f32;

        for (s, t) in states.iter().zip(targets.iter()) {
            let pred = readout.predict_from_raw_state(s);
            for j in 0..k {
                let err = pred[j] - t[j];
                total_mse += err * err;
                let dev = t[j] - target_mean[j];
                total_var += dev * dev;
            }
        }

        if total_var < 1e-12 {
            return 0.0; // Constant target — perfect fit
        }
        total_mse / total_var
    }

    /// Predict the next action for the Icarus agent.
    ///
    /// Uses the current world state as input, drives the ensemble,
    /// and decodes the prediction into an Action.
    /// Returns `None` if the readout hasn't been trained yet.
    pub fn predict_agent_action(&mut self, world: &World) -> Result<Option<Action>> {
        if !self.ensemble.is_trained() {
            return Ok(None);
        }

        let input = world.encode_state();

        // Drive the ensemble forward (encode + tick on all instances)
        let batch_prediction = self.ensemble.step(
            &input,
            &self.encoder,
            self.config.ticks_per_input,
            self.config.leak_rate,
        )?;

        // Use online RLS if available — more adaptive than batch readout
        let prediction = if let Some(ref rls) = self.online_rls {
            let state = self.ensemble.concatenated_state();
            rls.predict(&state)
        } else {
            batch_prediction
        };

        let action = World::decode_action(&prediction);
        self.last_predicted_action = Some(action.clone());
        Ok(Some(action))
    }

    /// Get the current training confidence (1.0 - NMSE, clamped to [0, 1]).
    pub fn confidence(&self) -> f32 {
        self.last_train_result
            .as_ref()
            .map(|r| (1.0 - r.nmse).clamp(0.0, 1.0))
            .unwrap_or(0.0)
    }

    /// Get the last NMSE (or 1.0 if not yet trained).
    pub fn nmse(&self) -> f32 {
        self.last_train_result
            .as_ref()
            .map(|r| r.nmse)
            .unwrap_or(1.0)
    }

    /// Total training samples collected.
    pub fn total_samples(&self) -> u64 {
        self.total_samples
    }

    /// Whether the ensemble has a trained readout.
    pub fn is_trained(&self) -> bool {
        self.ensemble.is_trained()
    }

    /// Number of online RLS updates performed since last warm-start.
    pub fn online_updates(&self) -> u64 {
        self.online_rls.as_ref().map_or(0, |r| r.num_updates())
    }

    /// Number of EWC prior tasks registered (0 = plain ridge, 1+ = EWC active).
    pub fn ewc_tasks(&self) -> usize {
        self.ewc.num_prior_tasks()
    }

    /// Number of samples in the replay buffer.
    pub fn replay_size(&self) -> usize {
        self.replay.len()
    }

    /// Number of completed batch retrains.
    pub fn retrain_count(&self) -> u32 {
        self.retrain_count
    }

    /// Per-backend status for the HUD.
    pub fn backend_status(&self) -> Vec<BackendStatus> {
        self.ensemble
            .instance_status()
            .into_iter()
            .map(|s| BackendStatus {
                name: s.backend_name,
                state_dim: s.feature_dim,
                ticks: s.total_ticks,
            })
            .collect()
    }

    /// Get the current training mode.
    pub fn mode(&self) -> TrainingMode {
        self.mode
    }

    /// Set the training mode.
    pub fn set_mode(&mut self, mode: TrainingMode) {
        self.mode = mode;
        if mode == TrainingMode::DAgger {
            // Reset DAgger counters on mode entry
            self.dagger_steps = 0;
            self.interventions = 0;
        }
    }

    /// Total human interventions during DAgger mode.
    pub fn interventions(&self) -> u64 {
        self.interventions
    }

    /// NMSE history across all retrains.
    pub fn nmse_history(&self) -> &[f32] {
        &self.nmse_history
    }

    /// Recent event log entries (drained on read).
    pub fn drain_events(&mut self) -> Vec<String> {
        std::mem::take(&mut self.event_log)
    }

    /// Push an event to the log.
    #[allow(dead_code)]
    pub fn push_event(&mut self, event: String) {
        // Keep bounded at 100 entries
        if self.event_log.len() >= 100 {
            self.event_log.remove(0);
        }
        self.event_log.push(event);
    }

    /// Extract field magnitudes from the first ensemble instance.
    ///
    /// Returns sqrt(re² + im²) for each of the 241 E8 lattice sites.
    /// Returns empty vec if no instances exist.
    pub fn lattice_overlay_data(&self) -> Vec<f32> {
        let Some(emc) = self.ensemble.instance(0) else {
            return Vec::new();
        };
        if emc.manifold.layers.is_empty() {
            return Vec::new();
        }
        let field = &emc.manifold.layers[0].field;
        let n = field.num_sites;
        let mut magnitudes = Vec::with_capacity(n);
        for i in 0..n {
            let re = field.values_re[i];
            let im = field.values_im[i];
            magnitudes.push((re * re + im * im).sqrt());
        }
        magnitudes
    }

    /// Compute E8 root vector positions projected to 3D.
    ///
    /// Takes the first 3 components of each 8D root vector, normalized,
    /// plus the origin at [0, 0, 0]. Returns 241 positions.
    pub fn lattice_positions() -> Vec<[f32; 3]> {
        let lattice = E8Lattice::new();
        let roots = lattice.root_vectors();
        let mut positions = Vec::with_capacity(roots.len() + 1);
        // Origin site
        positions.push([0.0, 0.0, 0.0]);
        for root in roots {
            // Project 8D to 3D using first 3 coordinates
            let x = root[0] as f32;
            let y = root[1] as f32;
            let z = root[2] as f32;
            // Normalize to unit sphere for display
            let len = (x * x + y * y + z * z).sqrt();
            if len > 1e-6 {
                positions.push([x / len, y / len, z / len]);
            } else {
                positions.push([0.0, 0.0, 0.0]);
            }
        }
        positions
    }

    /// Convert the last predicted action to a 3D direction vector.
    ///
    /// Move → (dx, 0, dz), Push → (dir[0], 0, dir[1]), Jump → (0, 1, 0),
    /// others → None.
    pub fn predicted_action_direction(&self) -> Option<[f32; 3]> {
        match &self.last_predicted_action {
            Some(Action::Move { dx, dz }) => Some([*dx, 0.0, *dz]),
            Some(Action::Push { direction }) => Some([direction[0], 0.0, direction[1]]),
            Some(Action::Jump) => Some([0.0, 1.0, 0.0]),
            _ => None,
        }
    }

    /// Intervention rate: interventions / dagger_steps (0.0 if no steps yet).
    #[allow(dead_code)]
    pub fn intervention_rate(&self) -> f32 {
        if self.dagger_steps == 0 {
            0.0
        } else {
            self.interventions as f32 / self.dagger_steps as f32
        }
    }

    /// Process one DAgger step.
    ///
    /// The agent predicts an action from the current world state.
    /// If the human provides an override (any key pressed), that override
    /// is recorded as an intervention and used as the training target.
    /// Otherwise the agent's own prediction is used as the target.
    ///
    /// Returns the action that should drive physics (agent prediction if
    /// no human override, or `None` if the ensemble isn't trained yet).
    pub fn on_dagger_step(
        &mut self,
        world: &World,
        human_override: Option<&Action>,
    ) -> Result<Option<Action>> {
        self.ensure_warmup()?;
        self.dagger_steps += 1;

        // If not trained yet, fall back to observation behavior
        if !self.ensemble.is_trained() {
            if let Some(action) = human_override {
                self.on_player_action(world, action)?;
            }
            return Ok(None);
        }

        if human_override.is_some() {
            self.interventions += 1;
        }

        // The training target is the human action (if overriding) or the
        // agent's own prediction (self-training on its own policy).
        let input = world.encode_state();

        // Get agent prediction first (drives the reservoir forward)
        let prediction = self.ensemble.step(
            &input,
            &self.encoder,
            self.config.ticks_per_input,
            self.config.leak_rate,
        )?;
        let agent_action = World::decode_action(&prediction);

        // Record the target: human override if present, else agent's own action
        let target_action = human_override.unwrap_or(&agent_action);
        let target = World::encode_action(target_action);

        // Store the training pair (state was already collected by step())
        // We need to manually add the target since step() doesn't store one
        let state = self.ensemble.drive(
            &input,
            &self.encoder,
            self.config.ticks_per_input,
            self.config.leak_rate,
            Some(&target),
        )?;

        // Online RLS: per-tick weight update for immediate adaptation
        if let Some(ref mut rls) = self.online_rls {
            rls.update(&state, &target);
        }

        // Store in replay buffer for diverse retrain data
        self.replay.add_batch(&[state], &[target]);

        self.total_samples += 1;
        self.samples_since_retrain += 1;

        Ok(Some(agent_action))
    }
}
