# Curriculum Learning Scheduler Design

## Executive Summary

This document presents a comprehensive curriculum learning system for the Icarus Training Game, enabling progressive difficulty scaling for imitation learning. The curriculum scheduler orchestrates phased learning progression from simple navigation tasks to complex multi-step goal completion, with adaptive difficulty adjustment based on agent performance metrics.

**Key Features:**
- Four-phase progressive curriculum (navigation → interaction → multi-step → adaptive)
- Performance-based phase transition conditions
- World configuration management per curriculum phase
- Real-time difficulty adjustment within phases
- Integration with existing TrainingBridge and EnsembleTrainer infrastructure

---

## 1. Curriculum Learning Foundations

### 1.1 Motivation

Curriculum learning accelerates training by ordering tasks from simple to complex, mirroring how humans learn. For the Icarus Training Game:

1. **Faster Convergence**: Simple tasks establish foundational policies before complex behaviors
2. **Reduced Sample Complexity**: Early success on easy tasks provides more informative gradients
3. **Catastrophic Forgetting Prevention**: Progressive difficulty with replay buffer diversity
4. **Better Generalization**: Exposure to varied difficulty levels improves robustness

### 1.2 Research Context

From academic research (Causal-Paced Deep RL, ARLBench):
- Task sequencing should balance **novelty** (exploration) and **similarity** (transfer)
- Metrics for progression: success rate, reward gain, variance reduction
- Adaptive pacing outperforms fixed schedules
- Curriculum design benefits from causal task structure awareness

### 1.3 Existing Infrastructure Analysis

**TrainingBridge (`icarus-game/src/bridge.rs`)**:
- Collects (state, action) pairs via `on_player_action()`
- Drives ensemble with encoded inputs
- Retrains every `retrain_interval` samples
- Provides confidence metric: `1.0 - NMSE`
- Supports DAgger mode for online corrections

**World (`icarus-engine/src/world.rs`)**:
- 128×128 Perlin terrain, 256.0 world-space extent
- Entities: Trees (40), Rocks (25), Orbs (8), Crates (12), Lights (6)
- Actions: Move, Jump, PickUp, Drop, Push, ToggleLight, Idle
- State encoding: 20-dim vector (position, rotation, velocity, nearest 3 entities, flags)

**EnsembleTrainer (`icarus-engine/src/ensemble.rs`)**:
- Multi-EMC reservoir computing (CPU/CUDA/NPU backends)
- Ridge regression readout with auto-lambda selection
- EWC (Elastic Weight Consolidation) for continual learning
- Online RLS for per-tick adaptation

---

## 2. Curriculum Phase Design

### Phase 1: Simple Navigation (Foundation)

**Objective**: Learn basic movement and terrain traversal.

**Task Characteristics:**
- Single waypoint target (visible marker)
- Flat terrain region (±2.0 height variance maximum)
- No obstacles between spawn and target
- Short distance (15-30 units)
- No entities nearby (cleared radius)

**Success Criteria:**
- Distance to target < 2.0 units
- Time limit: 60 seconds

**World Configuration:**
```rust
WorldConfig {
    terrain_smoothness: 0.8,     // Reduce Perlin octaves
    spawn_position: [0.0, _, 0.0],
    target_position: [20.0, _, 0.0],
    entity_density: 0.0,          // No entities
    spawn_cleared_radius: 50.0,
}
```

**Transition Condition:**
- Success rate ≥ 80% over last 50 episodes
- Average time-to-goal < 40 seconds
- Confidence ≥ 0.7

---

### Phase 2: Object Interaction (Skill Acquisition)

**Objective**: Learn to identify, approach, and interact with entities.

**Task Characteristics:**
- Pickup tasks: Collect N orbs (start N=1, scale to N=3)
- Avoidance tasks: Navigate around obstacles (rocks, trees)
- Moderate terrain (±5.0 height variance)
- Medium distance (30-60 units)
- Entities placed on direct path

**Success Criteria:**
- Pickup: All target orbs collected
- Avoidance: No collisions (distance to obstacles > 1.5 units)
- Time limit: 120 seconds

**World Configuration:**
```rust
WorldConfig {
    terrain_smoothness: 0.5,
    spawn_position: random_in_region(safe_zone),
    target_orbs: vec![id1, id2, ...],  // Specified orb IDs
    obstacle_density: 0.3,              // Sparse obstacles
    entity_density: 0.5,                // Half normal density
}
```

**Sub-phases:**
2a. Single orb pickup (static position)
2b. Multiple orb pickup (3 orbs, shortest path)
2c. Pickup with obstacle avoidance

**Transition Condition:**
- Sub-phase 2a: 85% success over 30 episodes
- Sub-phase 2b: 80% success over 40 episodes
- Sub-phase 2c: 75% success over 50 episodes
- Average orbs collected per episode ≥ 2.5
- Confidence ≥ 0.75

---

### Phase 3: Multi-Step Tasks (Goal Composition)

**Objective**: Execute sequential goal plans with dependencies.

**Task Characteristics:**
- Key-door-goal sequences: Collect key orb → approach light (door) → toggle light → reach final waypoint
- Resource collection: Gather 3 crates to designated drop zone
- Complex terrain (±8.0 height variance, full Perlin)
- Long distance (60-100 units)
- Full entity density

**Success Criteria:**
- All sub-goals completed in order
- Final goal reached
- Time limit: 180 seconds

**World Configuration:**
```rust
WorldConfig {
    terrain_smoothness: 0.3,          // Full complexity
    task_type: MultiStep {
        steps: vec![
            TaskStep::CollectOrb { orb_id: 5 },
            TaskStep::ToggleLight { light_id: 2 },
            TaskStep::ReachWaypoint { pos: [80.0, _, 60.0] },
        ],
    },
    entity_density: 1.0,              // Full density
    spawn_cleared_radius: 5.0,        // Minimal clearing
}
```

**Sub-phases:**
3a. Two-step sequence (collect → reach)
3b. Three-step sequence (collect → toggle → reach)
3c. Four-step sequence with backtracking

**Transition Condition:**
- Sub-phase 3a: 70% success over 50 episodes
- Sub-phase 3b: 65% success over 60 episodes
- Sub-phase 3c: 60% success over 70 episodes
- Average steps completed ≥ 2.5 / 3.0
- Confidence ≥ 0.8

---

### Phase 4: Adaptive Difficulty (Mastery)

**Objective**: Generalize across task variations and maintain performance.

**Task Characteristics:**
- Random task sampling from Phases 1-3
- Procedurally generated terrain (new seeds)
- Dynamic difficulty adjustment based on rolling performance
- Adversarial entity placement (blocking optimal paths)
- Time pressure variations

**Success Criteria:**
- Adaptive per task type (Phase 1: 85%, Phase 2: 80%, Phase 3: 65%)
- Generalization: Success on unseen terrain seeds

**World Configuration:**
```rust
WorldConfig {
    task_sampler: AdaptiveSampler {
        phase_weights: [0.2, 0.3, 0.5],  // Favor harder tasks
        difficulty_multiplier: calculate_from_recent_performance(),
    },
    terrain_seed: random(),              // New seed each episode
    entity_seed: random(),
    adversarial_mode: true,              // Place obstacles on paths
}
```

**Difficulty Adjustment Rules:**
- If rolling success rate > 85%: increase difficulty_multiplier by 0.1
- If rolling success rate < 50%: decrease difficulty_multiplier by 0.1
- Difficulty affects: entity density, time limits, terrain roughness, waypoint distance

**Exit Condition:**
- Sustained performance: 70%+ success over 200 episodes
- Cross-phase consistency: >60% on all task types
- Confidence ≥ 0.85
- No exit (continual learning mode)

---

## 3. Curriculum Scheduler Architecture

### 3.1 Core Struct Definitions

```rust
/// Curriculum phase identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CurriculumPhase {
    Phase1Navigation,
    Phase2Interaction { sub_phase: u8 },  // 0=single, 1=multi, 2=avoid
    Phase3MultiStep { sub_phase: u8 },    // 0=two-step, 1=three-step, 2=four-step
    Phase4Adaptive,
}

/// Task type within a curriculum phase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskType {
    /// Navigate to a single waypoint.
    NavigateToWaypoint { target: [f32; 3] },
    
    /// Collect specified orbs.
    CollectOrbs { orb_ids: Vec<u32> },
    
    /// Navigate while avoiding obstacles.
    AvoidObstacles { waypoint: [f32; 3], obstacles: Vec<u32> },
    
    /// Multi-step goal sequence.
    MultiStepSequence { steps: Vec<TaskStep> },
}

/// A single step in a multi-step task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskStep {
    CollectOrb { orb_id: u32 },
    ToggleLight { light_id: u32 },
    ReachWaypoint { position: [f32; 3] },
    DropCrate { zone: [f32; 3] },
}

/// Configuration for world generation per curriculum phase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldConfig {
    /// Terrain smoothness (0.0 = rough, 1.0 = smooth).
    pub terrain_smoothness: f32,
    
    /// Entity density multiplier (0.0 = none, 1.0 = full).
    pub entity_density: f32,
    
    /// Radius around spawn with no entities.
    pub spawn_cleared_radius: f32,
    
    /// Task type for this episode.
    pub task_type: TaskType,
    
    /// Terrain seed (for reproducibility or randomization).
    pub terrain_seed: u64,
    
    /// Entity spawn seed.
    pub entity_seed: u64,
    
    /// Enable adversarial entity placement (block paths).
    pub adversarial_mode: bool,
}

/// Performance metrics for an episode.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodeMetrics {
    /// Episode completed successfully.
    pub success: bool,
    
    /// Time to complete (seconds), or time limit if failed.
    pub time_to_goal: f32,
    
    /// Number of sub-goals completed (for multi-step tasks).
    pub sub_goals_completed: u32,
    
    /// Total sub-goals in task.
    pub total_sub_goals: u32,
    
    /// Average distance error from optimal path.
    pub path_efficiency: f32,
    
    /// Number of collisions with obstacles.
    pub collision_count: u32,
    
    /// Final distance to goal (if failed).
    pub final_distance: f32,
    
    /// Prediction confidence during episode (avg NMSE).
    pub prediction_confidence: f32,
}

/// Rolling window of performance metrics for transition decisions.
#[derive(Debug, Clone)]
pub struct PerformanceWindow {
    /// Recent episode metrics (bounded circular buffer).
    metrics: VecDeque<EpisodeMetrics>,
    
    /// Window size (number of episodes to consider).
    window_size: usize,
}

impl PerformanceWindow {
    pub fn new(window_size: usize) -> Self {
        Self {
            metrics: VecDeque::with_capacity(window_size),
            window_size,
        }
    }
    
    pub fn add(&mut self, metrics: EpisodeMetrics) {
        if self.metrics.len() >= self.window_size {
            self.metrics.pop_front();
        }
        self.metrics.push_back(metrics);
    }
    
    /// Calculate success rate over the window.
    pub fn success_rate(&self) -> f32 {
        if self.metrics.is_empty() {
            return 0.0;
        }
        let successes = self.metrics.iter().filter(|m| m.success).count();
        successes as f32 / self.metrics.len() as f32
    }
    
    /// Average time to goal (only successful episodes).
    pub fn avg_time_to_goal(&self) -> f32 {
        let successful: Vec<_> = self.metrics.iter().filter(|m| m.success).collect();
        if successful.is_empty() {
            return f32::MAX;
        }
        successful.iter().map(|m| m.time_to_goal).sum::<f32>() / successful.len() as f32
    }
    
    /// Average prediction confidence.
    pub fn avg_confidence(&self) -> f32 {
        if self.metrics.is_empty() {
            return 0.0;
        }
        self.metrics.iter().map(|m| m.prediction_confidence).sum::<f32>() / self.metrics.len() as f32
    }
    
    /// Average sub-goal completion ratio.
    pub fn avg_sub_goal_ratio(&self) -> f32 {
        if self.metrics.is_empty() {
            return 0.0;
        }
        self.metrics.iter().map(|m| {
            if m.total_sub_goals == 0 {
                1.0
            } else {
                m.sub_goals_completed as f32 / m.total_sub_goals as f32
            }
        }).sum::<f32>() / self.metrics.len() as f32
    }
    
    /// Variance of success rate (stability metric).
    pub fn success_variance(&self) -> f32 {
        let rate = self.success_rate();
        let n = self.metrics.len() as f32;
        if n < 2.0 {
            return 1.0;
        }
        let variance = self.metrics.iter().map(|m| {
            let val = if m.success { 1.0 } else { 0.0 };
            (val - rate).powi(2)
        }).sum::<f32>() / (n - 1.0);
        variance
    }
}

/// Transition conditions for curriculum phase progression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionCondition {
    /// Minimum success rate required.
    pub min_success_rate: f32,
    
    /// Minimum average confidence required.
    pub min_confidence: f32,
    
    /// Maximum average time to goal (seconds).
    pub max_avg_time: f32,
    
    /// Minimum episodes in window before considering transition.
    pub min_episodes: usize,
    
    /// Additional phase-specific conditions.
    pub custom_checks: Vec<CustomCheck>,
}

/// Custom phase-specific checks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CustomCheck {
    MinSubGoalRatio { threshold: f32 },
    MaxCollisionRate { threshold: f32 },
    MinPathEfficiency { threshold: f32 },
    SuccessStability { max_variance: f32 },
}

/// The curriculum scheduler — orchestrates phase progression.
pub struct CurriculumScheduler {
    /// Current curriculum phase.
    current_phase: CurriculumPhase,
    
    /// Performance window for current phase.
    performance: PerformanceWindow,
    
    /// Total episodes completed.
    total_episodes: u64,
    
    /// Episodes in current phase.
    phase_episodes: u64,
    
    /// Phase transition history.
    phase_history: Vec<(CurriculumPhase, u64)>,  // (phase, episode_count)
    
    /// Configuration for world generation.
    world_config: WorldConfig,
    
    /// Difficulty multiplier for adaptive phase (0.5 = easier, 2.0 = harder).
    difficulty_multiplier: f32,
    
    /// Random number generator for task generation.
    rng: SimpleRng,
    
    /// Episode metrics history for analysis (bounded).
    metrics_history: VecDeque<EpisodeMetrics>,
}

impl CurriculumScheduler {
    /// Create a new curriculum scheduler starting at Phase 1.
    pub fn new(seed: u64) -> Self {
        let initial_config = WorldConfig {
            terrain_smoothness: 0.8,
            entity_density: 0.0,
            spawn_cleared_radius: 50.0,
            task_type: TaskType::NavigateToWaypoint {
                target: [20.0, 0.0, 0.0],
            },
            terrain_seed: seed,
            entity_seed: seed.wrapping_add(1000),
            adversarial_mode: false,
        };
        
        Self {
            current_phase: CurriculumPhase::Phase1Navigation,
            performance: PerformanceWindow::new(50),
            total_episodes: 0,
            phase_episodes: 0,
            phase_history: vec![(CurriculumPhase::Phase1Navigation, 0)],
            world_config: initial_config,
            difficulty_multiplier: 1.0,
            rng: SimpleRng::new(seed),
            metrics_history: VecDeque::with_capacity(1000),
        }
    }
    
    /// Generate a world configuration for the next episode.
    pub fn generate_world_config(&mut self) -> WorldConfig {
        // Update seeds for randomization
        self.world_config.terrain_seed = self.rng.next_u64();
        self.world_config.entity_seed = self.rng.next_u64();
        
        // Generate task based on current phase
        self.world_config.task_type = self.generate_task();
        
        self.world_config.clone()
    }
    
    /// Generate a task appropriate for the current curriculum phase.
    fn generate_task(&mut self) -> TaskType {
        match self.current_phase {
            CurriculumPhase::Phase1Navigation => {
                // Simple waypoint in flat terrain
                let distance = self.rng.next_range(15.0, 30.0);
                let angle = self.rng.next_range(0.0, std::f32::consts::TAU);
                let target = [
                    distance * angle.cos(),
                    0.0,  // Height set by terrain
                    distance * angle.sin(),
                ];
                TaskType::NavigateToWaypoint { target }
            }
            
            CurriculumPhase::Phase2Interaction { sub_phase } => {
                match sub_phase {
                    0 => {
                        // Single orb pickup
                        TaskType::CollectOrbs { orb_ids: vec![0] }
                    }
                    1 => {
                        // Multiple orb pickup
                        TaskType::CollectOrbs { orb_ids: vec![0, 1, 2] }
                    }
                    2 => {
                        // Avoidance task
                        let distance = self.rng.next_range(40.0, 60.0);
                        let angle = self.rng.next_range(0.0, std::f32::consts::TAU);
                        let waypoint = [
                            distance * angle.cos(),
                            0.0,
                            distance * angle.sin(),
                        ];
                        TaskType::AvoidObstacles {
                            waypoint,
                            obstacles: vec![],  // Populated by world generator
                        }
                    }
                    _ => unreachable!(),
                }
            }
            
            CurriculumPhase::Phase3MultiStep { sub_phase } => {
                match sub_phase {
                    0 => {
                        // Two-step: collect → reach
                        TaskType::MultiStepSequence {
                            steps: vec![
                                TaskStep::CollectOrb { orb_id: 0 },
                                TaskStep::ReachWaypoint { position: [60.0, 0.0, 60.0] },
                            ],
                        }
                    }
                    1 => {
                        // Three-step: collect → toggle → reach
                        TaskType::MultiStepSequence {
                            steps: vec![
                                TaskStep::CollectOrb { orb_id: 0 },
                                TaskStep::ToggleLight { light_id: 0 },
                                TaskStep::ReachWaypoint { position: [80.0, 0.0, 60.0] },
                            ],
                        }
                    }
                    2 => {
                        // Four-step with backtracking
                        TaskType::MultiStepSequence {
                            steps: vec![
                                TaskStep::CollectOrb { orb_id: 0 },
                                TaskStep::ReachWaypoint { position: [40.0, 0.0, 40.0] },
                                TaskStep::ToggleLight { light_id: 0 },
                                TaskStep::ReachWaypoint { position: [80.0, 0.0, 80.0] },
                            ],
                        }
                    }
                    _ => unreachable!(),
                }
            }
            
            CurriculumPhase::Phase4Adaptive => {
                // Randomly sample from all previous task types
                let phase_choice = self.rng.next_range(0.0, 1.0);
                if phase_choice < 0.3 {
                    self.generate_phase1_task()
                } else if phase_choice < 0.6 {
                    self.generate_phase2_task()
                } else {
                    self.generate_phase3_task()
                }
            }
        }
    }
    
    fn generate_phase1_task(&mut self) -> TaskType {
        let distance = self.rng.next_range(15.0, 30.0) * self.difficulty_multiplier;
        let angle = self.rng.next_range(0.0, std::f32::consts::TAU);
        TaskType::NavigateToWaypoint {
            target: [distance * angle.cos(), 0.0, distance * angle.sin()],
        }
    }
    
    fn generate_phase2_task(&mut self) -> TaskType {
        if self.rng.next_f32() < 0.5 {
            let count = ((self.rng.next_range(1.0, 4.0) * self.difficulty_multiplier) as usize).min(3).max(1);
            TaskType::CollectOrbs {
                orb_ids: (0..count as u32).collect(),
            }
        } else {
            let distance = self.rng.next_range(40.0, 60.0) * self.difficulty_multiplier;
            let angle = self.rng.next_range(0.0, std::f32::consts::TAU);
            TaskType::AvoidObstacles {
                waypoint: [distance * angle.cos(), 0.0, distance * angle.sin()],
                obstacles: vec![],
            }
        }
    }
    
    fn generate_phase3_task(&mut self) -> TaskType {
        let step_count = ((self.rng.next_range(2.0, 5.0) * self.difficulty_multiplier) as usize).min(4).max(2);
        let mut steps = Vec::with_capacity(step_count);
        for i in 0..step_count {
            let step = match i % 3 {
                0 => TaskStep::CollectOrb { orb_id: (i % 8) as u32 },
                1 => TaskStep::ToggleLight { light_id: (i % 6) as u32 },
                2 => {
                    let dist = self.rng.next_range(40.0, 80.0);
                    let angle = self.rng.next_range(0.0, std::f32::consts::TAU);
                    TaskStep::ReachWaypoint {
                        position: [dist * angle.cos(), 0.0, dist * angle.sin()],
                    }
                }
                _ => unreachable!(),
            };
            steps.push(step);
        }
        TaskType::MultiStepSequence { steps }
    }
    
    /// Record the outcome of an episode.
    pub fn record_episode(&mut self, metrics: EpisodeMetrics) {
        self.performance.add(metrics.clone());
        self.metrics_history.push_back(metrics);
        if self.metrics_history.len() > 1000 {
            self.metrics_history.pop_front();
        }
        self.total_episodes += 1;
        self.phase_episodes += 1;
    }
    
    /// Check if the agent should transition to the next phase.
    pub fn should_transition(&self) -> bool {
        let condition = self.get_transition_condition();
        
        // Minimum episodes check
        if self.phase_episodes < condition.min_episodes as u64 {
            return false;
        }
        
        // Core metrics
        let success_rate = self.performance.success_rate();
        let confidence = self.performance.avg_confidence();
        let avg_time = self.performance.avg_time_to_goal();
        
        if success_rate < condition.min_success_rate {
            return false;
        }
        if confidence < condition.min_confidence {
            return false;
        }
        if avg_time > condition.max_avg_time {
            return false;
        }
        
        // Custom checks
        for check in &condition.custom_checks {
            if !self.check_custom_condition(check) {
                return false;
            }
        }
        
        true
    }
    
    /// Get the transition condition for the current phase.
    fn get_transition_condition(&self) -> TransitionCondition {
        match self.current_phase {
            CurriculumPhase::Phase1Navigation => TransitionCondition {
                min_success_rate: 0.8,
                min_confidence: 0.7,
                max_avg_time: 40.0,
                min_episodes: 50,
                custom_checks: vec![
                    CustomCheck::SuccessStability { max_variance: 0.15 },
                ],
            },
            
            CurriculumPhase::Phase2Interaction { sub_phase } => {
                let (rate, conf, time) = match sub_phase {
                    0 => (0.85, 0.7, 80.0),
                    1 => (0.80, 0.72, 100.0),
                    2 => (0.75, 0.75, 110.0),
                    _ => unreachable!(),
                };
                TransitionCondition {
                    min_success_rate: rate,
                    min_confidence: conf,
                    max_avg_time: time,
                    min_episodes: 30 + sub_phase as usize * 10,
                    custom_checks: vec![
                        CustomCheck::MinSubGoalRatio { threshold: 0.8 },
                    ],
                }
            }
            
            CurriculumPhase::Phase3MultiStep { sub_phase } => {
                let (rate, conf, time) = match sub_phase {
                    0 => (0.70, 0.75, 140.0),
                    1 => (0.65, 0.78, 160.0),
                    2 => (0.60, 0.80, 170.0),
                    _ => unreachable!(),
                };
                TransitionCondition {
                    min_success_rate: rate,
                    min_confidence: conf,
                    max_avg_time: time,
                    min_episodes: 50 + sub_phase as usize * 10,
                    custom_checks: vec![
                        CustomCheck::MinSubGoalRatio { threshold: 0.7 },
                    ],
                }
            }
            
            CurriculumPhase::Phase4Adaptive => TransitionCondition {
                min_success_rate: 0.70,
                min_confidence: 0.85,
                max_avg_time: f32::MAX,  // No time constraint
                min_episodes: 200,
                custom_checks: vec![],
            },
        }
    }
    
    /// Check a custom condition.
    fn check_custom_condition(&self, check: &CustomCheck) -> bool {
        match check {
            CustomCheck::MinSubGoalRatio { threshold } => {
                self.performance.avg_sub_goal_ratio() >= *threshold
            }
            CustomCheck::MaxCollisionRate { threshold } => {
                let avg_collisions = if self.performance.metrics.is_empty() {
                    0.0
                } else {
                    self.performance.metrics.iter()
                        .map(|m| m.collision_count as f32)
                        .sum::<f32>() / self.performance.metrics.len() as f32
                };
                avg_collisions <= *threshold
            }
            CustomCheck::MinPathEfficiency { threshold } => {
                let avg_efficiency = if self.performance.metrics.is_empty() {
                    0.0
                } else {
                    self.performance.metrics.iter()
                        .map(|m| m.path_efficiency)
                        .sum::<f32>() / self.performance.metrics.len() as f32
                };
                avg_efficiency >= *threshold
            }
            CustomCheck::SuccessStability { max_variance } => {
                self.performance.success_variance() <= *max_variance
            }
        }
    }
    
    /// Transition to the next curriculum phase.
    pub fn transition_to_next_phase(&mut self) {
        let next_phase = match self.current_phase {
            CurriculumPhase::Phase1Navigation => {
                CurriculumPhase::Phase2Interaction { sub_phase: 0 }
            }
            CurriculumPhase::Phase2Interaction { sub_phase } => {
                if sub_phase < 2 {
                    CurriculumPhase::Phase2Interaction { sub_phase: sub_phase + 1 }
                } else {
                    CurriculumPhase::Phase3MultiStep { sub_phase: 0 }
                }
            }
            CurriculumPhase::Phase3MultiStep { sub_phase } => {
                if sub_phase < 2 {
                    CurriculumPhase::Phase3MultiStep { sub_phase: sub_phase + 1 }
                } else {
                    CurriculumPhase::Phase4Adaptive
                }
            }
            CurriculumPhase::Phase4Adaptive => {
                // No transition — continual learning
                return;
            }
        };
        
        self.phase_history.push((next_phase, self.total_episodes));
        self.current_phase = next_phase;
        self.phase_episodes = 0;
        self.performance = PerformanceWindow::new(50);
        
        // Update world config for new phase
        self.update_world_config_for_phase();
    }
    
    /// Update world configuration when transitioning phases.
    fn update_world_config_for_phase(&mut self) {
        match self.current_phase {
            CurriculumPhase::Phase1Navigation => {
                self.world_config.terrain_smoothness = 0.8;
                self.world_config.entity_density = 0.0;
                self.world_config.spawn_cleared_radius = 50.0;
                self.world_config.adversarial_mode = false;
            }
            CurriculumPhase::Phase2Interaction { sub_phase } => {
                self.world_config.terrain_smoothness = 0.5;
                self.world_config.entity_density = 0.3 + (sub_phase as f32 * 0.1);
                self.world_config.spawn_cleared_radius = 30.0;
                self.world_config.adversarial_mode = false;
            }
            CurriculumPhase::Phase3MultiStep { sub_phase } => {
                self.world_config.terrain_smoothness = 0.3 - (sub_phase as f32 * 0.05);
                self.world_config.entity_density = 0.7 + (sub_phase as f32 * 0.1);
                self.world_config.spawn_cleared_radius = 10.0;
                self.world_config.adversarial_mode = sub_phase >= 1;
            }
            CurriculumPhase::Phase4Adaptive => {
                self.world_config.terrain_smoothness = 0.3;
                self.world_config.entity_density = 1.0;
                self.world_config.spawn_cleared_radius = 5.0;
                self.world_config.adversarial_mode = true;
            }
        }
    }
    
    /// Adjust difficulty in adaptive phase based on recent performance.
    pub fn adjust_adaptive_difficulty(&mut self) {
        if !matches!(self.current_phase, CurriculumPhase::Phase4Adaptive) {
            return;
        }
        
        let success_rate = self.performance.success_rate();
        
        // Increase difficulty if too easy
        if success_rate > 0.85 && self.difficulty_multiplier < 2.0 {
            self.difficulty_multiplier += 0.1;
        }
        // Decrease difficulty if too hard
        else if success_rate < 0.50 && self.difficulty_multiplier > 0.5 {
            self.difficulty_multiplier -= 0.1;
        }
        
        // Clamp to reasonable bounds
        self.difficulty_multiplier = self.difficulty_multiplier.clamp(0.5, 2.0);
    }
    
    /// Get current phase.
    pub fn current_phase(&self) -> CurriculumPhase {
        self.current_phase
    }
    
    /// Get performance summary for current phase.
    pub fn performance_summary(&self) -> String {
        format!(
            "Phase: {:?} | Episodes: {} | Success: {:.1}% | Confidence: {:.2} | Avg Time: {:.1}s",
            self.current_phase,
            self.phase_episodes,
            self.performance.success_rate() * 100.0,
            self.performance.avg_confidence(),
            self.performance.avg_time_to_goal(),
        )
    }
    
    /// Export curriculum state for visualization/analysis.
    pub fn export_state(&self) -> serde_json::Value {
        serde_json::json!({
            "current_phase": format!("{:?}", self.current_phase),
            "total_episodes": self.total_episodes,
            "phase_episodes": self.phase_episodes,
            "success_rate": self.performance.success_rate(),
            "confidence": self.performance.avg_confidence(),
            "avg_time": self.performance.avg_time_to_goal(),
            "difficulty_multiplier": self.difficulty_multiplier,
            "phase_history": self.phase_history.iter().map(|(p, e)| {
                serde_json::json!({
                    "phase": format!("{:?}", p),
                    "started_at_episode": e,
                })
            }).collect::<Vec<_>>(),
        })
    }
}

// Simple RNG (xorshift64) for task generation
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }
    
    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }
    
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() & 0xFFFFFF) as f32 / 0xFFFFFF as f32
    }
    
    fn next_range(&mut self, min: f32, max: f32) -> f32 {
        min + self.next_f32() * (max - min)
    }
}
```

---

## 4. Integration with TrainingBridge

### 4.1 Modified Training Loop

```rust
/// Extended training bridge with curriculum support.
pub struct CurriculumTrainingBridge {
    bridge: TrainingBridge,
    curriculum: CurriculumScheduler,
    current_episode_start: Instant,
    current_task: TaskType,
    sub_goals_completed: u32,
}

impl CurriculumTrainingBridge {
    pub fn new(bridge_config: BridgeConfig, curriculum_seed: u64) -> Result<Self> {
        Ok(Self {
            bridge: TrainingBridge::new(bridge_config)?,
            curriculum: CurriculumScheduler::new(curriculum_seed),
            current_episode_start: Instant::now(),
            current_task: TaskType::NavigateToWaypoint {
                target: [20.0, 0.0, 0.0],
            },
            sub_goals_completed: 0,
        })
    }
    
    /// Start a new episode with curriculum-generated task.
    pub fn start_episode(&mut self, world: &mut World) -> WorldConfig {
        self.current_episode_start = Instant::now();
        self.sub_goals_completed = 0;
        
        let config = self.curriculum.generate_world_config();
        self.current_task = config.task_type.clone();
        
        // Apply config to world (procedural generation)
        self.apply_world_config(world, &config);
        
        config
    }
    
    /// Apply world configuration to the game world.
    fn apply_world_config(&self, world: &mut World, config: &WorldConfig) {
        // Regenerate terrain with smoothness parameter
        world.terrain = self.generate_curriculum_terrain(
            config.terrain_seed,
            config.terrain_smoothness,
        );
        
        // Regenerate entities with density parameter
        world.entities = self.generate_curriculum_entities(
            &world.terrain,
            config.entity_seed,
            config.entity_density,
            config.spawn_cleared_radius,
            config.adversarial_mode,
        );
        
        // Reset player/agent positions
        let spawn_y = world.terrain.height_at(0.0, 0.0) + 1.0;
        world.player.position = [0.0, spawn_y, 0.0];
        world.agent.position = [5.0, spawn_y + 1.0, 5.0];
    }
    
    fn generate_curriculum_terrain(&self, seed: u64, smoothness: f32) -> Terrain {
        // Adjust Perlin octaves based on smoothness
        let octaves = (4.0 * (1.0 - smoothness) + 1.0) as u32;
        let persistence = 0.3 + smoothness * 0.3;
        
        // Custom terrain generation (modify world.rs Terrain::generate)
        Terrain::generate_curriculum(128, 256.0, seed, octaves, persistence)
    }
    
    fn generate_curriculum_entities(
        &self,
        terrain: &Terrain,
        seed: u64,
        density: f32,
        cleared_radius: f32,
        adversarial: bool,
    ) -> Vec<Entity> {
        // Generate entities with density scaling and spawn clearing
        // (implementation details omitted for brevity)
        generate_entities_curriculum(terrain, seed, density, cleared_radius, adversarial)
    }
    
    /// End the current episode and record metrics.
    pub fn end_episode(&mut self, world: &World, success: bool) {
        let elapsed = self.current_episode_start.elapsed().as_secs_f32();
        
        let metrics = EpisodeMetrics {
            success,
            time_to_goal: elapsed,
            sub_goals_completed: self.sub_goals_completed,
            total_sub_goals: self.count_sub_goals(&self.current_task),
            path_efficiency: self.calculate_path_efficiency(world),
            collision_count: 0,  // Track in world.rs
            final_distance: self.calculate_goal_distance(world),
            prediction_confidence: self.bridge.confidence(),
        };
        
        self.curriculum.record_episode(metrics);
        
        // Check for phase transition
        if self.curriculum.should_transition() {
            self.curriculum.transition_to_next_phase();
            println!("🎓 Curriculum Phase Transition: {}", self.curriculum.performance_summary());
        }
        
        // Adjust adaptive difficulty
        self.curriculum.adjust_adaptive_difficulty();
    }
    
    fn count_sub_goals(&self, task: &TaskType) -> u32 {
        match task {
            TaskType::NavigateToWaypoint { .. } => 1,
            TaskType::CollectOrbs { orb_ids } => orb_ids.len() as u32,
            TaskType::AvoidObstacles { .. } => 1,
            TaskType::MultiStepSequence { steps } => steps.len() as u32,
        }
    }
    
    fn calculate_path_efficiency(&self, world: &World) -> f32 {
        // Compare actual path length to optimal (A* or Euclidean)
        // Higher = more efficient (1.0 = optimal)
        0.8  // Placeholder
    }
    
    fn calculate_goal_distance(&self, world: &World) -> f32 {
        // Distance to final goal based on current task
        match &self.current_task {
            TaskType::NavigateToWaypoint { target } => {
                let p = world.player.position;
                let dx = target[0] - p[0];
                let dz = target[2] - p[2];
                (dx * dx + dz * dz).sqrt()
            }
            _ => 0.0,  // Task-specific logic
        }
    }
    
    /// Increment sub-goal counter when agent completes a sub-goal.
    pub fn mark_sub_goal_completed(&mut self) {
        self.sub_goals_completed += 1;
    }
}
```

### 4.2 Game Server Integration

The game server (`icarus-game/src/main.rs`) would be extended:

```rust
// In game loop
if episode_ended {
    let success = check_task_success(&world, &curriculum_bridge.current_task);
    curriculum_bridge.end_episode(&world, success);
    
    // Start new episode with next curriculum task
    let config = curriculum_bridge.start_episode(&mut world);
    println!("📚 New Episode: Phase={:?}, Task={:?}", 
             curriculum_bridge.curriculum.current_phase(), config.task_type);
}
```

---

## 5. Visualization and Monitoring

### 5.1 Curriculum Dashboard

Real-time HUD overlay showing:
- **Current Phase**: "Phase 2b: Multi-Orb Collection"
- **Progress Bar**: Episode count toward transition
- **Metrics**: Success rate, confidence, avg time
- **Difficulty Gauge**: Multiplier visualization (Phase 4)

### 5.2 Performance Graphs

WebSocket-pushed data for client-side charts:
- Success rate over episodes (phase-segmented)
- NMSE reduction curve
- Sub-goal completion ratio
- Phase transition timeline

### 5.3 Export API

MCP tool: `icarus_curriculum_export`
```json
{
  "current_phase": "Phase3MultiStep(sub=1)",
  "total_episodes": 457,
  "phase_episodes": 62,
  "success_rate": 0.68,
  "confidence": 0.79,
  "phase_history": [
    {"phase": "Phase1Navigation", "episodes": 73},
    {"phase": "Phase2Interaction(0)", "episodes": 35},
    ...
  ]
}
```

---

## 6. Implementation Roadmap

### Phase 1: Core Structures (Week 1)
- [ ] Implement `CurriculumPhase`, `TaskType`, `TaskStep` enums
- [ ] Implement `WorldConfig`, `EpisodeMetrics`, `PerformanceWindow` structs
- [ ] Implement `TransitionCondition` and `CustomCheck`
- [ ] Write unit tests for `PerformanceWindow` metrics

### Phase 2: Scheduler Logic (Week 2)
- [ ] Implement `CurriculumScheduler` struct
- [ ] Task generation methods per phase
- [ ] Transition condition checking
- [ ] Difficulty adjustment for Phase 4
- [ ] Integration tests with mock data

### Phase 3: World Generation (Week 2-3)
- [ ] Extend `Terrain::generate` with curriculum parameters
- [ ] Implement `generate_entities_curriculum` with density/clearing
- [ ] Add adversarial placement logic
- [ ] Procedural task marker placement (waypoints, target entities)

### Phase 4: Bridge Integration (Week 3)
- [ ] Create `CurriculumTrainingBridge` wrapper
- [ ] Episode lifecycle: start, tick, end
- [ ] Sub-goal tracking logic
- [ ] Metrics calculation (path efficiency, distance)

### Phase 5: Game Server Integration (Week 4)
- [ ] Episode boundary detection
- [ ] Task success evaluation
- [ ] Curriculum state in game protocol
- [ ] HUD rendering for curriculum status

### Phase 6: Monitoring & Tuning (Week 4-5)
- [ ] WebSocket curriculum state streaming
- [ ] Performance visualization frontend
- [ ] MCP export tool
- [ ] Hyperparameter tuning (transition thresholds, window sizes)

---

## 7. Open Questions & Future Directions

### 7.1 Hyperparameter Sensitivity
- **Window sizes**: 50 episodes sufficient? Phase-specific tuning?
- **Transition thresholds**: Are 80%/75%/70% success rates optimal?
- **Difficulty scaling**: Linear multiplier or exponential?

### 7.2 Advanced Curriculum Strategies
- **Causal task ordering**: Use SCM-based distance (from CP-DRL paper)
- **Prioritized experience replay**: Weight samples by curriculum phase
- **Meta-learning**: Learn curriculum policy itself (AutoRL)
- **Teacher-student**: Use expert trajectories for hard tasks

### 7.3 Generalization Testing
- **Zero-shot transfer**: Can Phase 4 agent handle novel task compositions?
- **Procedural stress testing**: Extreme terrain/entity configurations
- **Cross-seed robustness**: Performance variance across seeds

### 7.4 Human-in-the-Loop
- **Manual phase override**: Allow curriculum reset/skip for debugging
- **Demonstration injection**: Provide expert demos for stuck phases
- **Adaptive intervention**: DAgger mode activated on low confidence

---

## 8. References

### Academic Papers
1. **Causal-Paced Deep Reinforcement Learning** (Cho et al., 2025)
   - Curriculum pacing via SCM task similarity
   - Combines novelty (exploration) with learnability (transfer)

2. **ARLBench: Flexible and Efficient Benchmarking for HPO in RL** (Becktepe et al., 2024)
   - Hyperparameter landscapes for RL algorithms
   - Task selection for representative evaluation

3. **A Tutorial on Meta-Reinforcement Learning** (Beck et al., 2023)
   - Task distribution adaptation
   - Few-shot learning in new environments

### Codebase Files
- `/root/workspace-v2/Icarus/icarus-engine/src/world.rs`
- `/root/workspace-v2/Icarus/icarus-game/src/bridge.rs`
- `/root/workspace-v2/Icarus/icarus-engine/src/ensemble.rs`
- `/root/workspace-v2/Icarus/icarus-engine/src/continual.rs` (EWC)
- `/root/workspace-v2/Icarus/icarus-engine/src/training.rs` (Online RLS)

### External Resources
- **Curriculum Learning Survey**: Bengio et al. (2009)
- **Reverse Curriculum Generation**: Florensa et al. (2017)
- **Teacher-Student Curriculum**: Matiisen et al. (2017)

---

## 9. Conclusion

This curriculum learning design provides a structured path from simple navigation to complex multi-step goal completion. The four-phase progression, combined with adaptive difficulty in Phase 4, creates a robust training pipeline that leverages the existing Icarus infrastructure (EnsembleTrainer, EWC, Online RLS).

**Key advantages:**
1. **Gradual complexity scaling** prevents catastrophic forgetting
2. **Performance-based transitions** ensure readiness before advancement
3. **Adaptive difficulty** maintains challenge in mastery phase
4. **Modular integration** preserves existing bridge/ensemble architecture

**Next steps:**
1. Prototype `CurriculumScheduler` with Phase 1 tasks
2. Validate transition conditions via simulation
3. Integrate with TrainingBridge for end-to-end testing
4. Tune hyperparameters based on empirical performance

The system is designed for extensibility — new phases, task types, and transition conditions can be added without disrupting the core scheduler logic.

---

**Document Version**: 1.0  
**Last Updated**: 2026-02-12  
**Author**: Claude Opus 4.6 (Deep Researcher)  
**Status**: Design Complete — Ready for Implementation
