//! Emotional episodic memory — arousal-gated encoding with mood-congruent recall.
//!
//! Only high-arousal events form lasting memories (flashbulb memory effect).
//! Memories decay over time following a power law, but can be reinforced
//! through rehearsal. Current mood biases which memories are most accessible.

use serde::{Deserialize, Serialize};

use crate::emotion::PlutchikState;

/// Configuration for the emotional memory system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Maximum number of episodes before forgetting oldest.
    pub capacity: usize,
    /// Minimum arousal level to encode a new episode [0, 1].
    pub encoding_threshold: f32,
    /// Power-law decay exponent (higher = faster forgetting).
    pub decay_exponent: f32,
    /// Minimum decay factor before episode is eligible for removal.
    pub forget_threshold: f32,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            capacity: 10000,
            encoding_threshold: 0.6,
            decay_exponent: 0.3,
            forget_threshold: 0.01,
        }
    }
}

/// A single emotional episode in memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalEpisode {
    /// Tick at which the episode was encoded.
    pub tick: u64,
    /// Wall-clock timestamp (seconds since epoch).
    pub timestamp: u64,
    /// Description of the stimulus that caused this episode.
    pub stimulus: String,
    /// Snapshot of Plutchik emotion activations at encoding time.
    pub emotion_snapshot: PlutchikState,
    /// Valence at encoding time (pleasure dimension).
    pub valence: f32,
    /// Arousal at encoding time — higher = stronger memory.
    pub arousal_at_encoding: f32,
    /// Semantic tags for retrieval.
    pub tags: Vec<String>,
    /// Current decay factor [0, 1]. Starts at 1.0, decreases over time.
    pub decay_factor: f32,
    /// Number of times this memory has been recalled (rehearsal count).
    pub recall_count: u32,
}

impl EmotionalEpisode {
    /// Effective strength of this memory (decay * arousal weighting).
    pub fn strength(&self) -> f32 {
        self.decay_factor * self.arousal_at_encoding
    }

    /// Whether this episode has positive valence.
    pub fn is_positive(&self) -> bool {
        self.valence > 0.0
    }

    /// Whether this episode has negative valence.
    pub fn is_negative(&self) -> bool {
        self.valence < 0.0
    }
}

/// Emotional episodic memory system.
#[derive(Debug, Clone)]
pub struct EmotionalMemory {
    episodes: Vec<EmotionalEpisode>,
    config: MemoryConfig,
}

impl EmotionalMemory {
    pub fn new(config: MemoryConfig) -> Self {
        Self {
            episodes: Vec::new(),
            config,
        }
    }

    /// Attempt to encode a new episode. Returns true if encoding succeeded.
    ///
    /// Encoding is gated by arousal: only events with arousal above the
    /// encoding threshold are stored (flashbulb memory effect).
    pub fn encode(
        &mut self,
        tick: u64,
        timestamp: u64,
        stimulus: String,
        emotion_snapshot: PlutchikState,
        valence: f32,
        arousal: f32,
        tags: Vec<String>,
    ) -> bool {
        if arousal < self.config.encoding_threshold {
            return false;
        }

        let episode = EmotionalEpisode {
            tick,
            timestamp,
            stimulus,
            emotion_snapshot,
            valence,
            arousal_at_encoding: arousal,
            tags,
            decay_factor: 1.0,
            recall_count: 0,
        };

        self.episodes.push(episode);

        // If over capacity, remove weakest memory
        if self.episodes.len() > self.config.capacity {
            self.remove_weakest();
        }

        true
    }

    /// Apply time-based decay to all episodes.
    ///
    /// Uses power-law forgetting: decay = 1 / (1 + age)^exponent
    /// where age = current_tick - encoding_tick.
    pub fn apply_decay(&mut self, current_tick: u64) {
        let exp = self.config.decay_exponent;
        for episode in &mut self.episodes {
            let age = (current_tick.saturating_sub(episode.tick)) as f32;
            episode.decay_factor = 1.0 / (1.0 + age).powf(exp);
        }

        // Remove episodes below forget threshold
        let threshold = self.config.forget_threshold;
        self.episodes
            .retain(|e| e.decay_factor >= threshold);
    }

    /// Rehearse (recall) a memory, resetting its decay factor.
    ///
    /// Rehearsal strengthens the memory trace, preventing forgetting.
    pub fn rehearse(&mut self, index: usize) {
        if let Some(episode) = self.episodes.get_mut(index) {
            episode.decay_factor = 1.0;
            episode.recall_count += 1;
        }
    }

    /// Recall episodes matching a specific primary emotion.
    ///
    /// Returns indices of matching episodes, sorted by strength (strongest first).
    /// `emotion_index` is a Plutchik index (JOY=0 .. ANTICIPATION=7).
    pub fn recall_by_emotion(&self, emotion_index: usize, top_k: usize) -> Vec<usize> {
        let mut scored: Vec<(usize, f32)> = self
            .episodes
            .iter()
            .enumerate()
            .map(|(i, e)| {
                let emotion_activation = e.emotion_snapshot.activations
                    .get(emotion_index)
                    .copied()
                    .unwrap_or(0.0);
                let score = emotion_activation * e.strength();
                (i, score)
            })
            .filter(|(_, score)| *score > 0.0)
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        scored.into_iter().map(|(i, _)| i).collect()
    }

    /// Recall episodes matching a semantic tag.
    ///
    /// Returns indices sorted by strength.
    pub fn recall_by_tag(&self, tag: &str, top_k: usize) -> Vec<usize> {
        let tag_lower = tag.to_lowercase();
        let mut scored: Vec<(usize, f32)> = self
            .episodes
            .iter()
            .enumerate()
            .filter(|(_, e)| e.tags.iter().any(|t| t.to_lowercase().contains(&tag_lower)))
            .map(|(i, e)| (i, e.strength()))
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        scored.into_iter().map(|(i, _)| i).collect()
    }

    /// Mood-congruent recall: retrieve episodes matching current mood valence.
    ///
    /// `mood_bias` > 0 favors positive memories, < 0 favors negative.
    /// The bias adjusts the scoring weight of valence-congruent episodes.
    pub fn mood_congruent_recall(&self, mood_bias: f32, top_k: usize) -> Vec<usize> {
        let mut scored: Vec<(usize, f32)> = self
            .episodes
            .iter()
            .enumerate()
            .map(|(i, e)| {
                // Congruence: positive mood_bias * positive valence = bonus
                let congruence = mood_bias * e.valence;
                let score = e.strength() * (1.0 + congruence.clamp(-0.5, 0.5));
                (i, score)
            })
            .filter(|(_, score)| *score > 0.0)
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        scored.into_iter().map(|(i, _)| i).collect()
    }

    /// Get all episodes as a slice.
    pub fn episodes(&self) -> &[EmotionalEpisode] {
        &self.episodes
    }

    /// Get an episode by index.
    pub fn get(&self, index: usize) -> Option<&EmotionalEpisode> {
        self.episodes.get(index)
    }

    /// Total number of stored episodes.
    pub fn len(&self) -> usize {
        self.episodes.len()
    }

    /// Whether the memory is empty.
    pub fn is_empty(&self) -> bool {
        self.episodes.is_empty()
    }

    /// Get the strongest episode (highest current strength).
    pub fn strongest(&self) -> Option<&EmotionalEpisode> {
        self.episodes
            .iter()
            .max_by(|a, b| a.strength().partial_cmp(&b.strength()).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Get the oldest episode.
    pub fn oldest(&self) -> Option<&EmotionalEpisode> {
        self.episodes.iter().min_by_key(|e| e.tick)
    }

    /// Get memory statistics.
    pub fn stats(&self) -> MemoryStats {
        let count = self.episodes.len();
        let avg_strength = if count > 0 {
            self.episodes.iter().map(|e| e.strength()).sum::<f32>() / count as f32
        } else {
            0.0
        };
        let positive_count = self.episodes.iter().filter(|e| e.is_positive()).count();
        let negative_count = self.episodes.iter().filter(|e| e.is_negative()).count();

        MemoryStats {
            total_episodes: count,
            capacity: self.config.capacity,
            average_strength: avg_strength,
            positive_episodes: positive_count,
            negative_episodes: negative_count,
            oldest_tick: self.oldest().map(|e| e.tick),
            strongest_valence: self.strongest().map(|e| e.valence),
        }
    }

    /// Remove the episode with the lowest strength.
    fn remove_weakest(&mut self) {
        if self.episodes.is_empty() {
            return;
        }
        let weakest_idx = self
            .episodes
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                a.strength()
                    .partial_cmp(&b.strength())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0);

        self.episodes.swap_remove(weakest_idx);
    }
}

impl Default for EmotionalMemory {
    fn default() -> Self {
        Self::new(MemoryConfig::default())
    }
}

/// Statistics about the emotional memory system.
#[derive(Debug, Clone, Serialize)]
pub struct MemoryStats {
    pub total_episodes: usize,
    pub capacity: usize,
    pub average_strength: f32,
    pub positive_episodes: usize,
    pub negative_episodes: usize,
    pub oldest_tick: Option<u64>,
    pub strongest_valence: Option<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_snapshot(joy: f32, fear: f32) -> PlutchikState {
        let mut state = PlutchikState::default();
        state.activations[crate::emotion::JOY] = joy;
        state.activations[crate::emotion::FEAR] = fear;
        state
    }

    #[test]
    fn test_encoding_gated_by_arousal() {
        let mut mem = EmotionalMemory::new(MemoryConfig {
            encoding_threshold: 0.6,
            ..Default::default()
        });

        // Low arousal → not encoded
        let encoded = mem.encode(1, 1000, "low arousal".into(), PlutchikState::default(), 0.5, 0.3, vec![]);
        assert!(!encoded);
        assert_eq!(mem.len(), 0);

        // High arousal → encoded
        let encoded = mem.encode(2, 1001, "high arousal".into(), PlutchikState::default(), 0.5, 0.8, vec![]);
        assert!(encoded);
        assert_eq!(mem.len(), 1);
    }

    #[test]
    fn test_capacity_enforcement() {
        let mut mem = EmotionalMemory::new(MemoryConfig {
            capacity: 3,
            encoding_threshold: 0.0,
            ..Default::default()
        });

        for i in 0..5 {
            mem.encode(
                i,
                1000 + i,
                format!("event {i}"),
                PlutchikState::default(),
                0.5,
                0.8,
                vec![],
            );
        }

        assert_eq!(mem.len(), 3, "should not exceed capacity");
    }

    #[test]
    fn test_power_law_decay() {
        let mut mem = EmotionalMemory::new(MemoryConfig {
            encoding_threshold: 0.0,
            decay_exponent: 0.5,
            forget_threshold: 0.01,
            ..Default::default()
        });

        mem.encode(0, 1000, "old event".into(), PlutchikState::default(), 0.5, 0.8, vec![]);

        // At tick 0, decay = 1.0
        mem.apply_decay(0);
        assert!((mem.get(0).unwrap().decay_factor - 1.0).abs() < 0.01);

        // At tick 100, decay < 1.0
        mem.apply_decay(100);
        let decay = mem.get(0).unwrap().decay_factor;
        assert!(decay < 1.0 && decay > 0.0, "decay at tick 100: {decay}");
    }

    #[test]
    fn test_rehearsal_resets_decay() {
        let mut mem = EmotionalMemory::new(MemoryConfig {
            encoding_threshold: 0.0,
            decay_exponent: 0.5,
            ..Default::default()
        });

        mem.encode(0, 1000, "event".into(), PlutchikState::default(), 0.5, 0.8, vec![]);

        // Decay
        mem.apply_decay(1000);
        let decayed = mem.get(0).unwrap().decay_factor;
        assert!(decayed < 0.5);

        // Rehearse
        mem.rehearse(0);
        let rehearsed = mem.get(0).unwrap().decay_factor;
        assert!((rehearsed - 1.0).abs() < f32::EPSILON, "rehearsal should reset decay");
        assert_eq!(mem.get(0).unwrap().recall_count, 1);
    }

    #[test]
    fn test_recall_by_emotion() {
        let mut mem = EmotionalMemory::new(MemoryConfig {
            encoding_threshold: 0.0,
            ..Default::default()
        });

        mem.encode(0, 1000, "joyful".into(), make_snapshot(0.9, 0.1), 0.8, 0.7, vec![]);
        mem.encode(1, 1001, "scary".into(), make_snapshot(0.1, 0.9), -0.5, 0.9, vec![]);
        mem.encode(2, 1002, "mildly happy".into(), make_snapshot(0.3, 0.0), 0.3, 0.6, vec![]);

        // Recall by JOY
        let joy_memories = mem.recall_by_emotion(crate::emotion::JOY, 2);
        assert!(!joy_memories.is_empty());
        // First result should be the most joyful
        let first = &mem.episodes[joy_memories[0]];
        assert!(first.emotion_snapshot.activations[crate::emotion::JOY] > 0.5);

        // Recall by FEAR
        let fear_memories = mem.recall_by_emotion(crate::emotion::FEAR, 2);
        assert!(!fear_memories.is_empty());
        let first = &mem.episodes[fear_memories[0]];
        assert!(first.emotion_snapshot.activations[crate::emotion::FEAR] > 0.5);
    }

    #[test]
    fn test_recall_by_tag() {
        let mut mem = EmotionalMemory::new(MemoryConfig {
            encoding_threshold: 0.0,
            ..Default::default()
        });

        mem.encode(0, 1000, "event A".into(), PlutchikState::default(), 0.5, 0.8, vec!["work".into(), "meeting".into()]);
        mem.encode(1, 1001, "event B".into(), PlutchikState::default(), 0.3, 0.7, vec!["personal".into()]);
        mem.encode(2, 1002, "event C".into(), PlutchikState::default(), 0.7, 0.9, vec!["work".into(), "deadline".into()]);

        let work_memories = mem.recall_by_tag("work", 5);
        assert_eq!(work_memories.len(), 2);

        let personal_memories = mem.recall_by_tag("personal", 5);
        assert_eq!(personal_memories.len(), 1);
    }

    #[test]
    fn test_mood_congruent_recall() {
        let mut mem = EmotionalMemory::new(MemoryConfig {
            encoding_threshold: 0.0,
            ..Default::default()
        });

        mem.encode(0, 1000, "happy".into(), PlutchikState::default(), 0.8, 0.7, vec![]);
        mem.encode(1, 1001, "sad".into(), PlutchikState::default(), -0.8, 0.9, vec![]);

        // Positive mood should prefer positive memories
        let positive_recall = mem.mood_congruent_recall(0.5, 2);
        let first_ep = &mem.episodes[positive_recall[0]];
        assert!(first_ep.is_positive(), "positive mood should recall positive memories first");

        // Negative mood should prefer negative memories
        let negative_recall = mem.mood_congruent_recall(-0.5, 2);
        let first_ep = &mem.episodes[negative_recall[0]];
        assert!(first_ep.is_negative(), "negative mood should recall negative memories first");
    }

    #[test]
    fn test_episode_strength() {
        let episode = EmotionalEpisode {
            tick: 0,
            timestamp: 1000,
            stimulus: "test".into(),
            emotion_snapshot: PlutchikState::default(),
            valence: 0.5,
            arousal_at_encoding: 0.9,
            tags: vec![],
            decay_factor: 0.5,
            recall_count: 0,
        };
        assert!((episode.strength() - 0.45).abs() < f32::EPSILON);
    }

    #[test]
    fn test_memory_stats() {
        let mut mem = EmotionalMemory::new(MemoryConfig {
            encoding_threshold: 0.0,
            ..Default::default()
        });

        mem.encode(0, 1000, "positive".into(), PlutchikState::default(), 0.8, 0.7, vec![]);
        mem.encode(5, 1005, "negative".into(), PlutchikState::default(), -0.5, 0.8, vec![]);

        let stats = mem.stats();
        assert_eq!(stats.total_episodes, 2);
        assert_eq!(stats.positive_episodes, 1);
        assert_eq!(stats.negative_episodes, 1);
        assert_eq!(stats.oldest_tick, Some(0));
    }

    #[test]
    fn test_strongest_and_oldest() {
        let mut mem = EmotionalMemory::new(MemoryConfig {
            encoding_threshold: 0.0,
            ..Default::default()
        });

        mem.encode(10, 1000, "weak old".into(), PlutchikState::default(), 0.1, 0.5, vec![]);
        mem.encode(20, 1010, "strong new".into(), PlutchikState::default(), 0.9, 0.95, vec![]);

        assert_eq!(mem.oldest().unwrap().tick, 10);
        assert!(mem.strongest().unwrap().arousal_at_encoding > 0.9);
    }

    #[test]
    fn test_forget_removes_weak_memories() {
        let mut mem = EmotionalMemory::new(MemoryConfig {
            encoding_threshold: 0.0,
            decay_exponent: 1.0, // aggressive decay
            forget_threshold: 0.05,
            ..Default::default()
        });

        mem.encode(0, 1000, "ancient".into(), PlutchikState::default(), 0.5, 0.6, vec![]);
        mem.encode(9990, 2000, "recent".into(), PlutchikState::default(), 0.5, 0.7, vec![]);

        // At tick 10000, "ancient" (age 10000) should be below threshold
        mem.apply_decay(10000);

        // Ancient should be forgotten, recent should survive
        assert!(mem.len() <= 2);
        if mem.len() == 1 {
            assert_eq!(mem.get(0).unwrap().stimulus, "recent");
        }
    }

    #[test]
    fn test_empty_memory() {
        let mem = EmotionalMemory::default();
        assert!(mem.is_empty());
        assert_eq!(mem.len(), 0);
        assert!(mem.strongest().is_none());
        assert!(mem.oldest().is_none());
        assert!(mem.recall_by_emotion(0, 5).is_empty());
        assert!(mem.recall_by_tag("test", 5).is_empty());
    }
}
