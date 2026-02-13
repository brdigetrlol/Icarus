//! Handler functions for Cortana MCP tool calls.

use anyhow::Result;
use cortana_core::config::CortanaConfig;
use cortana_core::engine::CortanaEngine;
use serde_json::{json, Value};

/// Initialize engine from a preset name.
pub fn handle_init(engine: &mut Option<CortanaEngine>, args: &Value) -> Result<Value> {
    let preset = args
        .get("preset")
        .and_then(|v| v.as_str())
        .unwrap_or("cortana_default");

    let config = match preset {
        "cortana_default" => CortanaConfig::cortana_default(),
        "cortana_full" => CortanaConfig::cortana_full(),
        "stoic" => CortanaConfig::stoic(),
        "creative" => CortanaConfig::creative(),
        "anxious" => CortanaConfig::anxious(),
        other => {
            return Ok(json!({
                "error": format!("Unknown preset '{}'. Valid: cortana_default, cortana_full, stoic, creative, anxious", other)
            }));
        }
    };

    let issues = config.validate();
    if !issues.is_empty() {
        return Ok(json!({ "error": format!("Config validation failed: {:?}", issues) }));
    }

    let eng = CortanaEngine::new_cpu(config);
    *engine = Some(eng);

    Ok(json!({
        "status": "initialized",
        "preset": preset,
    }))
}

/// Run N ticks.
pub fn handle_tick(engine: &mut Option<CortanaEngine>, args: &Value) -> Result<Value> {
    let eng = match engine.as_mut() {
        Some(e) => e,
        None => return Ok(json!({ "error": "Engine not initialized. Call cortana_init first." })),
    };

    let ticks = args
        .get("ticks")
        .and_then(|v| v.as_u64())
        .unwrap_or(1);

    eng.run(ticks)?;

    Ok(json!({
        "ticks_run": ticks,
        "total_ticks": eng.total_ticks,
    }))
}

/// Get engine snapshot.
pub fn handle_snapshot(engine: &Option<CortanaEngine>) -> Result<Value> {
    let eng = match engine.as_ref() {
        Some(e) => e,
        None => return Ok(json!({ "error": "Engine not initialized. Call cortana_init first." })),
    };

    let snap = eng.snapshot();
    Ok(serde_json::to_value(&snap)?)
}

/// Get detailed emotion state.
pub fn handle_emotion(engine: &Option<CortanaEngine>) -> Result<Value> {
    let eng = match engine.as_ref() {
        Some(e) => e,
        None => return Ok(json!({ "error": "Engine not initialized. Call cortana_init first." })),
    };

    let affect = eng.affective_state();
    let plutchik = &affect.plutchik;
    let dominant = &affect.dominant_emotion;

    let primary_names = [
        "joy", "sadness", "trust", "disgust", "fear", "anger", "surprise", "anticipation",
    ];
    let activations: serde_json::Map<String, Value> = primary_names
        .iter()
        .zip(plutchik.activations.iter())
        .map(|(name, val)| (name.to_string(), json!(val)))
        .collect();

    let compounds: Vec<Value> = plutchik
        .active_compounds(0.3)
        .into_iter()
        .map(|c| json!(c.0.name()))
        .collect();

    Ok(json!({
        "dominant_emotion": dominant.name(),
        "activations": activations,
        "active_compounds": compounds,
        "emotional_intensity": affect.emotional_intensity(),
        "peak_activation": affect.peak_activation(),
        "pleasure": affect.pleasure,
        "arousal": affect.arousal,
        "dominance": affect.dominance,
    }))
}

/// Get mood state.
pub fn handle_mood(engine: &Option<CortanaEngine>) -> Result<Value> {
    let eng = match engine.as_ref() {
        Some(e) => e,
        None => return Ok(json!({ "error": "Engine not initialized. Call cortana_init first." })),
    };

    let mood = eng.mood_state();
    Ok(json!({
        "label": mood.label.name(),
        "hedonic_tone": mood.hedonic_tone,
        "tense_energy": mood.tense_energy,
        "confidence": mood.confidence,
    }))
}

/// Get or set personality traits.
pub fn handle_personality(engine: &mut Option<CortanaEngine>, args: &Value) -> Result<Value> {
    let eng = match engine.as_mut() {
        Some(e) => e,
        None => return Ok(json!({ "error": "Engine not initialized. Call cortana_init first." })),
    };

    let action = args
        .get("action")
        .and_then(|v| v.as_str())
        .unwrap_or("get");

    match action {
        "get" => {
            let p = eng.personality();
            Ok(json!({
                "openness": p.openness,
                "conscientiousness": p.conscientiousness,
                "extraversion": p.extraversion,
                "agreeableness": p.agreeableness,
                "neuroticism": p.neuroticism,
            }))
        }
        "set" => {
            let trait_name = args
                .get("trait_name")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let value = args
                .get("value")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.5) as f32;

            if eng.set_personality_trait(trait_name, value) {
                Ok(json!({
                    "status": "updated",
                    "trait": trait_name,
                    "value": value,
                }))
            } else {
                Ok(json!({
                    "error": format!("Unknown trait '{}'. Valid: openness, conscientiousness, extraversion, agreeableness, neuroticism", trait_name)
                }))
            }
        }
        other => Ok(json!({ "error": format!("Unknown action '{}'. Use 'get' or 'set'.", other) })),
    }
}

/// Get memory statistics.
pub fn handle_memory_stats(engine: &Option<CortanaEngine>) -> Result<Value> {
    let eng = match engine.as_ref() {
        Some(e) => e,
        None => return Ok(json!({ "error": "Engine not initialized. Call cortana_init first." })),
    };

    let stats = eng.memory_stats();
    Ok(serde_json::to_value(&stats)?)
}

/// Recall emotional memories.
pub fn handle_memory_recall(engine: &Option<CortanaEngine>, args: &Value) -> Result<Value> {
    let eng = match engine.as_ref() {
        Some(e) => e,
        None => return Ok(json!({ "error": "Engine not initialized. Call cortana_init first." })),
    };

    let mode = args
        .get("mode")
        .and_then(|v| v.as_str())
        .unwrap_or("recent");
    let limit = args
        .get("limit")
        .and_then(|v| v.as_u64())
        .unwrap_or(5) as usize;

    let indices: Vec<usize> = match mode {
        "by_tag" => {
            let tag = args
                .get("tag")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            eng.memory.recall_by_tag(tag, limit)
        }
        "mood_congruent" => {
            let mood = eng.mood_state();
            eng.memory.mood_congruent_recall(mood.hedonic_tone, limit)
        }
        _ => {
            // "recent" — return most recent episode indices
            let total = eng.memory.episodes().len();
            let start = total.saturating_sub(limit);
            (start..total).rev().collect()
        }
    };

    let results: Vec<Value> = indices
        .iter()
        .filter_map(|&idx| eng.memory.get(idx))
        .map(|ep| {
            json!({
                "stimulus": ep.stimulus,
                "valence": ep.valence,
                "arousal_at_encoding": ep.arousal_at_encoding,
                "tags": ep.tags,
                "decay_factor": ep.decay_factor,
                "tick": ep.tick,
            })
        })
        .collect();

    Ok(json!({
        "mode": mode,
        "count": results.len(),
        "memories": results,
    }))
}

/// Inject a stimulus event.
pub fn handle_inject_stimulus(engine: &mut Option<CortanaEngine>, args: &Value) -> Result<Value> {
    let eng = match engine.as_mut() {
        Some(e) => e,
        None => return Ok(json!({ "error": "Engine not initialized. Call cortana_init first." })),
    };

    let stimulus = args
        .get("stimulus")
        .and_then(|v| v.as_str())
        .unwrap_or("external event");
    let valence = args
        .get("valence")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0) as f32;
    let tags_str = args
        .get("tags")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let tags: Vec<String> = tags_str
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    eng.inject_stimulus(stimulus, valence, tags.clone());

    Ok(json!({
        "status": "injected",
        "stimulus": stimulus,
        "valence": valence,
        "tags": tags,
        "memory_count": eng.memory.episodes().len(),
    }))
}

/// Get expression metadata.
pub fn handle_expression(engine: &Option<CortanaEngine>) -> Result<Value> {
    let eng = match engine.as_ref() {
        Some(e) => e,
        None => return Ok(json!({ "error": "Engine not initialized. Call cortana_init first." })),
    };

    let expr = eng.expression();
    Ok(serde_json::to_value(expr)?)
}

/// Get creative state.
pub fn handle_creative(engine: &Option<CortanaEngine>) -> Result<Value> {
    let eng = match engine.as_ref() {
        Some(e) => e,
        None => return Ok(json!({ "error": "Engine not initialized. Call cortana_init first." })),
    };

    let creative = eng.creative_state();
    Ok(serde_json::to_value(creative)?)
}

/// Get social state.
pub fn handle_social(engine: &Option<CortanaEngine>) -> Result<Value> {
    let eng = match engine.as_ref() {
        Some(e) => e,
        None => return Ok(json!({ "error": "Engine not initialized. Call cortana_init first." })),
    };

    let social = eng.social_state();
    Ok(json!({
        "current_social_emotion": social.current_social_emotion.name(),
        "trust_level": social.trust_level,
        "empathy": social.empathy,
        "bonding": social.bonding,
        "positive_interactions": social.positive_interactions,
        "negative_interactions": social.negative_interactions,
    }))
}

/// Get neuromodulator levels.
pub fn handle_neuromodulators(engine: &Option<CortanaEngine>) -> Result<Value> {
    let eng = match engine.as_ref() {
        Some(e) => e,
        None => return Ok(json!({ "error": "Engine not initialized. Call cortana_init first." })),
    };

    let affect = eng.affective_state();
    let aether = &affect.extended_aether;

    Ok(json!({
        "dopamine": aether.base.dopamine,
        "norepinephrine": aether.base.norepinephrine,
        "acetylcholine": aether.base.acetylcholine,
        "serotonin": aether.base.serotonin,
        "oxytocin": aether.oxytocin,
        "endorphin": aether.endorphin,
        "cortisol": aether.cortisol,
        "gaba": aether.gaba,
        "derived": {
            "stress_suppression": aether.stress_suppression(),
            "regulation_capacity": aether.regulation_capacity(),
            "social_bonding": aether.social_bonding(),
            "flow_state": aether.flow_state(),
        }
    }))
}

/// Get processor status.
pub fn handle_processors(engine: &Option<CortanaEngine>) -> Result<Value> {
    let eng = match engine.as_ref() {
        Some(e) => e,
        None => return Ok(json!({ "error": "Engine not initialized. Call cortana_init first." })),
    };

    let procs = eng.processors();
    let status = procs.status_summary();
    Ok(serde_json::to_value(&status)?)
}
