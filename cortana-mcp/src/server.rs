//! CortanaServer — MCP server wrapping the Cortana emotion engine.

use std::sync::Mutex;

use async_trait::async_trait;
use cortana_core::engine::CortanaEngine;
use mcp_core::*;
use serde_json::Value;

use crate::handlers;
use crate::tools;

pub struct CortanaServer {
    engine: Mutex<Option<CortanaEngine>>,
}

impl CortanaServer {
    pub fn new() -> Self {
        Self {
            engine: Mutex::new(None),
        }
    }
}

#[async_trait]
impl McpServer for CortanaServer {
    fn server_info(&self) -> ServerInfo {
        ServerInfo::new("cortana-mcp", env!("CARGO_PKG_VERSION"))
    }

    fn tools(&self) -> Vec<Tool> {
        tools::all_tools()
    }

    async fn call_tool(&self, name: &str, args: Value) -> Result<CallToolResult, McpError> {
        let mut engine = self
            .engine
            .lock()
            .map_err(|e| McpError::InternalError(format!("Lock poisoned: {}", e)))?;

        let result = match name {
            "cortana_init" => handlers::handle_init(&mut engine, &args),
            "cortana_tick" => handlers::handle_tick(&mut engine, &args),
            "cortana_snapshot" => handlers::handle_snapshot(&engine),
            "cortana_emotion" => handlers::handle_emotion(&engine),
            "cortana_mood" => handlers::handle_mood(&engine),
            "cortana_personality" => handlers::handle_personality(&mut engine, &args),
            "cortana_memory_stats" => handlers::handle_memory_stats(&engine),
            "cortana_memory_recall" => handlers::handle_memory_recall(&engine, &args),
            "cortana_inject_stimulus" => handlers::handle_inject_stimulus(&mut engine, &args),
            "cortana_expression" => handlers::handle_expression(&engine),
            "cortana_creative" => handlers::handle_creative(&engine),
            "cortana_social" => handlers::handle_social(&engine),
            "cortana_neuromodulators" => handlers::handle_neuromodulators(&engine),
            "cortana_processors" => handlers::handle_processors(&engine),
            _ => return Err(McpError::UnknownTool(name.into())),
        };

        match result {
            Ok(val) => {
                let text = serde_json::to_string_pretty(&val)
                    .unwrap_or_else(|_| val.to_string());
                Ok(CallToolResult::text(text))
            }
            Err(e) => Ok(CallToolResult::text(format!("Error: {}", e))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mcp_core::Content;

    fn extract_text(result: &CallToolResult) -> &str {
        match &result.content[0] {
            Content::Text { text } => text.as_str(),
            _ => panic!("Expected Text content"),
        }
    }

    #[test]
    fn test_server_info() {
        let server = CortanaServer::new();
        let info = server.server_info();
        assert_eq!(info.name, "cortana-mcp");
    }

    #[test]
    fn test_tools_listing() {
        let server = CortanaServer::new();
        let tools = server.tools();
        assert_eq!(tools.len(), 14);
    }

    #[tokio::test]
    async fn test_init_default() {
        let server = CortanaServer::new();
        let result = server
            .call_tool("cortana_init", serde_json::json!({"preset": "cortana_default"}))
            .await
            .unwrap();
        let text = extract_text(&result);
        assert!(text.contains("initialized"));
    }

    #[tokio::test]
    async fn test_tick_without_init() {
        let server = CortanaServer::new();
        let result = server
            .call_tool("cortana_tick", serde_json::json!({}))
            .await
            .unwrap();
        let text = extract_text(&result);
        assert!(text.contains("not initialized"));
    }

    #[tokio::test]
    async fn test_full_workflow() {
        let server = CortanaServer::new();

        // Init
        server
            .call_tool("cortana_init", serde_json::json!({"preset": "cortana_default"}))
            .await
            .unwrap();

        // Tick
        let result = server
            .call_tool("cortana_tick", serde_json::json!({"ticks": 50}))
            .await
            .unwrap();
        let text = extract_text(&result);
        assert!(text.contains("50"));

        // Snapshot
        let result = server
            .call_tool("cortana_snapshot", serde_json::json!({}))
            .await
            .unwrap();
        let text = extract_text(&result);
        assert!(text.contains("tick"));
        assert!(text.contains("dominant_emotion"));

        // Emotion
        let result = server
            .call_tool("cortana_emotion", serde_json::json!({}))
            .await
            .unwrap();
        let text = extract_text(&result);
        assert!(text.contains("activations"));

        // Mood
        let result = server
            .call_tool("cortana_mood", serde_json::json!({}))
            .await
            .unwrap();
        let text = extract_text(&result);
        assert!(text.contains("hedonic_tone"));

        // Personality get
        let result = server
            .call_tool("cortana_personality", serde_json::json!({"action": "get"}))
            .await
            .unwrap();
        let text = extract_text(&result);
        assert!(text.contains("openness"));

        // Expression
        let result = server
            .call_tool("cortana_expression", serde_json::json!({}))
            .await
            .unwrap();
        let text = extract_text(&result);
        assert!(text.contains("tone"));

        // Creative
        let result = server
            .call_tool("cortana_creative", serde_json::json!({}))
            .await
            .unwrap();
        let text = extract_text(&result);
        assert!(text.contains("creative_drive"));

        // Social
        let result = server
            .call_tool("cortana_social", serde_json::json!({}))
            .await
            .unwrap();
        let text = extract_text(&result);
        assert!(text.contains("trust_level"));

        // Neuromodulators
        let result = server
            .call_tool("cortana_neuromodulators", serde_json::json!({}))
            .await
            .unwrap();
        let text = extract_text(&result);
        assert!(text.contains("dopamine"));
        assert!(text.contains("oxytocin"));

        // Inject stimulus
        let result = server
            .call_tool(
                "cortana_inject_stimulus",
                serde_json::json!({"stimulus": "test event", "valence": 0.8, "tags": "test,positive"}),
            )
            .await
            .unwrap();
        let text = extract_text(&result);
        assert!(text.contains("injected"));

        // Memory stats
        let result = server
            .call_tool("cortana_memory_stats", serde_json::json!({}))
            .await
            .unwrap();
        let text = extract_text(&result);
        assert!(text.contains("total_episodes"));

        // Memory recall
        let result = server
            .call_tool(
                "cortana_memory_recall",
                serde_json::json!({"mode": "by_tag", "tag": "test"}),
            )
            .await
            .unwrap();
        let text = extract_text(&result);
        assert!(text.contains("memories"));

        // Processors
        let result = server
            .call_tool("cortana_processors", serde_json::json!({}))
            .await
            .unwrap();
        let text = extract_text(&result);
        assert!(!text.is_empty());
    }

    #[tokio::test]
    async fn test_unknown_tool() {
        let server = CortanaServer::new();
        let result = server.call_tool("nonexistent", serde_json::json!({})).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_personality_set() {
        let server = CortanaServer::new();
        server
            .call_tool("cortana_init", serde_json::json!({"preset": "cortana_default"}))
            .await
            .unwrap();

        let result = server
            .call_tool(
                "cortana_personality",
                serde_json::json!({"action": "set", "trait_name": "neuroticism", "value": 0.9}),
            )
            .await
            .unwrap();
        let text = extract_text(&result);
        assert!(text.contains("updated"));

        // Verify
        let result = server
            .call_tool("cortana_personality", serde_json::json!({"action": "get"}))
            .await
            .unwrap();
        let text = extract_text(&result);
        // f32 serialization of 0.9 may produce "0.9" or "0.90000..." — check the field exists
        // and the value changed from default (cortana_default neuroticism is ~0.35)
        let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
        let n = parsed["neuroticism"].as_f64().unwrap();
        assert!((n - 0.9).abs() < 0.01, "neuroticism should be ~0.9, got {}", n);
    }

    #[tokio::test]
    async fn test_all_presets() {
        for preset in &["cortana_default", "cortana_full", "stoic", "creative", "anxious"] {
            let server = CortanaServer::new();
            let result = server
                .call_tool("cortana_init", serde_json::json!({"preset": preset}))
                .await
                .unwrap();
            let text = extract_text(&result);
            assert!(text.contains("initialized"), "Preset {} failed", preset);
        }
    }

    #[tokio::test]
    async fn test_invalid_preset() {
        let server = CortanaServer::new();
        let result = server
            .call_tool("cortana_init", serde_json::json!({"preset": "invalid"}))
            .await
            .unwrap();
        let text = extract_text(&result);
        assert!(text.contains("Unknown preset"));
    }
}
