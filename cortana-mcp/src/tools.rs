//! Tool definitions for the Cortana MCP server.

use mcp_core::ToolBuilder;

pub fn all_tools() -> Vec<mcp_core::Tool> {
    vec![
        ToolBuilder::new("cortana_init", "Initialize a Cortana emotion engine with a personality preset")
            .string_param("preset", "Personality preset: cortana_default, cortana_full, stoic, creative, anxious", false)
            .build(),
        ToolBuilder::new("cortana_tick", "Run N ticks of the Cortana emotion engine")
            .number_param("ticks", "Number of ticks to run (default 1)", false)
            .build(),
        ToolBuilder::new("cortana_snapshot", "Get a summary snapshot of the engine state")
            .build(),
        ToolBuilder::new("cortana_emotion", "Get detailed Plutchik emotion state")
            .build(),
        ToolBuilder::new("cortana_mood", "Get current mood state (hedonic tone, energy, confidence, label)")
            .build(),
        ToolBuilder::new("cortana_personality", "Get or set personality traits")
            .string_param("action", "get or set", false)
            .string_param("trait_name", "Big Five trait: openness, conscientiousness, extraversion, agreeableness, neuroticism", false)
            .number_param("value", "New trait value [0,1] (only for set)", false)
            .build(),
        ToolBuilder::new("cortana_memory_stats", "Get emotional memory statistics")
            .build(),
        ToolBuilder::new("cortana_memory_recall", "Recall emotional memories by tag or mood congruence")
            .string_param("mode", "Recall mode: by_tag, mood_congruent, recent", false)
            .string_param("tag", "Tag to search for (by_tag mode)", false)
            .number_param("limit", "Maximum memories to return (default 5)", false)
            .build(),
        ToolBuilder::new("cortana_inject_stimulus", "Inject an external stimulus event into emotional memory")
            .string_param("stimulus", "Description of the stimulus event", true)
            .number_param("valence", "Emotional valence [-1, 1] (negative=bad, positive=good)", false)
            .string_param("tags", "Comma-separated tags for the event", false)
            .build(),
        ToolBuilder::new("cortana_expression", "Get language expression metadata (tone, energy, formality)")
            .build(),
        ToolBuilder::new("cortana_creative", "Get creative agent state (drive, ideation, inspiration)")
            .build(),
        ToolBuilder::new("cortana_social", "Get social agent state (trust, empathy, bonding, social emotions)")
            .build(),
        ToolBuilder::new("cortana_neuromodulators", "Get extended neuromodulator levels (DA, NE, ACh, 5-HT, oxytocin, endorphin, cortisol, GABA)")
            .build(),
        ToolBuilder::new("cortana_processors", "Get processor orchestrator status and capabilities")
            .build(),
    ]
}
