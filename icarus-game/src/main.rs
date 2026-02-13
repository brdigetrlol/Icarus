//! Icarus Game Server
//!
//! Axum-based HTTP + WebSocket server that:
//! 1. Serves a procedurally-rendered 3D game page
//! 2. Accepts WebSocket connections for real-time input
//! 3. Runs a 20Hz physics tick loop
//! 4. Trains an ensemble of EMCs by observing player behavior
//! 5. Controls an AI agent (Icarus) that learns to imitate the player

mod bridge;
mod protocol;

use std::sync::Arc;
use std::time::Duration;

use axum::extract::ws::{Message, WebSocket};
use axum::extract::WebSocketUpgrade;
use axum::response::{Html, Json};
use axum::routing::get;
use axum::Router;
use futures::stream::StreamExt;
use futures::SinkExt;
use tokio::sync::Mutex;
use tokio::time::interval;
use tower_http::cors::CorsLayer;

use bridge::{BridgeConfig, TrainingBridge};
use cortana_core::config::CortanaConfig;
use cortana_core::engine::CortanaEngine;
use icarus_engine::world::{Action, KeyState, World};
use protocol::{ClientMsg, EntityInit, EntityUpdate, ServerMsg, TrainingMode};

/// Shared game state behind a mutex.
struct GameState {
    world: World,
    bridge: TrainingBridge,
    cortana: CortanaEngine,
    /// Latest input from the client.
    last_keys: KeyState,
    last_mouse_dx: f32,
    last_mouse_dy: f32,
}

/// Check if the player is actively providing input.
fn has_input(keys: &KeyState, mouse_dx: f32) -> bool {
    keys.forward || keys.backward || keys.left || keys.right
        || keys.jump || keys.interact
        || mouse_dx.abs() > 1.0
}

/// Convert an agent-predicted Action into KeyState for physics_tick.
fn action_to_keys(action: &Action) -> (KeyState, f32) {
    let mut keys = KeyState::default();
    let mut mouse_dx = 0.0f32;
    match action {
        Action::Move { dx, dz } => {
            if *dz > 0.1 {
                keys.forward = true;
            }
            if *dz < -0.1 {
                keys.backward = true;
            }
            if *dx < -0.1 {
                keys.left = true;
            }
            if *dx > 0.1 {
                keys.right = true;
            }
            if dx.abs() > 0.1 {
                mouse_dx = *dx * 3.0;
            }
        }
        Action::Jump => keys.jump = true,
        Action::PickUp { .. } | Action::ToggleLight { .. } => keys.interact = true,
        Action::Drop | Action::Push { .. } | Action::Idle => {}
    }
    (keys, mouse_dx)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let port = std::env::var("ICARUS_PORT")
        .ok()
        .and_then(|p| p.parse::<u16>().ok())
        .unwrap_or(3000);

    let seed = std::env::var("ICARUS_SEED")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(42);

    let world = World::new(seed);
    let bridge_config = BridgeConfig {
        seed,
        ..BridgeConfig::default()
    };
    let bridge = TrainingBridge::new(bridge_config)?;
    let cortana = CortanaEngine::new_cpu(CortanaConfig::cortana_default());

    let state = Arc::new(Mutex::new(GameState {
        world,
        bridge,
        cortana,
        last_keys: KeyState::default(),
        last_mouse_dx: 0.0,
        last_mouse_dy: 0.0,
    }));

    let app = Router::new()
        .route("/", get(serve_game_page))
        .route("/status", get({
            let state = Arc::clone(&state);
            move || serve_status(state)
        }))
        .route("/ws", get({
            let state = Arc::clone(&state);
            move |ws| ws_handler(ws, state)
        }))
        .layer(CorsLayer::permissive());

    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{port}")).await?;
    eprintln!("Icarus Game running on http://localhost:{port}");
    axum::serve(listener, app).await?;

    Ok(())
}

/// Serve the game HTML page.
async fn serve_game_page() -> Html<String> {
    Html(build_game_html())
}

/// Serve JSON status for MCP queries.
async fn serve_status(state: Arc<Mutex<GameState>>) -> Json<serde_json::Value> {
    let gs = state.lock().await;
    let backends: Vec<serde_json::Value> = gs
        .bridge
        .backend_status()
        .iter()
        .map(|b| {
            serde_json::json!({
                "name": b.name,
                "state_dim": b.state_dim,
                "ticks": b.ticks,
            })
        })
        .collect();

    Json(serde_json::json!({
        "tick": gs.world.tick,
        "time_of_day": gs.world.time_of_day,
        "samples": gs.bridge.total_samples(),
        "nmse": gs.bridge.nmse(),
        "confidence": gs.bridge.confidence(),
        "trained": gs.bridge.is_trained(),
        "player_position": gs.world.player.position,
        "agent_active": gs.world.agent.active,
        "agent_confidence": gs.world.agent.confidence,
        "backends": backends,
    }))
}

/// Handle a WebSocket connection: game loop + input processing.
async fn ws_handler(ws: WebSocketUpgrade, state: Arc<Mutex<GameState>>) -> axum::response::Response {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

async fn handle_socket(socket: WebSocket, state: Arc<Mutex<GameState>>) {
    let (mut sender, mut receiver) = socket.split();

    // Wait for Ready message from client
    loop {
        match receiver.next().await {
            Some(Ok(Message::Text(text))) => {
                if let Ok(ClientMsg::Ready) = serde_json::from_str(&text) {
                    break;
                }
            }
            Some(Ok(Message::Close(_))) | None => return,
            _ => continue,
        }
    }

    // Send initial world state
    {
        let gs = state.lock().await;
        let init_msg = ServerMsg::InitWorld {
            terrain_grid: gs.world.terrain.grid(),
            terrain_extent: gs.world.terrain.extent,
            entities: gs.world.entities.iter().map(EntityInit::from).collect(),
            seed: gs.world.seed,
            lattice_positions: TrainingBridge::lattice_positions(),
        };
        if let Ok(json) = serde_json::to_string(&init_msg) {
            let _ = sender.send(Message::Text(json.into())).await;
        }
    }

    // Channel for mode change notifications (input task → game loop)
    let (mode_tx, mut mode_rx) = tokio::sync::mpsc::channel::<TrainingMode>(4);

    // Spawn a task to process incoming messages
    let input_state = Arc::clone(&state);
    let input_task = tokio::spawn(async move {
        while let Some(Ok(msg)) = receiver.next().await {
            if let Message::Text(text) = msg {
                if let Ok(client_msg) = serde_json::from_str::<ClientMsg>(&text) {
                    let mut gs = input_state.lock().await;
                    match client_msg {
                        ClientMsg::Input {
                            keys,
                            mouse_dx,
                            mouse_dy,
                        } => {
                            gs.last_keys = keys;
                            gs.last_mouse_dx = mouse_dx;
                            gs.last_mouse_dy = mouse_dy;
                        }
                        ClientMsg::PlayerAction(action) => {
                            let GameState {
                                ref world,
                                ref mut bridge,
                                ..
                            } = *gs;
                            let _ = bridge.on_player_action(world, &action);
                        }
                        ClientMsg::ModeSwitch { mode } => {
                            gs.bridge.set_mode(mode);
                            let _ = mode_tx.send(mode).await;
                        }
                        ClientMsg::Ready => {}
                    }
                }
            }
        }
    });

    // Game loop: 20Hz tick
    let mut tick_interval = interval(Duration::from_millis(50));
    let mut training_counter = 0u64;

    loop {
        tick_interval.tick().await;

        // Check for mode change notifications (non-blocking)
        let mut mode_changed_msg: Option<ServerMsg> = None;
        if let Ok(new_mode) = mode_rx.try_recv() {
            mode_changed_msg = Some(ServerMsg::ModeChanged {
                mode: new_mode,
                message: format!("Switched to {} mode", new_mode),
            });
        }

        let (world_msg, training_msg, lattice_msg, cortana_msg) = {
            let mut gs = state.lock().await;
            let current_mode = gs.bridge.mode();

            // Snapshot input
            let keys = gs.last_keys.clone();
            let mdx = gs.last_mouse_dx;
            let mdy = gs.last_mouse_dy;
            gs.last_mouse_dx = 0.0;
            gs.last_mouse_dy = 0.0;

            match current_mode {
                TrainingMode::Observation => {
                    // Human plays, EMC observes
                    let action = World::keys_to_action(&keys, mdx);

                    // Record for training (every other tick)
                    if gs.world.tick % 2 == 0 {
                        let GameState {
                            ref world,
                            ref mut bridge,
                            ..
                        } = *gs;
                        let _ = bridge.on_player_action(world, &action);
                    }

                    // Physics tick with human input
                    gs.world.physics_tick(&keys, mdx, mdy);

                    // Predict and show agent (shadow mode — doesn't control anything)
                    if gs.bridge.is_trained() {
                        let GameState {
                            ref world,
                            ref mut bridge,
                            ..
                        } = *gs;
                        if let Ok(Some(agent_action)) = bridge.predict_agent_action(world) {
                            gs.world.apply_agent_action(&agent_action);
                            gs.world.agent.active = true;
                            gs.world.agent.confidence = gs.bridge.confidence();
                        }
                    }
                }

                TrainingMode::DAgger => {
                    // Agent proposes, human can override
                    let human_active = has_input(&keys, mdx);
                    let human_action = if human_active {
                        Some(World::keys_to_action(&keys, mdx))
                    } else {
                        None
                    };

                    // DAgger step: agent predicts, human correction recorded
                    let agent_action = {
                        let GameState {
                            ref world,
                            ref mut bridge,
                            ..
                        } = *gs;
                        bridge.on_dagger_step(world, human_action.as_ref())
                    };

                    if human_active {
                        // Human is overriding — use human keys for physics
                        gs.world.physics_tick(&keys, mdx, mdy);
                    } else if let Ok(Some(ref action)) = agent_action {
                        // Agent drives physics
                        let (agent_keys, agent_mdx) = action_to_keys(action);
                        gs.world.physics_tick(&agent_keys, agent_mdx, 0.0);
                    } else {
                        // No agent prediction yet, use human input (even if idle)
                        gs.world.physics_tick(&keys, mdx, mdy);
                    }

                    // Show agent
                    if let Ok(Some(ref action)) = agent_action {
                        gs.world.apply_agent_action(action);
                        gs.world.agent.active = true;
                        gs.world.agent.confidence = gs.bridge.confidence();
                    }
                }

                TrainingMode::Autonomous => {
                    // Agent controls everything, no training
                    if gs.bridge.is_trained() {
                        let agent_action = {
                            let GameState {
                                ref world,
                                ref mut bridge,
                                ..
                            } = *gs;
                            bridge.predict_agent_action(world)
                        };

                        if let Ok(Some(ref action)) = agent_action {
                            let (agent_keys, agent_mdx) = action_to_keys(action);
                            gs.world.physics_tick(&agent_keys, agent_mdx, 0.0);
                            gs.world.apply_agent_action(action);
                            gs.world.agent.active = true;
                            gs.world.agent.confidence = gs.bridge.confidence();
                        } else {
                            // Fallback: idle tick
                            gs.world.physics_tick(&KeyState::default(), 0.0, 0.0);
                        }
                    } else {
                        // Not trained yet — just tick with no input
                        gs.world.physics_tick(&KeyState::default(), 0.0, 0.0);
                        gs.world.agent.active = false;
                    }
                }
            }

            // Tick Cortana emotion engine
            let _ = gs.cortana.tick();
            if has_input(&keys, mdx) {
                gs.cortana.inject_stimulus("player_input", 0.1, vec!["gameplay".into()]);
            }

            // Retrain periodically (in Observation and DAgger modes)
            training_counter += 1;
            let train_msg = if training_counter % 20 == 0 {
                if current_mode != TrainingMode::Autonomous {
                    let _ = gs.bridge.retrain_if_needed();
                }
                Some(ServerMsg::TrainingUpdate {
                    mode: current_mode,
                    nmse: gs.bridge.nmse(),
                    samples: gs.bridge.total_samples(),
                    confidence: gs.bridge.confidence(),
                    backends: gs.bridge.backend_status(),
                    trained: gs.bridge.is_trained(),
                    interventions: gs.bridge.interventions(),
                    online_updates: gs.bridge.online_updates(),
                    ewc_tasks: gs.bridge.ewc_tasks(),
                    replay_size: gs.bridge.replay_size(),
                    retrain_count: gs.bridge.retrain_count(),
                    nmse_history: gs.bridge.nmse_history().to_vec(),
                    events: gs.bridge.drain_events(),
                })
            } else {
                None
            };

            // Build world state message
            let world_msg = ServerMsg::WorldState {
                player: gs.world.player.clone(),
                agent: gs.world.agent.clone(),
                entities: gs.world.entities.iter().map(EntityUpdate::from).collect(),
                time_of_day: gs.world.time_of_day,
                tick: gs.world.tick,
                agent_action_vec: gs.bridge.predicted_action_direction(),
            };

            // Send lattice overlay every 10 ticks
            let lattice_msg = if training_counter % 10 == 0 {
                let values = gs.bridge.lattice_overlay_data();
                if !values.is_empty() {
                    Some(ServerMsg::LatticeOverlay { values })
                } else {
                    None
                }
            } else {
                None
            };

            // Send Cortana emotion update every 20 ticks
            let cortana_msg = if training_counter % 20 == 0 {
                let affect = gs.cortana.affective_state();
                let aether = &affect.extended_aether;
                let social = gs.cortana.social_state();
                let creative = gs.cortana.creative_state();
                Some(ServerMsg::CortanaUpdate {
                    dominant_emotion: affect.dominant_emotion.name().to_string(),
                    mood: gs.cortana.mood_state().label.name().to_string(),
                    pleasure: affect.pleasure,
                    arousal: affect.arousal,
                    dominance: affect.dominance,
                    plutchik: affect.plutchik.activations,
                    emotional_intensity: affect.emotional_intensity(),
                    memory_count: gs.cortana.memory_stats().total_episodes,
                    creative_drive: creative.creative_drive,
                    social_bonding: social.bonding,
                    neuromodulators: [
                        aether.base.dopamine,
                        aether.base.norepinephrine,
                        aether.base.acetylcholine,
                        aether.base.serotonin,
                        aether.oxytocin,
                        aether.endorphin,
                        aether.cortisol,
                        aether.gaba,
                    ],
                })
            } else {
                None
            };

            (world_msg, train_msg, lattice_msg, cortana_msg)
        };

        // Send mode change notification if any
        if let Some(msg) = mode_changed_msg {
            if let Ok(json) = serde_json::to_string(&msg) {
                let _ = sender.send(Message::Text(json.into())).await;
            }
        }

        // Send world state
        if let Ok(json) = serde_json::to_string(&world_msg) {
            if sender.send(Message::Text(json.into())).await.is_err() {
                break;
            }
        }

        // Send training update if available
        if let Some(msg) = training_msg {
            if let Ok(json) = serde_json::to_string(&msg) {
                let _ = sender.send(Message::Text(json.into())).await;
            }
        }

        // Send lattice overlay if available
        if let Some(msg) = lattice_msg {
            if let Ok(json) = serde_json::to_string(&msg) {
                let _ = sender.send(Message::Text(json.into())).await;
            }
        }

        // Send Cortana emotion update if available
        if let Some(msg) = cortana_msg {
            if let Ok(json) = serde_json::to_string(&msg) {
                let _ = sender.send(Message::Text(json.into())).await;
            }
        }
    }

    input_task.abort();
}

// ─── Game HTML Page ─────────────────────────────────

/// Build the complete game HTML page with procedural Three.js rendering.
fn build_game_html() -> String {
    format!(
        r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Icarus World</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ overflow: hidden; background: #0a0a0f; font-family: 'Courier New', monospace; color: #e0e0e0; cursor: crosshair; }}
canvas {{ display: block; }}

.hud-panel {{
    position: fixed; padding: 12px 16px; background: rgba(10, 10, 20, 0.85);
    border: 1px solid rgba(0, 229, 255, 0.3); border-radius: 4px;
    font-size: 12px; line-height: 1.6; pointer-events: none; z-index: 100;
}}
#hud-training {{ top: 10px; right: 10px; min-width: 220px; }}
#hud-controls {{ bottom: 10px; left: 10px; }}
#hud-stats {{ top: 10px; left: 10px; min-width: 180px; }}
#hud-crosshair {{
    position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
    width: 20px; height: 20px; z-index: 100; pointer-events: none;
}}
#hud-crosshair::before, #hud-crosshair::after {{
    content: ''; position: absolute; background: rgba(0, 229, 255, 0.7);
}}
#hud-crosshair::before {{ left: 50%; top: 0; width: 1px; height: 100%; transform: translateX(-50%); }}
#hud-crosshair::after {{ top: 50%; left: 0; height: 1px; width: 100%; transform: translateY(-50%); }}

.accent {{ color: #00e5ff; }}
.dim {{ color: #666; }}
.bar {{ display: inline-block; height: 8px; background: #00e5ff; border-radius: 2px; transition: width 0.3s; }}
.bar-bg {{ display: inline-block; width: 100px; height: 8px; background: rgba(255,255,255,0.1); border-radius: 2px; }}
.mode-observation {{ border-color: rgba(0, 229, 255, 0.5) !important; }}
.mode-dagger {{ border-color: rgba(255, 165, 0, 0.5) !important; }}
.mode-autonomous {{ border-color: rgba(0, 255, 100, 0.5) !important; }}
.mode-badge {{ display: inline-block; padding: 2px 8px; border-radius: 3px; font-weight: bold; font-size: 11px; }}
.mode-badge.obs {{ background: rgba(0,229,255,0.2); color: #00e5ff; }}
.mode-badge.dag {{ background: rgba(255,165,0,0.2); color: #ffa500; }}
.mode-badge.auto {{ background: rgba(0,255,100,0.2); color: #00ff64; }}
#mode-toast {{
    position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
    padding: 16px 32px; background: rgba(10, 10, 20, 0.95);
    border: 2px solid #00e5ff; border-radius: 8px; font-size: 20px;
    font-weight: bold; z-index: 300; pointer-events: none;
    opacity: 0; transition: opacity 0.3s;
}}

#click-to-play {{
    position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    background: rgba(10, 10, 20, 0.95); z-index: 200;
    display: flex; align-items: center; justify-content: center; flex-direction: column;
    cursor: pointer;
}}
#click-to-play h1 {{ font-size: 48px; color: #00e5ff; margin-bottom: 20px; }}
#click-to-play p {{ font-size: 16px; color: #888; }}

#hud-events {{
    position: fixed; bottom: 10px; right: 10px; width: 280px; max-height: 180px;
    overflow-y: auto; padding: 10px 14px; background: rgba(10, 10, 20, 0.85);
    border: 1px solid rgba(0, 229, 255, 0.2); border-radius: 4px;
    font-size: 11px; line-height: 1.5; z-index: 100; pointer-events: none;
}}
#hud-events .ev {{ color: #999; border-bottom: 1px solid rgba(255,255,255,0.05); padding: 2px 0; }}
#nmse-canvas {{ display: block; margin-top: 6px; border: 1px solid rgba(0,229,255,0.15); border-radius: 2px; }}
</style>
</head>
<body>

<div id="click-to-play">
    <h1>ICARUS WORLD</h1>
    <p>Click to play — WASD to move, Mouse to look, Space to jump, E to interact</p>
</div>

<div id="hud-crosshair"></div>

<div class="hud-panel" id="hud-stats">
    <div>Tick: <span class="accent" id="stat-tick">0</span></div>
    <div>Pos: <span class="accent" id="stat-pos">0, 0, 0</span></div>
    <div>Time: <span class="accent" id="stat-time">morning</span></div>
    <div>FPS: <span class="accent" id="stat-fps">0</span></div>
</div>

<div class="hud-panel mode-observation" id="hud-training">
    <div style="margin-bottom:6px; font-weight:bold;">ICARUS TRAINING <span class="mode-badge obs" id="mode-badge">OBSERVATION</span></div>
    <div>Status: <span class="accent" id="train-status">Observing...</span></div>
    <div>Samples: <span class="accent" id="train-samples">0</span></div>
    <div>NMSE: <span class="accent" id="train-nmse">&mdash;</span></div>
    <div>Confidence: <span id="train-conf-bar" class="bar-bg"><span class="bar" id="train-conf" style="width:0%"></span></span></div>
    <div id="train-interventions" style="display:none;">Interventions: <span class="accent" id="train-intv-count">0</span></div>
    <div id="train-backends" class="dim" style="margin-top:6px;"></div>
    <div id="train-extra" class="dim" style="margin-top:4px;"></div>
    <canvas id="nmse-canvas" width="200" height="60"></canvas>
</div>

<div class="hud-panel" id="hud-controls">
    <span class="dim">WASD</span> Move &nbsp; <span class="dim">Mouse</span> Look &nbsp;
    <span class="dim">Space</span> Jump &nbsp; <span class="dim">E</span> Interact &nbsp;
    <span class="dim">Tab</span> Mode &nbsp; <span class="dim">L</span> Lattice
</div>

<div id="mode-toast"></div>

<div id="hud-events">
    <div style="margin-bottom:4px;font-weight:bold;color:#00e5ff;font-size:12px;">EVENTS</div>
    <div id="event-list"></div>
</div>

<div class="hud-panel" id="hud-cortana" style="bottom:200px; right:10px; min-width:220px;">
    <div style="margin-bottom:6px;font-weight:bold;">CORTANA <span style="color:#ff66aa;font-size:11px;" id="cortana-emotion">Neutral</span></div>
    <div>Mood: <span class="accent" id="cortana-mood">Neutral</span></div>
    <div>PAD: <span class="dim" id="cortana-pad">0.00 / 0.00 / 0.00</span></div>
    <div>Intensity: <span id="cortana-intensity-bar" class="bar-bg"><span class="bar" id="cortana-intensity" style="width:0%;background:#ff66aa;"></span></span></div>
    <div style="margin-top:4px;font-size:11px;" id="cortana-plutchik"></div>
    <div style="margin-top:4px;" class="dim" id="cortana-neuro"></div>
    <div class="dim">Memories: <span class="accent" id="cortana-mem">0</span> | Creative: <span class="accent" id="cortana-creative">0.00</span> | Social: <span class="accent" id="cortana-social">0.00</span></div>
</div>

<script>
// ─── State ──────────────────────────────────────
const keys = {{ forward:false, backward:false, left:false, right:false, jump:false, interact:false }};
let mouseDx = 0, mouseDy = 0;
let wsReady = false;
let ws = null;
let terrainMesh = null;
let entityMeshes = {{}};
let agentGroup = null;
let scene, camera, renderer;
let terrainData = null;
let terrainExtent = 256;
let locked = false;

// Training mode state
const MODES = ['Observation', 'DAgger', 'Autonomous'];
let currentModeIdx = 0;
let currentMode = 'Observation';
let toastTimer = null;

// Phase 4: lattice overlay, action arrow, NMSE graph, event log
let latticePoints = null;
let latticePositions = null;
let latticeVisible = true;
let actionArrow = null;
let nmseHistory = [];

// ─── Three.js Setup ─────────────────────────────
scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a2e);
scene.fog = new THREE.FogExp2(0x1a1a2e, 0.008);

camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 500);
camera.position.set(0, 5, 0);

renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = false;
document.body.appendChild(renderer.domElement);

// ─── Lighting ───────────────────────────────────
const ambientLight = new THREE.AmbientLight(0x334455, 0.4);
scene.add(ambientLight);

const sunLight = new THREE.DirectionalLight(0xffeedd, 1.0);
sunLight.position.set(50, 80, 30);
sunLight.castShadow = false;
scene.add(sunLight);

const hemisphereLight = new THREE.HemisphereLight(0x88aacc, 0x443322, 0.3);
scene.add(hemisphereLight);

// ─── Water plane ────────────────────────────────
const waterGeo = new THREE.PlaneGeometry(300, 300, 16, 16);
const waterMat = new THREE.MeshPhongMaterial({{
    color: 0x1155aa, transparent: true, opacity: 0.6,
    shininess: 100, specular: 0x88bbff
}});
const waterMesh = new THREE.Mesh(waterGeo, waterMat);
waterMesh.rotation.x = -Math.PI / 2;
waterMesh.position.y = -0.5;
scene.add(waterMesh);

// ─── Sky dome ───────────────────────────────────
const skyGeo = new THREE.SphereGeometry(400, 32, 16);
const skyMat = new THREE.MeshBasicMaterial({{ color: 0x1a1a3e, side: THREE.BackSide }});
const skyMesh = new THREE.Mesh(skyGeo, skyMat);
scene.add(skyMesh);

// ─── Entity builders ────────────────────────────
function buildTree(ent) {{
    const g = new THREE.Group();
    // Trunk
    const trunk = new THREE.Mesh(
        new THREE.CylinderGeometry(0.15*ent.scale, 0.25*ent.scale, ent.scale*2, 6),
        new THREE.MeshLambertMaterial({{ color: 0x553311 }})
    );
    trunk.position.y = ent.scale;
    g.add(trunk);
    // Foliage layers
    for (let i = 0; i < 3; i++) {{
        const r = ent.scale * (1.2 - i*0.3);
        const foliage = new THREE.Mesh(
            new THREE.SphereGeometry(r, 8, 6),
            new THREE.MeshLambertMaterial({{ color: new THREE.Color(ent.color[0], ent.color[1], ent.color[2]) }})
        );
        foliage.position.y = ent.scale * (1.8 + i*0.6);
        g.add(foliage);
    }}
    g.position.set(ent.position[0], ent.position[1], ent.position[2]);
    return g;
}}

function buildRock(ent) {{
    const geo = new THREE.IcosahedronGeometry(ent.scale, 1);
    const posArr = geo.attributes.position.array;
    for (let i = 0; i < posArr.length; i++) {{
        posArr[i] += (Math.random() - 0.5) * 0.2 * ent.scale;
    }}
    geo.computeVertexNormals();
    const mat = new THREE.MeshLambertMaterial({{
        color: new THREE.Color(ent.color[0], ent.color[1], ent.color[2])
    }});
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(ent.position[0], ent.position[1] + ent.scale*0.5, ent.position[2]);
    return mesh;
}}

function buildOrb(ent) {{
    const g = new THREE.Group();
    const mesh = new THREE.Mesh(
        new THREE.SphereGeometry(ent.scale, 16, 12),
        new THREE.MeshPhongMaterial({{
            color: new THREE.Color(ent.color[0], ent.color[1], ent.color[2]),
            emissive: new THREE.Color(ent.color[0]*0.5, ent.color[1]*0.5, ent.color[2]*0.5),
            transparent: true, opacity: 0.8, shininess: 150
        }})
    );
    const light = new THREE.PointLight(
        new THREE.Color(ent.color[0], ent.color[1], ent.color[2]).getHex(), 0.6, 8
    );
    g.add(mesh);
    g.add(light);
    g.position.set(ent.position[0], ent.position[1], ent.position[2]);
    return g;
}}

function buildCrate(ent) {{
    const mesh = new THREE.Mesh(
        new THREE.BoxGeometry(ent.scale, ent.scale, ent.scale),
        new THREE.MeshLambertMaterial({{ color: 0x996633 }})
    );
    mesh.position.set(ent.position[0], ent.position[1] + ent.scale*0.5, ent.position[2]);
    mesh.rotation.y = ent.rotation_y;
    return mesh;
}}

function buildLight(ent) {{
    const g = new THREE.Group();
    // Pole
    const pole = new THREE.Mesh(
        new THREE.CylinderGeometry(0.05, 0.08, 2.5, 4),
        new THREE.MeshLambertMaterial({{ color: 0x444444 }})
    );
    pole.position.y = 1.25;
    g.add(pole);
    // Lamp
    const lamp = new THREE.Mesh(
        new THREE.SphereGeometry(0.2, 8, 8),
        new THREE.MeshBasicMaterial({{ color: 0xffee88 }})
    );
    lamp.position.y = 2.6;
    g.add(lamp);
    const light = new THREE.PointLight(0xffee88, 1.0, 15);
    light.position.y = 2.6;
    g.add(light);
    g.position.set(ent.position[0], ent.position[1], ent.position[2]);
    return g;
}}

// ─── Icarus Agent (E8 lattice wireframe) ────────
function buildIcarusAgent() {{
    const g = new THREE.Group();
    // Body: translucent sphere
    const body = new THREE.Mesh(
        new THREE.SphereGeometry(0.6, 16, 12),
        new THREE.MeshPhongMaterial({{
            color: 0x00e5ff, emissive: 0x003344,
            transparent: true, opacity: 0.3, wireframe: true
        }})
    );
    g.add(body);
    // Inner glow
    const core = new THREE.Mesh(
        new THREE.SphereGeometry(0.2, 12, 8),
        new THREE.MeshBasicMaterial({{ color: 0x00e5ff, transparent: true, opacity: 0.8 }})
    );
    g.add(core);
    // Point light
    const glow = new THREE.PointLight(0x00e5ff, 0.5, 10);
    g.add(glow);
    g.visible = false;
    return g;
}}

// ─── Build terrain mesh from heightmap grid ─────
function buildTerrain(grid, extent) {{
    const size = grid.length;
    const geo = new THREE.PlaneGeometry(extent, extent, size-1, size-1);
    geo.rotateX(-Math.PI / 2);
    const posArr = geo.attributes.position.array;
    // Assign heights from grid
    for (let z = 0; z < size; z++) {{
        for (let x = 0; x < size; x++) {{
            const idx = z * size + x;
            posArr[idx * 3 + 1] = grid[z][x]; // y = height
        }}
    }}
    geo.computeVertexNormals();
    // Vertex color by height
    const colors = new Float32Array(posArr.length);
    for (let i = 0; i < posArr.length / 3; i++) {{
        const h = posArr[i*3+1];
        const t = (h + 8) / 16; // normalize roughly
        colors[i*3]   = 0.15 + t * 0.3; // R
        colors[i*3+1] = 0.3 + t * 0.4;  // G
        colors[i*3+2] = 0.1 + t * 0.15; // B
    }}
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    const mat = new THREE.MeshLambertMaterial({{ vertexColors: true }});
    const mesh = new THREE.Mesh(geo, mat);
    return mesh;
}}

// ─── Input ──────────────────────────────────────
function cycleMode() {{
    currentModeIdx = (currentModeIdx + 1) % MODES.length;
    const newMode = MODES[currentModeIdx];
    if (wsReady && ws.readyState === 1) {{
        ws.send(JSON.stringify({{ type: 'ModeSwitch', mode: newMode }}));
    }}
}}

function showModeToast(mode, message) {{
    const toast = document.getElementById('mode-toast');
    const colors = {{ Observation: '#00e5ff', DAgger: '#ffa500', Autonomous: '#00ff64' }};
    toast.style.borderColor = colors[mode] || '#00e5ff';
    toast.style.color = colors[mode] || '#00e5ff';
    toast.textContent = message || ('Mode: ' + mode);
    toast.style.opacity = '1';
    if (toastTimer) clearTimeout(toastTimer);
    toastTimer = setTimeout(() => {{ toast.style.opacity = '0'; }}, 1500);
}}

function updateModeUI(mode) {{
    currentMode = mode;
    currentModeIdx = MODES.indexOf(mode);
    if (currentModeIdx < 0) currentModeIdx = 0;

    const panel = document.getElementById('hud-training');
    const badge = document.getElementById('mode-badge');
    const intv = document.getElementById('train-interventions');

    panel.className = 'hud-panel mode-' + mode.toLowerCase();

    const badgeMap = {{
        Observation: ['obs', 'OBSERVATION'],
        DAgger: ['dag', 'DAGGER'],
        Autonomous: ['auto', 'AUTONOMOUS']
    }};
    const [cls, label] = badgeMap[mode] || ['obs', mode.toUpperCase()];
    badge.className = 'mode-badge ' + cls;
    badge.textContent = label;

    intv.style.display = mode === 'DAgger' ? 'block' : 'none';
}}

document.addEventListener('keydown', e => {{
    switch(e.code) {{
        case 'KeyW': keys.forward = true; break;
        case 'KeyS': keys.backward = true; break;
        case 'KeyA': keys.left = true; break;
        case 'KeyD': keys.right = true; break;
        case 'Space': keys.jump = true; e.preventDefault(); break;
        case 'KeyE': keys.interact = true; break;
        case 'Tab': cycleMode(); e.preventDefault(); break;
        case 'KeyL': latticeVisible = !latticeVisible; if (latticePoints) latticePoints.visible = latticeVisible; break;
    }}
}});
document.addEventListener('keyup', e => {{
    switch(e.code) {{
        case 'KeyW': keys.forward = false; break;
        case 'KeyS': keys.backward = false; break;
        case 'KeyA': keys.left = false; break;
        case 'KeyD': keys.right = false; break;
        case 'Space': keys.jump = false; break;
        case 'KeyE': keys.interact = false; break;
    }}
}});
document.addEventListener('mousemove', e => {{
    if (!locked) return;
    mouseDx += e.movementX;
    mouseDy += e.movementY;
}});

// ─── Pointer Lock ───────────────────────────────
const overlay = document.getElementById('click-to-play');
overlay.addEventListener('click', () => {{
    document.body.requestPointerLock();
}});
document.addEventListener('pointerlockchange', () => {{
    locked = !!document.pointerLockElement;
    overlay.style.display = locked ? 'none' : 'flex';
}});

// ─── WebSocket ──────────────────────────────────
function connectWS() {{
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(proto + '//' + location.host + '/ws');

    ws.onopen = () => {{
        ws.send(JSON.stringify({{ type: 'Ready' }}));
        wsReady = true;
    }};

    ws.onmessage = (evt) => {{
        const msg = JSON.parse(evt.data);
        handleServerMsg(msg);
    }};

    ws.onclose = () => {{
        wsReady = false;
        setTimeout(connectWS, 2000);
    }};
}}

function handleServerMsg(msg) {{
    switch(msg.type) {{
        case 'InitWorld':
            terrainData = msg;
            terrainExtent = msg.terrain_extent;
            initWorld(msg);
            break;
        case 'WorldState':
            updateWorld(msg);
            break;
        case 'TrainingUpdate':
            updateTrainingHUD(msg);
            break;
        case 'LatticeOverlay':
            updateLatticeOverlay(msg.values);
            break;
        case 'ModeChanged':
            updateModeUI(msg.mode);
            showModeToast(msg.mode, msg.message);
            break;
        case 'CortanaUpdate':
            updateCortanaHUD(msg);
            break;
    }}
}}

function initWorld(msg) {{
    // Build terrain
    if (terrainMesh) scene.remove(terrainMesh);
    terrainMesh = buildTerrain(msg.terrain_grid, msg.terrain_extent);
    scene.add(terrainMesh);

    // Build entities
    for (const ent of msg.entities) {{
        let mesh;
        switch(ent.kind) {{
            case 'tree': mesh = buildTree(ent); break;
            case 'rock': mesh = buildRock(ent); break;
            case 'orb': mesh = buildOrb(ent); break;
            case 'crate': mesh = buildCrate(ent); break;
            case 'light': mesh = buildLight(ent); break;
            default: continue;
        }}
        scene.add(mesh);
        entityMeshes[ent.id] = mesh;
    }}

    // Build Icarus agent
    agentGroup = buildIcarusAgent();
    scene.add(agentGroup);

    // Build E8 lattice point cloud overlay
    if (msg.lattice_positions && msg.lattice_positions.length > 0) {{
        if (latticePoints) scene.remove(latticePoints);
        const count = msg.lattice_positions.length;
        const geo = new THREE.BufferGeometry();
        const positions = new Float32Array(count * 3);
        const colors = new Float32Array(count * 3);
        for (let i = 0; i < count; i++) {{
            positions[i*3]   = msg.lattice_positions[i][0];
            positions[i*3+1] = msg.lattice_positions[i][1];
            positions[i*3+2] = msg.lattice_positions[i][2];
            colors[i*3] = 0.0; colors[i*3+1] = 0.9; colors[i*3+2] = 1.0;
        }}
        latticePositions = positions;
        geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(positions), 3));
        geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        const mat = new THREE.PointsMaterial({{
            size: 0.3, vertexColors: true, transparent: true, opacity: 0.6,
            blending: THREE.AdditiveBlending, depthWrite: false, sizeAttenuation: true
        }});
        latticePoints = new THREE.Points(geo, mat);
        latticePoints.visible = latticeVisible;
        scene.add(latticePoints);
    }}
}}

function updateWorld(msg) {{
    const p = msg.player;

    // Camera: third-person behind agent in Autonomous mode, first-person otherwise
    if (currentMode === 'Autonomous' && msg.agent.active) {{
        const a = msg.agent;
        const targetX = a.position[0];
        const targetY = a.position[1] + 1.0;
        const targetZ = a.position[2];
        const angle = -p.rotation_y;
        const idealX = targetX - Math.sin(angle) * 6;
        const idealZ = targetZ - Math.cos(angle) * 6;
        const idealY = targetY + 3;
        camera.position.x += (idealX - camera.position.x) * 0.1;
        camera.position.y += (idealY - camera.position.y) * 0.1;
        camera.position.z += (idealZ - camera.position.z) * 0.1;
        camera.lookAt(targetX, targetY, targetZ);
    }} else {{
        camera.position.set(p.position[0], p.position[1] + 1.6, p.position[2]);
        camera.rotation.order = 'YXZ';
        camera.rotation.y = -p.rotation_y;
        camera.rotation.x = p.pitch;
    }}

    // Update entity positions
    for (const eu of msg.entities) {{
        const mesh = entityMeshes[eu.id];
        if (mesh) {{
            mesh.position.set(eu.position[0], eu.position[1], eu.position[2]);
            mesh.visible = !eu.held || eu.active;
        }}
    }}

    // Update Icarus agent
    if (agentGroup && msg.agent.active) {{
        agentGroup.visible = true;
        agentGroup.position.set(msg.agent.position[0], msg.agent.position[1] + 0.8, msg.agent.position[2]);
        // Pulse based on confidence
        const s = 1.0 + msg.agent.confidence * 0.3 * Math.sin(Date.now() * 0.003);
        agentGroup.scale.set(s, s, s);
    }}

    // Action arrow (agent predicted direction) — reuse object
    if (msg.agent_action_vec) {{
        const av = msg.agent_action_vec;
        const dir = new THREE.Vector3(av[0], av[1], av[2]);
        const len = dir.length();
        if (len > 0.01) {{
            dir.normalize();
            if (!actionArrow) {{
                actionArrow = new THREE.ArrowHelper(dir, new THREE.Vector3(), 1, 0x00ff64, 0.3, 0.15);
                scene.add(actionArrow);
            }}
            actionArrow.position.set(msg.agent.position[0], msg.agent.position[1] + 1.5, msg.agent.position[2]);
            actionArrow.setDirection(dir);
            actionArrow.setLength(Math.min(len * 2 + 1, 4), 0.3, 0.15);
            actionArrow.visible = true;
        }} else if (actionArrow) {{
            actionArrow.visible = false;
        }}
    }} else if (actionArrow) {{
        actionArrow.visible = false;
    }}

    // Lattice point cloud follows agent
    if (latticePoints && msg.agent.active) {{
        latticePoints.position.set(msg.agent.position[0], msg.agent.position[1] + 0.8, msg.agent.position[2]);
    }}

    // Update sky color based on time of day
    const tod = msg.time_of_day;
    updateSky(tod);

    // Update water animation (throttled to every 3rd update)
    waterFrame = (waterFrame || 0) + 1;
    if (waterFrame % 3 === 0) {{
        const wPos = waterMesh.geometry.attributes.position.array;
        const t = Date.now() * 0.001;
        for (let i = 0; i < wPos.length; i += 3) {{
            wPos[i+1] = Math.sin(wPos[i]*0.1 + t) * 0.15 + Math.sin(wPos[i+2]*0.12 + t*0.8) * 0.1;
        }}
        waterMesh.geometry.attributes.position.needsUpdate = true;
    }}

    // HUD stats
    document.getElementById('stat-tick').textContent = msg.tick;
    document.getElementById('stat-pos').textContent =
        `${{p.position[0].toFixed(1)}}, ${{p.position[1].toFixed(1)}}, ${{p.position[2].toFixed(1)}}`;
    const timeNames = ['Night', 'Dawn', 'Morning', 'Noon', 'Afternoon', 'Dusk', 'Evening', 'Night'];
    document.getElementById('stat-time').textContent = timeNames[Math.floor(tod * 8) % 8];
}}

function updateSky(tod) {{
    // Cycle: 0=midnight, 0.25=dawn, 0.5=noon, 0.75=dusk
    let r, g, b;
    if (tod < 0.25) {{ // night → dawn
        const t = tod / 0.25;
        r = 0.05 + t * 0.3; g = 0.05 + t * 0.2; b = 0.15 + t * 0.3;
    }} else if (tod < 0.5) {{ // dawn → noon
        const t = (tod - 0.25) / 0.25;
        r = 0.35 + t * 0.3; g = 0.25 + t * 0.4; b = 0.45 + t * 0.3;
    }} else if (tod < 0.75) {{ // noon → dusk
        const t = (tod - 0.5) / 0.25;
        r = 0.65 - t * 0.3; g = 0.65 - t * 0.35; b = 0.75 - t * 0.3;
    }} else {{ // dusk → night
        const t = (tod - 0.75) / 0.25;
        r = 0.35 - t * 0.3; g = 0.3 - t * 0.25; b = 0.45 - t * 0.3;
    }}
    scene.background.setRGB(r, g, b);
    scene.fog.color.setRGB(r, g, b);
    skyMesh.material.color.setRGB(r, g, b);

    // Sun position
    const sunAngle = tod * Math.PI * 2 - Math.PI / 2;
    sunLight.position.set(Math.cos(sunAngle) * 80, Math.sin(sunAngle) * 80, 30);
    sunLight.intensity = Math.max(0, Math.sin(sunAngle)) * 1.5;
    ambientLight.intensity = 0.15 + Math.max(0, Math.sin(sunAngle)) * 0.35;
}}

function updateTrainingHUD(msg) {{
    // Update mode UI if mode field is present
    if (msg.mode && msg.mode !== currentMode) {{
        updateModeUI(msg.mode);
    }}

    const statusMap = {{
        Observation: msg.trained ? 'Trained (Observing)' : 'Observing...',
        DAgger: msg.trained ? 'DAgger Active' : 'DAgger (Warming Up)',
        Autonomous: msg.trained ? 'Autonomous' : 'Not Trained'
    }};
    document.getElementById('train-status').textContent = statusMap[currentMode] || (msg.trained ? 'Trained' : 'Observing...');
    document.getElementById('train-samples').textContent = msg.samples;
    document.getElementById('train-nmse').textContent = msg.nmse < 1 ? msg.nmse.toFixed(4) : '\u2014';
    const confPct = Math.round(msg.confidence * 100);
    document.getElementById('train-conf').style.width = confPct + '%';

    // Update intervention count for DAgger mode
    if (msg.interventions !== undefined) {{
        document.getElementById('train-intv-count').textContent = msg.interventions;
    }}

    let backendsHtml = '';
    for (const b of msg.backends) {{
        backendsHtml += `<div>${{b.name}}: ${{b.state_dim}}d, ${{b.ticks}} ticks</div>`;
    }}
    document.getElementById('train-backends').innerHTML = backendsHtml;

    // Phase 4: extended training info
    let extraHtml = '';
    if (msg.online_updates !== undefined) extraHtml += `<div>RLS updates: ${{msg.online_updates}}</div>`;
    if (msg.ewc_tasks !== undefined && msg.ewc_tasks > 0) extraHtml += `<div>EWC tasks: ${{msg.ewc_tasks}}</div>`;
    if (msg.replay_size !== undefined) extraHtml += `<div>Replay: ${{msg.replay_size}}</div>`;
    if (msg.retrain_count !== undefined && msg.retrain_count > 0) extraHtml += `<div>Retrains: ${{msg.retrain_count}}</div>`;
    document.getElementById('train-extra').innerHTML = extraHtml;

    // NMSE history graph
    if (msg.nmse_history && msg.nmse_history.length > 0) {{
        nmseHistory = msg.nmse_history;
        drawNmseGraph();
    }}

    // Event log
    if (msg.events && msg.events.length > 0) {{
        appendEvents(msg.events);
    }}
}}

// ─── Phase 4 Helpers ─────────────────────────────

function drawNmseGraph() {{
    const canvas = document.getElementById('nmse-canvas');
    if (!canvas || nmseHistory.length === 0) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    // Background
    ctx.fillStyle = 'rgba(10, 10, 20, 0.5)';
    ctx.fillRect(0, 0, w, h);

    // Find range
    const maxVal = Math.max(...nmseHistory, 0.01);
    const minVal = Math.min(...nmseHistory, 0);

    // Grid lines
    ctx.strokeStyle = 'rgba(0, 229, 255, 0.1)';
    ctx.lineWidth = 0.5;
    for (let i = 0; i < 4; i++) {{
        const y = (i / 3) * h;
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }}

    // Plot NMSE line
    ctx.strokeStyle = '#00e5ff';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    const n = nmseHistory.length;
    for (let i = 0; i < n; i++) {{
        const x = (i / Math.max(n - 1, 1)) * w;
        const y = h - ((nmseHistory[i] - minVal) / (maxVal - minVal + 1e-8)) * (h - 4) - 2;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }}
    ctx.stroke();

    // Labels
    ctx.fillStyle = '#666';
    ctx.font = '9px monospace';
    ctx.fillText('NMSE: ' + nmseHistory[n - 1].toFixed(4), 4, 10);
    ctx.fillText('n=' + n, w - 30, 10);
}}

function appendEvents(events) {{
    const list = document.getElementById('event-list');
    if (!list) return;
    for (const ev of events) {{
        const div = document.createElement('div');
        div.className = 'ev';
        div.textContent = ev;
        list.appendChild(div);
    }}
    // Keep max 50 entries
    while (list.children.length > 50) list.removeChild(list.firstChild);
    // Auto-scroll
    const container = document.getElementById('hud-events');
    if (container) container.scrollTop = container.scrollHeight;
}}

function updateCortanaHUD(msg) {{
    const el = (id) => document.getElementById(id);
    const e = el('cortana-emotion'); if (e) e.textContent = msg.dominant_emotion;
    const m = el('cortana-mood'); if (m) m.textContent = msg.mood;
    const p = el('cortana-pad');
    if (p) p.textContent = msg.pleasure.toFixed(2) + ' / ' + msg.arousal.toFixed(2) + ' / ' + msg.dominance.toFixed(2);
    const intBar = el('cortana-intensity');
    if (intBar) intBar.style.width = Math.round(msg.emotional_intensity * 100) + '%';

    const emoNames = ['Joy','Sadness','Trust','Disgust','Fear','Anger','Surprise','Anticipation'];
    let plutHtml = '';
    for (let i = 0; i < 8; i++) {{
        if (msg.plutchik[i] > 0.05) {{
            plutHtml += '<span style="color:#ff66aa;">' + emoNames[i] + '</span>: ' + msg.plutchik[i].toFixed(2) + ' ';
        }}
    }}
    const pk = el('cortana-plutchik');
    if (pk) pk.innerHTML = plutHtml || '<span class="dim">quiet</span>';

    const neuroNames = ['DA','NE','ACh','5-HT','OT','End','Cort','GABA'];
    let neuroHtml = '';
    for (let i = 0; i < 8; i++) {{
        neuroHtml += neuroNames[i] + ':' + msg.neuromodulators[i].toFixed(2) + ' ';
    }}
    const n = el('cortana-neuro'); if (n) n.textContent = neuroHtml;

    const mc = el('cortana-mem'); if (mc) mc.textContent = msg.memory_count;
    const cr = el('cortana-creative'); if (cr) cr.textContent = msg.creative_drive.toFixed(2);
    const sb = el('cortana-social'); if (sb) sb.textContent = msg.social_bonding.toFixed(2);
}}

function updateLatticeOverlay(values) {{
    if (!latticePoints || !values) return;
    const geo = latticePoints.geometry;
    const colors = geo.attributes.color.array;
    const count = values.length;
    // Normalize magnitudes
    let maxVal = 0;
    for (let i = 0; i < count; i++) {{
        if (values[i] > maxVal) maxVal = values[i];
    }}
    const inv = maxVal > 0 ? 1.0 / maxVal : 1.0;
    for (let i = 0; i < count; i++) {{
        const t = values[i] * inv;
        colors[i * 3]     = t;             // R: 0→1 (cyan→white)
        colors[i * 3 + 1] = 0.5 + t * 0.5; // G: 0.5→1
        colors[i * 3 + 2] = 1.0;           // B: always 1
    }}
    geo.attributes.color.needsUpdate = true;
}}

// ─── Render + Send Loop ─────────────────────────
let lastFrame = performance.now();
let frameCount = 0;
let lastFpsUpdate = performance.now();
let lastInputSend = 0;
let waterFrame = 0;

function animate() {{
    requestAnimationFrame(animate);

    const now = performance.now();
    frameCount++;
    if (now - lastFpsUpdate > 1000) {{
        document.getElementById('stat-fps').textContent = frameCount;
        frameCount = 0;
        lastFpsUpdate = now;
    }}

    // Send input to server (throttled to 20Hz)
    if (wsReady && ws.readyState === 1 && now - lastInputSend >= 50) {{
        lastInputSend = now;
        ws.send(JSON.stringify({{
            type: 'Input',
            keys: keys,
            mouse_dx: mouseDx,
            mouse_dy: mouseDy
        }}));
        mouseDx = 0;
        mouseDy = 0;
    }}

    renderer.render(scene, camera);
}}

window.addEventListener('resize', () => {{
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}});

connectWS();
animate();
</script>
</body>
</html>"##
    )
}
