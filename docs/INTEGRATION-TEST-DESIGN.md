# Icarus Training Game Integration Test Suite Design

## Executive Summary

This document specifies a comprehensive integration test suite for the Icarus Training Game pipeline, covering end-to-end WebSocket communication, training cycle verification, and multi-client scenarios. The test infrastructure uses tokio-tungstenite for WebSocket client simulation and runs the Axum server programmatically in-process.

## Architecture Overview

### Components Under Test

1. **Game Server** (`icarus-game/src/main.rs`)
   - Axum HTTP + WebSocket server
   - 20Hz physics tick loop
   - Protocol message routing

2. **Training Bridge** (`icarus-game/src/bridge.rs`)
   - EnsembleTrainer integration
   - Training mode switching (Observation/DAgger/Autonomous)
   - Online RLS + EWC continual learning
   - Replay buffer management

3. **Protocol Layer** (`icarus-game/src/protocol.rs`)
   - JSON serialization/deserialization
   - ClientMsg/ServerMsg enums
   - Training mode state machine

4. **World Simulation** (`icarus-engine/src/world.rs`)
   - Physics tick (20Hz)
   - Terrain collision
   - Entity interaction
   - Action encoding/decoding

## Test Infrastructure

### Dependencies

Add to `icarus-game/Cargo.toml`:

```toml
[dev-dependencies]
tokio-tungstenite = "0.26"
http = "1.0"
tower = "0.5"
hyper = "1.0"
tokio-test = "0.4"
```

### Test Module Structure

```
icarus-game/tests/
├── integration_test.rs         # Main test orchestrator
├── common/
│   ├── mod.rs                  # Shared test utilities
│   ├── server.rs               # Server spawn/teardown helpers
│   ├── client.rs               # Mock WebSocket client
│   └── assertions.rs           # Custom assertion macros
```

### Server Spawning Strategy

**In-Process with Dynamic Port Allocation**

Spawn the Axum server in a background tokio task within the test runtime, using port 0 to get a dynamically assigned port from the OS. This avoids port conflicts when running tests in parallel.

```rust
use std::net::{SocketAddr, TcpListener};
use tokio::task::JoinHandle;

pub struct TestServer {
    pub addr: SocketAddr,
    pub handle: JoinHandle<()>,
}

impl TestServer {
    pub async fn spawn() -> anyhow::Result<Self> {
        // Bind to port 0 to get a random available port
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let addr = listener.local_addr()?;
        
        // Convert to tokio listener
        listener.set_nonblocking(true)?;
        let listener = tokio::net::TcpListener::from_std(listener)?;

        // Build the Axum app (same as main.rs but with test seed)
        let seed = 42;
        let world = World::new(seed);
        let bridge_config = BridgeConfig {
            seed,
            warmup_ticks: 10,  // Reduced for tests
            retrain_interval: 20,  // Faster retrains for tests
            ..BridgeConfig::default()
        };
        let bridge = TrainingBridge::new(bridge_config)?;

        let state = Arc::new(Mutex::new(GameState {
            world,
            bridge,
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

        // Spawn server in background
        let handle = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        // Wait for server to be ready
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        Ok(Self { addr, handle })
    }

    pub fn ws_url(&self) -> String {
        format!("ws://{}/ws", self.addr)
    }

    pub fn status_url(&self) -> String {
        format!("http://{}/status", self.addr)
    }
}

impl Drop for TestServer {
    fn drop(&mut self) {
        self.handle.abort();
    }
}
```

### Mock WebSocket Client

```rust
use tokio_tungstenite::{connect_async, tungstenite::Message};
use futures::{SinkExt, StreamExt};
use std::time::Duration;

pub struct MockClient {
    ws_stream: WebSocketStream<MaybeTlsStream<TcpStream>>,
    received: Vec<ServerMsg>,
}

impl MockClient {
    pub async fn connect(url: &str) -> anyhow::Result<Self> {
        let (ws_stream, _) = connect_async(url).await?;
        Ok(Self {
            ws_stream,
            received: Vec::new(),
        })
    }

    pub async fn send(&mut self, msg: ClientMsg) -> anyhow::Result<()> {
        let json = serde_json::to_string(&msg)?;
        self.ws_stream.send(Message::Text(json)).await?;
        Ok(())
    }

    pub async fn recv(&mut self) -> anyhow::Result<ServerMsg> {
        loop {
            match tokio::time::timeout(Duration::from_secs(5), self.ws_stream.next()).await {
                Ok(Some(Ok(Message::Text(text)))) => {
                    let msg: ServerMsg = serde_json::from_str(&text)?;
                    self.received.push(msg.clone());
                    return Ok(msg);
                }
                Ok(Some(Ok(Message::Close(_)))) => {
                    anyhow::bail!("WebSocket closed");
                }
                Ok(Some(Err(e))) => {
                    anyhow::bail!("WebSocket error: {}", e);
                }
                Ok(None) => {
                    anyhow::bail!("WebSocket stream ended");
                }
                Err(_) => {
                    anyhow::bail!("Timeout waiting for message");
                }
                _ => continue,
            }
        }
    }

    pub async fn recv_timeout(&mut self, timeout: Duration) -> anyhow::Result<Option<ServerMsg>> {
        match tokio::time::timeout(timeout, self.recv()).await {
            Ok(Ok(msg)) => Ok(Some(msg)),
            Ok(Err(e)) => Err(e),
            Err(_) => Ok(None),
        }
    }

    pub async fn wait_for<F>(&mut self, predicate: F, timeout: Duration) -> anyhow::Result<ServerMsg>
    where
        F: Fn(&ServerMsg) -> bool,
    {
        let start = tokio::time::Instant::now();
        loop {
            if start.elapsed() > timeout {
                anyhow::bail!("Timeout waiting for message matching predicate");
            }
            let msg = self.recv().await?;
            if predicate(&msg) {
                return Ok(msg);
            }
        }
    }

    pub fn messages(&self) -> &[ServerMsg] {
        &self.received
    }

    pub fn clear_received(&mut self) {
        self.received.clear();
    }
}
```

### Custom Assertion Macros

```rust
#[macro_export]
macro_rules! assert_server_msg {
    ($msg:expr, InitWorld) => {
        match $msg {
            ServerMsg::InitWorld { .. } => {},
            _ => panic!("Expected InitWorld, got {:?}", $msg),
        }
    };
    ($msg:expr, WorldState) => {
        match $msg {
            ServerMsg::WorldState { .. } => {},
            _ => panic!("Expected WorldState, got {:?}", $msg),
        }
    };
    ($msg:expr, TrainingUpdate) => {
        match $msg {
            ServerMsg::TrainingUpdate { .. } => {},
            _ => panic!("Expected TrainingUpdate, got {:?}", $msg),
        }
    };
    ($msg:expr, ModeChanged) => {
        match $msg {
            ServerMsg::ModeChanged { .. } => {},
            _ => panic!("Expected ModeChanged, got {:?}", $msg),
        }
    };
}

#[macro_export]
macro_rules! assert_training_converged {
    ($msg:expr, $min_confidence:expr) => {
        match $msg {
            ServerMsg::TrainingUpdate { confidence, trained, .. } => {
                assert!(trained, "Training not complete");
                assert!(confidence >= $min_confidence, 
                    "Confidence {} below threshold {}", confidence, $min_confidence);
            }
            _ => panic!("Expected TrainingUpdate, got {:?}", $msg),
        }
    };
}
```

## Test Scenarios

### 1. Server Startup and Status Endpoint

**Objective**: Verify server starts, binds to port, and responds to HTTP status queries.

```rust
#[tokio::test]
async fn test_server_startup_and_status() -> anyhow::Result<()> {
    let server = TestServer::spawn().await?;
    
    // Query status endpoint
    let response = reqwest::get(server.status_url()).await?;
    assert_eq!(response.status(), 200);
    
    let status: serde_json::Value = response.json().await?;
    assert_eq!(status["tick"], 0);
    assert!(status["backends"].is_array());
    assert!(!status["trained"].as_bool().unwrap());
    
    Ok(())
}
```

### 2. WebSocket Connection and Initial WorldState

**Objective**: Establish WebSocket connection, send Ready, receive InitWorld.

```rust
#[tokio::test]
async fn test_websocket_connection_and_init() -> anyhow::Result<()> {
    let server = TestServer::spawn().await?;
    let mut client = MockClient::connect(&server.ws_url()).await?;
    
    // Send Ready message
    client.send(ClientMsg::Ready).await?;
    
    // Expect InitWorld
    let msg = client.recv().await?;
    assert_server_msg!(msg, InitWorld);
    
    match msg {
        ServerMsg::InitWorld { terrain_grid, entities, lattice_positions, seed, .. } => {
            assert_eq!(terrain_grid.len(), 128);
            assert_eq!(terrain_grid[0].len(), 128);
            assert_eq!(entities.len(), 91);  // 40 trees + 25 rocks + 8 orbs + 12 crates + 6 lights
            assert_eq!(lattice_positions.len(), 241);  // E8 has 240 roots + origin
            assert_eq!(seed, 42);
        }
        _ => unreachable!(),
    }
    
    Ok(())
}
```

### 3. Player Action → Server Processes → WorldState Update

**Objective**: Send player input, verify physics tick updates world state.

```rust
#[tokio::test]
async fn test_player_action_physics_update() -> anyhow::Result<()> {
    let server = TestServer::spawn().await?;
    let mut client = MockClient::connect(&server.ws_url()).await?;
    
    client.send(ClientMsg::Ready).await?;
    let init = client.recv().await?;
    assert_server_msg!(init, InitWorld);
    
    // Record initial player position
    let initial_pos = loop {
        let msg = client.recv().await?;
        if let ServerMsg::WorldState { player, .. } = msg {
            break player.position;
        }
    };
    
    // Send forward movement for 10 ticks
    for _ in 0..10 {
        client.send(ClientMsg::Input {
            keys: KeyState {
                forward: true,
                backward: false,
                left: false,
                right: false,
                jump: false,
                interact: false,
            },
            mouse_dx: 0.0,
            mouse_dy: 0.0,
        }).await?;
        
        tokio::time::sleep(Duration::from_millis(50)).await;  // 20Hz tick rate
    }
    
    // Verify player moved
    let final_pos = client.wait_for(
        |msg| matches!(msg, ServerMsg::WorldState { .. }),
        Duration::from_secs(2),
    ).await?;
    
    if let ServerMsg::WorldState { player, .. } = final_pos {
        let distance = (
            (player.position[0] - initial_pos[0]).powi(2) +
            (player.position[2] - initial_pos[2]).powi(2)
        ).sqrt();
        
        assert!(distance > 1.0, "Player should have moved at least 1 unit, moved {}", distance);
    }
    
    Ok(())
}
```

### 4. Training Cycle: Warmup → Retrain → Prediction

**Objective**: Verify training pipeline from observation to prediction.

```rust
#[tokio::test]
async fn test_training_cycle_observation_mode() -> anyhow::Result<()> {
    let server = TestServer::spawn().await?;
    let mut client = MockClient::connect(&server.ws_url()).await?;
    
    client.send(ClientMsg::Ready).await?;
    client.recv().await?;  // InitWorld
    
    // Send 100 training samples (enough to trigger multiple retrains)
    for tick in 0..100 {
        client.send(ClientMsg::Input {
            keys: KeyState {
                forward: tick % 3 == 0,
                left: tick % 5 == 0,
                jump: tick % 10 == 0,
                ..Default::default()
            },
            mouse_dx: (tick as f32 * 0.1).sin(),
            mouse_dy: 0.0,
        }).await?;
        
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    
    // Wait for TrainingUpdate with trained=true
    let training_msg = client.wait_for(
        |msg| {
            matches!(msg, ServerMsg::TrainingUpdate { trained: true, .. })
        },
        Duration::from_secs(10),
    ).await?;
    
    if let ServerMsg::TrainingUpdate { nmse, confidence, samples, retrain_count, .. } = training_msg {
        assert!(samples >= 20, "Should have at least 20 samples");
        assert!(retrain_count >= 1, "Should have completed at least 1 retrain");
        assert!(nmse < 1.0, "NMSE should be < 1.0, got {}", nmse);
        assert!(confidence > 0.0, "Confidence should be > 0.0");
    }
    
    Ok(())
}
```

### 5. DAgger Mode: Correction → Retrain → Improved Prediction

**Objective**: Verify DAgger intervention recording and training improvement.

```rust
#[tokio::test]
async fn test_dagger_mode_intervention() -> anyhow::Result<()> {
    let server = TestServer::spawn().await?;
    let mut client = MockClient::connect(&server.ws_url()).await?;
    
    client.send(ClientMsg::Ready).await?;
    client.recv().await?;  // InitWorld
    
    // First train in observation mode
    for _ in 0..50 {
        client.send(ClientMsg::Input {
            keys: KeyState { forward: true, ..Default::default() },
            mouse_dx: 0.0,
            mouse_dy: 0.0,
        }).await?;
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    
    // Wait for initial training
    client.wait_for(
        |msg| matches!(msg, ServerMsg::TrainingUpdate { trained: true, .. }),
        Duration::from_secs(10),
    ).await?;
    
    // Switch to DAgger mode
    client.send(ClientMsg::ModeSwitch {
        mode: TrainingMode::DAgger,
    }).await?;
    
    let mode_change = client.recv().await?;
    assert_server_msg!(mode_change, ModeChanged);
    
    // Record initial intervention count
    let initial_interventions = loop {
        if let ServerMsg::TrainingUpdate { interventions, mode, .. } = client.recv().await? {
            if mode == TrainingMode::DAgger {
                break interventions;
            }
        }
    };
    
    // Send human corrections (different from agent predictions)
    for _ in 0..20 {
        client.send(ClientMsg::Input {
            keys: KeyState { left: true, ..Default::default() },
            mouse_dx: 5.0,  // Strong correction
            mouse_dy: 0.0,
        }).await?;
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    
    // Verify interventions were recorded
    let final_training = client.wait_for(
        |msg| {
            if let ServerMsg::TrainingUpdate { interventions, mode, .. } = msg {
                *mode == TrainingMode::DAgger && *interventions > initial_interventions
            } else {
                false
            }
        },
        Duration::from_secs(5),
    ).await?;
    
    if let ServerMsg::TrainingUpdate { interventions, retrain_count, .. } = final_training {
        assert!(interventions > initial_interventions, 
            "Interventions should have increased from {}", initial_interventions);
        assert!(retrain_count >= 2, "Should have multiple retrains");
    }
    
    Ok(())
}
```

### 6. Autonomous Mode: Agent Controls Physics

**Objective**: Verify agent can control player when trained.

```rust
#[tokio::test]
async fn test_autonomous_mode_agent_control() -> anyhow::Result<()> {
    let server = TestServer::spawn().await?;
    let mut client = MockClient::connect(&server.ws_url()).await?;
    
    client.send(ClientMsg::Ready).await?;
    client.recv().await?;
    
    // Train first
    for _ in 0..60 {
        client.send(ClientMsg::Input {
            keys: KeyState { forward: true, ..Default::default() },
            mouse_dx: 0.0,
            mouse_dy: 0.0,
        }).await?;
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    
    client.wait_for(
        |msg| matches!(msg, ServerMsg::TrainingUpdate { trained: true, .. }),
        Duration::from_secs(10),
    ).await?;
    
    // Switch to Autonomous
    client.send(ClientMsg::ModeSwitch {
        mode: TrainingMode::Autonomous,
    }).await?;
    
    client.recv().await?;  // ModeChanged
    
    // Record agent position
    let initial_pos = loop {
        if let ServerMsg::WorldState { agent, .. } = client.recv().await? {
            if agent.active {
                break agent.position;
            }
        }
    };
    
    // Wait and verify agent moved autonomously (no input sent)
    tokio::time::sleep(Duration::from_secs(2)).await;
    
    let final_state = client.wait_for(
        |msg| matches!(msg, ServerMsg::WorldState { .. }),
        Duration::from_secs(1),
    ).await?;
    
    if let ServerMsg::WorldState { agent, .. } = final_state {
        assert!(agent.active, "Agent should be active");
        assert!(agent.confidence > 0.0, "Agent should have confidence");
        
        let distance = (
            (agent.position[0] - initial_pos[0]).powi(2) +
            (agent.position[2] - initial_pos[2]).powi(2)
        ).sqrt();
        
        assert!(distance > 0.5, "Agent should have moved autonomously");
    }
    
    Ok(())
}
```

### 7. Multiple Concurrent Connections

**Objective**: Verify server handles multiple WebSocket clients simultaneously.

```rust
#[tokio::test]
async fn test_multiple_concurrent_clients() -> anyhow::Result<()> {
    let server = TestServer::spawn().await?;
    
    // Spawn 3 clients concurrently
    let mut clients = Vec::new();
    for _ in 0..3 {
        let mut client = MockClient::connect(&server.ws_url()).await?;
        client.send(ClientMsg::Ready).await?;
        clients.push(client);
    }
    
    // All clients should receive InitWorld
    for client in &mut clients {
        let msg = client.recv().await?;
        assert_server_msg!(msg, InitWorld);
    }
    
    // Send input from all clients simultaneously
    let mut handles = Vec::new();
    for client in clients {
        let handle = tokio::spawn(async move {
            let mut c = client;
            for _ in 0..20 {
                c.send(ClientMsg::Input {
                    keys: KeyState { forward: true, ..Default::default() },
                    mouse_dx: 0.0,
                    mouse_dy: 0.0,
                }).await.unwrap();
                
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
            
            // Verify receiving WorldState
            let msg = c.recv().await.unwrap();
            assert_server_msg!(msg, WorldState);
        });
        handles.push(handle);
    }
    
    // Wait for all clients to complete
    for handle in handles {
        handle.await?;
    }
    
    Ok(())
}
```

### 8. Continual Learning: EWC Task Registration

**Objective**: Verify EWC preserves prior knowledge across retrains.

```rust
#[tokio::test]
async fn test_ewc_continual_learning() -> anyhow::Result<()> {
    let server = TestServer::spawn().await?;
    let mut client = MockClient::connect(&server.ws_url()).await?;
    
    client.send(ClientMsg::Ready).await?;
    client.recv().await?;
    
    // Phase 1: Train on forward movement
    for _ in 0..60 {
        client.send(ClientMsg::Input {
            keys: KeyState { forward: true, ..Default::default() },
            mouse_dx: 0.0,
            mouse_dy: 0.0,
        }).await?;
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    
    let phase1 = client.wait_for(
        |msg| matches!(msg, ServerMsg::TrainingUpdate { trained: true, ewc_tasks: 1, .. }),
        Duration::from_secs(10),
    ).await?;
    
    if let ServerMsg::TrainingUpdate { ewc_tasks, nmse: nmse1, .. } = phase1 {
        assert_eq!(ewc_tasks, 1, "Should have 1 EWC task after first retrain");
    }
    
    // Phase 2: Train on left movement (different behavior)
    for _ in 0..60 {
        client.send(ClientMsg::Input {
            keys: KeyState { left: true, ..Default::default() },
            mouse_dx: 0.0,
            mouse_dy: 0.0,
        }).await?;
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    
    let phase2 = client.wait_for(
        |msg| {
            if let ServerMsg::TrainingUpdate { ewc_tasks, retrain_count, .. } = msg {
                *ewc_tasks >= 2 && *retrain_count >= 2
            } else {
                false
            }
        },
        Duration::from_secs(10),
    ).await?;
    
    if let ServerMsg::TrainingUpdate { ewc_tasks, nmse, retrain_count, .. } = phase2 {
        assert!(ewc_tasks >= 2, "Should have multiple EWC tasks");
        assert!(retrain_count >= 2, "Should have multiple retrains");
        assert!(nmse < 1.0, "NMSE should remain reasonable despite task shift");
    }
    
    Ok(())
}
```

### 9. Online RLS Updates Between Retrains

**Objective**: Verify online RLS adapts weights incrementally.

```rust
#[tokio::test]
async fn test_online_rls_updates() -> anyhow::Result<()> {
    let server = TestServer::spawn().await?;
    let mut client = MockClient::connect(&server.ws_url()).await?;
    
    client.send(ClientMsg::Ready).await?;
    client.recv().await?;
    
    // Initial training
    for _ in 0..60 {
        client.send(ClientMsg::Input {
            keys: KeyState { forward: true, ..Default::default() },
            mouse_dx: 0.0,
            mouse_dy: 0.0,
        }).await?;
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    
    client.wait_for(
        |msg| matches!(msg, ServerMsg::TrainingUpdate { trained: true, .. }),
        Duration::from_secs(10),
    ).await?;
    
    // Clear received buffer
    client.clear_received();
    
    // Send 10 more samples (not enough to trigger retrain)
    for _ in 0..10 {
        client.send(ClientMsg::Input {
            keys: KeyState { right: true, ..Default::default() },
            mouse_dx: 0.0,
            mouse_dy: 0.0,
        }).await?;
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    
    // Check that online_updates increased without retrain_count increasing
    let update_msg = client.wait_for(
        |msg| {
            if let ServerMsg::TrainingUpdate { online_updates, .. } = msg {
                *online_updates > 0
            } else {
                false
            }
        },
        Duration::from_secs(5),
    ).await?;
    
    if let ServerMsg::TrainingUpdate { online_updates, retrain_count, .. } = update_msg {
        assert!(online_updates > 0, "Online RLS should have updated");
        assert_eq!(retrain_count, 1, "Retrain count should still be 1");
    }
    
    Ok(())
}
```

### 10. Lattice Overlay Data

**Objective**: Verify E8 lattice field magnitudes are computed and sent.

```rust
#[tokio::test]
async fn test_lattice_overlay_data() -> anyhow::Result<()> {
    let server = TestServer::spawn().await?;
    let mut client = MockClient::connect(&server.ws_url()).await?;
    
    client.send(ClientMsg::Ready).await?;
    client.recv().await?;
    
    // Train first
    for _ in 0..60 {
        client.send(ClientMsg::Input {
            keys: KeyState { forward: true, ..Default::default() },
            mouse_dx: 0.0,
            mouse_dy: 0.0,
        }).await?;
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    
    client.wait_for(
        |msg| matches!(msg, ServerMsg::TrainingUpdate { trained: true, .. }),
        Duration::from_secs(10),
    ).await?;
    
    // Wait for LatticeOverlay message
    let lattice_msg = client.wait_for(
        |msg| matches!(msg, ServerMsg::LatticeOverlay { .. }),
        Duration::from_secs(10),
    ).await?;
    
    if let ServerMsg::LatticeOverlay { values } = lattice_msg {
        assert_eq!(values.len(), 241, "Should have 241 E8 lattice sites");
        
        // All values should be finite and non-negative (magnitudes)
        for (i, &v) in values.iter().enumerate() {
            assert!(v.is_finite(), "Value at index {} is not finite", i);
            assert!(v >= 0.0, "Magnitude at index {} is negative", i);
        }
    }
    
    Ok(())
}
```

## Test Helper Functions

### HTTP Status Query Helper

```rust
pub async fn query_status(server: &TestServer) -> anyhow::Result<serde_json::Value> {
    let response = reqwest::get(server.status_url()).await?;
    let status = response.json().await?;
    Ok(status)
}

pub async fn wait_for_trained(server: &TestServer, timeout: Duration) -> anyhow::Result<bool> {
    let start = tokio::time::Instant::now();
    loop {
        if start.elapsed() > timeout {
            return Ok(false);
        }
        
        let status = query_status(server).await?;
        if status["trained"].as_bool().unwrap_or(false) {
            return Ok(true);
        }
        
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
}
```

### Message Collection Helper

```rust
pub async fn collect_messages<F>(
    client: &mut MockClient,
    predicate: F,
    count: usize,
    timeout: Duration,
) -> anyhow::Result<Vec<ServerMsg>>
where
    F: Fn(&ServerMsg) -> bool,
{
    let mut collected = Vec::new();
    let start = tokio::time::Instant::now();
    
    while collected.len() < count {
        if start.elapsed() > timeout {
            anyhow::bail!(
                "Timeout: collected only {}/{} messages",
                collected.len(),
                count
            );
        }
        
        let msg = client.recv().await?;
        if predicate(&msg) {
            collected.push(msg);
        }
    }
    
    Ok(collected)
}
```

### Training Data Generator

```rust
pub async fn generate_training_samples(
    client: &mut MockClient,
    pattern: TrainingPattern,
    count: usize,
) -> anyhow::Result<()> {
    for i in 0..count {
        let keys = match pattern {
            TrainingPattern::Forward => KeyState { forward: true, ..Default::default() },
            TrainingPattern::Circle => KeyState {
                forward: true,
                left: i % 4 == 0,
                right: i % 4 == 2,
                ..Default::default()
            },
            TrainingPattern::Jump => KeyState {
                forward: true,
                jump: i % 10 == 0,
                ..Default::default()
            },
            TrainingPattern::Random => KeyState {
                forward: (i * 3) % 7 == 0,
                left: (i * 5) % 11 == 0,
                right: (i * 7) % 13 == 0,
                jump: (i * 11) % 17 == 0,
                ..Default::default()
            },
        };
        
        client.send(ClientMsg::Input {
            keys,
            mouse_dx: (i as f32 * 0.1).sin(),
            mouse_dy: 0.0,
        }).await?;
        
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    
    Ok(())
}

pub enum TrainingPattern {
    Forward,
    Circle,
    Jump,
    Random,
}
```

## Performance and Load Tests

### High-Frequency Input Stress Test

```rust
#[tokio::test]
async fn test_high_frequency_input() -> anyhow::Result<()> {
    let server = TestServer::spawn().await?;
    let mut client = MockClient::connect(&server.ws_url()).await?;
    
    client.send(ClientMsg::Ready).await?;
    client.recv().await?;
    
    // Send 1000 inputs as fast as possible
    let start = tokio::time::Instant::now();
    for _ in 0..1000 {
        client.send(ClientMsg::Input {
            keys: KeyState { forward: true, ..Default::default() },
            mouse_dx: 0.0,
            mouse_dy: 0.0,
        }).await?;
    }
    let elapsed = start.elapsed();
    
    println!("Sent 1000 inputs in {:?}", elapsed);
    
    // Server should still be responsive
    let msg = client.recv().await?;
    assert_server_msg!(msg, WorldState);
    
    Ok(())
}
```

### Memory Leak Detection

```rust
#[tokio::test]
async fn test_memory_stability_long_session() -> anyhow::Result<()> {
    let server = TestServer::spawn().await?;
    let mut client = MockClient::connect(&server.ws_url()).await?;
    
    client.send(ClientMsg::Ready).await?;
    client.recv().await?;
    
    // Run for 5000 ticks (~4 minutes at 20Hz)
    for _ in 0..5000 {
        client.send(ClientMsg::Input {
            keys: KeyState { forward: true, ..Default::default() },
            mouse_dx: 0.0,
            mouse_dy: 0.0,
        }).await?;
        
        // Poll less frequently to avoid flooding
        if tokio::time::timeout(Duration::from_millis(10), client.recv()).await.is_ok() {
            // Message received, continue
        }
    }
    
    // Server should still be responsive
    let status = query_status(&server).await?;
    assert!(status["tick"].as_u64().unwrap() > 4000);
    
    Ok(())
}
```

## CI/CD Integration

### GitHub Actions Workflow

```yaml
name: Integration Tests

on: [push, pull_request]

jobs:
  integration-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Install Rust
        uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
          override: true
      
      - name: Cache cargo registry
        uses: actions/cache@v3
        with:
          path: ~/.cargo/registry
          key: ${{ runner.os }}-cargo-registry-${{ hashFiles('**/Cargo.lock') }}
      
      - name: Cache cargo build
        uses: actions/cache@v3
        with:
          path: target
          key: ${{ runner.os }}-cargo-build-${{ hashFiles('**/Cargo.lock') }}
      
      - name: Run integration tests
        run: |
          cd Icarus/icarus-game
          cargo test --test integration_test -- --test-threads=1 --nocapture
        env:
          RUST_BACKTRACE: 1
```

### Local Test Runner Script

```bash
#!/bin/bash
# run-integration-tests.sh

set -e

cd "$(dirname "$0")"

echo "Building icarus-game..."
cargo build --release -p icarus-game

echo "Running integration tests..."
cargo test --test integration_test -- --test-threads=1 --nocapture

echo "All integration tests passed!"
```

## Known Limitations and Future Work

### Current Limitations

1. **No GPU Testing**: Integration tests run CPU-only EMC instances. GPU backend testing requires CUDA-enabled CI runners.

2. **Determinism**: Physics simulation with floating-point arithmetic may have minor non-determinism across platforms. Use approximate assertions (`assert!((a - b).abs() < epsilon)`) rather than exact equality.

3. **Timing Sensitivity**: Tests assume 20Hz tick rate. If system is under heavy load, timing-based assertions may fail. Use generous timeouts.

4. **Replay Buffer Sampling**: Replay buffer uses random sampling, so exact NMSE values may vary between runs.

### Future Enhancements

1. **Property-Based Testing**: Use `proptest` or `quickcheck` to generate random input sequences and verify invariants (e.g., "player should never fall through terrain").

2. **Snapshot Testing**: Record full game sessions and replay them deterministically to catch regressions.

3. **Performance Benchmarking**: Integrate with `criterion` to track latency metrics (WebSocket RTT, physics tick duration, training throughput).

4. **Visual Regression Testing**: Render frames to headless browser and compare screenshots (requires Selenium/Playwright integration).

5. **Chaos Engineering**: Inject random disconnections, packet loss, and message corruption to test error handling.

## References

### Documentation

- [tokio-tungstenite GitHub](https://github.com/snapview/tokio-tungstenite) - WebSocket library
- [axum testing-websockets example](https://github.com/tokio-rs/axum/blob/main/examples/testing-websockets/src/main.rs) - Official Axum WebSocket testing pattern
- [tokio-tungstenite docs.rs](https://docs.rs/tokio-tungstenite) - API reference

### Related Files

- `/root/workspace-v2/Icarus/icarus-game/src/main.rs` - Server implementation
- `/root/workspace-v2/Icarus/icarus-game/src/bridge.rs` - Training bridge
- `/root/workspace-v2/Icarus/icarus-game/src/protocol.rs` - Protocol definitions
- `/root/workspace-v2/Icarus/icarus-engine/src/world.rs` - World simulation

---

**Document Version**: 1.0  
**Last Updated**: 2026-02-12  
**Author**: Claude (Sonnet 4.5)  
**Status**: Ready for Implementation
