//! Icarus Fitness Visualizer — 3D E8 Lattice + Training Progress
//!
//! Generates a self-contained HTML file with Three.js WebGL rendering:
//! - Rotating E8 root system (240 vectors projected to 3D)
//! - Fully-meshed muscular "Icarus" figure — Greek god physique
//! - Individual muscle groups grow as NMSE drops (swole factor)
//! - Three exercises: pushups, deadlifts, bench press
//! - Live training curves + auto-refresh

use std::fs;
use std::io::{BufRead, BufReader};

/// E8 root system: 240 vectors in 8D
fn e8_roots() -> Vec<[f64; 8]> {
    let mut roots = Vec::with_capacity(240);

    for i in 0..8 {
        for j in (i + 1)..8 {
            for si in [-1.0f64, 1.0] {
                for sj in [-1.0, 1.0] {
                    let mut v = [0.0; 8];
                    v[i] = si;
                    v[j] = sj;
                    roots.push(v);
                }
            }
        }
    }

    for bits in 0u32..256 {
        let neg_count = bits.count_ones();
        if neg_count % 2 != 0 {
            continue;
        }
        let mut v = [0.5f64; 8];
        for k in 0..8 {
            if bits & (1 << k) != 0 {
                v[k] = -0.5;
            }
        }
        roots.push(v);
    }

    roots
}

/// Project 8D -> 3D using a nice projection matrix
fn project_to_3d(v: &[f64; 8]) -> [f64; 3] {
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let inv_phi = 1.0 / phi;

    let x = v[0] + phi * v[1] + inv_phi * v[2] + 0.3 * v[3]
        + 0.1 * v[4] - 0.2 * v[5] + 0.4 * v[6] - 0.15 * v[7];
    let y = -inv_phi * v[0] + 0.2 * v[1] + phi * v[2] + 0.5 * v[3]
        - 0.3 * v[4] + 0.1 * v[5] + 0.15 * v[6] + 0.4 * v[7];
    let z = 0.3 * v[0] - 0.1 * v[1] + 0.4 * v[2] - inv_phi * v[3]
        + phi * v[4] + 0.2 * v[5] - 0.5 * v[6] + 0.3 * v[7];

    let scale = 1.5;
    [x * scale, y * scale, z * scale]
}

/// Parse the disco results file for visualization data
fn parse_results(path: &str) -> (Vec<(u64, f64)>, f64, u64, u64) {
    let mut best_over_time: Vec<(u64, f64)> = Vec::new();
    let mut current_best = f64::MAX;
    let mut total_trials: u64 = 0;
    let mut max_trials: u64 = 172800;

    if let Ok(file) = fs::File::open(path) {
        let reader = BufReader::new(file);
        for line in reader.lines().flatten() {
            if let Some(bracket_end) = line.find(']') {
                let bracket_content = &line[1..bracket_end];
                if let Some(slash) = bracket_content.find('/') {
                    if let Ok(trial) = bracket_content[..slash].trim().parse::<u64>() {
                        total_trials = total_trials.max(trial);
                        let after_slash = &bracket_content[slash + 1..];
                        if let Some(space) = after_slash.find(' ') {
                            if let Ok(mt) = after_slash[..space].parse::<u64>() {
                                max_trials = mt;
                            }
                        }
                    }
                }
            }

            if line.contains("NEW BEST") || line.contains("best so far") {
                if let Some(nmse_pos) = line.find("NMSE=") {
                    let after = &line[nmse_pos + 5..];
                    let end = after
                        .find(|c: char| !c.is_ascii_digit() && c != '.' && c != '-')
                        .unwrap_or(after.len());
                    if let Ok(nmse) = after[..end].parse::<f64>() {
                        if nmse < current_best {
                            current_best = nmse;
                            best_over_time.push((total_trials, nmse));
                        }
                    }
                }
            }
        }
    }

    (best_over_time, current_best, total_trials, max_trials)
}

fn generate_html(roots_3d: &[[f64; 3]], results_path: &str) -> String {
    let (best_over_time, current_best, total_trials, max_trials) = parse_results(results_path);

    let points_json: Vec<String> = roots_3d
        .iter()
        .map(|p| format!("[{:.4},{:.4},{:.4}]", p[0], p[1], p[2]))
        .collect();

    let mut edges: Vec<[usize; 2]> = Vec::new();
    let threshold = 2.5;
    for i in 0..roots_3d.len() {
        for j in (i + 1)..roots_3d.len() {
            let dx = roots_3d[i][0] - roots_3d[j][0];
            let dy = roots_3d[i][1] - roots_3d[j][1];
            let dz = roots_3d[i][2] - roots_3d[j][2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            if dist < threshold {
                edges.push([i, j]);
            }
        }
    }

    let edges_json: Vec<String> = edges.iter().map(|e| format!("[{},{}]", e[0], e[1])).collect();

    let progress_json: Vec<String> = best_over_time
        .iter()
        .map(|(t, n)| format!("[{},{}]", t, n))
        .collect();

    let progress_pct = if max_trials > 0 {
        (total_trials as f64 / max_trials as f64 * 100.0).min(100.0)
    } else {
        0.0
    };

    let swole_factor = ((1.5 - current_best.min(1.5)) / 1.5 * 100.0).max(5.0);

    format!(
        r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ICARUS FITNESS - E8 Lattice Training Visualizer</title>
<meta http-equiv="refresh" content="30">
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    background: #000;
    color: #0ff;
    font-family: 'Courier New', monospace;
    overflow: hidden;
    height: 100vh;
  }}
  #canvas-container {{
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 100%;
    z-index: 1;
  }}
  #hud {{
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 100%;
    z-index: 10;
    pointer-events: none;
  }}
  .panel {{
    background: rgba(0, 20, 40, 0.85);
    border: 1px solid rgba(0, 255, 255, 0.3);
    border-radius: 8px;
    padding: 15px;
    backdrop-filter: blur(10px);
  }}
  #title-panel {{
    position: absolute;
    top: 15px; left: 50%;
    transform: translateX(-50%);
    text-align: center;
    font-size: 14px;
  }}
  #title-panel h1 {{
    font-size: 28px;
    background: linear-gradient(90deg, #f0f, #0ff, #ff0, #f0f);
    background-size: 400% 100%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: disco-text 3s linear infinite;
  }}
  @keyframes disco-text {{
    0% {{ background-position: 0% 50%; }}
    100% {{ background-position: 400% 50%; }}
  }}
  #stats-panel {{
    position: absolute;
    top: 15px; right: 15px;
    width: 280px;
    font-size: 13px;
    line-height: 1.8;
  }}
  #progress-panel {{
    position: absolute;
    bottom: 15px; left: 15px;
    right: 15px;
    font-size: 12px;
  }}
  .progress-bar {{
    width: 100%;
    height: 20px;
    background: rgba(0, 255, 255, 0.1);
    border: 1px solid rgba(0, 255, 255, 0.3);
    border-radius: 10px;
    overflow: hidden;
    margin: 8px 0;
  }}
  .progress-fill {{
    height: 100%;
    background: linear-gradient(90deg, #f0f, #0ff, #ff0);
    border-radius: 10px;
    transition: width 1s ease;
    animation: pulse-bar 2s ease-in-out infinite;
  }}
  @keyframes pulse-bar {{
    0%, 100% {{ opacity: 0.8; }}
    50% {{ opacity: 1; }}
  }}
  #swole-panel {{
    position: absolute;
    top: 15px; left: 15px;
    width: 200px;
    text-align: center;
  }}
  .swole-meter {{
    width: 100%;
    height: 16px;
    background: rgba(255, 0, 255, 0.1);
    border: 1px solid rgba(255, 0, 255, 0.4);
    border-radius: 8px;
    overflow: hidden;
    margin: 6px 0;
  }}
  .swole-fill {{
    height: 100%;
    background: linear-gradient(90deg, #f00, #ff0, #0f0);
    border-radius: 8px;
    animation: swole-pulse 1.5s ease-in-out infinite;
  }}
  @keyframes swole-pulse {{
    0%, 100% {{ transform: scaleY(1); }}
    50% {{ transform: scaleY(1.3); }}
  }}
  #exercise-panel {{
    position: absolute;
    top: 180px; left: 15px;
    width: 200px;
    text-align: center;
  }}
  #exercise-name {{
    font-size: 22px;
    color: #ff0;
    text-shadow: 0 0 15px #ff0, 0 0 30px #f80;
    font-weight: bold;
    letter-spacing: 2px;
  }}
  #rep-count {{
    font-size: 13px;
    color: #0ff;
    margin-top: 6px;
  }}
  @keyframes pump {{
    0%, 100% {{ text-shadow: 0 0 15px #ff0, 0 0 30px #f80; }}
    50% {{ text-shadow: 0 0 25px #f00, 0 0 50px #f80, 0 0 80px #ff0; }}
  }}
  #chart-panel {{
    position: absolute;
    bottom: 70px; right: 15px;
    width: 350px;
    height: 180px;
  }}
  canvas#chart {{
    width: 100%;
    height: 140px;
  }}
  .label {{ color: #888; }}
  .value {{ color: #0ff; font-weight: bold; }}
  .highlight {{ color: #ff0; }}
  .good {{ color: #0f0; }}
  .bad {{ color: #f44; }}
</style>
</head>
<body>
<div id="canvas-container"></div>
<div id="hud">
  <div id="title-panel" class="panel">
    <h1>ICARUS FITNESS</h1>
    <div>E8 Lattice Emergent Manifold Computer &mdash; Getting RIPPED</div>
  </div>

  <div id="swole-panel" class="panel">
    <div style="font-size:16px; color:#f0f;">SWOLE METER</div>
    <div class="swole-meter">
      <div class="swole-fill" style="width:{swole_pct:.1}%"></div>
    </div>
    <div style="font-size:22px; color:#ff0;">{swole_pct:.1}%</div>
    <div style="font-size:10px; color:#888; margin-top:4px;">
      {swole_label}
    </div>
  </div>

  <div id="exercise-panel" class="panel">
    <div id="exercise-name" style="animation: pump 1s ease-in-out infinite;">PUSHUPS</div>
    <div id="rep-count">Rep 0</div>
  </div>

  <div id="stats-panel" class="panel">
    <div><span class="label">Best NMSE:</span> <span class="{best_class}">{best_nmse}</span></div>
    <div><span class="label">Trials:</span> <span class="value">{total_trials} / {max_trials}</span></div>
    <div><span class="label">Progress:</span> <span class="value">{progress_pct:.1}%</span></div>
    <div><span class="label">E8 Root System:</span> <span class="value">240 vectors in 8D</span></div>
    <div><span class="label">Lattice Sites:</span> <span class="value">241 per layer</span></div>
    <div><span class="label">Hardware:</span> <span class="value">8 CPU + RTX 5070 Ti</span></div>
    <div style="margin-top:8px; font-size:11px; color:#888;">Auto-refresh: 30s</div>
  </div>

  <div id="chart-panel" class="panel">
    <div style="margin-bottom:5px;">NMSE Convergence</div>
    <canvas id="chart" width="320" height="140"></canvas>
  </div>

  <div id="progress-panel" class="panel">
    <div style="display:flex; justify-content:space-between;">
      <span>Phase 2: Coarse Grid Sweep</span>
      <span class="value">{progress_pct:.1}%</span>
    </div>
    <div class="progress-bar">
      <div class="progress-fill" style="width:{progress_pct:.1}%"></div>
    </div>
    <div style="font-size:11px; color:#888;">
      172,800 parameter combinations &times; 3 encoders &times; 8 threads &mdash; All cores blazing
    </div>
  </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
// === THREE.JS SCENE ===
const container = document.getElementById('canvas-container');
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(0, 0, 12);

const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
container.appendChild(renderer.domElement);

scene.background = new THREE.Color(0x000510);
scene.fog = new THREE.FogExp2(0x000510, 0.02);

// E8 lattice points
const points = [{points_str}];
const edges = [{edges_str}];

const pointsGeometry = new THREE.BufferGeometry();
const positions = new Float32Array(points.length * 3);
const colors = new Float32Array(points.length * 3);

for (let i = 0; i < points.length; i++) {{
    positions[i * 3] = points[i][0];
    positions[i * 3 + 1] = points[i][1];
    positions[i * 3 + 2] = points[i][2];
    const hue = (Math.atan2(points[i][1], points[i][0]) / Math.PI + 1) * 0.5;
    const color = new THREE.Color().setHSL(hue, 1, 0.6);
    colors[i * 3] = color.r;
    colors[i * 3 + 1] = color.g;
    colors[i * 3 + 2] = color.b;
}}

pointsGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
pointsGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

const pointsMaterial = new THREE.PointsMaterial({{
    size: 0.15,
    vertexColors: true,
    transparent: true,
    opacity: 0.9,
    blending: THREE.AdditiveBlending,
}});
const pointCloud = new THREE.Points(pointsGeometry, pointsMaterial);
scene.add(pointCloud);

const edgeGeometry = new THREE.BufferGeometry();
const edgePositions = [];
for (const [i, j] of edges) {{
    edgePositions.push(points[i][0], points[i][1], points[i][2]);
    edgePositions.push(points[j][0], points[j][1], points[j][2]);
}}
edgeGeometry.setAttribute('position', new THREE.Float32BufferAttribute(edgePositions, 3));
const edgeMaterial = new THREE.LineBasicMaterial({{
    color: 0x0088ff,
    transparent: true,
    opacity: 0.15,
    blending: THREE.AdditiveBlending,
}});
const edgeLines = new THREE.LineSegments(edgeGeometry, edgeMaterial);
scene.add(edgeLines);

// ============================================================
// === ICARUS — GREEK GOD PHYSIQUE WITH VISIBLE MUSCLES ===
// ============================================================
const swoleFactor = {swole_raw};
const sw = swoleFactor / 100.0; // 0.0 to 1.0 normalized

// Muscle growth multiplier: starts lean, ends massive
const mg = 1.0 + sw * 2.0; // 1.0 at start, 3.0 at max swole
const S = 0.5; // base body scale

// --- MATERIALS: Golden bronze Greek god ---
const skinMat = new THREE.MeshPhongMaterial({{
    color: new THREE.Color().setHSL(0.08, 0.55 + sw * 0.2, 0.55 + sw * 0.1),
    emissive: new THREE.Color().setHSL(0.06, 0.6, 0.05 + sw * 0.08),
    shininess: 30 + sw * 90,
    transparent: true,
    opacity: 0.95,
}});
const muscleMat = new THREE.MeshPhongMaterial({{
    color: new THREE.Color().setHSL(0.07, 0.6 + sw * 0.2, 0.5 + sw * 0.12),
    emissive: new THREE.Color().setHSL(0.05, 0.7, 0.06 + sw * 0.1),
    shininess: 50 + sw * 100,
    transparent: true,
    opacity: 0.95,
}});
const veinMat = new THREE.MeshPhongMaterial({{
    color: 0x884422,
    emissive: 0x331100,
    shininess: 20,
    transparent: true,
    opacity: sw * 0.6,
}});

const icarus = new THREE.Group();

// Helper: create a muscle bulge (scaled sphere)
function muscleBulge(sx, sy, sz, mat) {{
    const geo = new THREE.SphereGeometry(1, 12, 10);
    const mesh = new THREE.Mesh(geo, mat || muscleMat);
    mesh.scale.set(sx, sy, sz);
    return mesh;
}}

// =====================
// TORSO — V-taper
// =====================
// Main torso cylinder (tapers from wide chest to narrow waist)
const torsoW = 0.32 * S * mg;
const waistW = 0.22 * S * (0.8 + sw * 0.3);
const torsoH = 1.3 * S;
const torsoGeo = new THREE.CylinderGeometry(waistW, torsoW, torsoH, 12);
const torso = new THREE.Mesh(torsoGeo, skinMat);
icarus.add(torso);

// --- PECS (two large hemispheres on front of chest) ---
const pecSize = 0.16 * S * mg;
const lPec = muscleBulge(pecSize, pecSize * 0.7, pecSize * 0.9);
lPec.position.set(-0.1 * S * mg, 0.2 * S, 0.2 * S * mg);
icarus.add(lPec);
const rPec = muscleBulge(pecSize, pecSize * 0.7, pecSize * 0.9);
rPec.position.set(0.1 * S * mg, 0.2 * S, 0.2 * S * mg);
icarus.add(rPec);

// --- ABS (6-pack: 3 rows x 2 columns) ---
const absSize = 0.05 * S * mg;
for (let row = 0; row < 3; row++) {{
    for (let col = -1; col <= 1; col += 2) {{
        const ab = muscleBulge(absSize, absSize * 0.8, absSize * 0.7);
        ab.position.set(
            col * 0.055 * S * mg,
            -0.05 * S - row * 0.1 * S,
            0.2 * S * mg
        );
        icarus.add(ab);
    }}
}}

// --- OBLIQUES (side ab muscles) ---
const obliqueSize = 0.08 * S * mg;
for (let side = -1; side <= 1; side += 2) {{
    for (let i = 0; i < 2; i++) {{
        const ob = muscleBulge(obliqueSize * 0.5, obliqueSize * 0.8, obliqueSize * 0.6);
        ob.position.set(
            side * 0.22 * S * mg,
            -0.05 * S - i * 0.12 * S,
            0.1 * S
        );
        icarus.add(ob);
    }}
}}

// --- LATS (wide back muscles — give the V shape) ---
const latSize = 0.18 * S * mg;
const lLat = muscleBulge(latSize, latSize * 1.3, latSize * 0.5);
lLat.position.set(-0.25 * S * mg, 0.05 * S, -0.08 * S);
icarus.add(lLat);
const rLat = muscleBulge(latSize, latSize * 1.3, latSize * 0.5);
rLat.position.set(0.25 * S * mg, 0.05 * S, -0.08 * S);
icarus.add(rLat);

// --- TRAPS (neck to shoulder connection) ---
const trapSize = 0.1 * S * mg;
const lTrap = muscleBulge(trapSize * 1.2, trapSize * 0.8, trapSize * 0.7);
lTrap.position.set(-0.14 * S * mg, 0.5 * S, 0);
icarus.add(lTrap);
const rTrap = muscleBulge(trapSize * 1.2, trapSize * 0.8, trapSize * 0.7);
rTrap.position.set(0.14 * S * mg, 0.5 * S, 0);
icarus.add(rTrap);

// --- LOWER BACK / ERECTORS ---
const erectorSize = 0.06 * S * mg;
for (let side = -1; side <= 1; side += 2) {{
    const er = muscleBulge(erectorSize * 0.6, erectorSize * 1.5, erectorSize * 0.7);
    er.position.set(side * 0.08 * S, -0.15 * S, -0.18 * S * mg);
    icarus.add(er);
}}

// =====================
// HEAD
// =====================
const headGeo = new THREE.SphereGeometry(0.17 * S, 16, 16);
const head = new THREE.Mesh(headGeo, skinMat);
head.position.y = 0.82 * S;
icarus.add(head);

// Neck (thickens with traps)
const neckGeo = new THREE.CylinderGeometry(0.07 * S * mg, 0.08 * S, 0.12 * S, 8);
const neck = new THREE.Mesh(neckGeo, skinMat);
neck.position.y = 0.7 * S;
icarus.add(neck);

// =====================
// LEFT ARM — fully muscled
// =====================
const lShoulderPivot = new THREE.Group();
lShoulderPivot.position.set(-(torsoW + 0.04 * S), 0.38 * S, 0);
icarus.add(lShoulderPivot);

// Deltoid (shoulder cap — 3 heads)
const deltSize = 0.1 * S * mg;
const lDeltF = muscleBulge(deltSize * 0.8, deltSize, deltSize * 0.7);
lDeltF.position.set(0, 0.02 * S, 0.04 * S);
lShoulderPivot.add(lDeltF);
const lDeltS = muscleBulge(deltSize, deltSize * 0.9, deltSize * 0.6);
lDeltS.position.set(-0.04 * S, 0.02 * S, 0);
lShoulderPivot.add(lDeltS);
const lDeltR = muscleBulge(deltSize * 0.7, deltSize * 0.9, deltSize * 0.7);
lDeltR.position.set(0, 0.02 * S, -0.04 * S);
lShoulderPivot.add(lDeltR);

// Upper arm bone
const bicepR = 0.07 * S * mg;
const lUpperGeo = new THREE.CylinderGeometry(bicepR * 0.8, bicepR * 0.7, 0.5 * S, 8);
const lUpperArm = new THREE.Mesh(lUpperGeo, skinMat);
lUpperArm.position.y = -0.25 * S;
lShoulderPivot.add(lUpperArm);

// Bicep bulge (front of upper arm)
const lBicep = muscleBulge(bicepR * 1.2, 0.13 * S * mg, bicepR);
lBicep.position.set(0, -0.2 * S, 0.04 * S * mg);
lShoulderPivot.add(lBicep);

// Tricep bulge (back of upper arm)
const triR = 0.06 * S * mg;
const lTricep = muscleBulge(triR, 0.12 * S * mg, triR);
lTricep.position.set(0, -0.22 * S, -0.04 * S * mg);
lShoulderPivot.add(lTricep);

// Vein on bicep (visible at high swole)
if (sw > 0.3) {{
    const veinGeo = new THREE.CylinderGeometry(0.005 * S, 0.005 * S, 0.18 * S, 4);
    const lVein = new THREE.Mesh(veinGeo, veinMat);
    lVein.position.set(0.03 * S, -0.18 * S, 0.06 * S * mg);
    lVein.rotation.z = 0.2;
    lShoulderPivot.add(lVein);
}}

const lElbowPivot = new THREE.Group();
lElbowPivot.position.set(0, -0.5 * S, 0);
lShoulderPivot.add(lElbowPivot);

// Forearm
const forearmR = 0.055 * S * mg;
const lForearmGeo = new THREE.CylinderGeometry(forearmR, forearmR * 0.7, 0.45 * S, 8);
const lForearm = new THREE.Mesh(lForearmGeo, skinMat);
lForearm.position.y = -0.22 * S;
lElbowPivot.add(lForearm);

// Forearm muscle (brachioradialis)
const lBrachR = muscleBulge(forearmR * 0.8, 0.08 * S * mg, forearmR * 0.6);
lBrachR.position.set(0, -0.12 * S, 0.02 * S);
lElbowPivot.add(lBrachR);

const lHandGeo = new THREE.SphereGeometry(0.04 * S, 8, 8);
const lHand = new THREE.Mesh(lHandGeo, skinMat);
lHand.position.y = -0.45 * S;
lElbowPivot.add(lHand);

// =====================
// RIGHT ARM (mirror)
// =====================
const rShoulderPivot = new THREE.Group();
rShoulderPivot.position.set(torsoW + 0.04 * S, 0.38 * S, 0);
icarus.add(rShoulderPivot);

const rDeltF = muscleBulge(deltSize * 0.8, deltSize, deltSize * 0.7);
rDeltF.position.set(0, 0.02 * S, 0.04 * S);
rShoulderPivot.add(rDeltF);
const rDeltS = muscleBulge(deltSize, deltSize * 0.9, deltSize * 0.6);
rDeltS.position.set(0.04 * S, 0.02 * S, 0);
rShoulderPivot.add(rDeltS);
const rDeltR = muscleBulge(deltSize * 0.7, deltSize * 0.9, deltSize * 0.7);
rDeltR.position.set(0, 0.02 * S, -0.04 * S);
rShoulderPivot.add(rDeltR);

const rUpperGeo = new THREE.CylinderGeometry(bicepR * 0.8, bicepR * 0.7, 0.5 * S, 8);
const rUpperArm = new THREE.Mesh(rUpperGeo, skinMat);
rUpperArm.position.y = -0.25 * S;
rShoulderPivot.add(rUpperArm);

const rBicep = muscleBulge(bicepR * 1.2, 0.13 * S * mg, bicepR);
rBicep.position.set(0, -0.2 * S, 0.04 * S * mg);
rShoulderPivot.add(rBicep);

const rTricep = muscleBulge(triR, 0.12 * S * mg, triR);
rTricep.position.set(0, -0.22 * S, -0.04 * S * mg);
rShoulderPivot.add(rTricep);

if (sw > 0.3) {{
    const veinGeo = new THREE.CylinderGeometry(0.005 * S, 0.005 * S, 0.18 * S, 4);
    const rVein = new THREE.Mesh(veinGeo, veinMat);
    rVein.position.set(-0.03 * S, -0.18 * S, 0.06 * S * mg);
    rVein.rotation.z = -0.2;
    rShoulderPivot.add(rVein);
}}

const rElbowPivot = new THREE.Group();
rElbowPivot.position.set(0, -0.5 * S, 0);
rShoulderPivot.add(rElbowPivot);

const rForearmGeo = new THREE.CylinderGeometry(forearmR, forearmR * 0.7, 0.45 * S, 8);
const rForearm = new THREE.Mesh(rForearmGeo, skinMat);
rForearm.position.y = -0.22 * S;
rElbowPivot.add(rForearm);

const rBrachR = muscleBulge(forearmR * 0.8, 0.08 * S * mg, forearmR * 0.6);
rBrachR.position.set(0, -0.12 * S, 0.02 * S);
rElbowPivot.add(rBrachR);

const rHandGeo = new THREE.SphereGeometry(0.04 * S, 8, 8);
const rHand = new THREE.Mesh(rHandGeo, skinMat);
rHand.position.y = -0.45 * S;
rElbowPivot.add(rHand);

// =====================
// LEFT LEG — fully muscled
// =====================
const lHipPivot = new THREE.Group();
lHipPivot.position.set(-0.13 * S * mg, -0.62 * S, 0);
icarus.add(lHipPivot);

// Glute
const gluteSize = 0.1 * S * mg;
const lGlute = muscleBulge(gluteSize, gluteSize * 0.9, gluteSize * 0.8);
lGlute.position.set(0, 0.02 * S, -0.08 * S * mg);
lHipPivot.add(lGlute);

// Thigh bone
const quadR = 0.09 * S * mg;
const lThighGeo = new THREE.CylinderGeometry(quadR, quadR * 0.85, 0.6 * S, 10);
const lThigh = new THREE.Mesh(lThighGeo, skinMat);
lThigh.position.y = -0.3 * S;
lHipPivot.add(lThigh);

// Quad (front of thigh — massive teardrop)
const lQuad = muscleBulge(quadR * 1.1, 0.18 * S * mg, quadR * 0.9);
lQuad.position.set(0, -0.22 * S, 0.05 * S * mg);
lHipPivot.add(lQuad);

// Inner quad (vastus medialis — the teardrop)
const lVMO = muscleBulge(quadR * 0.6, 0.1 * S * mg, quadR * 0.5);
lVMO.position.set(0.04 * S, -0.35 * S, 0.04 * S * mg);
lHipPivot.add(lVMO);

// Hamstring (back of thigh)
const lHam = muscleBulge(quadR * 0.8, 0.15 * S * mg, quadR * 0.7);
lHam.position.set(0, -0.2 * S, -0.05 * S * mg);
lHipPivot.add(lHam);

const lKneePivot = new THREE.Group();
lKneePivot.position.set(0, -0.6 * S, 0);
lHipPivot.add(lKneePivot);

// Shin / calf
const calfR = 0.065 * S * mg;
const lShinGeo = new THREE.CylinderGeometry(calfR * 0.8, calfR * 0.5, 0.55 * S, 8);
const lShin = new THREE.Mesh(lShinGeo, skinMat);
lShin.position.y = -0.27 * S;
lKneePivot.add(lShin);

// Calf muscle (gastrocnemius — diamond shaped)
const lCalf = muscleBulge(calfR * 0.8, 0.1 * S * mg, calfR * 1.0);
lCalf.position.set(0, -0.12 * S, -0.03 * S * mg);
lKneePivot.add(lCalf);

const lFootGeo = new THREE.BoxGeometry(0.08 * S, 0.04 * S, 0.16 * S);
const lFoot = new THREE.Mesh(lFootGeo, skinMat);
lFoot.position.set(0, -0.56 * S, 0.03 * S);
lKneePivot.add(lFoot);

// =====================
// RIGHT LEG (mirror)
// =====================
const rHipPivot = new THREE.Group();
rHipPivot.position.set(0.13 * S * mg, -0.62 * S, 0);
icarus.add(rHipPivot);

const rGlute = muscleBulge(gluteSize, gluteSize * 0.9, gluteSize * 0.8);
rGlute.position.set(0, 0.02 * S, -0.08 * S * mg);
rHipPivot.add(rGlute);

const rThighGeo = new THREE.CylinderGeometry(quadR, quadR * 0.85, 0.6 * S, 10);
const rThigh = new THREE.Mesh(rThighGeo, skinMat);
rThigh.position.y = -0.3 * S;
rHipPivot.add(rThigh);

const rQuad = muscleBulge(quadR * 1.1, 0.18 * S * mg, quadR * 0.9);
rQuad.position.set(0, -0.22 * S, 0.05 * S * mg);
rHipPivot.add(rQuad);

const rVMO = muscleBulge(quadR * 0.6, 0.1 * S * mg, quadR * 0.5);
rVMO.position.set(-0.04 * S, -0.35 * S, 0.04 * S * mg);
rHipPivot.add(rVMO);

const rHam = muscleBulge(quadR * 0.8, 0.15 * S * mg, quadR * 0.7);
rHam.position.set(0, -0.2 * S, -0.05 * S * mg);
rHipPivot.add(rHam);

const rKneePivot = new THREE.Group();
rKneePivot.position.set(0, -0.6 * S, 0);
rHipPivot.add(rKneePivot);

const rShinGeo = new THREE.CylinderGeometry(calfR * 0.8, calfR * 0.5, 0.55 * S, 8);
const rShin = new THREE.Mesh(rShinGeo, skinMat);
rShin.position.y = -0.27 * S;
rKneePivot.add(rShin);

const rCalf = muscleBulge(calfR * 0.8, 0.1 * S * mg, calfR * 1.0);
rCalf.position.set(0, -0.12 * S, -0.03 * S * mg);
rKneePivot.add(rCalf);

const rFootGeo = new THREE.BoxGeometry(0.08 * S, 0.04 * S, 0.16 * S);
const rFoot = new THREE.Mesh(rFootGeo, skinMat);
rFoot.position.set(0, -0.56 * S, 0.03 * S);
rKneePivot.add(rFoot);

// =====================
// WINGS — majestic, scale with swole
// =====================
const wingMat = new THREE.MeshPhongMaterial({{
    color: 0x00ffff,
    emissive: new THREE.Color().setHSL(0.5, 0.8, 0.05 + sw * 0.15),
    side: THREE.DoubleSide,
    transparent: true,
    opacity: 0.5 + sw * 0.3,
}});

const wingSpan = 1.8 * S * (1 + sw * 0.8);
const wingShape = new THREE.Shape();
wingShape.moveTo(0, 0);
wingShape.quadraticCurveTo(-wingSpan * 0.5, wingSpan * 0.6, -wingSpan, wingSpan * 0.5);
wingShape.quadraticCurveTo(-wingSpan * 0.7, 0, -wingSpan * 0.3, -wingSpan * 0.3);
wingShape.lineTo(0, 0);
const wingGeo = new THREE.ShapeGeometry(wingShape);
const leftWing = new THREE.Mesh(wingGeo, wingMat);
leftWing.position.set(-0.15 * S, 0.25 * S, -0.12);
icarus.add(leftWing);

const rWingShape = new THREE.Shape();
rWingShape.moveTo(0, 0);
rWingShape.quadraticCurveTo(wingSpan * 0.5, wingSpan * 0.6, wingSpan, wingSpan * 0.5);
rWingShape.quadraticCurveTo(wingSpan * 0.7, 0, wingSpan * 0.3, -wingSpan * 0.3);
rWingShape.lineTo(0, 0);
const rWingGeo = new THREE.ShapeGeometry(rWingShape);
const rightWing = new THREE.Mesh(rWingGeo, wingMat);
rightWing.position.set(0.15 * S, 0.25 * S, -0.12);
icarus.add(rightWing);

scene.add(icarus);

// === BARBELL ===
const barbellMat = new THREE.MeshPhongMaterial({{ color: 0xcccccc, emissive: 0x333333, shininess: 150 }});
const plateMat = new THREE.MeshPhongMaterial({{ color: 0x222222, emissive: 0x080808, shininess: 40 }});

const barbell = new THREE.Group();
const barGeo = new THREE.CylinderGeometry(0.02, 0.02, 1.8, 8);
barGeo.rotateZ(Math.PI / 2);
const bar = new THREE.Mesh(barGeo, barbellMat);
barbell.add(bar);

const plateR = 0.13 + sw * 0.12;
const plateGeo = new THREE.CylinderGeometry(plateR, plateR, 0.04, 16);
plateGeo.rotateZ(Math.PI / 2);
for (const xOff of [-0.82, -0.72, 0.72, 0.82]) {{
    const plate = new THREE.Mesh(plateGeo, plateMat);
    plate.position.x = xOff;
    barbell.add(plate);
}}
// Extra plates at high swole
if (sw > 0.4) {{
    for (const xOff of [-0.62, 0.62]) {{
        const plate = new THREE.Mesh(plateGeo, plateMat);
        plate.position.x = xOff;
        barbell.add(plate);
    }}
}}
barbell.visible = false;
scene.add(barbell);

// === BENCH ===
const benchMat = new THREE.MeshPhongMaterial({{ color: 0x444444, emissive: 0x111111 }});
const benchPadMat = new THREE.MeshPhongMaterial({{ color: 0x880000, emissive: 0x220000 }});

const benchGroup = new THREE.Group();
const benchPadGeo = new THREE.BoxGeometry(0.4, 0.08, 1.2);
const benchPad = new THREE.Mesh(benchPadGeo, benchPadMat);
benchPad.position.y = 0.35;
benchGroup.add(benchPad);
const benchLegGeo = new THREE.CylinderGeometry(0.03, 0.03, 0.35, 6);
for (const pos of [[-0.15, 0.175, -0.5], [0.15, 0.175, -0.5], [-0.15, 0.175, 0.5], [0.15, 0.175, 0.5]]) {{
    const leg = new THREE.Mesh(benchLegGeo, benchMat);
    leg.position.set(pos[0], pos[1], pos[2]);
    benchGroup.add(leg);
}}
benchGroup.visible = false;
scene.add(benchGroup);

// === LIGHTS ===
const ambientLight = new THREE.AmbientLight(0x222244);
scene.add(ambientLight);

// Key light — shows muscle definition
const keyLight = new THREE.DirectionalLight(0xffeedd, 0.8);
keyLight.position.set(3, 5, 5);
scene.add(keyLight);

const pointLight1 = new THREE.PointLight(0xff00ff, 2, 30);
pointLight1.position.set(5, 5, 5);
scene.add(pointLight1);

const pointLight2 = new THREE.PointLight(0x00ffff, 2, 30);
pointLight2.position.set(-5, -3, 5);
scene.add(pointLight2);

const pointLight3 = new THREE.PointLight(0xffff00, 1.5, 30);
pointLight3.position.set(0, 5, -5);
scene.add(pointLight3);

// Rim light for silhouette definition
const rimLight = new THREE.PointLight(0xff4400, 1.0 + sw * 1.5, 20);
rimLight.position.set(0, 2, -6);
scene.add(rimLight);

// Disco particles
const particleCount = 200;
const particleGeometry = new THREE.BufferGeometry();
const particlePositions = new Float32Array(particleCount * 3);
const particleColors = new Float32Array(particleCount * 3);
const particleVelocities = [];

for (let i = 0; i < particleCount; i++) {{
    particlePositions[i * 3] = (Math.random() - 0.5) * 20;
    particlePositions[i * 3 + 1] = (Math.random() - 0.5) * 20;
    particlePositions[i * 3 + 2] = (Math.random() - 0.5) * 20;
    const hue = Math.random();
    const color = new THREE.Color().setHSL(hue, 1, 0.7);
    particleColors[i * 3] = color.r;
    particleColors[i * 3 + 1] = color.g;
    particleColors[i * 3 + 2] = color.b;
    particleVelocities.push({{
        x: (Math.random() - 0.5) * 0.02,
        y: (Math.random() - 0.5) * 0.02,
        z: (Math.random() - 0.5) * 0.02,
    }});
}}

particleGeometry.setAttribute('position', new THREE.BufferAttribute(particlePositions, 3));
particleGeometry.setAttribute('color', new THREE.BufferAttribute(particleColors, 3));

const particleMaterial = new THREE.PointsMaterial({{
    size: 0.08,
    vertexColors: true,
    transparent: true,
    opacity: 0.7,
    blending: THREE.AdditiveBlending,
}});
const particles = new THREE.Points(particleGeometry, particleMaterial);
scene.add(particles);

// === EXERCISE STATE MACHINE ===
const EXERCISE_DURATION = 10.0;
const exerciseNames = ['PUSHUPS', 'DEADLIFTS', 'BENCH PRESS'];
let exerciseIndex = 0;
let exerciseTimer = 0;
let repCount = 0;
let lastRepHalf = false;

const exerciseNameEl = document.getElementById('exercise-name');
const repCountEl = document.getElementById('rep-count');

// === ANIMATION LOOP ===
let time = 0;
let lastTime = performance.now() / 1000;

function animate() {{
    requestAnimationFrame(animate);
    const now = performance.now() / 1000;
    const dt = Math.min(now - lastTime, 0.05);
    lastTime = now;
    time += dt;

    // --- Exercise cycling ---
    exerciseTimer += dt;
    if (exerciseTimer >= EXERCISE_DURATION) {{
        exerciseTimer -= EXERCISE_DURATION;
        exerciseIndex = (exerciseIndex + 1) % 3;
        repCount = 0;
        lastRepHalf = false;
    }}

    const repSpeed = 2.5 + sw * 1.5;
    const phase = exerciseTimer * repSpeed;
    const p = (Math.sin(phase) + 1) * 0.5;

    const currentHalf = Math.sin(phase) > 0;
    if (currentHalf && !lastRepHalf) repCount++;
    lastRepHalf = currentHalf;

    // --- Reset joints ---
    lShoulderPivot.rotation.set(0, 0, 0);
    rShoulderPivot.rotation.set(0, 0, 0);
    lElbowPivot.rotation.set(0, 0, 0);
    rElbowPivot.rotation.set(0, 0, 0);
    lHipPivot.rotation.set(0, 0, 0);
    rHipPivot.rotation.set(0, 0, 0);
    lKneePivot.rotation.set(0, 0, 0);
    rKneePivot.rotation.set(0, 0, 0);
    icarus.rotation.set(0, 0, 0);
    icarus.position.set(0, 0, 0);

    // --- Muscle flex pump (subtle scaling pulse on current exercise muscles) ---
    const flexPulse = 1.0 + Math.sin(phase * 2) * 0.06 * mg;

    if (exerciseIndex === 0) {{
        // === PUSHUPS ===
        barbell.visible = false;
        benchGroup.visible = false;

        icarus.rotation.x = -Math.PI * 0.42;
        icarus.position.y = -0.3 + p * 0.35;
        icarus.position.z = 0.4;

        lShoulderPivot.rotation.x = Math.PI * 0.5;
        rShoulderPivot.rotation.x = Math.PI * 0.5;
        lShoulderPivot.rotation.z = 0.3;
        rShoulderPivot.rotation.z = -0.3;

        const elbowBend = (1.0 - p) * 1.8;
        lElbowPivot.rotation.x = -elbowBend;
        rElbowPivot.rotation.x = -elbowBend;

        lHipPivot.rotation.x = 0.5;
        rHipPivot.rotation.x = 0.5;

        // Flex pecs and triceps during pushups
        lPec.scale.z = pecSize * 0.9 * flexPulse;
        rPec.scale.z = pecSize * 0.9 * flexPulse;

    }} else if (exerciseIndex === 1) {{
        // === DEADLIFTS ===
        barbell.visible = true;
        benchGroup.visible = false;

        const hipAngle = (1.0 - p) * 0.9;
        icarus.rotation.x = -hipAngle;
        icarus.position.y = -0.2 + p * 0.3;

        lShoulderPivot.rotation.x = hipAngle * 0.8;
        rShoulderPivot.rotation.x = hipAngle * 0.8;
        lElbowPivot.rotation.x = -0.15;
        rElbowPivot.rotation.x = -0.15;

        const kneeBend = (1.0 - p) * 0.5;
        lHipPivot.rotation.x = -kneeBend * 0.3;
        rHipPivot.rotation.x = -kneeBend * 0.3;
        lKneePivot.rotation.x = kneeBend;
        rKneePivot.rotation.x = kneeBend;

        barbell.position.set(0, -1.2 + p * 0.7, 0.3 - hipAngle * 0.3);
        barbell.rotation.set(0, 0, 0);

        // Flex back and quads during deadlift
        lQuad.scale.y = 0.18 * S * mg * flexPulse;
        rQuad.scale.y = 0.18 * S * mg * flexPulse;

    }} else {{
        // === BENCH PRESS ===
        barbell.visible = true;
        benchGroup.visible = true;
        benchGroup.position.set(0, 0, 0);

        icarus.rotation.x = Math.PI * 0.5;
        icarus.position.y = 0.7;
        icarus.position.z = -0.2;

        lShoulderPivot.rotation.x = Math.PI * 0.5;
        rShoulderPivot.rotation.x = Math.PI * 0.5;
        lShoulderPivot.rotation.z = 0.25;
        rShoulderPivot.rotation.z = -0.25;

        const benchElbow = (1.0 - p) * 1.6;
        lElbowPivot.rotation.x = -benchElbow;
        rElbowPivot.rotation.x = -benchElbow;

        lHipPivot.rotation.x = -0.3;
        rHipPivot.rotation.x = -0.3;
        lKneePivot.rotation.x = 0.6;
        rKneePivot.rotation.x = 0.6;

        barbell.position.set(0, 0.9 + p * 0.6, -0.2);
        barbell.rotation.set(0, 0, 0);

        // Flex pecs and biceps during bench
        lBicep.scale.x = bicepR * 1.2 * flexPulse;
        rBicep.scale.x = bicepR * 1.2 * flexPulse;
    }}

    // --- HUD ---
    exerciseNameEl.textContent = exerciseNames[exerciseIndex];
    repCountEl.textContent = 'Rep ' + repCount;

    // --- Wings flap ---
    leftWing.rotation.y = Math.sin(time * 2.5) * 0.35;
    rightWing.rotation.y = -Math.sin(time * 2.5) * 0.35;

    // --- E8 lattice rotation ---
    pointCloud.rotation.y = time * 0.15;
    pointCloud.rotation.x = Math.sin(time * 0.1) * 0.2;
    edgeLines.rotation.y = time * 0.15;
    edgeLines.rotation.x = Math.sin(time * 0.1) * 0.2;

    // --- Disco lights ---
    pointLight1.position.x = Math.cos(time * 0.7) * 8;
    pointLight1.position.z = Math.sin(time * 0.7) * 8;
    pointLight2.position.x = Math.cos(time * 0.5 + 2) * 6;
    pointLight2.position.z = Math.sin(time * 0.5 + 2) * 6;
    pointLight3.position.y = Math.sin(time * 0.3) * 5 + 3;
    rimLight.position.x = Math.sin(time * 0.2) * 4;

    pointLight1.color.setHSL((time * 0.1) % 1, 1, 0.5);
    pointLight2.color.setHSL((time * 0.1 + 0.33) % 1, 1, 0.5);
    pointLight3.color.setHSL((time * 0.1 + 0.66) % 1, 1, 0.5);

    // --- Particles ---
    const pPositions = particles.geometry.attributes.position.array;
    for (let i = 0; i < particleCount; i++) {{
        pPositions[i * 3] += particleVelocities[i].x;
        pPositions[i * 3 + 1] += particleVelocities[i].y;
        pPositions[i * 3 + 2] += particleVelocities[i].z;
        for (let j = 0; j < 3; j++) {{
            if (Math.abs(pPositions[i * 3 + j]) > 10) {{
                pPositions[i * 3 + j] *= -0.9;
            }}
        }}
    }}
    particles.geometry.attributes.position.needsUpdate = true;

    // --- Camera ---
    camera.position.x = Math.sin(time * 0.08) * 8;
    camera.position.z = Math.cos(time * 0.08) * 8;
    camera.position.y = Math.sin(time * 0.05) * 2 + 1.5;
    camera.lookAt(0, 0, 0);

    renderer.render(scene, camera);
}}

animate();

// === NMSE CHART ===
const chartCanvas = document.getElementById('chart');
const ctx = chartCanvas.getContext('2d');
const progressData = [{progress_str}];

function drawChart() {{
    const w = chartCanvas.width;
    const h = chartCanvas.height;
    ctx.clearRect(0, 0, w, h);

    if (progressData.length < 2) {{
        ctx.fillStyle = '#888';
        ctx.font = '12px Courier New';
        ctx.fillText('Waiting for data...', 10, h / 2);
        return;
    }}

    const maxTrial = progressData[progressData.length - 1][0];
    const maxNmse = progressData[0][1];
    const minNmse = progressData[progressData.length - 1][1];
    const nmseRange = maxNmse - minNmse || 0.1;

    ctx.strokeStyle = 'rgba(0, 255, 255, 0.1)';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {{
        const y = (i / 4) * h;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
    }}

    ctx.beginPath();
    ctx.strokeStyle = '#0ff';
    ctx.lineWidth = 2;
    ctx.shadowColor = '#0ff';
    ctx.shadowBlur = 8;

    for (let i = 0; i < progressData.length; i++) {{
        const x = (progressData[i][0] / (maxTrial || 1)) * w;
        const y = h - ((progressData[i][1] - minNmse) / nmseRange) * (h * 0.8) - h * 0.1;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }}
    ctx.stroke();
    ctx.shadowBlur = 0;

    ctx.fillStyle = '#0ff';
    ctx.font = '10px Courier New';
    ctx.fillText(maxNmse.toFixed(4), 2, 12);
    ctx.fillText(minNmse.toFixed(4), 2, h - 2);
}}

drawChart();

window.addEventListener('resize', () => {{
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}});
</script>
</body>
</html>"##,
        swole_pct = swole_factor,
        swole_label = if swole_factor < 15.0 {
            "Just warming up..."
        } else if swole_factor < 30.0 {
            "Getting warmed up!"
        } else if swole_factor < 50.0 {
            "Feeling the burn!"
        } else if swole_factor < 70.0 {
            "LOOKING GOOD!"
        } else if swole_factor < 90.0 {
            "ABSOLUTELY SHREDDED"
        } else {
            "MAXIMUM SWOLE ACHIEVED"
        },
        best_class = if current_best < 0.5 {
            "good"
        } else if current_best < 0.9 {
            "highlight"
        } else {
            "bad"
        },
        best_nmse = if current_best < f64::MAX {
            format!("{:.4}", current_best)
        } else {
            "N/A".to_string()
        },
        total_trials = total_trials,
        max_trials = max_trials,
        progress_pct = progress_pct,
        points_str = points_json.join(","),
        edges_str = edges_json.join(","),
        progress_str = progress_json.join(","),
        swole_raw = swole_factor,
    )
}

fn main() {
    println!("Generating E8 root system...");
    let roots = e8_roots();
    println!("  {} root vectors generated", roots.len());

    let roots_3d: Vec<[f64; 3]> = roots.iter().map(|r| project_to_3d(r)).collect();
    println!("  Projected to 3D");

    let results_path = "/tmp/icarus-disco-results.txt";
    let html = generate_html(&roots_3d, results_path);

    let output_path = "/tmp/icarus-disco.html";
    fs::write(output_path, &html).expect("Failed to write HTML");
    println!("\n  Wrote {} bytes to {}", html.len(), output_path);
    println!("  Open in browser: file://{}", output_path);

    let win_path = "/mnt/c/Users";
    if std::path::Path::new(win_path).exists() {
        if let Ok(entries) = fs::read_dir(win_path) {
            for entry in entries.flatten() {
                let desktop = entry.path().join("Desktop");
                if desktop.exists() {
                    let dest = desktop.join("icarus-disco.html");
                    if fs::copy(output_path, &dest).is_ok() {
                        println!("  Copied to Desktop: {}", dest.display());
                    }
                    break;
                }
            }
        }
    }

    println!("\n  The page auto-refreshes every 30 seconds.");
    println!("  Re-run this visualizer to update with latest training results.");
    println!("\n  ICARUS FITNESS!");
}
