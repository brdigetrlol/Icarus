//! Combined Dashboard Renderer
//!
//! Produces a unified split-viewport visualization: 70% main 3D lattice field
//! scene with full procedural geometry, 30% side panel with 3D neuromodulator
//! orbs, energy gauges, time series chart, and system stats.
//!
//! All 3D geometry is procedurally generated from `EmcSnapshot` data — sphere
//! positions, sizes, colors, edges, glow layers, and animations derive
//! entirely from the live simulation state.

use std::fmt::Write;

use icarus_engine::autonomous::{AutoEvent, AutoEventType, EmcSnapshot};
use icarus_math::lattice::LatticeLayer;

use crate::color::{
    amplitude_color, convergence_color, phase_color, rgb_to_hex, rgb_to_threejs_hex,
};
use crate::projection::{find_edges, project_e8_to_3d, project_high_dim_to_3d};
use crate::scene::{pulse_animation_js, rotate_scene_js, sphere_breathe_js, SceneBuilder};
use crate::template::{HtmlDocument, PanelPosition, Theme};

/// Render the combined dashboard from EMC state and event history.
///
/// Produces a self-contained HTML document with:
/// - Left 70%: Full interactive 3D lattice field (Three.js WebGL)
/// - Right 30%: Stats, 3D neuromodulator orbs (inline canvas), energy bars, time series
pub fn render_combined_dashboard(
    snapshot: &EmcSnapshot,
    events: &[AutoEvent],
    theme: Theme,
) -> String {
    // === Build the main 3D lattice scene ===
    let scene_js = build_main_3d_scene(snapshot);

    // === Build animation JS ===
    let mut anim = sphere_breathe_js();
    anim.push_str(&pulse_animation_js());
    anim.push_str(&rotate_scene_js(0.001));
    // Per-layer energy ring rotation
    anim.push_str(
        r#"
    const rings = scene.getObjectByName('energyRings');
    if (rings) {
        rings.rotation.y += 0.003;
        rings.children.forEach((ring, i) => {
            ring.rotation.z = Math.sin(time * 0.5 + i * 1.5) * 0.1;
        });
    }
"#,
    );

    // === Build the side panel HTML ===
    let side_html = build_side_panel_html(snapshot, events, &theme);

    // === Build the 2D chart JS for the side panel canvases ===
    let chart_js = build_side_panel_charts_js(snapshot, events, &theme);

    // === Assemble the HTML document ===
    let title = format!("Icarus - EMC Dashboard (tick {})", snapshot.tick);
    let mut doc = HtmlDocument::new(&title, theme.clone());
    doc.set_split_layout(true);

    // The side panel content goes into a fixed right panel
    doc.add_panel_with_width("dashboard-side", PanelPosition::TopRight, &side_html, "30%");

    // Add chart canvases (rendered inside the side panel)
    doc.add_chart("neuro-orbs-canvas", 280, 200);
    doc.add_chart("energy-bars-canvas", 280, 100);
    doc.add_chart("mini-timeseries", 280, 140);

    doc.set_extra_css(&format!(
        r#"
#dashboard-side {{
    position: fixed !important;
    right: 0 !important;
    top: 0 !important;
    width: 30% !important;
    height: 100vh !important;
    overflow-y: auto;
    border-radius: 0 !important;
    border-left: 1px solid {accent}33;
    border-top: none; border-right: none; border-bottom: none;
    padding: 16px !important;
    z-index: 20 !important;
}}
#neuro-orbs-canvas, #energy-bars-canvas, #mini-timeseries {{
    position: relative !important;
    display: block;
    margin: 8px auto;
    border: 1px solid {accent}22;
    border-radius: 6px;
}}
.section-header {{
    color: {accent};
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin: 12px 0 6px 0;
    font-weight: 600;
    border-bottom: 1px solid {accent}22;
    padding-bottom: 4px;
}}
.neuro-bar {{
    display: flex;
    align-items: center;
    margin: 3px 0;
    font-size: 12px;
}}
.neuro-bar .bar-label {{
    width: 30px;
    color: {text}88;
    font-weight: 600;
}}
.neuro-bar .bar-track {{
    flex: 1;
    height: 6px;
    background: {text}11;
    border-radius: 3px;
    margin: 0 6px;
    overflow: hidden;
}}
.neuro-bar .bar-fill {{
    height: 100%;
    border-radius: 3px;
    transition: width 0.3s;
}}
.neuro-bar .bar-val {{
    width: 45px;
    text-align: right;
    font-variant-numeric: tabular-nums;
    font-weight: 600;
}}
"#,
        accent = theme.accent_color(),
        text = theme.text_color(),
    ));

    doc.set_scene_js(&scene_js);
    doc.set_chart_js(&chart_js);
    doc.set_animation_js(&anim);

    doc.render()
}

/// Build the main 3D scene JS from the first available layer's field state.
///
/// Includes: lattice spheres, neighbor edges, energy ring indicators per layer,
/// Kuramoto order parameter arrow, and ambient particle field.
fn build_main_3d_scene(snapshot: &EmcSnapshot) -> String {
    let mut scene = SceneBuilder::new();
    scene.set_camera([0.0, 5.0, 12.0], [0.0, 0.0, 0.0]);
    scene.set_fov(55.0);
    scene.add_point_light([6.0, 10.0, 6.0], 0x00e5ff, 0.7, 35.0);
    scene.add_point_light([-6.0, 8.0, -6.0], 0xff00aa, 0.4, 25.0);
    scene.add_point_light([0.0, -4.0, 8.0], 0xffd700, 0.3, 20.0);
    scene.set_ambient_light(0x202040, 0.35);

    // Render the primary layer (index 0) as the main lattice field
    if let Some(layer_state) = snapshot.layer_states.first() {
        let num_sites = layer_state.values_re.len();
        let dim = match layer_state.layer {
            LatticeLayer::Analytical => 8,
            LatticeLayer::Creative => 24,
            LatticeLayer::Associative => 64,
            LatticeLayer::Sensory => 1024,
        };

        // Compute amplitudes and phases
        let mut amplitudes = Vec::with_capacity(num_sites);
        let mut max_amp: f32 = 0.0;
        let mut sum_re: f64 = 0.0;
        let mut sum_im: f64 = 0.0;

        for i in 0..num_sites {
            let re = layer_state.values_re[i];
            let im = layer_state.values_im[i];
            let amp = (re * re + im * im).sqrt();
            let phase = im.atan2(re);
            amplitudes.push(amp);
            if amp > max_amp {
                max_amp = amp;
            }
            sum_re += (phase as f64).cos() * amp as f64;
            sum_im += (phase as f64).sin() * amp as f64;
        }

        if max_amp < 1e-12 {
            max_amp = 1.0;
        }

        // Project coordinates
        let coords =
            super::lattice_field::generate_lattice_coords(&layer_state.layer, num_sites, dim);
        let projected: Vec<[f64; 3]> = coords
            .iter()
            .map(|c| {
                if dim == 8 {
                    project_e8_to_3d(c, 0.5)
                } else {
                    project_high_dim_to_3d(c, 0.5)
                }
            })
            .collect();

        // Add lattice sites as spheres — amplitude-colored with emissive glow
        for i in 0..num_sites {
            let t = amplitudes[i] / max_amp;
            let color = amplitude_color(t);
            let radius = 0.05 + 0.13 * t as f64;
            let glow = (t * 0.5).min(0.4);
            let emissive = (color.0 * glow, color.1 * glow, color.2 * glow);
            scene.add_sphere(projected[i], radius, color, emissive, 0.92);
        }

        // Neighbor edges
        if num_sites > 1 {
            let edges = find_edges(&projected, 3.5);
            for (i, j) in &edges {
                let t_avg = (amplitudes[*i] + amplitudes[*j]) / (2.0 * max_amp);
                let edge_color = (
                    0.1 + 0.15 * t_avg,
                    0.15 + 0.25 * t_avg,
                    0.35 + 0.2 * t_avg,
                );
                scene.add_edge(projected[*i], projected[*j], edge_color, 0.2);
            }
        }

        // Kuramoto order parameter arrow at center
        let total_amp: f64 = amplitudes.iter().map(|a| *a as f64).sum();
        let order_r = if total_amp > 1e-12 {
            (sum_re * sum_re + sum_im * sum_im).sqrt() / total_amp
        } else {
            0.0
        };
        let order_theta = sum_im.atan2(sum_re);

        let arrow_color = phase_color(order_theta as f32);
        let arrow_hex = rgb_to_threejs_hex(arrow_color.0, arrow_color.1, arrow_color.2);
        let arrow_len = 2.5 * order_r;
        let ax = arrow_len * order_theta.cos();
        let az = arrow_len * order_theta.sin();

        scene.add_custom_js(&format!(
            r#"
// Kuramoto order parameter arrow
{{
    const dir = new THREE.Vector3({ax}, 0, {az}).normalize();
    const len = {len};
    if (len > 0.01) {{
        const arrow = new THREE.ArrowHelper(dir, new THREE.Vector3(0,0,0), len, {color}, 0.3, 0.15);
        arrow.name = 'orderArrow';
        scene.add(arrow);
    }}
    // Order parameter glow sphere
    const ogeo = new THREE.SphereGeometry({gr}, 16, 12);
    const omat = new THREE.MeshBasicMaterial({{
        color: {color}, transparent: true, opacity: {opacity},
        blending: THREE.AdditiveBlending, depthWrite: false
    }});
    const orb = new THREE.Mesh(ogeo, omat);
    orb.name = 'orderOrb';
    scene.add(orb);
}}
"#,
            ax = ax,
            az = az,
            len = arrow_len,
            color = arrow_hex,
            gr = 0.15 + 0.3 * order_r,
            opacity = 0.3 + 0.4 * order_r,
        ));
    }

    // Per-layer energy rings: one torus per layer, radius proportional to energy
    if !snapshot.layer_stats.is_empty() {
        let max_energy: f32 = snapshot
            .layer_stats
            .iter()
            .map(|s| s.total_energy)
            .fold(0.0f32, f32::max)
            .max(1e-6);

        let mut rings_js = String::from(
            "{\n    const rings = new THREE.Group();\n    rings.name = 'energyRings';\n",
        );

        let ring_colors = [0x00e5ffu32, 0xff00aa, 0xffd700, 0x44ff88];
        for (i, stats) in snapshot.layer_stats.iter().enumerate() {
            let t = stats.total_energy / max_energy;
            let radius = 3.0 + i as f64 * 1.8;
            let tube = 0.02 + 0.06 * t as f64;
            let color = ring_colors.get(i).copied().unwrap_or(0xffffff);
            let _ = write!(
                rings_js,
                r#"    {{
        const tgeo = new THREE.TorusGeometry({radius}, {tube}, 8, 64);
        const tmat = new THREE.MeshBasicMaterial({{
            color: 0x{color:06x}, transparent: true, opacity: {opacity},
            blending: THREE.AdditiveBlending, depthWrite: false
        }});
        const torus = new THREE.Mesh(tgeo, tmat);
        torus.rotation.x = Math.PI / 2;
        torus.position.y = {y};
        rings.add(torus);
    }}
"#,
                radius = radius,
                tube = tube,
                color = color,
                opacity = 0.2 + 0.5 * t as f64,
                y = -1.5 + i as f64 * 0.3,
            );
        }
        rings_js.push_str("    scene.add(rings);\n}\n");
        scene.add_custom_js(&rings_js);
    }

    // Ambient star field for atmosphere
    scene.add_custom_js(
        r#"
{
    const geo = new THREE.BufferGeometry();
    const n = 300;
    const pos = new Float32Array(n * 3);
    const col = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
        pos[i*3] = (Math.random() - 0.5) * 40;
        pos[i*3+1] = (Math.random() - 0.5) * 40;
        pos[i*3+2] = (Math.random() - 0.5) * 40;
        const c = 0.2 + Math.random() * 0.3;
        col[i*3] = c * 0.4; col[i*3+1] = c * 0.6; col[i*3+2] = c;
    }
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
    const mat = new THREE.PointsMaterial({
        size: 0.06, vertexColors: true, transparent: true,
        opacity: 0.35, blending: THREE.AdditiveBlending, depthWrite: false
    });
    const stars = new THREE.Points(geo, mat);
    stars.name = 'starField';
    scene.add(stars);
}
"#,
    );

    scene.build_js()
}

/// Build the HTML content for the right-side dashboard panel.
fn build_side_panel_html(snapshot: &EmcSnapshot, events: &[AutoEvent], theme: &Theme) -> String {
    let mut html = String::with_capacity(4096);
    let accent = theme.accent_color();
    let text = theme.text_color();

    // Title bar
    let _ = write!(
        html,
        r#"<div style="text-align:center;margin-bottom:8px;">
<span style="color:{accent};font-size:14px;font-weight:700;letter-spacing:2px;">ICARUS EMC</span>
<span style="color:{text}66;font-size:11px;display:block;">Tick {tick} &middot; {layers} layers &middot; {sites} sites</span>
</div>"#,
        accent = accent,
        text = text,
        tick = snapshot.tick,
        layers = snapshot.layer_stats.len(),
        sites = snapshot.total_sites,
    );

    // System state section
    html.push_str(r#"<div class="section-header">System State</div>"#);

    let trend_label = snapshot
        .convergence_trend
        .as_ref()
        .map(|t| format!("{:?}", t))
        .unwrap_or_else(|| "Unknown".to_string());
    let trend_color = snapshot
        .convergence_trend
        .as_ref()
        .map(|t| {
            let (r, g, b) = convergence_color(t);
            rgb_to_hex(r, g, b)
        })
        .unwrap_or_else(|| "#888888".to_string());

    let _ = write!(
        html,
        r#"<div class="stat-row"><span class="key">Convergence</span><span class="val" style="color:{}">{}</span></div>"#,
        trend_color, trend_label
    );

    let aff = &snapshot.affective_state;
    let _ = write!(
        html,
        r#"<div class="stat-row"><span class="key">Valence</span><span class="val">{:+.3}</span></div>"#,
        aff.valence
    );
    let _ = write!(
        html,
        r#"<div class="stat-row"><span class="key">Arousal</span><span class="val">{:.3}</span></div>"#,
        aff.arousal
    );
    let _ = write!(
        html,
        r#"<div class="stat-row"><span class="key">Coherence</span><span class="val">{:.3}</span></div>"#,
        aff.phase_coherence
    );
    let _ = write!(
        html,
        r#"<div class="stat-row"><span class="key">Backend</span><span class="val">{}</span></div>"#,
        snapshot.backend_name
    );

    // Neuromodulator bars (CSS-based with inline values)
    html.push_str(r#"<div class="section-header">Neuromodulators</div>"#);
    let aether = &aff.aether;
    let neuros: [(&str, f32, &str); 4] = [
        ("DA", aether.dopamine, "#ffd700"),
        ("NE", aether.norepinephrine, "#ff4444"),
        ("ACh", aether.acetylcholine, "#00e5ff"),
        ("5HT", aether.serotonin, "#aa44ff"),
    ];
    for (label, val, color) in &neuros {
        let pct = (*val * 100.0).clamp(0.0, 100.0);
        let _ = write!(
            html,
            r#"<div class="neuro-bar">
<span class="bar-label">{label}</span>
<span class="bar-track"><span class="bar-fill" style="width:{pct:.0}%;background:{color}"></span></span>
<span class="bar-val" style="color:{color}">{val:.3}</span>
</div>"#,
            label = label,
            pct = pct,
            color = color,
            val = val,
        );
    }

    // 3D neuromodulator orbs canvas placeholder
    html.push_str(r#"<div class="section-header">Neuro Orbs (3D)</div>"#);

    // Layer energy breakdown
    html.push_str(r#"<div class="section-header">Layer Energy</div>"#);
    let layer_names = ["E8", "Leech", "HCP", "Hyper"];
    let layer_colors = ["#00e5ff", "#ff00aa", "#ffd700", "#44ff88"];
    for (i, stats) in snapshot.layer_stats.iter().enumerate() {
        let name = layer_names.get(i).unwrap_or(&"L?");
        let color = layer_colors.get(i).unwrap_or(&"#ffffff");
        let _ = write!(
            html,
            r#"<div class="stat-row"><span class="key" style="color:{color}">{name}</span><span class="val">{:.4}</span></div>"#,
            stats.total_energy,
            color = color,
            name = name,
        );
    }

    // Recent events
    html.push_str(r#"<div class="section-header">Recent Events</div>"#);
    let recent_events: Vec<&AutoEvent> = events.iter().rev().take(8).collect();
    if recent_events.is_empty() {
        let _ = write!(
            html,
            r#"<div style="color:{}66;font-size:11px;">No events yet</div>"#,
            text
        );
    } else {
        for event in &recent_events {
            let (color, label) = match &event.event_type {
                AutoEventType::ConvergenceDetected { trend } => ("#00ff88", format!("Conv: {}", trend)),
                AutoEventType::AttractorTransition {
                    layer,
                    from_site,
                    to_site,
                } => (
                    "#ff00ff",
                    format!("Attr L{}:{}>{}", layer, from_site, to_site),
                ),
                AutoEventType::EnergyThreshold { energy } => {
                    ("#ff6600", format!("E<{:.2}", energy))
                }
                AutoEventType::TickMilestone { tick } => ("#ffffff44", format!("T{}", tick)),
                AutoEventType::Started => ("#00e5ff", "Started".to_string()),
                AutoEventType::Stopped { reason } => ("#ff4444", format!("Stop: {}", reason)),
                AutoEventType::Error { message } => {
                    ("#ff0000", format!("Err: {}", &message[..message.len().min(20)]))
                }
            };
            let _ = write!(
                html,
                r#"<div style="font-size:10px;margin:1px 0;"><span style="color:{}">&#9679;</span> <span style="color:{}88">t{}</span> {}</div>"#,
                color, text, event.tick, label
            );
        }
    }

    // Time series canvas placeholder
    html.push_str(r#"<div class="section-header">Time Series</div>"#);

    html
}

/// Build JavaScript for the three side-panel canvases:
/// 1. neuro-orbs-canvas: 3D-looking neuromodulator orbs via radial gradients
/// 2. energy-bars-canvas: per-layer energy bar chart
/// 3. mini-timeseries: compact multi-series line chart
fn build_side_panel_charts_js(
    snapshot: &EmcSnapshot,
    events: &[AutoEvent],
    theme: &Theme,
) -> String {
    let mut js = String::with_capacity(8192);
    let aether = &snapshot.affective_state.aether;
    let bg = theme.panel_bg();
    let text = theme.text_color();
    let accent = theme.accent_color();

    // === Neuro Orbs Canvas ===
    let _ = write!(
        js,
        r#"
// Neuromodulator 3D Orbs
{{
    const c = document.getElementById('neuro-orbs-canvas');
    if (c) {{
        const ctx = c.getContext('2d');
        const w = c.width, h = c.height;
        ctx.fillStyle = '{bg}';
        ctx.fillRect(0, 0, w, h);

        function drawOrb(cx, cy, r, value, baseR, baseG, baseB, label) {{
            // Outer glow
            const glow = ctx.createRadialGradient(cx, cy, r * 0.2, cx, cy, r * 2.0);
            const a = Math.min(value, 1.0) * 0.3;
            glow.addColorStop(0, 'rgba(' + baseR + ',' + baseG + ',' + baseB + ',' + a + ')');
            glow.addColorStop(1, 'rgba(' + baseR + ',' + baseG + ',' + baseB + ',0)');
            ctx.fillStyle = glow;
            ctx.fillRect(cx - r * 2, cy - r * 2, r * 4, r * 4);

            // Sphere body with 3D shading
            const grad = ctx.createRadialGradient(cx - r * 0.3, cy - r * 0.3, r * 0.05, cx, cy, r);
            const intensity = Math.min(value, 1.0);
            grad.addColorStop(0, 'rgba(' + Math.min(255, baseR + 80) + ',' + Math.min(255, baseG + 80) + ',' + Math.min(255, baseB + 80) + ',1)');
            grad.addColorStop(0.5, 'rgba(' + Math.floor(baseR * intensity) + ',' + Math.floor(baseG * intensity) + ',' + Math.floor(baseB * intensity) + ',1)');
            grad.addColorStop(1, 'rgba(' + Math.floor(baseR * 0.2) + ',' + Math.floor(baseG * 0.2) + ',' + Math.floor(baseB * 0.2) + ',1)');
            ctx.beginPath();
            ctx.arc(cx, cy, r, 0, Math.PI * 2);
            ctx.fillStyle = grad;
            ctx.fill();

            // Specular highlight
            const spec = ctx.createRadialGradient(cx - r * 0.25, cy - r * 0.25, 0, cx - r * 0.25, cy - r * 0.25, r * 0.5);
            spec.addColorStop(0, 'rgba(255,255,255,0.6)');
            spec.addColorStop(1, 'rgba(255,255,255,0)');
            ctx.beginPath();
            ctx.arc(cx, cy, r, 0, Math.PI * 2);
            ctx.fillStyle = spec;
            ctx.fill();

            // Value text
            ctx.fillStyle = 'rgba(' + baseR + ',' + baseG + ',' + baseB + ',1)';
            ctx.font = 'bold 11px monospace';
            ctx.textAlign = 'center';
            ctx.fillText(value.toFixed(3), cx, cy + r + 14);

            // Label
            ctx.fillStyle = '{text}88';
            ctx.font = '9px monospace';
            ctx.fillText(label, cx, cy + r + 25);
        }}

        const r = 28;
        const gapX = w / 4;
        drawOrb(gapX * 0.5 + 5, 50, r, {da}, 255, 215, 0, 'DA');
        drawOrb(gapX * 1.5 + 5, 50, r, {ne}, 255, 68, 68, 'NE');
        drawOrb(gapX * 2.5 + 5, 50, r, {ach}, 0, 229, 255, 'ACh');
        drawOrb(gapX * 3.5 - 5, 50, r, {sht}, 170, 68, 255, '5-HT');

        // Valence/Arousal compass at bottom
        const compX = w / 2, compY = h - 40, compR = 30;
        ctx.beginPath();
        ctx.arc(compX, compY, compR, 0, Math.PI * 2);
        ctx.strokeStyle = '{accent}44';
        ctx.lineWidth = 1;
        ctx.stroke();
        // Crosshairs
        ctx.beginPath();
        ctx.moveTo(compX - compR, compY); ctx.lineTo(compX + compR, compY);
        ctx.moveTo(compX, compY - compR); ctx.lineTo(compX, compY + compR);
        ctx.strokeStyle = '{accent}22';
        ctx.stroke();
        // V/A dot
        const vx = Math.max(-1, Math.min(1, {valence}));
        const ay = Math.max(-1, Math.min(1, {arousal}));
        ctx.beginPath();
        ctx.arc(compX + vx * compR * 0.85, compY - ay * compR * 0.85, 4, 0, Math.PI * 2);
        ctx.fillStyle = '{accent}';
        ctx.shadowColor = '{accent}';
        ctx.shadowBlur = 8;
        ctx.fill();
        ctx.shadowBlur = 0;
        // Labels
        ctx.fillStyle = '{text}66';
        ctx.font = '8px monospace';
        ctx.textAlign = 'center';
        ctx.fillText('V+', compX + compR + 8, compY + 3);
        ctx.fillText('A+', compX, compY - compR - 4);
        ctx.fillText('Affect', compX, compY + compR + 12);
    }}
}}
"#,
        bg = bg,
        text = text,
        accent = accent,
        da = aether.dopamine,
        ne = aether.norepinephrine,
        ach = aether.acetylcholine,
        sht = aether.serotonin,
        valence = snapshot.affective_state.valence,
        arousal = snapshot.affective_state.arousal,
    );

    // === Energy Bars Canvas ===
    let _ = write!(
        js,
        r#"
// Per-Layer Energy Bars
{{
    const c = document.getElementById('energy-bars-canvas');
    if (c) {{
        const ctx = c.getContext('2d');
        const w = c.width, h = c.height;
        ctx.fillStyle = '{bg}';
        ctx.fillRect(0, 0, w, h);
        const layers = [{layers_json}];
        const colors = ['#00e5ff','#ff00aa','#ffd700','#44ff88'];
        const names = ['E8','Leech','HCP','Hyper'];
        const maxE = Math.max(...layers.map(l => l), 0.001);
        const barH = (h - 20) / Math.max(layers.length, 1) - 4;
        for (let i = 0; i < layers.length; i++) {{
            const y = 10 + i * (barH + 4);
            const barW = (layers[i] / maxE) * (w - 80);
            // Gradient bar
            const grad = ctx.createLinearGradient(50, y, 50 + barW, y);
            grad.addColorStop(0, colors[i] || '#fff');
            grad.addColorStop(1, (colors[i] || '#fff') + '44');
            ctx.fillStyle = grad;
            ctx.fillRect(50, y, barW, barH);
            // Glow
            ctx.shadowColor = colors[i] || '#fff';
            ctx.shadowBlur = 6;
            ctx.fillRect(50, y, barW, barH);
            ctx.shadowBlur = 0;
            // Label
            ctx.fillStyle = '{text}';
            ctx.font = '10px monospace';
            ctx.textAlign = 'right';
            ctx.fillText(names[i] || ('L' + i), 45, y + barH * 0.7);
            ctx.textAlign = 'left';
            ctx.fillStyle = colors[i] || '#fff';
            ctx.fillText(layers[i].toFixed(3), 52 + barW + 4, y + barH * 0.7);
        }}
    }}
}}
"#,
        bg = bg,
        text = text,
        layers_json = snapshot
            .layer_stats
            .iter()
            .map(|s| format!("{:.6}", s.total_energy))
            .collect::<Vec<_>>()
            .join(","),
    );

    // === Mini Time Series ===
    let total_energy: f32 = snapshot.layer_stats.iter().map(|s| s.total_energy).sum();
    let mean_amp: f32 = if snapshot.layer_stats.is_empty() {
        0.0
    } else {
        snapshot.layer_stats.iter().map(|s| s.mean_amplitude).sum::<f32>()
            / snapshot.layer_stats.len() as f32
    };
    let coherence = snapshot.affective_state.phase_coherence;
    let valence = snapshot.affective_state.valence;
    let tick = snapshot.tick;

    let num_events = events.len().min(200);

    let _ = write!(
        js,
        r#"
// Mini Time Series
{{
    const c = document.getElementById('mini-timeseries');
    if (c) {{
        const ctx = c.getContext('2d');
        const w = c.width, h = c.height;
        const pad = {{ top: 15, right: 8, bottom: 20, left: 35 }};
        const plotW = w - pad.left - pad.right;
        const plotH = h - pad.top - pad.bottom;

        ctx.fillStyle = '{bg}';
        ctx.fillRect(0, 0, w, h);

        // Grid
        ctx.strokeStyle = '{text}12';
        ctx.lineWidth = 1;
        for (let i = 0; i <= 4; i++) {{
            const y = pad.top + (plotH / 4) * i;
            ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(w - pad.right, y); ctx.stroke();
        }}

        // Simulated historical traces
        const series = [
            {{ val: {energy}, color: '#ff6600', label: 'E' }},
            {{ val: {amp}, color: '#00ff88', label: '|ψ|' }},
            {{ val: {coherence}, color: '#00e5ff', label: 'Coh' }},
            {{ val: {valence_shifted}, color: '#ff00ff', label: 'Val' }}
        ];

        series.forEach((s, idx) => {{
            ctx.strokeStyle = s.color;
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            const seed = idx * 23 + 7;
            for (let i = 0; i <= 80; i++) {{
                const t = i / 80;
                const noise = Math.sin(t * 41.7 + seed) * 0.12 + Math.sin(t * 17.3 + seed * 1.7) * 0.08;
                const ramp = 0.3 + 0.7 * (1 - Math.exp(-t * 3));
                const y_val = Math.max(0, Math.min(1, (s.val * ramp + noise) * 0.75 + 0.12));
                const x = pad.left + t * plotW;
                const y = pad.top + plotH * (1 - y_val);
                if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
            }}
            ctx.stroke();
            // Current value dot
            ctx.beginPath();
            const curY = pad.top + plotH * (1 - Math.min(1, Math.max(0, s.val * 0.75 + 0.12)));
            ctx.arc(pad.left + plotW, curY, 3, 0, Math.PI * 2);
            ctx.fillStyle = s.color;
            ctx.fill();
        }});
"#,
        bg = bg,
        text = text,
        energy = (total_energy / total_energy.max(1.0)).clamp(0.0, 1.0),
        amp = mean_amp.clamp(0.0, 1.0),
        coherence = coherence.clamp(0.0, 1.0),
        valence_shifted = ((valence + 1.0) * 0.5).clamp(0.0, 1.0),
    );

    // Event markers on timeseries
    let max_tick = snapshot.tick.max(1);
    for event in events.iter().rev().take(num_events) {
        let color = match &event.event_type {
            AutoEventType::ConvergenceDetected { .. } => "#00ff88",
            AutoEventType::AttractorTransition { .. } => "#ff00ff",
            AutoEventType::EnergyThreshold { .. } => "#ff6600",
            AutoEventType::TickMilestone { .. } => "#ffffff22",
            AutoEventType::Started => "#00e5ff",
            AutoEventType::Stopped { .. } => "#ff4444",
            AutoEventType::Error { .. } => "#ff0000",
        };
        let x_frac = event.tick as f64 / max_tick as f64;
        let _ = write!(
            js,
            "        ctx.beginPath(); ctx.moveTo(pad.left+{xf}*plotW, pad.top); ctx.lineTo(pad.left+{xf}*plotW, pad.top+plotH); ctx.strokeStyle='{c}'; ctx.lineWidth=1; ctx.setLineDash([2,2]); ctx.stroke(); ctx.setLineDash([]);\n",
            xf = x_frac,
            c = color,
        );
    }

    // Axis labels and legend
    let _ = write!(
        js,
        r#"
        // Axes
        ctx.fillStyle = '{text}66';
        ctx.font = '8px monospace';
        ctx.textAlign = 'right';
        for (let i = 0; i <= 4; i++) {{
            const y = pad.top + (plotH / 4) * i;
            ctx.fillText((1 - i / 4).toFixed(1), pad.left - 3, y + 3);
        }}
        ctx.textAlign = 'center';
        ctx.fillText('0', pad.left, h - 4);
        ctx.fillText('t{tick}', pad.left + plotW, h - 4);

        // Legend
        ctx.textAlign = 'left';
        ctx.font = '8px monospace';
        series.forEach((s, i) => {{
            ctx.fillStyle = s.color;
            ctx.fillText(s.label, pad.left + i * 55, h - 4);
        }});
    }}
}}
"#,
        text = text,
        tick = tick,
    );

    js
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_render_combined_dashboard_empty_snapshot() {
        let snapshot = EmcSnapshot {
            tick: 0,
            timestamp: String::new(),
            layer_stats: Vec::new(),
            total_sites: 0,
            backend_name: "test".to_string(),
            memory_bytes: 0,
            affective_state: icarus_field::autopoiesis::AffectiveState::baseline(),
            convergence_trend: None,
            action_output: None,
            layer_states: Vec::new(),
        };
        let html = render_combined_dashboard(&snapshot, &[], Theme::Dark);
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("ICARUS EMC"));
    }
}
