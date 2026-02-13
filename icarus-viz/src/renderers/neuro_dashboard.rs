//! Neuromodulator Dashboard Renderer
//!
//! Procedurally generates radial gauge visualizations for the four
//! neuromodulators (DA, NE, ACh, 5-HT) plus valence/arousal compass.
//! All geometry is derived from `AffectiveState` values.

use std::fmt::Write;

use icarus_engine::autonomous::EmcSnapshot;

use crate::template::{HtmlDocument, PanelPosition, Theme};

/// Color assignments for each neuromodulator.
const DA_COLOR: &str = "#ffd700";   // Gold
const NE_COLOR: &str = "#ff4444";   // Red
const ACH_COLOR: &str = "#00e5ff";  // Cyan
const SEROTONIN_COLOR: &str = "#aa44ff"; // Violet

/// Render the neuromodulator dashboard.
///
/// Produces a 2D Canvas-based gauge cluster showing all four neuromodulators
/// and a valence/arousal indicator. All values procedurally drawn from state.
pub fn render_neuro_dashboard(snapshot: &EmcSnapshot, theme: Theme) -> String {
    let aff = &snapshot.affective_state;
    let aether = &aff.aether;

    // Build the canvas-based gauge visualization
    let mut chart_js = String::new();
    let _ = write!(chart_js, r#"
// Neuromodulator Gauges
{{
    const canvas = document.getElementById('neuro-gauges');
    if (canvas) {{
        const ctx = canvas.getContext('2d');
        const w = canvas.width, h = canvas.height;
        ctx.fillStyle = '{bg}';
        ctx.fillRect(0, 0, w, h);

        function drawGauge(cx, cy, radius, value, color, label) {{
            // Background arc
            ctx.beginPath();
            ctx.arc(cx, cy, radius, Math.PI * 0.75, Math.PI * 2.25, false);
            ctx.strokeStyle = color + '33';
            ctx.lineWidth = 8;
            ctx.stroke();

            // Value arc
            const sweep = Math.PI * 1.5 * Math.min(Math.max(value, 0), 1);
            ctx.beginPath();
            ctx.arc(cx, cy, radius, Math.PI * 0.75, Math.PI * 0.75 + sweep, false);
            ctx.strokeStyle = color;
            ctx.lineWidth = 8;
            ctx.lineCap = 'round';
            ctx.stroke();

            // Glow
            ctx.shadowColor = color;
            ctx.shadowBlur = 12;
            ctx.beginPath();
            ctx.arc(cx, cy, radius, Math.PI * 0.75, Math.PI * 0.75 + sweep, false);
            ctx.strokeStyle = color;
            ctx.lineWidth = 3;
            ctx.stroke();
            ctx.shadowBlur = 0;

            // Value text
            ctx.fillStyle = color;
            ctx.font = 'bold 16px monospace';
            ctx.textAlign = 'center';
            ctx.fillText(value.toFixed(3), cx, cy + 6);

            // Label
            ctx.fillStyle = '{text}';
            ctx.font = '11px monospace';
            ctx.fillText(label, cx, cy + radius + 18);
        }}

        // 4 gauges in a 2x2 grid
        const r = 45;
        const pad = 30;
        const startX = pad + r;
        const startY = pad + r;
        const gapX = (w - 2 * pad) / 2;
        const gapY = (h - 2 * pad - 50) / 2;

        drawGauge(startX,         startY,         r, {da},  '{da_c}',  'Dopamine');
        drawGauge(startX + gapX,  startY,         r, {ne},  '{ne_c}',  'Norepinephrine');
        drawGauge(startX,         startY + gapY,  r, {ach}, '{ach_c}', 'Acetylcholine');
        drawGauge(startX + gapX,  startY + gapY,  r, {sht}, '{sht_c}', 'Serotonin');

        // Valence/Arousal compass at bottom
        const compassY = h - 50;
        const compassX = w / 2;
        const compassR = 30;

        // Compass circle
        ctx.beginPath();
        ctx.arc(compassX, compassY, compassR, 0, Math.PI * 2);
        ctx.strokeStyle = '{accent}44';
        ctx.lineWidth = 1;
        ctx.stroke();

        // Crosshairs
        ctx.beginPath();
        ctx.moveTo(compassX - compassR, compassY);
        ctx.lineTo(compassX + compassR, compassY);
        ctx.moveTo(compassX, compassY - compassR);
        ctx.lineTo(compassX, compassY + compassR);
        ctx.strokeStyle = '{accent}22';
        ctx.lineWidth = 1;
        ctx.stroke();

        // V/A point
        const vx = Math.max(-1, Math.min(1, {valence}));
        const ay = Math.max(-1, Math.min(1, {arousal}));
        const px = compassX + vx * compassR * 0.9;
        const py = compassY - ay * compassR * 0.9;

        ctx.beginPath();
        ctx.arc(px, py, 4, 0, Math.PI * 2);
        ctx.fillStyle = '{accent}';
        ctx.fill();
        ctx.shadowColor = '{accent}';
        ctx.shadowBlur = 8;
        ctx.fill();
        ctx.shadowBlur = 0;

        // Labels
        ctx.fillStyle = '{text}88';
        ctx.font = '9px monospace';
        ctx.textAlign = 'center';
        ctx.fillText('V+', compassX + compassR + 10, compassY + 3);
        ctx.fillText('V-', compassX - compassR - 10, compassY + 3);
        ctx.fillText('A+', compassX, compassY - compassR - 5);
        ctx.fillText('A-', compassX, compassY + compassR + 12);
        ctx.fillStyle = '{accent}';
        ctx.fillText('Affect', compassX, compassY + compassR + 24);
    }}
}}
"#,
        bg = theme.panel_bg(),
        text = theme.text_color(),
        accent = theme.accent_color(),
        da = aether.dopamine,
        ne = aether.norepinephrine,
        ach = aether.acetylcholine,
        sht = aether.serotonin,
        da_c = DA_COLOR,
        ne_c = NE_COLOR,
        ach_c = ACH_COLOR,
        sht_c = SEROTONIN_COLOR,
        valence = aff.valence,
        arousal = aff.arousal,
    );

    // Stats panel
    let mut stats_html = String::new();
    let _ = write!(stats_html, "<h3>Affective State</h3>");
    let _ = write!(stats_html, r#"<div class="stat-row"><span class="key">Tick</span><span class="val">{}</span></div>"#, snapshot.tick);
    let _ = write!(stats_html, r#"<div class="stat-row"><span class="key">Valence</span><span class="val">{:+.3}</span></div>"#, aff.valence);
    let _ = write!(stats_html, r#"<div class="stat-row"><span class="key">Arousal</span><span class="val">{:.3}</span></div>"#, aff.arousal);
    let _ = write!(stats_html, r#"<div class="stat-row"><span class="key">Coherence</span><span class="val">{:.3}</span></div>"#, aff.phase_coherence);

    // Emotion quadrant
    let emotion = if aff.valence > 0.0 && aff.arousal > 0.5 {
        "Eureka"
    } else if aff.valence > 0.0 {
        "Joy/Relief"
    } else if aff.arousal > 0.5 {
        "Confusion"
    } else {
        "Sadness"
    };
    let _ = write!(stats_html, r#"<div class="stat-row"><span class="key">Emotion</span><span class="val">{}</span></div>"#, emotion);

    let title = format!("Icarus - Neuromodulator Dashboard (tick {})", snapshot.tick);
    let mut doc = HtmlDocument::new(&title, theme);
    doc.add_panel_with_width("stats-panel", PanelPosition::TopRight, &stats_html, "220px");
    doc.add_chart("neuro-gauges", 360, 340);

    doc.set_extra_css(r#"
#neuro-gauges {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    border: 1px solid #00e5ff33;
    border-radius: 12px;
}
"#);

    // Minimal 3D scene — just ambient particles for atmosphere
    doc.set_scene_js(r#"
camera.position.set(0, 2, 6);
scene.add(new THREE.AmbientLight(0x404060, 0.3));

// Atmospheric particle field
{
    const geo = new THREE.BufferGeometry();
    const n = 200;
    const positions = new Float32Array(n * 3);
    const colors = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
        positions[i*3] = (Math.random() - 0.5) * 20;
        positions[i*3+1] = (Math.random() - 0.5) * 20;
        positions[i*3+2] = (Math.random() - 0.5) * 20;
        const c = Math.random();
        colors[i*3] = c * 0.3;
        colors[i*3+1] = c * 0.5;
        colors[i*3+2] = c;
    }
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    const mat = new THREE.PointsMaterial({
        size: 0.08, vertexColors: true, transparent: true,
        opacity: 0.4, blending: THREE.AdditiveBlending, depthWrite: false
    });
    scene.add(new THREE.Points(geo, mat));
}
"#);

    doc.set_chart_js(&chart_js);
    doc.set_animation_js("");

    doc.render()
}
