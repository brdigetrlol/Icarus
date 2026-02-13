//! Phase Coherence Visualizer
//!
//! Procedurally renders phase angles as colored arrows on lattice sites
//! and shows the Kuramoto order parameter as a central vector. A polar
//! phase histogram is rendered as an overlay chart.

use std::fmt::Write;

use icarus_engine::autonomous::EmcSnapshot;
use icarus_math::lattice::LatticeLayer;

use crate::color::{phase_color, rgb_to_threejs_hex};
use crate::projection::{project_e8_to_3d, project_high_dim_to_3d};
use crate::scene::SceneBuilder;
use crate::template::{HtmlDocument, PanelPosition, Theme};

/// Render a phase coherence visualization for a given layer.
///
/// Each lattice site shows a colored arrow indicating its phase angle.
/// The global Kuramoto order parameter is shown as a bright central arrow.
pub fn render_phase_portrait(snapshot: &EmcSnapshot, layer_index: usize, theme: Theme) -> String {
    let layer_state = snapshot.layer_states
        .get(layer_index)
        .expect("Layer index out of range");

    let num_sites = layer_state.values_re.len();
    let dim = match layer_state.layer {
        LatticeLayer::Analytical => 8,
        LatticeLayer::Creative => 24,
        LatticeLayer::Associative => 64,
        LatticeLayer::Sensory => 1024,
    };

    // Compute amplitudes and phases
    let mut phases = Vec::with_capacity(num_sites);
    let mut amplitudes = Vec::with_capacity(num_sites);
    let mut sum_re: f64 = 0.0;
    let mut sum_im: f64 = 0.0;
    let mut max_amp: f32 = 0.0;

    for i in 0..num_sites {
        let re = layer_state.values_re[i];
        let im = layer_state.values_im[i];
        let amp = (re * re + im * im).sqrt();
        let phase = im.atan2(re);
        phases.push(phase);
        amplitudes.push(amp);
        if amp > max_amp { max_amp = amp; }
        // Weighted Kuramoto sum
        sum_re += (phase as f64).cos() * amp as f64;
        sum_im += (phase as f64).sin() * amp as f64;
    }

    let total_amp: f64 = amplitudes.iter().map(|a| *a as f64).sum();
    let order_r = if total_amp > 1e-12 {
        (sum_re * sum_re + sum_im * sum_im).sqrt() / total_amp
    } else { 0.0 };
    let order_theta = sum_im.atan2(sum_re);
    if max_amp < 1e-12 { max_amp = 1.0; }

    // Phase histogram (8 bins)
    let mut histogram = [0u32; 8];
    for &phase in &phases {
        let normalized = (phase + std::f32::consts::PI) / (2.0 * std::f32::consts::PI);
        let bin = ((normalized * 8.0).floor() as usize).min(7);
        histogram[bin] += 1;
    }
    let hist_max = *histogram.iter().max().unwrap_or(&1);

    // Generate coordinates and project
    let coords = super::lattice_field::generate_lattice_coords(&layer_state.layer, num_sites, dim);
    let projected: Vec<[f64; 3]> = coords.iter()
        .map(|c| {
            if dim == 8 { project_e8_to_3d(c, 0.5) }
            else { project_high_dim_to_3d(c, 0.5) }
        })
        .collect();

    // Build scene
    let mut scene = SceneBuilder::new();
    scene.set_camera([0.0, 5.0, 10.0], [0.0, 0.0, 0.0]);
    scene.add_point_light([4.0, 8.0, 4.0], 0xffffff, 0.6, 30.0);

    // Phase-colored spheres at each site
    for i in 0..num_sites {
        let color = phase_color(phases[i]);
        let t = amplitudes[i] / max_amp;
        let radius = 0.04 + 0.1 * t as f64;
        // Desaturate when amplitude is low (less relevant phase)
        let desat = t.max(0.3);
        let adjusted = (
            color.0 * desat + (1.0 - desat) * 0.3,
            color.1 * desat + (1.0 - desat) * 0.3,
            color.2 * desat + (1.0 - desat) * 0.3,
        );
        scene.add_sphere(projected[i], radius, adjusted, (0.0, 0.0, 0.0), 0.85);
    }

    // Kuramoto order parameter arrow — a bright glowing sphere at center
    // with radius proportional to |R| (the order parameter magnitude)
    let arrow_radius = 0.3 * order_r;
    let arrow_color = phase_color(order_theta as f32);
    scene.add_sphere(
        [0.0, 0.0, 0.0],
        arrow_radius,
        arrow_color,
        (arrow_color.0 * 0.8, arrow_color.1 * 0.8, arrow_color.2 * 0.8),
        0.95,
    );

    // Add directional indicator: a line from origin toward order parameter direction
    let arrow_len = 2.0 * order_r;
    let ax = arrow_len * order_theta.cos();
    let az = arrow_len * order_theta.sin();
    scene.add_edge([0.0, 0.0, 0.0], [ax, 0.0, az], arrow_color, 0.8);

    // Custom JS: add an ArrowHelper for the order parameter
    let arrow_hex = rgb_to_threejs_hex(arrow_color.0, arrow_color.1, arrow_color.2);
    scene.add_custom_js(&format!(r#"
// Kuramoto order parameter arrow
{{
    const dir = new THREE.Vector3({ax}, 0, {az}).normalize();
    const arrow = new THREE.ArrowHelper(dir, new THREE.Vector3(0,0,0), {len}, {color}, 0.3, 0.15);
    arrow.name = 'orderArrow';
    scene.add(arrow);
}}
"#, ax = ax, az = az, len = arrow_len, color = arrow_hex));

    // Build phase histogram as canvas chart (inline JS)
    let mut chart_js = String::new();
    let _ = write!(chart_js, r#"
// Phase histogram (polar-ish bar chart)
{{
    const canvas = document.getElementById('phase-hist');
    if (canvas) {{
        const ctx = canvas.getContext('2d');
        const w = canvas.width, h = canvas.height;
        ctx.fillStyle = 'rgba(10,10,20,0.8)';
        ctx.fillRect(0, 0, w, h);
        const bins = [{b0},{b1},{b2},{b3},{b4},{b5},{b6},{b7}];
        const maxVal = {hist_max};
        const barW = w / 8 - 4;
        const colors = ['#ff0000','#ff8800','#ffff00','#00ff00','#00ffff','#0088ff','#8800ff','#ff00ff'];
        for (let i = 0; i < 8; i++) {{
            const barH = (bins[i] / maxVal) * (h - 30);
            ctx.fillStyle = colors[i];
            ctx.fillRect(i * (barW + 4) + 2, h - barH - 15, barW, barH);
            ctx.fillStyle = '#888';
            ctx.font = '9px monospace';
            ctx.fillText((i * 45) + '°', i * (barW + 4) + 2, h - 2);
        }}
        ctx.fillStyle = '#00e5ff';
        ctx.font = '10px monospace';
        ctx.fillText('Phase Distribution', 4, 12);
    }}
}}
"#,
        b0 = histogram[0], b1 = histogram[1], b2 = histogram[2], b3 = histogram[3],
        b4 = histogram[4], b5 = histogram[5], b6 = histogram[6], b7 = histogram[7],
        hist_max = hist_max,
    );

    // HUD
    let mut stats_html = String::new();
    let _ = write!(stats_html, "<h3>Phase Coherence</h3>");
    let _ = write!(stats_html, r#"<div class="stat-row"><span class="key">Tick</span><span class="val">{}</span></div>"#, snapshot.tick);
    let _ = write!(stats_html, r#"<div class="stat-row"><span class="key">Order |R|</span><span class="val">{:.4}</span></div>"#, order_r);
    let _ = write!(stats_html, r#"<div class="stat-row"><span class="key">Order θ</span><span class="val">{:.2}°</span></div>"#, order_theta.to_degrees());
    let _ = write!(stats_html, r#"<div class="stat-row"><span class="key">Coherence</span><span class="val">{:.3}</span></div>"#, snapshot.affective_state.phase_coherence);
    let _ = write!(stats_html, r#"<div class="stat-row"><span class="key">Sites</span><span class="val">{}</span></div>"#, num_sites);

    let title = format!("Icarus - Phase Portrait (tick {})", snapshot.tick);
    let mut doc = HtmlDocument::new(&title, theme);
    doc.add_panel_with_width("stats-panel", PanelPosition::TopLeft, &stats_html, "220px");
    doc.add_chart("phase-hist", 280, 120);

    doc.set_extra_css(r#"
#phase-hist {
    position: absolute;
    bottom: 10px;
    left: 10px;
    border: 1px solid #00e5ff33;
    border-radius: 8px;
}
"#);

    doc.set_scene_js(&scene.build_js());
    doc.set_chart_js(&chart_js);

    // Animate: gently rotate order arrow
    doc.set_animation_js(r#"
    const arrow = scene.getObjectByName('orderArrow');
    if (arrow) {
        arrow.rotation.y = Math.sin(time * 0.3) * 0.1;
    }
"#);

    doc.render()
}
