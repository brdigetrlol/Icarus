//! 3D Lattice Field Renderer — the hero visualization.
//!
//! Procedurally generates an interactive 3D point cloud from EMC layer state:
//! - Sphere size proportional to amplitude
//! - Color mapped from phase, amplitude, or energy
//! - Edges connect nearest neighbors in projected space
//! - Emissive glow on high-amplitude sites
//! - HUD panels show layer stats, convergence, and color legend

use std::fmt::Write;

use icarus_engine::autonomous::EmcSnapshot;

use icarus_math::lattice::LatticeLayer;

use crate::color::{
    amplitude_color, energy_color, phase_color, convergence_color,
    rgb_to_hex,
};
use crate::projection::{project_e8_to_3d, project_high_dim_to_3d, find_edges};
use crate::scene::{SceneBuilder, pulse_animation_js, sphere_breathe_js};
use crate::template::{HtmlDocument, PanelPosition, Theme};

/// Color mapping mode for lattice sites.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorMode {
    /// Blue→cyan→green→yellow→red based on |ψ|
    Amplitude,
    /// HSL hue wheel based on arg(ψ)
    Phase,
    /// Deep blue→magenta→white based on site energy
    Energy,
}

impl ColorMode {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "phase" => Self::Phase,
            "energy" => Self::Energy,
            _ => Self::Amplitude,
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::Amplitude => "Amplitude",
            Self::Phase => "Phase",
            Self::Energy => "Energy",
        }
    }
}

/// Configuration for the lattice field renderer.
pub struct LatticeFieldConfig {
    pub layer_index: usize,
    pub color_mode: ColorMode,
    pub show_edges: bool,
    pub edge_threshold: f64,
    pub theme: Theme,
    pub coord_scale: f64,
}

impl Default for LatticeFieldConfig {
    fn default() -> Self {
        Self {
            layer_index: 0,
            color_mode: ColorMode::Amplitude,
            show_edges: true,
            edge_threshold: 3.5,
            theme: Theme::Dark,
            coord_scale: 0.5,
        }
    }
}

/// Render a single layer's field state as an interactive 3D lattice.
///
/// Returns a self-contained HTML string with Three.js WebGL.
pub fn render_lattice_field(snapshot: &EmcSnapshot, config: &LatticeFieldConfig) -> String {
    let layer_state = snapshot.layer_states
        .get(config.layer_index)
        .expect("Layer index out of range");

    let layer_stats = snapshot.layer_stats
        .get(config.layer_index);

    let num_sites = layer_state.values_re.len();
    let layer_name = layer_label(&layer_state.layer);
    let dim = layer_dimension(&layer_state.layer);

    // Compute amplitudes and phases for all sites
    let mut amplitudes = Vec::with_capacity(num_sites);
    let mut phases = Vec::with_capacity(num_sites);
    let mut max_amp: f32 = 0.0;

    for i in 0..num_sites {
        let re = layer_state.values_re[i];
        let im = layer_state.values_im[i];
        let amp = (re * re + im * im).sqrt();
        let phase = im.atan2(re);
        amplitudes.push(amp);
        phases.push(phase);
        if amp > max_amp {
            max_amp = amp;
        }
    }

    // Avoid division by zero
    if max_amp < 1e-12 {
        max_amp = 1.0;
    }

    // Generate lattice coordinates and project to 3D
    let coords = generate_lattice_coords(&layer_state.layer, num_sites, dim);
    let projected: Vec<[f64; 3]> = coords.iter()
        .map(|c| {
            if dim == 8 {
                project_e8_to_3d(c, config.coord_scale)
            } else {
                project_high_dim_to_3d(c, config.coord_scale)
            }
        })
        .collect();

    // Build the 3D scene procedurally
    let mut scene = SceneBuilder::new();
    scene.set_camera([0.0, 4.0, 10.0], [0.0, 0.0, 0.0]);
    scene.add_point_light([5.0, 8.0, 5.0], 0x00e5ff, 0.8, 30.0);
    scene.add_point_light([-5.0, 6.0, -5.0], 0xff00aa, 0.5, 25.0);

    // Add lattice sites as spheres — size and color procedurally from field state
    for i in 0..num_sites {
        let t = amplitudes[i] / max_amp;
        let color = match config.color_mode {
            ColorMode::Amplitude => amplitude_color(t),
            ColorMode::Phase => phase_color(phases[i]),
            ColorMode::Energy => {
                let site_energy = amplitudes[i] * amplitudes[i];
                let energy_max = max_amp * max_amp;
                energy_color(if energy_max > 0.0 { site_energy / energy_max } else { 0.0 })
            }
        };

        let radius = 0.06 + 0.14 * t as f64;
        let emissive_strength = (t * 0.6).min(0.5);
        let emissive = (
            color.0 * emissive_strength,
            color.1 * emissive_strength,
            color.2 * emissive_strength,
        );

        scene.add_sphere(projected[i], radius, color, emissive, 0.95);
    }

    // Find and add edges (nearest neighbors in projected space)
    if config.show_edges && num_sites > 1 {
        let edges = find_edges(&projected, config.edge_threshold);
        for (i, j) in &edges {
            let t_i = amplitudes[*i] / max_amp;
            let t_j = amplitudes[*j] / max_amp;
            let avg_t = (t_i + t_j) * 0.5;
            let edge_color = (0.15 + 0.15 * avg_t, 0.2 + 0.3 * avg_t, 0.4 + 0.2 * avg_t);
            scene.add_edge(projected[*i], projected[*j], edge_color, 0.25);
        }
    }

    // Build HUD content
    let stats_html = build_stats_panel(snapshot, layer_stats, layer_name, num_sites, &config.color_mode);
    let legend_html = build_color_legend(&config.color_mode, &config.theme);
    let convergence_html = build_convergence_panel(snapshot);

    // Assemble HTML document
    let title = format!("Icarus - {} Field (tick {})", layer_name, snapshot.tick);
    let mut doc = HtmlDocument::new(&title, config.theme.clone());

    doc.add_panel_with_width("stats-panel", PanelPosition::TopLeft, &stats_html, "260px");
    doc.add_panel_with_width("legend-panel", PanelPosition::BottomLeft, &legend_html, "200px");
    doc.add_panel_with_width("convergence-panel", PanelPosition::TopRight, &convergence_html, "220px");

    doc.set_scene_js(&scene.build_js());

    // Procedural animation: pulse + breathe based on convergence state
    let mut anim = sphere_breathe_js();
    anim.push_str(&pulse_animation_js());
    doc.set_animation_js(&anim);

    doc.render()
}

/// Generate integer lattice coordinates for a given layer.
pub(crate) fn generate_lattice_coords(layer: &LatticeLayer, num_sites: usize, dim: usize) -> Vec<Vec<i64>> {
    match layer {
        LatticeLayer::Analytical => {
            // E8: origin + 240 root vectors (use the standard E8 root system)
            let mut coords = Vec::with_capacity(num_sites);
            // Site 0: origin
            coords.push(vec![0i64; 8]);
            // Sites 1..=240: E8 root vectors
            // Type 1: all permutations of (±1, ±1, 0, 0, 0, 0, 0, 0) — 112 roots
            for i in 0..8 {
                for j in (i + 1)..8 {
                    for si in [-1i64, 1] {
                        for sj in [-1i64, 1] {
                            let mut v = vec![0i64; 8];
                            v[i] = si * 2; // doubled coordinates
                            v[j] = sj * 2;
                            coords.push(v);
                            if coords.len() >= num_sites {
                                return coords;
                            }
                        }
                    }
                }
            }
            // Type 2: (±1, ±1, ±1, ±1, ±1, ±1, ±1, ±1) with even number of minus signs — 128 roots
            for bits in 0u32..256 {
                if bits.count_ones() % 2 != 0 {
                    continue;
                }
                let mut v = vec![1i64; 8];
                for k in 0..8 {
                    if bits & (1 << k) != 0 {
                        v[k] = -1;
                    }
                }
                coords.push(v);
                if coords.len() >= num_sites {
                    return coords;
                }
            }
            coords.truncate(num_sites);
            coords
        }
        _ => {
            // For non-E8 layers, generate a simple star-like pattern
            // that captures the lattice structure visually
            let mut coords = Vec::with_capacity(num_sites);
            coords.push(vec![0i64; dim]);

            // Generate axis-aligned neighbors at distance 1
            for d in 0..dim {
                for sign in [-1i64, 1] {
                    let mut v = vec![0i64; dim];
                    v[d] = sign;
                    coords.push(v);
                    if coords.len() >= num_sites {
                        return coords;
                    }
                }
            }

            // Generate diagonal neighbors
            for d1 in 0..dim.min(16) {
                for d2 in (d1 + 1)..dim.min(16) {
                    for s1 in [-1i64, 1] {
                        for s2 in [-1i64, 1] {
                            let mut v = vec![0i64; dim];
                            v[d1] = s1;
                            v[d2] = s2;
                            coords.push(v);
                            if coords.len() >= num_sites {
                                return coords;
                            }
                        }
                    }
                }
            }

            // Fill remaining with random-ish coords if needed
            let phi = 1.618033988749895_f64;
            let mut idx = coords.len();
            while coords.len() < num_sites {
                let mut v = vec![0i64; dim];
                for d in 0..dim.min(8) {
                    let val = ((idx as f64 * phi * (d + 1) as f64) % 5.0 - 2.0) as i64;
                    v[d] = val;
                }
                coords.push(v);
                idx += 1;
            }

            coords
        }
    }
}

fn layer_label(layer: &LatticeLayer) -> &'static str {
    match layer {
        LatticeLayer::Analytical => "E8 Analytical",
        LatticeLayer::Creative => "Leech Creative",
        LatticeLayer::Associative => "HCP Associative",
        LatticeLayer::Sensory => "Hypercubic Sensory",
    }
}

fn layer_dimension(layer: &LatticeLayer) -> usize {
    match layer {
        LatticeLayer::Analytical => 8,
        LatticeLayer::Creative => 24,
        LatticeLayer::Associative => 64,
        LatticeLayer::Sensory => 1024,
    }
}

fn build_stats_panel(
    snapshot: &EmcSnapshot,
    layer_stats: Option<&icarus_engine::manifold::LayerStats>,
    layer_name: &str,
    num_sites: usize,
    color_mode: &ColorMode,
) -> String {
    let mut html = String::new();
    let _ = write!(html, "<h3>{}</h3>", layer_name);

    let _ = write!(html, r#"<div class="stat-row"><span class="key">Tick</span><span class="val">{}</span></div>"#, snapshot.tick);
    let _ = write!(html, r#"<div class="stat-row"><span class="key">Sites</span><span class="val">{}</span></div>"#, num_sites);
    let _ = write!(html, r#"<div class="stat-row"><span class="key">Color</span><span class="val">{}</span></div>"#, color_mode.label());

    if let Some(stats) = layer_stats {
        let _ = write!(html, r#"<div class="stat-row"><span class="key">Energy</span><span class="val">{:.4}</span></div>"#, stats.total_energy);
        let _ = write!(html, r#"<div class="stat-row"><span class="key">Mean |ψ|</span><span class="val">{:.4}</span></div>"#, stats.mean_amplitude);
        let _ = write!(html, r#"<div class="stat-row"><span class="key">Kinetic</span><span class="val">{:.4}</span></div>"#, stats.kinetic_energy);
        let _ = write!(html, r#"<div class="stat-row"><span class="key">Potential</span><span class="val">{:.4}</span></div>"#, stats.potential_energy);
    }

    let _ = write!(html, r#"<div class="stat-row"><span class="key">Backend</span><span class="val">{}</span></div>"#, snapshot.backend_name);

    html
}

fn build_color_legend(color_mode: &ColorMode, _theme: &Theme) -> String {
    let mut html = String::new();
    html.push_str("<h3>Color Legend</h3>");

    let entries: Vec<(&str, &str)> = match color_mode {
        ColorMode::Amplitude => vec![
            ("Low |ψ|", "#0000ff"),
            ("Mid-low", "#00ffff"),
            ("Mid", "#00ff00"),
            ("Mid-high", "#ffff00"),
            ("High |ψ|", "#ff0000"),
        ],
        ColorMode::Phase => vec![
            ("0°", "#ff0000"),
            ("72°", "#ccff00"),
            ("144°", "#00ff66"),
            ("216°", "#0066ff"),
            ("288°", "#cc00ff"),
        ],
        ColorMode::Energy => vec![
            ("Low E", "#000033"),
            ("Mid-low", "#3300cc"),
            ("Mid", "#cc00ff"),
            ("Mid-high", "#ff4dff"),
            ("High E", "#ffffff"),
        ],
    };

    for (label, color) in &entries {
        let _ = write!(html,
            r#"<div class="color-legend"><span class="swatch" style="background:{color}"></span><span>{label}</span></div>"#,
            color = color, label = label,
        );
    }

    html
}

fn build_convergence_panel(snapshot: &EmcSnapshot) -> String {
    let mut html = String::new();
    html.push_str("<h3>System State</h3>");

    // Convergence trend
    let trend_label = snapshot.convergence_trend
        .as_ref()
        .map(|t| format!("{:?}", t))
        .unwrap_or_else(|| "Unknown".to_string());
    let trend_color = snapshot.convergence_trend
        .as_ref()
        .map(|t| {
            let (r, g, b) = convergence_color(t);
            rgb_to_hex(r, g, b)
        })
        .unwrap_or_else(|| "#888888".to_string());

    let _ = write!(html,
        r#"<div class="stat-row"><span class="key">Trend</span><span class="val" style="color:{}">{}</span></div>"#,
        trend_color, trend_label
    );

    // Affective state
    let aff = &snapshot.affective_state;
    let _ = write!(html, r#"<div class="stat-row"><span class="key">Valence</span><span class="val">{:+.3}</span></div>"#, aff.valence);
    let _ = write!(html, r#"<div class="stat-row"><span class="key">Arousal</span><span class="val">{:.3}</span></div>"#, aff.arousal);
    let _ = write!(html, r#"<div class="stat-row"><span class="key">Coherence</span><span class="val">{:.3}</span></div>"#, aff.phase_coherence);

    // Neuromodulators
    html.push_str("<h3 style=\"margin-top:8px\">Neuromodulators</h3>");
    let aether = &aff.aether;
    let _ = write!(html, r#"<div class="stat-row"><span class="key">DA</span><span class="val" style="color:#ffd700">{:.3}</span></div>"#, aether.dopamine);
    let _ = write!(html, r#"<div class="stat-row"><span class="key">NE</span><span class="val" style="color:#ff4444">{:.3}</span></div>"#, aether.norepinephrine);
    let _ = write!(html, r#"<div class="stat-row"><span class="key">ACh</span><span class="val" style="color:#00e5ff">{:.3}</span></div>"#, aether.acetylcholine);
    let _ = write!(html, r#"<div class="stat-row"><span class="key">5-HT</span><span class="val" style="color:#aa44ff">{:.3}</span></div>"#, aether.serotonin);

    html
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_e8_coords() {
        let coords = generate_lattice_coords(&LatticeLayer::Analytical, 241, 8);
        assert_eq!(coords.len(), 241);
        // Origin should be first
        assert!(coords[0].iter().all(|&c| c == 0));
    }

    #[test]
    fn test_layer_labels() {
        assert_eq!(layer_label(&LatticeLayer::Analytical), "E8 Analytical");
        assert_eq!(layer_label(&LatticeLayer::Creative), "Leech Creative");
    }

    #[test]
    fn test_color_mode_from_str() {
        assert_eq!(ColorMode::from_str("phase"), ColorMode::Phase);
        assert_eq!(ColorMode::from_str("energy"), ColorMode::Energy);
        assert_eq!(ColorMode::from_str("amplitude"), ColorMode::Amplitude);
        assert_eq!(ColorMode::from_str("unknown"), ColorMode::Amplitude);
    }

    #[test]
    fn test_generate_generic_coords() {
        let coords = generate_lattice_coords(&LatticeLayer::Sensory, 100, 1024);
        assert_eq!(coords.len(), 100);
    }
}
