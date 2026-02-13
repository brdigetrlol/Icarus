// Copyright (c) 2025-2026 brdigetrlol. All rights reserved.
// SPDX-License-Identifier: LicenseRef-Icarus-Proprietary
// See LICENSE in the repository root for full license terms.

//! Energy Landscape Renderer
//!
//! Procedurally generates a 3D height map from per-site energy values
//! projected over the lattice topology. Color gradient encodes energy
//! intensity from cool blue (low) to hot red (high).

use std::fmt::Write;

use icarus_engine::autonomous::EmcSnapshot;
use icarus_math::lattice::LatticeLayer;

use crate::color::energy_color;
use crate::projection::{project_e8_to_3d, project_high_dim_to_3d, find_edges};
use crate::scene::SceneBuilder;
use crate::template::{HtmlDocument, PanelPosition, Theme};

/// Render the energy landscape for a given layer.
///
/// Sites are positioned in projected 3D space with Y offset proportional
/// to their local energy (|ψ|²). Wireframe mesh connects neighbors.
pub fn render_energy_landscape(snapshot: &EmcSnapshot, layer_index: usize, theme: Theme) -> String {
    let layer_state = snapshot.layer_states
        .get(layer_index)
        .expect("Layer index out of range");
    let layer_stats = snapshot.layer_stats.get(layer_index);

    let num_sites = layer_state.values_re.len();
    let dim = match layer_state.layer {
        LatticeLayer::Analytical => 8,
        LatticeLayer::Creative => 24,
        LatticeLayer::Associative => 64,
        LatticeLayer::Sensory => 1024,
    };

    // Compute per-site energy (|ψ|²)
    let mut energies = Vec::with_capacity(num_sites);
    let mut max_energy: f32 = 0.0;
    for i in 0..num_sites {
        let re = layer_state.values_re[i];
        let im = layer_state.values_im[i];
        let e = re * re + im * im;
        energies.push(e);
        if e > max_energy {
            max_energy = e;
        }
    }
    if max_energy < 1e-12 {
        max_energy = 1.0;
    }

    // Generate coordinates and project
    let coords = super::lattice_field::generate_lattice_coords(&layer_state.layer, num_sites, dim);
    let base_projected: Vec<[f64; 3]> = coords.iter()
        .map(|c| {
            if dim == 8 { project_e8_to_3d(c, 0.5) }
            else { project_high_dim_to_3d(c, 0.5) }
        })
        .collect();

    // Offset Y by energy to create height map
    let height_scale = 4.0;
    let projected: Vec<[f64; 3]> = base_projected.iter().enumerate()
        .map(|(i, p)| {
            let t = energies[i] / max_energy;
            [p[0], p[1] + t as f64 * height_scale, p[2]]
        })
        .collect();

    // Build scene
    let mut scene = SceneBuilder::new();
    scene.set_camera([0.0, 8.0, 12.0], [0.0, 2.0, 0.0]);
    scene.add_point_light([5.0, 12.0, 5.0], 0xff6600, 1.0, 40.0);
    scene.add_point_light([-5.0, 8.0, -5.0], 0x0066ff, 0.6, 30.0);
    scene.set_grid_helper(true);

    // Add energy spheres — color and height procedurally from |ψ|²
    for i in 0..num_sites {
        let t = energies[i] / max_energy;
        let color = energy_color(t);
        let radius = 0.05 + 0.12 * t as f64;
        let emissive = (color.0 * t * 0.4, color.1 * t * 0.4, color.2 * t * 0.4);
        scene.add_sphere(projected[i], radius, color, emissive, 0.9);
    }

    // Wireframe edges connecting neighbors
    let edges = find_edges(&base_projected, 3.5);
    for (i, j) in &edges {
        let t_avg = (energies[*i] + energies[*j]) / (2.0 * max_energy);
        let edge_color = energy_color(t_avg * 0.5);
        scene.add_edge(projected[*i], projected[*j], edge_color, 0.2);
    }

    // HUD panels
    let layer_name = match layer_state.layer {
        LatticeLayer::Analytical => "E8",
        LatticeLayer::Creative => "Leech",
        LatticeLayer::Associative => "HCP",
        LatticeLayer::Sensory => "Hypercubic",
    };

    let mut stats_html = String::new();
    let _ = write!(stats_html, "<h3>Energy Landscape</h3>");
    let _ = write!(stats_html, r#"<div class="stat-row"><span class="key">Layer</span><span class="val">{}</span></div>"#, layer_name);
    let _ = write!(stats_html, r#"<div class="stat-row"><span class="key">Tick</span><span class="val">{}</span></div>"#, snapshot.tick);
    let _ = write!(stats_html, r#"<div class="stat-row"><span class="key">Sites</span><span class="val">{}</span></div>"#, num_sites);
    let _ = write!(stats_html, r#"<div class="stat-row"><span class="key">Max |ψ|²</span><span class="val">{:.4}</span></div>"#, max_energy);

    if let Some(stats) = layer_stats {
        let _ = write!(stats_html, r#"<div class="stat-row"><span class="key">Total E</span><span class="val">{:.4}</span></div>"#, stats.total_energy);
        let _ = write!(stats_html, r#"<div class="stat-row"><span class="key">Kinetic</span><span class="val">{:.4}</span></div>"#, stats.kinetic_energy);
        let _ = write!(stats_html, r#"<div class="stat-row"><span class="key">Potential</span><span class="val">{:.4}</span></div>"#, stats.potential_energy);
        let ratio = if stats.total_energy.abs() > 1e-12 {
            stats.kinetic_energy / stats.total_energy * 100.0
        } else { 0.0 };
        let _ = write!(stats_html, r#"<div class="stat-row"><span class="key">K/E ratio</span><span class="val">{:.1}%</span></div>"#, ratio);
    }

    let title = format!("Icarus - Energy Landscape (tick {})", snapshot.tick);
    let mut doc = HtmlDocument::new(&title, theme);
    doc.add_panel_with_width("stats-panel", PanelPosition::TopLeft, &stats_html, "240px");
    doc.set_scene_js(&scene.build_js());

    // Procedural animation: gentle wave through the height map
    doc.set_animation_js(r#"
    scene.children.forEach((obj, idx) => {
        if (obj.name && obj.name.startsWith('sphere_')) {
            const baseY = obj.userData.baseY || obj.position.y;
            if (!obj.userData.baseY) obj.userData.baseY = baseY;
            obj.position.y = baseY + 0.05 * Math.sin(time * 1.5 + idx * 0.3);
        }
    });
"#);

    doc.render()
}
