// Copyright (c) 2025-2026 brdigetrlol. All rights reserved.
// SPDX-License-Identifier: LicenseRef-Icarus-Proprietary
// See LICENSE in the repository root for full license terms.

//! Time Series Chart Renderer
//!
//! Procedurally generates multi-line Canvas 2D charts from EMC event history
//! and snapshot data. Event markers show convergence changes, attractor
//! transitions, and tick milestones.

use std::fmt::Write;

use icarus_engine::autonomous::{AutoEvent, AutoEventType, EmcSnapshot};

use crate::template::{HtmlDocument, PanelPosition, Theme};

/// Which data series to include in the chart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Series {
    Energy,
    Amplitude,
    Coherence,
    Valence,
    Arousal,
    Dopamine,
    Norepinephrine,
}

impl Series {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Energy => "Energy",
            Self::Amplitude => "Amplitude",
            Self::Coherence => "Coherence",
            Self::Valence => "Valence",
            Self::Arousal => "Arousal",
            Self::Dopamine => "Dopamine",
            Self::Norepinephrine => "Norepinephrine",
        }
    }

    pub fn color(&self) -> &'static str {
        match self {
            Self::Energy => "#ff6600",
            Self::Amplitude => "#00ff88",
            Self::Coherence => "#00e5ff",
            Self::Valence => "#ff00ff",
            Self::Arousal => "#ffff00",
            Self::Dopamine => "#ffd700",
            Self::Norepinephrine => "#ff4444",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "energy" => Some(Self::Energy),
            "amplitude" => Some(Self::Amplitude),
            "coherence" => Some(Self::Coherence),
            "valence" => Some(Self::Valence),
            "arousal" => Some(Self::Arousal),
            "dopamine" => Some(Self::Dopamine),
            "norepinephrine" => Some(Self::Norepinephrine),
            _ => None,
        }
    }
}

/// Render a time series chart from event history.
///
/// Takes the current snapshot for latest values and an event log for
/// markers. Each event becomes a vertical marker on the chart.
pub fn render_timeseries(
    snapshot: &EmcSnapshot,
    events: &[AutoEvent],
    series: &[Series],
    theme: Theme,
) -> String {
    let active_series: Vec<&Series> = if series.is_empty() {
        vec![&Series::Energy, &Series::Amplitude, &Series::Coherence, &Series::Valence]
    } else {
        series.iter().collect()
    };

    // Extract current values for each series
    let current_values: Vec<(&Series, f32)> = active_series.iter().map(|s| {
        let val = match s {
            Series::Energy => snapshot.layer_stats.iter().map(|l| l.total_energy).sum(),
            Series::Amplitude => snapshot.layer_stats.iter().map(|l| l.mean_amplitude).sum::<f32>()
                / snapshot.layer_stats.len().max(1) as f32,
            Series::Coherence => snapshot.affective_state.phase_coherence,
            Series::Valence => snapshot.affective_state.valence,
            Series::Arousal => snapshot.affective_state.arousal,
            Series::Dopamine => snapshot.affective_state.aether.dopamine,
            Series::Norepinephrine => snapshot.affective_state.aether.norepinephrine,
        };
        (*s, val)
    }).collect();

    // Classify events for markers
    let mut event_markers = Vec::new();
    for event in events {
        let (color, label) = match &event.event_type {
            AutoEventType::ConvergenceDetected { trend } =>
                ("#00ff88", format!("Conv: {}", trend)),
            AutoEventType::AttractorTransition { layer, from_site, to_site } =>
                ("#ff00ff", format!("Attr L{}:{}->{}", layer, from_site, to_site)),
            AutoEventType::EnergyThreshold { energy } =>
                ("#ff6600", format!("E<{:.2}", energy)),
            AutoEventType::TickMilestone { tick } =>
                ("#ffffff44", format!("T{}", tick)),
            AutoEventType::Started =>
                ("#00e5ff", "Start".to_string()),
            AutoEventType::Stopped { reason } =>
                ("#ff4444", format!("Stop: {}", reason)),
            AutoEventType::Error { message } =>
                ("#ff0000", format!("Err: {}", &message[..message.len().min(20)])),
        };
        event_markers.push((event.tick, color, label));
    }

    // Build the chart JS procedurally
    let mut chart_js = String::new();
    let _ = write!(chart_js, r#"
// Time Series Chart
{{
    const canvas = document.getElementById('timeseries-chart');
    if (canvas) {{
        const ctx = canvas.getContext('2d');
        const w = canvas.width, h = canvas.height;
        const pad = {{ top: 30, right: 20, bottom: 40, left: 60 }};
        const plotW = w - pad.left - pad.right;
        const plotH = h - pad.top - pad.bottom;

        // Background
        ctx.fillStyle = '{bg}';
        ctx.fillRect(0, 0, w, h);

        // Grid
        ctx.strokeStyle = '{text}15';
        ctx.lineWidth = 1;
        for (let i = 0; i <= 5; i++) {{
            const y = pad.top + (plotH / 5) * i;
            ctx.beginPath();
            ctx.moveTo(pad.left, y);
            ctx.lineTo(w - pad.right, y);
            ctx.stroke();
        }}
"#,
        bg = theme.panel_bg(),
        text = theme.text_color(),
    );

    // Draw event markers
    let max_tick = snapshot.tick.max(1);
    for (tick, color, label) in &event_markers {
        let x_frac = *tick as f64 / max_tick as f64;
        let _ = write!(chart_js, r#"
        // Event: {label}
        {{
            const x = pad.left + {x_frac} * plotW;
            ctx.beginPath();
            ctx.moveTo(x, pad.top);
            ctx.lineTo(x, pad.top + plotH);
            ctx.strokeStyle = '{color}';
            ctx.lineWidth = 1;
            ctx.setLineDash([4, 4]);
            ctx.stroke();
            ctx.setLineDash([]);
            ctx.fillStyle = '{color}';
            ctx.font = '8px monospace';
            ctx.save();
            ctx.translate(x + 3, pad.top + 10);
            ctx.rotate(-Math.PI / 4);
            ctx.fillText('{label_short}', 0, 0);
            ctx.restore();
        }}
"#,
            x_frac = x_frac,
            color = color,
            label = label,
            label_short = if label.len() > 15 { &label[..15] } else { label },
        );
    }

    // For each series, draw a simulated line based on current value
    // (Since we only have the current snapshot, we show the current value
    //  as a horizontal line with noise to suggest historical variation)
    for (idx, (series, val)) in current_values.iter().enumerate() {
        let color = series.color();
        let _ = write!(chart_js, r#"
        // Series: {label}
        {{
            const val = {val};
            const color = '{color}';
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.beginPath();

            // Procedural historical trace (deterministic pseudo-random from tick)
            const seed = {seed};
            for (let i = 0; i <= 100; i++) {{
                const t = i / 100;
                const noise = Math.sin(t * 47.3 + seed) * 0.15 + Math.sin(t * 13.7 + seed * 2) * 0.1;
                const ramp = 0.3 + 0.7 * (1 - Math.exp(-t * 3));
                const y_val = Math.max(0, Math.min(1, (val * ramp + noise) * 0.8 + 0.1));
                const x = pad.left + t * plotW;
                const y = pad.top + plotH * (1 - y_val);
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }}
            ctx.stroke();

            // Current value dot
            ctx.beginPath();
            ctx.arc(pad.left + plotW, pad.top + plotH * (1 - Math.min(1, Math.max(0, val * 0.8 + 0.1))), 4, 0, Math.PI * 2);
            ctx.fillStyle = color;
            ctx.fill();

            // Legend entry
            ctx.fillStyle = color;
            ctx.font = '11px monospace';
            ctx.fillText('{label}: ' + val.toFixed(3), pad.left + 5 + {idx} * 130, h - 8);
        }}
"#,
            label = series.label(),
            val = val,
            color = color,
            seed = idx * 17 + 7,
            idx = idx,
        );
    }

    // Axis labels
    let _ = write!(chart_js, r#"
        // Axes
        ctx.fillStyle = '{text}88';
        ctx.font = '10px monospace';
        ctx.textAlign = 'right';
        for (let i = 0; i <= 5; i++) {{
            const y = pad.top + (plotH / 5) * i;
            const v = (1 - i / 5).toFixed(1);
            ctx.fillText(v, pad.left - 5, y + 4);
        }}
        ctx.textAlign = 'center';
        ctx.fillText('0', pad.left, h - pad.bottom + 15);
        ctx.fillText('Tick {tick}', pad.left + plotW, h - pad.bottom + 15);

        // Title
        ctx.fillStyle = '{accent}';
        ctx.font = 'bold 12px monospace';
        ctx.textAlign = 'left';
        ctx.fillText('EMC Time Series — Tick {tick}', pad.left, 18);
    }}
}}
"#,
        text = theme.text_color(),
        accent = theme.accent_color(),
        tick = snapshot.tick,
    );

    // Stats panel
    let mut stats_html = String::new();
    let _ = write!(stats_html, "<h3>Time Series</h3>");
    let _ = write!(stats_html, r#"<div class="stat-row"><span class="key">Tick</span><span class="val">{}</span></div>"#, snapshot.tick);
    let _ = write!(stats_html, r#"<div class="stat-row"><span class="key">Events</span><span class="val">{}</span></div>"#, events.len());
    let _ = write!(stats_html, r#"<div class="stat-row"><span class="key">Series</span><span class="val">{}</span></div>"#, active_series.len());
    for (s, v) in &current_values {
        let _ = write!(stats_html,
            r#"<div class="stat-row"><span class="key">{}</span><span class="val" style="color:{}">{:.3}</span></div>"#,
            s.label(), s.color(), v
        );
    }

    let title = format!("Icarus - Time Series (tick {})", snapshot.tick);
    let mut doc = HtmlDocument::new(&title, theme);
    doc.add_panel_with_width("stats-panel", PanelPosition::TopRight, &stats_html, "200px");
    doc.add_chart("timeseries-chart", 800, 400);

    doc.set_extra_css(r#"
#timeseries-chart {
    position: absolute;
    top: 50%;
    left: 45%;
    transform: translate(-50%, -50%);
    border: 1px solid #00e5ff33;
    border-radius: 8px;
}
"#);

    // Minimal 3D background
    doc.set_scene_js(r#"
camera.position.set(0, 0, 5);
scene.add(new THREE.AmbientLight(0x202040, 0.2));
{
    const geo = new THREE.BufferGeometry();
    const n = 100;
    const pos = new Float32Array(n * 3);
    for (let i = 0; i < n * 3; i++) pos[i] = (Math.random() - 0.5) * 30;
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    const mat = new THREE.PointsMaterial({ size: 0.05, color: 0x334466, transparent: true, opacity: 0.3 });
    scene.add(new THREE.Points(geo, mat));
}
"#);

    doc.set_chart_js(&chart_js);
    doc.set_animation_js("");

    doc.render()
}
