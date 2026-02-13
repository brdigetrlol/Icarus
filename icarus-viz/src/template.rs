//! HTML document template engine for Icarus visualizations.
//!
//! Generates self-contained HTML documents with embedded Three.js,
//! OrbitControls, and optional CSS2DRenderer for HUD overlays.

use std::fmt::Write;

/// Position anchor for HUD panels.
#[derive(Debug, Clone)]
pub enum PanelPosition {
    TopLeft,
    TopRight,
    TopCenter,
    BottomLeft,
    BottomRight,
    BottomCenter,
}

impl PanelPosition {
    fn css(&self) -> &'static str {
        match self {
            Self::TopLeft => "top: 10px; left: 10px;",
            Self::TopRight => "top: 10px; right: 10px;",
            Self::TopCenter => "top: 10px; left: 50%; transform: translateX(-50%);",
            Self::BottomLeft => "bottom: 10px; left: 10px;",
            Self::BottomRight => "bottom: 10px; right: 10px;",
            Self::BottomCenter => "bottom: 10px; left: 50%; transform: translateX(-50%);",
        }
    }
}

/// A HUD overlay panel rendered on top of the 3D scene.
#[derive(Debug, Clone)]
pub struct HudPanel {
    pub id: String,
    pub position: PanelPosition,
    pub content_html: String,
    pub width: Option<String>,
}

/// A 2D canvas element for chart rendering.
#[derive(Debug, Clone)]
pub struct ChartCanvas {
    pub id: String,
    pub width: u32,
    pub height: u32,
}

/// Theme configuration for the visualization.
#[derive(Debug, Clone)]
pub enum Theme {
    Dark,
    Light,
}

impl Theme {
    pub fn bg_color(&self) -> &'static str {
        match self {
            Self::Dark => "#0a0a0f",
            Self::Light => "#f0f0f5",
        }
    }

    pub fn text_color(&self) -> &'static str {
        match self {
            Self::Dark => "#e0e0e0",
            Self::Light => "#1a1a2e",
        }
    }

    pub fn panel_bg(&self) -> &'static str {
        match self {
            Self::Dark => "rgba(10, 10, 20, 0.85)",
            Self::Light => "rgba(240, 240, 245, 0.9)",
        }
    }

    pub fn accent_color(&self) -> &'static str {
        match self {
            Self::Dark => "#00e5ff",
            Self::Light => "#0066cc",
        }
    }

    pub fn fog_color_hex(&self) -> &'static str {
        match self {
            Self::Dark => "0x0a0a0f",
            Self::Light => "0xf0f0f5",
        }
    }
}

/// Builder for self-contained HTML visualization documents.
pub struct HtmlDocument {
    title: String,
    theme: Theme,
    panels: Vec<HudPanel>,
    charts: Vec<ChartCanvas>,
    scene_js: String,
    animation_js: String,
    chart_js: String,
    extra_css: String,
    auto_refresh_secs: u32,
    use_orbit_controls: bool,
    split_layout: bool,
}

impl HtmlDocument {
    pub fn new(title: &str, theme: Theme) -> Self {
        Self {
            title: title.to_string(),
            theme,
            panels: Vec::new(),
            charts: Vec::new(),
            scene_js: String::new(),
            animation_js: String::new(),
            chart_js: String::new(),
            extra_css: String::new(),
            auto_refresh_secs: 0,
            use_orbit_controls: true,
            split_layout: false,
        }
    }

    pub fn add_panel(&mut self, id: &str, position: PanelPosition, content: &str) {
        self.panels.push(HudPanel {
            id: id.to_string(),
            position,
            content_html: content.to_string(),
            width: None,
        });
    }

    pub fn add_panel_with_width(&mut self, id: &str, position: PanelPosition, content: &str, width: &str) {
        self.panels.push(HudPanel {
            id: id.to_string(),
            position,
            content_html: content.to_string(),
            width: Some(width.to_string()),
        });
    }

    pub fn add_chart(&mut self, id: &str, width: u32, height: u32) {
        self.charts.push(ChartCanvas {
            id: id.to_string(),
            width,
            height,
        });
    }

    pub fn set_scene_js(&mut self, js: &str) {
        self.scene_js = js.to_string();
    }

    pub fn set_animation_js(&mut self, js: &str) {
        self.animation_js = js.to_string();
    }

    pub fn set_chart_js(&mut self, js: &str) {
        self.chart_js = js.to_string();
    }

    pub fn set_extra_css(&mut self, css: &str) {
        self.extra_css = css.to_string();
    }

    pub fn set_auto_refresh(&mut self, secs: u32) {
        self.auto_refresh_secs = secs;
    }

    pub fn set_split_layout(&mut self, split: bool) {
        self.split_layout = split;
    }

    /// Render the complete self-contained HTML document.
    pub fn render(&self) -> String {
        let mut html = String::with_capacity(32768);

        // DOCTYPE and head
        let _ = write!(html, r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
"#, title = self.title);

        if self.auto_refresh_secs > 0 {
            let _ = write!(html, r#"<meta http-equiv="refresh" content="{}">"#, self.auto_refresh_secs);
        }

        // Three.js imports
        html.push_str(r#"<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
"#);
        if self.use_orbit_controls {
            html.push_str(r#"<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
"#);
        }

        // CSS
        let _ = write!(html, r#"<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    background: {bg};
    color: {text};
    font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
    overflow: hidden;
}}
#canvas-container {{
    position: {canvas_pos};
    {canvas_layout}
}}
.hud-panel {{
    position: absolute;
    background: {panel_bg};
    border: 1px solid {accent}44;
    border-radius: 8px;
    padding: 12px 16px;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    font-size: 13px;
    line-height: 1.5;
    z-index: 10;
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.4);
}}
.hud-panel h3 {{
    color: {accent};
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 6px;
    font-weight: 600;
}}
.hud-panel .value {{
    font-size: 18px;
    font-weight: 700;
    color: {accent};
}}
.hud-panel .label {{
    font-size: 11px;
    color: {text}99;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}
.stat-row {{
    display: flex;
    justify-content: space-between;
    padding: 2px 0;
}}
.stat-row .key {{ color: {text}88; }}
.stat-row .val {{ color: {accent}; font-weight: 600; font-variant-numeric: tabular-nums; }}
.color-legend {{
    display: flex;
    align-items: center;
    gap: 4px;
    margin-top: 6px;
}}
.color-legend .swatch {{
    width: 12px;
    height: 12px;
    border-radius: 2px;
}}
.chart-container {{
    position: absolute;
    background: {panel_bg};
    border: 1px solid {accent}33;
    border-radius: 8px;
    padding: 8px;
    z-index: 10;
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.4);
}}
{extra_css}
</style>
"#,
            bg = self.theme.bg_color(),
            text = self.theme.text_color(),
            panel_bg = self.theme.panel_bg(),
            accent = self.theme.accent_color(),
            canvas_pos = if self.split_layout { "relative" } else { "fixed" },
            canvas_layout = if self.split_layout {
                "width: 70%; height: 100vh; float: left;"
            } else {
                "top: 0; left: 0; width: 100%; height: 100%;"
            },
            extra_css = self.extra_css,
        );

        html.push_str("</head>\n<body>\n");

        // Canvas container
        html.push_str("<div id=\"canvas-container\"></div>\n");

        // Side panel container for split layout
        if self.split_layout {
            let _ = write!(html, r#"<div id="side-panel" style="
                position: fixed; right: 0; top: 0; width: 30%; height: 100vh;
                background: {bg}; border-left: 1px solid {accent}33;
                overflow-y: auto; padding: 12px;
            ">"#,
                bg = self.theme.bg_color(),
                accent = self.theme.accent_color(),
            );
        }

        // HUD panels
        for panel in &self.panels {
            let width_css = panel.width.as_deref().map(|w| format!("width: {};", w)).unwrap_or_default();
            let _ = write!(html, r#"<div id="{id}" class="hud-panel" style="{pos} {width}">
{content}
</div>
"#,
                id = panel.id,
                pos = panel.position.css(),
                width = width_css,
                content = panel.content_html,
            );
        }

        // Chart canvases
        for chart in &self.charts {
            let _ = write!(html, r#"<canvas id="{id}" width="{w}" height="{h}" style="display:block;"></canvas>
"#,
                id = chart.id,
                w = chart.width,
                h = chart.height,
            );
        }

        if self.split_layout {
            html.push_str("</div>\n");
        }

        // JavaScript
        let _ = write!(html, r#"<script>
// Scene setup
const container = document.getElementById('canvas-container');
const scene = new THREE.Scene();
scene.background = new THREE.Color({fog_color});
scene.fog = new THREE.FogExp2({fog_color}, 0.015);

const camera = new THREE.PerspectiveCamera(60, container.clientWidth / container.clientHeight, 0.1, 1000);
camera.position.set(0, 3, 8);

const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
renderer.setSize(container.clientWidth, container.clientHeight);
renderer.setPixelRatio(window.devicePixelRatio);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
container.appendChild(renderer.domElement);
"#,
            fog_color = self.theme.fog_color_hex(),
        );

        if self.use_orbit_controls {
            html.push_str(r#"
const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.autoRotate = true;
controls.autoRotateSpeed = 0.5;
"#);
        }

        // Scene content
        html.push_str("\n// === Scene Content ===\n");
        html.push_str(&self.scene_js);

        // Chart initialization
        if !self.chart_js.is_empty() {
            html.push_str("\n// === Chart Initialization ===\n");
            html.push_str(&self.chart_js);
        }

        // Animation loop
        let _ = write!(html, r#"
// === Animation Loop ===
function animate() {{
    requestAnimationFrame(animate);
    const time = performance.now() * 0.001;
{controls}
{animation}
    renderer.render(scene, camera);
}}

// Resize handler
window.addEventListener('resize', () => {{
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
}});

animate();
</script>
</body>
</html>"#,
            controls = if self.use_orbit_controls { "    controls.update();" } else { "" },
            animation = self.animation_js,
        );

        html
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_document_renders() {
        let doc = HtmlDocument::new("Test", Theme::Dark);
        let html = doc.render();
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("Test"));
        assert!(html.contains("three.min.js"));
        assert!(html.contains("OrbitControls"));
    }

    #[test]
    fn test_panels_rendered() {
        let mut doc = HtmlDocument::new("Test", Theme::Dark);
        doc.add_panel("stats", PanelPosition::TopRight, "<h3>Stats</h3>");
        let html = doc.render();
        assert!(html.contains("id=\"stats\""));
        assert!(html.contains("<h3>Stats</h3>"));
    }

    #[test]
    fn test_auto_refresh() {
        let mut doc = HtmlDocument::new("Test", Theme::Dark);
        doc.set_auto_refresh(30);
        let html = doc.render();
        assert!(html.contains("http-equiv=\"refresh\" content=\"30\""));
    }

    #[test]
    fn test_light_theme() {
        let doc = HtmlDocument::new("Test", Theme::Light);
        let html = doc.render();
        assert!(html.contains("#f0f0f5"));
    }
}
