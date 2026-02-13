// Copyright (c) 2025-2026 brdigetrlol. All rights reserved.
// SPDX-License-Identifier: LicenseRef-Icarus-Proprietary
// See LICENSE in the repository root for full license terms.

//! Three.js scene builder for procedural 3D rendering.
//!
//! `SceneBuilder` accumulates scene objects (spheres, edges, points, labels,
//! lights) and serializes them to JavaScript that constructs a Three.js scene.
//! All geometry is emitted procedurally — no external model files.

use std::fmt::Write;

use crate::color::rgb_to_threejs_hex;

/// A point in a point cloud.
#[derive(Debug, Clone)]
pub struct ScenePoint {
    pub position: [f64; 3],
    pub color: (f32, f32, f32),
    pub size: f32,
}

/// An edge between two 3D positions.
#[derive(Debug, Clone)]
pub struct SceneEdge {
    pub from: [f64; 3],
    pub to: [f64; 3],
    pub color: (f32, f32, f32),
    pub opacity: f64,
}

/// A sphere with position, radius, color, emissive glow, and opacity.
#[derive(Debug, Clone)]
pub struct SceneSphere {
    pub position: [f64; 3],
    pub radius: f64,
    pub color: (f32, f32, f32),
    pub emissive: (f32, f32, f32),
    pub opacity: f64,
}

/// A text label in 3D space (rendered as a Sprite with canvas texture).
#[derive(Debug, Clone)]
pub struct SceneLabel {
    pub position: [f64; 3],
    pub text: String,
    pub color: (f32, f32, f32),
    pub scale: f64,
}

/// A point light source.
#[derive(Debug, Clone)]
pub struct PointLight {
    pub position: [f64; 3],
    pub color: u32,
    pub intensity: f64,
    pub distance: f64,
}

/// Camera configuration.
#[derive(Debug, Clone)]
pub struct CameraConfig {
    pub position: [f64; 3],
    pub look_at: [f64; 3],
    pub fov: f64,
}

impl Default for CameraConfig {
    fn default() -> Self {
        Self {
            position: [0.0, 4.0, 10.0],
            look_at: [0.0, 0.0, 0.0],
            fov: 60.0,
        }
    }
}

/// Builder that accumulates Three.js scene objects and emits JavaScript.
///
/// Usage:
/// ```ignore
/// let mut scene = SceneBuilder::new();
/// scene.set_camera([0.0, 5.0, 10.0], [0.0, 0.0, 0.0]);
/// scene.add_point_light([4.0, 8.0, 4.0], 0xffffff, 0.8, 30.0);
/// scene.add_sphere([0.0, 0.0, 0.0], 0.5, (1.0, 0.0, 0.0), (0.3, 0.0, 0.0), 1.0);
/// let js = scene.build_js();
/// ```
pub struct SceneBuilder {
    camera: CameraConfig,
    ambient_color: u32,
    ambient_intensity: f64,
    point_lights: Vec<PointLight>,
    points: Vec<ScenePoint>,
    edges: Vec<SceneEdge>,
    spheres: Vec<SceneSphere>,
    labels: Vec<SceneLabel>,
    custom_js: Vec<String>,
    grid_helper: bool,
}

impl SceneBuilder {
    pub fn new() -> Self {
        Self {
            camera: CameraConfig::default(),
            ambient_color: 0x404060,
            ambient_intensity: 0.4,
            point_lights: Vec::new(),
            points: Vec::new(),
            edges: Vec::new(),
            spheres: Vec::new(),
            labels: Vec::new(),
            custom_js: Vec::new(),
            grid_helper: false,
        }
    }

    pub fn set_camera(&mut self, position: [f64; 3], look_at: [f64; 3]) {
        self.camera.position = position;
        self.camera.look_at = look_at;
    }

    pub fn set_fov(&mut self, fov: f64) {
        self.camera.fov = fov;
    }

    pub fn set_ambient_light(&mut self, color: u32, intensity: f64) {
        self.ambient_color = color;
        self.ambient_intensity = intensity;
    }

    pub fn add_point_light(&mut self, position: [f64; 3], color: u32, intensity: f64, distance: f64) {
        self.point_lights.push(PointLight { position, color, intensity, distance });
    }

    pub fn add_point(&mut self, position: [f64; 3], color: (f32, f32, f32), size: f32) {
        self.points.push(ScenePoint { position, color, size });
    }

    pub fn add_points(&mut self, pts: Vec<ScenePoint>) {
        self.points.extend(pts);
    }

    pub fn add_edge(&mut self, from: [f64; 3], to: [f64; 3], color: (f32, f32, f32), opacity: f64) {
        self.edges.push(SceneEdge { from, to, color, opacity });
    }

    pub fn add_edges(&mut self, edges: Vec<SceneEdge>) {
        self.edges.extend(edges);
    }

    pub fn add_sphere(
        &mut self,
        position: [f64; 3],
        radius: f64,
        color: (f32, f32, f32),
        emissive: (f32, f32, f32),
        opacity: f64,
    ) {
        self.spheres.push(SceneSphere { position, radius, color, emissive, opacity });
    }

    pub fn add_label(&mut self, position: [f64; 3], text: &str, color: (f32, f32, f32), scale: f64) {
        self.labels.push(SceneLabel {
            position,
            text: text.to_string(),
            color,
            scale,
        });
    }

    pub fn set_grid_helper(&mut self, enabled: bool) {
        self.grid_helper = enabled;
    }

    pub fn add_custom_js(&mut self, js: &str) {
        self.custom_js.push(js.to_string());
    }

    /// Emit the complete Three.js scene construction as a JavaScript string.
    pub fn build_js(&self) -> String {
        let mut js = String::with_capacity(16384);

        // Camera
        let _ = write!(js, r#"
camera.position.set({cx}, {cy}, {cz});
camera.lookAt({lx}, {ly}, {lz});
"#,
            cx = self.camera.position[0], cy = self.camera.position[1], cz = self.camera.position[2],
            lx = self.camera.look_at[0], ly = self.camera.look_at[1], lz = self.camera.look_at[2],
        );

        // Ambient light
        let _ = write!(js, "scene.add(new THREE.AmbientLight({}, {}));\n",
            format_args!("0x{:06x}", self.ambient_color), self.ambient_intensity);

        // Point lights
        for light in &self.point_lights {
            let _ = write!(js, r#"{{
    const l = new THREE.PointLight(0x{color:06x}, {intensity}, {distance});
    l.position.set({x}, {y}, {z});
    scene.add(l);
}}
"#,
                color = light.color, intensity = light.intensity, distance = light.distance,
                x = light.position[0], y = light.position[1], z = light.position[2],
            );
        }

        // Grid helper
        if self.grid_helper {
            js.push_str("scene.add(new THREE.GridHelper(20, 40, 0x334455, 0x1a1a2e));\n");
        }

        // Point cloud (BufferGeometry + Points)
        if !self.points.is_empty() {
            self.build_point_cloud_js(&mut js);
        }

        // Spheres — InstancedMesh for large sets, individual Mesh for small
        if !self.spheres.is_empty() {
            if self.spheres.len() >= 50 {
                self.build_instanced_spheres_js(&mut js);
            } else {
                self.build_individual_spheres_js(&mut js);
            }
        }

        // Edges (LineSegments)
        if !self.edges.is_empty() {
            self.build_edges_js(&mut js);
        }

        // Labels (Sprite-based canvas textures)
        for (i, label) in self.labels.iter().enumerate() {
            self.build_label_js(&mut js, label, i);
        }

        // Custom JS
        for chunk in &self.custom_js {
            js.push_str(chunk);
            js.push('\n');
        }

        js
    }

    fn build_point_cloud_js(&self, js: &mut String) {
        let n = self.points.len();
        let _ = write!(js, r#"{{
    const n = {n};
    const geo = new THREE.BufferGeometry();
    const pos = new Float32Array(n * 3);
    const col = new Float32Array(n * 3);
    const sizes = new Float32Array(n);
"#, n = n);

        for (i, pt) in self.points.iter().enumerate() {
            let _ = write!(js,
                "    pos[{}]={};pos[{}]={};pos[{}]={}; col[{}]={};col[{}]={};col[{}]={}; sizes[{}]={};\n",
                i*3, pt.position[0], i*3+1, pt.position[1], i*3+2, pt.position[2],
                i*3, pt.color.0, i*3+1, pt.color.1, i*3+2, pt.color.2,
                i, pt.size,
            );
        }

        js.push_str(r#"    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
    const mat = new THREE.PointsMaterial({
        size: 0.1, vertexColors: true, transparent: true,
        opacity: 0.9, blending: THREE.AdditiveBlending, depthWrite: false, sizeAttenuation: true
    });
    scene.add(new THREE.Points(geo, mat));
}
"#);
    }

    fn build_instanced_spheres_js(&self, js: &mut String) {
        let n = self.spheres.len();
        let _ = write!(js, r#"{{
    const n = {n};
    const baseGeo = new THREE.SphereGeometry(1, 12, 8);
    const baseMat = new THREE.MeshPhongMaterial({{
        vertexColors: false, transparent: true, opacity: 0.95,
        shininess: 80, specular: 0x222222
    }});
    const mesh = new THREE.InstancedMesh(baseGeo, baseMat, n);
    mesh.name = 'sphereCluster';
    const dummy = new THREE.Object3D();
    const colors = new Float32Array(n * 3);
"#, n = n);

        for (i, s) in self.spheres.iter().enumerate() {
            let _ = write!(js, r#"    dummy.position.set({x},{y},{z}); dummy.scale.setScalar({r}); dummy.updateMatrix();
    mesh.setMatrixAt({i}, dummy.matrix);
    colors[{i3}]={cr}; colors[{i31}]={cg}; colors[{i32}]={cb};
"#,
                x = s.position[0], y = s.position[1], z = s.position[2],
                r = s.radius, i = i,
                i3 = i*3, i31 = i*3+1, i32 = i*3+2,
                cr = s.color.0, cg = s.color.1, cb = s.color.2,
            );
        }

        js.push_str(r#"    mesh.instanceMatrix.needsUpdate = true;
    mesh.instanceColor = new THREE.InstancedBufferAttribute(colors, 3);
    scene.add(mesh);

    // Emissive glow layer (additive blend)
    const glowMat = new THREE.MeshBasicMaterial({
        transparent: true, opacity: 0.3, blending: THREE.AdditiveBlending, depthWrite: false
    });
    const glow = new THREE.InstancedMesh(baseGeo, glowMat, n);
    glow.name = 'sphereGlow';
    const glowColors = new Float32Array(n * 3);
"#);

        for (i, s) in self.spheres.iter().enumerate() {
            let glow_scale = s.radius * 1.6;
            let _ = write!(js, r#"    dummy.position.set({x},{y},{z}); dummy.scale.setScalar({r}); dummy.updateMatrix();
    glow.setMatrixAt({i}, dummy.matrix);
    glowColors[{i3}]={er}; glowColors[{i31}]={eg}; glowColors[{i32}]={eb};
"#,
                x = s.position[0], y = s.position[1], z = s.position[2],
                r = glow_scale, i = i,
                i3 = i*3, i31 = i*3+1, i32 = i*3+2,
                er = s.emissive.0, eg = s.emissive.1, eb = s.emissive.2,
            );
        }

        js.push_str(r#"    glow.instanceMatrix.needsUpdate = true;
    glow.instanceColor = new THREE.InstancedBufferAttribute(glowColors, 3);
    scene.add(glow);
}
"#);
    }

    fn build_individual_spheres_js(&self, js: &mut String) {
        for (i, s) in self.spheres.iter().enumerate() {
            let color_hex = rgb_to_threejs_hex(s.color.0, s.color.1, s.color.2);
            let emissive_hex = rgb_to_threejs_hex(s.emissive.0, s.emissive.1, s.emissive.2);
            let _ = write!(js, r#"{{
    const geo = new THREE.SphereGeometry({r}, 16, 12);
    const mat = new THREE.MeshPhongMaterial({{
        color: {color}, emissive: {emissive}, transparent: true,
        opacity: {opacity}, shininess: 80
    }});
    const m = new THREE.Mesh(geo, mat);
    m.position.set({x}, {y}, {z});
    m.name = 'sphere_{idx}';
    scene.add(m);
}}
"#,
                r = s.radius, color = color_hex, emissive = emissive_hex,
                opacity = s.opacity, x = s.position[0], y = s.position[1], z = s.position[2],
                idx = i,
            );
        }
    }

    fn build_edges_js(&self, js: &mut String) {
        let n = self.edges.len();
        let _ = write!(js, r#"{{
    const edgeGeo = new THREE.BufferGeometry();
    const edgePos = new Float32Array({n6});
    const edgeCol = new Float32Array({n6});
"#, n6 = n * 6);

        for (i, e) in self.edges.iter().enumerate() {
            let _ = write!(js, r#"    edgePos[{i0}]={fx}; edgePos[{i1}]={fy}; edgePos[{i2}]={fz};
    edgePos[{i3}]={tx}; edgePos[{i4}]={ty}; edgePos[{i5}]={tz};
    edgeCol[{i0}]={cr}; edgeCol[{i1}]={cg}; edgeCol[{i2}]={cb};
    edgeCol[{i3}]={cr}; edgeCol[{i4}]={cg}; edgeCol[{i5}]={cb};
"#,
                i0=i*6, i1=i*6+1, i2=i*6+2, i3=i*6+3, i4=i*6+4, i5=i*6+5,
                fx=e.from[0], fy=e.from[1], fz=e.from[2],
                tx=e.to[0], ty=e.to[1], tz=e.to[2],
                cr=e.color.0, cg=e.color.1, cb=e.color.2,
            );
        }

        // Use average opacity
        let avg_opacity = if n > 0 {
            self.edges.iter().map(|e| e.opacity).sum::<f64>() / n as f64
        } else { 0.3 };

        let _ = write!(js, r#"    edgeGeo.setAttribute('position', new THREE.BufferAttribute(edgePos, 3));
    edgeGeo.setAttribute('color', new THREE.BufferAttribute(edgeCol, 3));
    const edgeMat = new THREE.LineBasicMaterial({{
        vertexColors: true, transparent: true, opacity: {opacity},
        blending: THREE.AdditiveBlending, depthWrite: false
    }});
    scene.add(new THREE.LineSegments(edgeGeo, edgeMat));
}}
"#, opacity = avg_opacity);
    }

    fn build_label_js(&self, js: &mut String, label: &SceneLabel, idx: usize) {
        let _ = write!(js, r#"{{
    const canvas = document.createElement('canvas');
    canvas.width = 256; canvas.height = 64;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = 'rgba({r},{g},{b},0.9)';
    ctx.font = 'bold 28px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('{text}', 128, 40);
    const tex = new THREE.CanvasTexture(canvas);
    const mat = new THREE.SpriteMaterial({{ map: tex, transparent: true, depthWrite: false }});
    const sprite = new THREE.Sprite(mat);
    sprite.position.set({x}, {y}, {z});
    sprite.scale.set({scale}, {sh}, 1);
    sprite.name = 'label_{idx}';
    scene.add(sprite);
}}
"#,
            r = (label.color.0 * 255.0) as u8,
            g = (label.color.1 * 255.0) as u8,
            b = (label.color.2 * 255.0) as u8,
            text = label.text.replace('\'', "\\'"),
            x = label.position[0], y = label.position[1], z = label.position[2],
            scale = label.scale, sh = label.scale * 0.25,
            idx = idx,
        );
    }
}

/// Generate JS for a pulsing glow animation on the sphere glow layer.
pub fn pulse_animation_js() -> String {
    r#"
    const glowMesh = scene.getObjectByName('sphereGlow');
    if (glowMesh && glowMesh.material) {
        glowMesh.material.opacity = 0.2 + 0.15 * Math.sin(time * 2.0);
    }
"#.to_string()
}

/// Generate JS for gentle rotation of the scene.
pub fn rotate_scene_js(speed: f64) -> String {
    format!(r#"
    scene.rotation.y += {};
"#, speed)
}

/// Generate JS for a breathing animation on instanced spheres.
pub fn sphere_breathe_js() -> String {
    r#"
    const cluster = scene.getObjectByName('sphereCluster');
    if (cluster) {
        const breathe = 1.0 + 0.03 * Math.sin(time * 1.2);
        cluster.scale.set(breathe, breathe, breathe);
    }
"#.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_scene() {
        let scene = SceneBuilder::new();
        let js = scene.build_js();
        assert!(js.contains("camera.position.set"));
        assert!(js.contains("AmbientLight"));
    }

    #[test]
    fn test_camera_config() {
        let mut scene = SceneBuilder::new();
        scene.set_camera([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]);
        let js = scene.build_js();
        assert!(js.contains("camera.position.set(1, 2, 3)"));
        assert!(js.contains("camera.lookAt(4, 5, 6)"));
    }

    #[test]
    fn test_point_light() {
        let mut scene = SceneBuilder::new();
        scene.add_point_light([1.0, 2.0, 3.0], 0xff0000, 0.8, 20.0);
        let js = scene.build_js();
        assert!(js.contains("PointLight"));
        assert!(js.contains("ff0000"));
    }

    #[test]
    fn test_individual_spheres() {
        let mut scene = SceneBuilder::new();
        for i in 0..5 {
            scene.add_sphere([i as f64, 0.0, 0.0], 0.1, (1.0, 0.0, 0.0), (0.3, 0.0, 0.0), 0.9);
        }
        let js = scene.build_js();
        assert!(js.contains("SphereGeometry"));
        assert!(js.contains("sphere_0"));
    }

    #[test]
    fn test_instanced_spheres() {
        let mut scene = SceneBuilder::new();
        for i in 0..100 {
            scene.add_sphere([i as f64 * 0.1, 0.0, 0.0], 0.05, (0.0, 1.0, 0.0), (0.0, 0.2, 0.0), 0.9);
        }
        let js = scene.build_js();
        assert!(js.contains("InstancedMesh"));
        assert!(js.contains("sphereCluster"));
    }

    #[test]
    fn test_edges() {
        let mut scene = SceneBuilder::new();
        scene.add_edge([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], (0.5, 0.5, 0.5), 0.4);
        let js = scene.build_js();
        assert!(js.contains("LineSegments"));
    }

    #[test]
    fn test_grid_helper() {
        let mut scene = SceneBuilder::new();
        scene.set_grid_helper(true);
        let js = scene.build_js();
        assert!(js.contains("GridHelper"));
    }

    #[test]
    fn test_custom_js() {
        let mut scene = SceneBuilder::new();
        scene.add_custom_js("console.log('hello');");
        let js = scene.build_js();
        assert!(js.contains("console.log('hello');"));
    }

    #[test]
    fn test_label() {
        let mut scene = SceneBuilder::new();
        scene.add_label([0.0, 1.0, 0.0], "Test", (1.0, 1.0, 1.0), 2.0);
        let js = scene.build_js();
        assert!(js.contains("Sprite"));
        assert!(js.contains("Test"));
    }
}
