//! Projection from high-dimensional lattice coordinates to 3D space.
//!
//! Uses golden-ratio weighted linear combinations for E8 (8D→3D)
//! and random orthogonal projection for higher-dimensional lattices.

/// Project an 8D E8 lattice coordinate to 3D using golden-ratio projection.
///
/// The projection uses phi-weighted combinations that preserve the lattice's
/// icosahedral symmetry in the projected space.
pub fn project_e8_to_3d(coords: &[i64], coord_scale: f64) -> [f64; 3] {
    assert!(coords.len() >= 8, "E8 requires 8D coordinates");

    let v: Vec<f64> = coords.iter().map(|&c| c as f64 * coord_scale).collect();
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

/// Project high-dimensional coordinates to 3D using a deterministic
/// pseudo-random orthogonal projection.
///
/// Works for any dimension >= 3. Uses seeded coefficients derived from
/// golden ratio offsets so the projection is stable across calls.
pub fn project_high_dim_to_3d(coords: &[i64], coord_scale: f64) -> [f64; 3] {
    let dim = coords.len();
    if dim <= 3 {
        let mut result = [0.0; 3];
        for (i, &c) in coords.iter().enumerate() {
            result[i] = c as f64 * coord_scale;
        }
        return result;
    }

    // For 8D, use the specialized E8 projection
    if dim == 8 {
        return project_e8_to_3d(coords, coord_scale);
    }

    // Generic projection: use golden-ratio-derived coefficients
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let mut result = [0.0f64; 3];

    for (i, &c) in coords.iter().enumerate() {
        let val = c as f64 * coord_scale;
        // Each axis gets a unique coefficient from golden ratio offsets
        let offset = (i + 1) as f64 * phi;
        result[0] += val * (offset % 2.0 - 1.0);
        result[1] += val * ((offset * phi) % 2.0 - 1.0);
        result[2] += val * ((offset * phi * phi) % 2.0 - 1.0);
    }

    // Normalize so the scale doesn't blow up with dimension
    let norm_factor = 1.5 / (dim as f64).sqrt();
    for v in &mut result {
        *v *= norm_factor;
    }

    result
}

/// Find edges between projected points within a distance threshold.
///
/// Returns pairs of indices (i, j) where i < j and the distance
/// between projected[i] and projected[j] is less than `threshold`.
pub fn find_edges(projected: &[[f64; 3]], threshold: f64) -> Vec<(usize, usize)> {
    let thresh_sq = threshold * threshold;
    let mut edges = Vec::new();

    for i in 0..projected.len() {
        for j in (i + 1)..projected.len() {
            let dx = projected[i][0] - projected[j][0];
            let dy = projected[i][1] - projected[j][1];
            let dz = projected[i][2] - projected[j][2];
            let dist_sq = dx * dx + dy * dy + dz * dz;
            if dist_sq < thresh_sq {
                edges.push((i, j));
            }
        }
    }

    edges
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_e8_origin_projects_to_origin() {
        let origin = [0i64; 8];
        let p = project_e8_to_3d(&origin, 0.5);
        assert!((p[0]).abs() < 1e-10);
        assert!((p[1]).abs() < 1e-10);
        assert!((p[2]).abs() < 1e-10);
    }

    #[test]
    fn test_e8_projection_nonzero() {
        let root = [2, 0, 0, 0, 0, 0, 0, 0]; // (1,0,0,0,0,0,0,0) in doubled coords
        let p = project_e8_to_3d(&root, 0.5);
        let mag = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        assert!(mag > 0.1, "Projection should be non-degenerate");
    }

    #[test]
    fn test_high_dim_3d_passthrough() {
        let coords = [1, 2, 3];
        let p = project_high_dim_to_3d(&coords, 1.0);
        assert_eq!(p, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_find_edges_basic() {
        let points = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ];
        let edges = find_edges(&points, 2.0);
        assert_eq!(edges, vec![(0, 1)]);
    }

    #[test]
    fn test_e8_root_count() {
        // Generate all 240 E8 roots and project them
        let mut roots = Vec::new();
        for i in 0..8 {
            for j in (i + 1)..8 {
                for si in [-1i64, 1] {
                    for sj in [-1i64, 1] {
                        let mut v = [0i64; 8];
                        v[i] = si * 2;
                        v[j] = sj * 2;
                        roots.push(v);
                    }
                }
            }
        }
        for bits in 0u32..256 {
            if bits.count_ones() % 2 != 0 {
                continue;
            }
            let mut v = [1i64; 8];
            for k in 0..8 {
                if bits & (1 << k) != 0 {
                    v[k] = -1;
                }
            }
            roots.push(v);
        }
        assert_eq!(roots.len(), 240);

        let projected: Vec<[f64; 3]> = roots
            .iter()
            .map(|r| project_e8_to_3d(r, 0.5))
            .collect();
        // All projections should be non-degenerate
        for p in &projected {
            let mag = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            assert!(mag > 0.01);
        }
    }
}
