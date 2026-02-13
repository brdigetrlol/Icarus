//! Color mapping functions for Icarus visualizations.
//!
//! All colors are returned as (r, g, b) tuples in [0.0, 1.0] range
//! or as hex strings "#RRGGBB" for direct HTML/CSS embedding.

use icarus_engine::agents::planning::ConvergenceTrend;

/// Map amplitude to a heat-ramp color: blue → cyan → green → yellow → red.
///
/// `t` should be in [0.0, 1.0] where 0 = minimum, 1 = maximum amplitude.
pub fn amplitude_color(t: f32) -> (f32, f32, f32) {
    let t = t.clamp(0.0, 1.0);
    if t < 0.25 {
        let s = t / 0.25;
        (0.0, s, 1.0) // blue → cyan
    } else if t < 0.5 {
        let s = (t - 0.25) / 0.25;
        (0.0, 1.0, 1.0 - s) // cyan → green
    } else if t < 0.75 {
        let s = (t - 0.5) / 0.25;
        (s, 1.0, 0.0) // green → yellow
    } else {
        let s = (t - 0.75) / 0.25;
        (1.0, 1.0 - s, 0.0) // yellow → red
    }
}

/// Map a complex phase angle (radians, [-pi, pi]) to HSL hue [0, 360).
pub fn phase_to_hue(phase: f32) -> f32 {
    let normalized = (phase + std::f32::consts::PI) / (2.0 * std::f32::consts::PI);
    normalized.clamp(0.0, 1.0) * 360.0
}

/// Map a phase angle to an RGB color via HSL with full saturation.
pub fn phase_color(phase: f32) -> (f32, f32, f32) {
    let hue = phase_to_hue(phase);
    hsl_to_rgb(hue, 1.0, 0.5)
}

/// Map energy to a color: deep blue (low) → white-hot (high).
pub fn energy_color(t: f32) -> (f32, f32, f32) {
    let t = t.clamp(0.0, 1.0);
    if t < 0.33 {
        let s = t / 0.33;
        (0.0, 0.0, 0.2 + 0.8 * s) // dark blue → blue
    } else if t < 0.66 {
        let s = (t - 0.33) / 0.33;
        (s, 0.3 * s, 1.0) // blue → magenta
    } else {
        let s = (t - 0.66) / 0.34;
        (1.0, 0.3 + 0.7 * s, 1.0) // magenta → white
    }
}

/// Map convergence trend to a signal color.
pub fn convergence_color(trend: &ConvergenceTrend) -> (f32, f32, f32) {
    match trend {
        ConvergenceTrend::Converging => (0.0, 1.0, 0.4),  // green
        ConvergenceTrend::Stable => (1.0, 0.85, 0.0),     // gold
        ConvergenceTrend::Diverging => (1.0, 0.15, 0.15),  // red
        ConvergenceTrend::Unknown => (0.5, 0.5, 0.5),     // gray
    }
}

/// Convert (r, g, b) in [0,1] to a CSS hex color string "#RRGGBB".
pub fn rgb_to_hex(r: f32, g: f32, b: f32) -> String {
    let ri = (r.clamp(0.0, 1.0) * 255.0) as u8;
    let gi = (g.clamp(0.0, 1.0) * 255.0) as u8;
    let bi = (b.clamp(0.0, 1.0) * 255.0) as u8;
    format!("#{:02x}{:02x}{:02x}", ri, gi, bi)
}

/// Convert (r, g, b) in [0,1] to a Three.js hex integer "0xRRGGBB".
pub fn rgb_to_threejs_hex(r: f32, g: f32, b: f32) -> String {
    let ri = (r.clamp(0.0, 1.0) * 255.0) as u32;
    let gi = (g.clamp(0.0, 1.0) * 255.0) as u32;
    let bi = (b.clamp(0.0, 1.0) * 255.0) as u32;
    format!("0x{:02x}{:02x}{:02x}", ri, gi, bi)
}

/// HSL to RGB conversion.
/// h: [0, 360), s: [0, 1], l: [0, 1]
pub fn hsl_to_rgb(h: f32, s: f32, l: f32) -> (f32, f32, f32) {
    if s == 0.0 {
        return (l, l, l);
    }

    let q = if l < 0.5 {
        l * (1.0 + s)
    } else {
        l + s - l * s
    };
    let p = 2.0 * l - q;
    let h_norm = h / 360.0;

    let r = hue_to_rgb(p, q, h_norm + 1.0 / 3.0);
    let g = hue_to_rgb(p, q, h_norm);
    let b = hue_to_rgb(p, q, h_norm - 1.0 / 3.0);

    (r, g, b)
}

fn hue_to_rgb(p: f32, q: f32, mut t: f32) -> f32 {
    if t < 0.0 {
        t += 1.0;
    }
    if t > 1.0 {
        t -= 1.0;
    }
    if t < 1.0 / 6.0 {
        return p + (q - p) * 6.0 * t;
    }
    if t < 1.0 / 2.0 {
        return q;
    }
    if t < 2.0 / 3.0 {
        return p + (q - p) * (2.0 / 3.0 - t) * 6.0;
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_amplitude_color_boundaries() {
        let (r, _g, b) = amplitude_color(0.0);
        assert!((r - 0.0).abs() < 0.01 && (b - 1.0).abs() < 0.01); // blue

        let (r, g, _b) = amplitude_color(1.0);
        assert!((r - 1.0).abs() < 0.01 && (g - 0.0).abs() < 0.01); // red

        let (_r, g, _b) = amplitude_color(0.5);
        assert!((g - 1.0).abs() < 0.01); // green peak
    }

    #[test]
    fn test_phase_to_hue_range() {
        let hue_neg = phase_to_hue(-std::f32::consts::PI);
        let hue_pos = phase_to_hue(std::f32::consts::PI);
        assert!(hue_neg >= 0.0 && hue_neg <= 360.0);
        assert!(hue_pos >= 0.0 && hue_pos <= 360.0);
    }

    #[test]
    fn test_rgb_to_hex() {
        assert_eq!(rgb_to_hex(1.0, 0.0, 0.0), "#ff0000");
        assert_eq!(rgb_to_hex(0.0, 1.0, 0.0), "#00ff00");
        assert_eq!(rgb_to_hex(0.0, 0.0, 1.0), "#0000ff");
    }

    #[test]
    fn test_hsl_to_rgb_red() {
        let (r, g, b) = hsl_to_rgb(0.0, 1.0, 0.5);
        assert!((r - 1.0).abs() < 0.01);
        assert!(g.abs() < 0.01);
        assert!(b.abs() < 0.01);
    }

    #[test]
    fn test_convergence_colors_distinct() {
        let c1 = convergence_color(&ConvergenceTrend::Converging);
        let c2 = convergence_color(&ConvergenceTrend::Diverging);
        // Green vs red - r components should differ significantly
        assert!((c2.0 - c1.0).abs() > 0.5);
    }
}
