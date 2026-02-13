pub mod projection;
pub mod color;
pub mod template;
pub mod scene;
pub mod renderers;

pub use renderers::{
    render_combined_dashboard,
    render_energy_landscape,
    render_lattice_field,
    render_neuro_dashboard,
    render_phase_portrait,
    render_timeseries,
};
