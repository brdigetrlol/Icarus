//! Visualization renderers for Icarus EMC data.
//!
//! Each renderer takes an `EmcSnapshot` and produces a self-contained HTML
//! document with embedded Three.js WebGL visualization. All geometry, color,
//! and animation are procedurally generated from the simulation state.

pub mod lattice_field;
pub mod energy_landscape;
pub mod phase_portrait;
pub mod neuro_dashboard;
pub mod timeseries;
pub mod combined;

pub use lattice_field::render_lattice_field;
pub use energy_landscape::render_energy_landscape;
pub use phase_portrait::render_phase_portrait;
pub use neuro_dashboard::render_neuro_dashboard;
pub use timeseries::render_timeseries;
pub use combined::render_combined_dashboard;
