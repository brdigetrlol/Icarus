// Copyright (c) 2025-2026 brdigetrlol. All rights reserved.
// SPDX-License-Identifier: LicenseRef-Icarus-Proprietary
// See LICENSE in the repository root for full license terms.

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
