// Copyright (c) 2025-2026 brdigetrlol. All rights reserved.
// SPDX-License-Identifier: LicenseRef-Icarus-Proprietary
// See LICENSE in the repository root for full license terms.

pub mod config;
pub mod manifold;
pub mod agents;
pub mod emc;
pub mod encoding;
pub mod readout;
pub mod training;
pub mod continual;
pub mod autonomous;
pub mod ensemble;
pub mod world;

pub use config::ManifoldConfig;
pub use emc::EmergentManifoldComputer;
pub use manifold::CausalCrystalManifold;
pub use encoding::{InputEncoder, SpatialEncoder, PhaseEncoder, SpectralEncoder};
pub use readout::{Readout, LinearReadout, DirectReadout, StateCollector, FeatureMode, extract_features, feature_dim};
pub use training::{RidgeRegression, ReservoirTrainer};
pub use continual::{ContinualTrainer, EwcRidgeRegression, ReplayBuffer, ContinualStats};
pub use ensemble::{EnsembleTrainer, EnsembleTrainResult};
