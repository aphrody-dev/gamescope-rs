//! Gamescope Core: Micro-compositor frame pacing, scaling math, layer composition, and HDR color science.

pub mod composition;
pub mod hdr;
pub mod pacing;
pub mod scaling;

pub use composition::{BlendMode, CompositionPlan, CompositorLayer, LayerType};
pub use hdr::{
    HdrTonemapParams, Mat3, TonemapOperator, MAT_BT2020_TO_BT709, MAT_BT709_TO_BT2020,
    pq_eotf, pq_inverse_eotf, scrgb_to_nits, nits_to_scrgb, transform_color,
};
pub use pacing::{LatencyBudgetCalculator, RefreshRate, RollingFrameHistogram};
pub use scaling::{Resolution, ScalingLayout, ScalingMode, ViewportRect};
