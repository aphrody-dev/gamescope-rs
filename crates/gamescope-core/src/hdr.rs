//! HDR10 (SMPTE ST.2084 PQ) <-> scRGB linear color space transfer matrices & tonemapping.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

// SMPTE ST.2084 PQ (Perceptual Quantizer) standard constants
pub const PQ_M1: f32 = 0.1593017578125; // 2610 / 16384
pub const PQ_M2: f32 = 78.84375;        // (2523 / 4096) * 128
pub const PQ_C1: f32 = 0.8359375;       // 3424 / 4096
pub const PQ_C2: f32 = 18.8515625;      // (2413 / 4096) * 32
pub const PQ_C3: f32 = 18.6875;         // (2392 / 4096) * 32
pub const PQ_MAX_NITS: f32 = 10000.0;
pub const REFERENCE_WHITE_NITS: f32 = 80.0;
pub const DEFAULT_PAPER_WHITE_NITS: f32 = 203.0; // ITU-R BT.2408

/// Converts normalized SMPTE ST.2084 non-linear signal [0.0, 1.0] to linear luminance in nits [0.0, 10000.0].
pub fn pq_eotf(signal: f32) -> f32 {
    let n = signal.clamp(0.0, 1.0);
    if n == 0.0 {
        return 0.0;
    }
    let n_pow = n.powf(1.0 / PQ_M2);
    let num = (n_pow - PQ_C1).max(0.0);
    let den = PQ_C2 - PQ_C3 * n_pow;
    let linear_ratio = (num / den).powf(1.0 / PQ_M1);
    linear_ratio * PQ_MAX_NITS
}

/// Converts linear luminance in nits [0.0, 10000.0] to normalized SMPTE ST.2084 non-linear signal [0.0, 1.0].
pub fn pq_inverse_eotf(nits: f32) -> f32 {
    let y = (nits / PQ_MAX_NITS).clamp(0.0, 1.0);
    if y == 0.0 {
        return 0.0;
    }
    let y_m1 = y.powf(PQ_M1);
    let num = PQ_C1 + PQ_C2 * y_m1;
    let den = 1.0 + PQ_C3 * y_m1;
    (num / den).powf(PQ_M2)
}

/// Convert scRGB linear value (where 1.0 = 80 nits) to absolute luminance in nits.
pub fn scrgb_to_nits(scrgb: f32) -> f32 {
    scrgb * REFERENCE_WHITE_NITS
}

/// Convert absolute luminance in nits to scRGB linear value.
pub fn nits_to_scrgb(nits: f32) -> f32 {
    nits / REFERENCE_WHITE_NITS
}

/// 3x3 Color primary transformation matrix.
pub type Mat3 = [f32; 9];

/// Transform matrix from BT.709 (sRGB) linear primaries to BT.2020 linear primaries (row-major).
pub const MAT_BT709_TO_BT2020: Mat3 = [
    0.6274040, 0.3292820, 0.0433136,
    0.0690970, 0.9195400, 0.0113615,
    0.0163916, 0.0880132, 0.8955950,
];

/// Transform matrix from BT.2020 linear primaries to BT.709 (sRGB) linear primaries (row-major).
pub const MAT_BT2020_TO_BT709: Mat3 = [
     1.6604910, -0.5876411, -0.0728499,
    -0.1245505,  1.1328999, -0.0083494,
    -0.0181508, -0.1005789,  1.1187297,
];

/// Apply 3x3 matrix to 3D vector [r, g, b].
pub fn transform_color(m: &Mat3, c: [f32; 3]) -> [f32; 3] {
    [
        m[0] * c[0] + m[1] * c[1] + m[2] * c[2],
        m[3] * c[0] + m[4] * c[1] + m[5] * c[2],
        m[6] * c[0] + m[7] * c[1] + m[8] * c[2],
    ]
}

/// HDR Tonemapping parameters for Gamescope compositing.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct HdrTonemapParams {
    /// Peak display capability in nits (e.g. 400.0, 600.0, 1000.0, 1500.0).
    pub display_peak_nits: f32,
    /// Paper white luminance for UI / SDR content in nits (standard 203 nits).
    pub paper_white_nits: f32,
    /// Target content max luminance in nits.
    pub content_peak_nits: f32,
    /// Tonemap algorithm operator.
    pub operator: TonemapOperator,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum TonemapOperator {
    /// Gamescope Max-RGB Reinhard preserving color purity without hue shift.
    GamescopeMaxRgb,
    /// ACES Film curve fit.
    AcesFilm,
    /// Standard Reinhard curve.
    Reinhard,
    /// Passthrough / unclipped HDR.
    Passthrough,
}

impl Default for HdrTonemapParams {
    fn default() -> Self {
        Self {
            display_peak_nits: 1000.0,
            paper_white_nits: DEFAULT_PAPER_WHITE_NITS,
            content_peak_nits: 1000.0,
            operator: TonemapOperator::GamescopeMaxRgb,
        }
    }
}

impl HdrTonemapParams {
    /// Tonemap a linear RGB color vector.
    pub fn apply(&self, rgb: [f32; 3]) -> [f32; 3] {
        match self.operator {
            TonemapOperator::Passthrough => rgb,
            TonemapOperator::Reinhard => [
                rgb[0] / (1.0 + rgb[0]),
                rgb[1] / (1.0 + rgb[1]),
                rgb[2] / (1.0 + rgb[2]),
            ],
            TonemapOperator::AcesFilm => {
                [
                    aces_film_curve(rgb[0]),
                    aces_film_curve(rgb[1]),
                    aces_film_curve(rgb[2]),
                ]
            }
            TonemapOperator::GamescopeMaxRgb => {
                let max_c = rgb[0].max(rgb[1]).max(rgb[2]);
                if max_c <= 0.0 {
                    return [0.0, 0.0, 0.0];
                }
                // Tone map based on maximum channel to preserve hue
                let scale = self.display_peak_nits / self.content_peak_nits.max(1.0);
                let mapped_max = (max_c * scale) / (1.0 + max_c * scale);
                let ratio = mapped_max / max_c;
                [
                    (rgb[0] * ratio).clamp(0.0, 1.0),
                    (rgb[1] * ratio).clamp(0.0, 1.0),
                    (rgb[2] * ratio).clamp(0.0, 1.0),
                ]
            }
        }
    }
}

/// Narkowicz 2015 ACES film curve approximation.
fn aces_film_curve(x: f32) -> f32 {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    ((x * (a * x + b)) / (x * (c * x + d) + e)).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pq_eotf_roundtrip() {
        let nits_cases = [1.0, 10.0, 80.0, 100.0, 203.0, 1000.0, 4000.0, 10000.0];
        for &expected_nits in &nits_cases {
            let signal = pq_inverse_eotf(expected_nits);
            let recovered_nits = pq_eotf(signal);
            let error = (recovered_nits - expected_nits).abs() / expected_nits;
            assert!(error < 0.005, "Mismatch for {} nits: got {}", expected_nits, recovered_nits);
        }
    }

    #[test]
    fn test_bt709_to_bt2020_white_point_unity() {
        // D65 white (1, 1, 1) should remain (1, 1, 1) after forward and inverse matrix
        let white = [1.0, 1.0, 1.0];
        let in_bt2020 = transform_color(&MAT_BT709_TO_BT2020, white);
        let back_in_bt709 = transform_color(&MAT_BT2020_TO_BT709, in_bt2020);
        for i in 0..3 {
            assert!((back_in_bt709[i] - 1.0).abs() < 0.001);
        }
    }

    #[test]
    fn test_gamescope_max_rgb_preserves_hue() {
        let params = HdrTonemapParams::default();
        let high_dynamic_red = [10.0, 2.0, 1.0]; // Bright saturated reddish color
        let mapped = params.apply(high_dynamic_red);

        // Verify output is in [0, 1] range
        assert!(mapped[0] <= 1.0 && mapped[0] >= 0.0);
        assert!(mapped[1] <= 1.0 && mapped[1] >= 0.0);
        assert!(mapped[2] <= 1.0 && mapped[2] >= 0.0);

        // Verify channel ratios (hue preservation)
        let original_ratio_g = high_dynamic_red[1] / high_dynamic_red[0];
        let mapped_ratio_g = mapped[1] / mapped[0];
        assert!((original_ratio_g - mapped_ratio_g).abs() < 0.0001);
    }
}
