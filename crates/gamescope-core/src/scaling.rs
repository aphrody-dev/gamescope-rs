//! Gamescope scaling mode math and matrix transformations.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// 2D Resolution (width, height) in integer pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Resolution {
    pub width: u32,
    pub height: u32,
}

impl Resolution {
    pub const HD_720P: Self = Self { width: 1280, height: 720 };
    pub const FHD_1080P: Self = Self { width: 1920, height: 1080 };
    pub const QHD_1440P: Self = Self { width: 2560, height: 1440 };
    pub const UHD_4K: Self = Self { width: 3840, height: 2160 };
    pub const STEAM_DECK: Self = Self { width: 1280, height: 800 };

    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    /// Aspect ratio as a floating point value (width / height).
    pub fn aspect_ratio(&self) -> f32 {
        if self.height == 0 {
            0.0
        } else {
            self.width as f32 / self.height as f32
        }
    }

    pub fn total_pixels(&self) -> u64 {
        (self.width as u64) * (self.height as u64)
    }
}

/// Viewport destination rectangle in screen coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ViewportRect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl ViewportRect {
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self { x, y, width, height }
    }

    pub fn to_int_bounds(&self) -> (u32, u32, u32, u32) {
        (
            self.x.round().max(0.0) as u32,
            self.y.round().max(0.0) as u32,
            self.width.round().max(0.0) as u32,
            self.height.round().max(0.0) as u32,
        )
    }
}

/// Gamescope scaling modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ScalingMode {
    /// Integer scaling: pixel-perfect nearest-integer multiplier, centered with black bars.
    Integer,
    /// Fit (Aspect): Preserve source aspect ratio, fitting within output bounds with letterbox/pillarbox.
    Fit,
    /// Stretch: Ignore aspect ratio, stretch completely across output canvas.
    Stretch,
    /// Fill: Zoom/crop source until entire output viewport is covered, preserving aspect ratio.
    Fill,
    /// Center: 1:1 unscaled pixel mapping, centered in output canvas.
    Center,
}

/// Scaling solver calculating layout rect, integer multipliers, and 4x4 NDC projection matrix.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ScalingLayout {
    pub mode: ScalingMode,
    pub source: Resolution,
    pub target: Resolution,
    pub destination_rect: ViewportRect,
    pub integer_scale: u32,
    /// 4x4 column-major affine transformation matrix for WGSL vertex positioning.
    pub transform_matrix: [f32; 16],
}

impl ScalingLayout {
    /// Calculate layout and matrix for a source and target resolution under the specified scaling mode.
    pub fn compute(source: Resolution, target: Resolution, mode: ScalingMode) -> Self {
        let (rect, int_scale) = match mode {
            ScalingMode::Stretch => {
                let rect = ViewportRect::new(0.0, 0.0, target.width as f32, target.height as f32);
                (rect, 1)
            }
            ScalingMode::Center => {
                let w = source.width.min(target.width) as f32;
                let h = source.height.min(target.height) as f32;
                let x = (target.width as f32 - w) * 0.5;
                let y = (target.height as f32 - h) * 0.5;
                (ViewportRect::new(x, y, w, h), 1)
            }
            ScalingMode::Integer => {
                let scale_x = target.width / source.width.max(1);
                let scale_y = target.height / source.height.max(1);
                let scale = scale_x.min(scale_y).max(1);
                let scaled_w = (source.width * scale) as f32;
                let scaled_h = (source.height * scale) as f32;
                let x = (target.width as f32 - scaled_w) * 0.5;
                let y = (target.height as f32 - scaled_h) * 0.5;
                (ViewportRect::new(x, y, scaled_w, scaled_h), scale)
            }
            ScalingMode::Fit => {
                let src_aspect = source.aspect_ratio();
                let dst_aspect = target.aspect_ratio();
                let (w, h) = if src_aspect > dst_aspect {
                    // Source is wider: fit width, letterbox top/bottom
                    let w = target.width as f32;
                    let h = w / src_aspect;
                    (w, h)
                } else {
                    // Source is taller: fit height, pillarbox left/right
                    let h = target.height as f32;
                    let w = h * src_aspect;
                    (w, h)
                };
                let x = (target.width as f32 - w) * 0.5;
                let y = (target.height as f32 - h) * 0.5;
                (ViewportRect::new(x, y, w, h), 1)
            }
            ScalingMode::Fill => {
                let src_aspect = source.aspect_ratio();
                let dst_aspect = target.aspect_ratio();
                let (w, h) = if src_aspect > dst_aspect {
                    // Fill height, crop sides
                    let h = target.height as f32;
                    let w = h * src_aspect;
                    (w, h)
                } else {
                    // Fill width, crop top/bottom
                    let w = target.width as f32;
                    let h = w / src_aspect;
                    (w, h)
                };
                let x = (target.width as f32 - w) * 0.5;
                let y = (target.height as f32 - h) * 0.5;
                (ViewportRect::new(x, y, w, h), 1)
            }
        };

        // Compute 4x4 NDC matrix mapping standard unit quad [0, 1] to target screen NDC [-1, 1]
        let tw = target.width as f32;
        let th = target.height as f32;

        let scale_x = (rect.width / tw) * 2.0;
        let scale_y = (rect.height / th) * 2.0;
        let translate_x = (rect.x / tw) * 2.0 - 1.0;
        let translate_y = 1.0 - (rect.y / th) * 2.0 - scale_y; // Invert Y for Vulkan/WebGPU NDC

        // Column-major 4x4 matrix
        let mut mat = [0.0f32; 16];
        mat[0] = scale_x;
        mat[5] = scale_y;
        mat[10] = 1.0;
        mat[12] = translate_x;
        mat[13] = translate_y;
        mat[15] = 1.0;

        Self {
            mode,
            source,
            target,
            destination_rect: rect,
            integer_scale: int_scale,
            transform_matrix: mat,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integer_scaling_720p_to_1440p() {
        let src = Resolution::HD_720P; // 1280x720
        let dst = Resolution::QHD_1440P; // 2560x1440
        let layout = ScalingLayout::compute(src, dst, ScalingMode::Integer);
        assert_eq!(layout.integer_scale, 2);
        assert_eq!(layout.destination_rect.width, 2560.0);
        assert_eq!(layout.destination_rect.height, 1440.0);
        assert_eq!(layout.destination_rect.x, 0.0);
        assert_eq!(layout.destination_rect.y, 0.0);
    }

    #[test]
    fn test_fit_scaling_pillarbox() {
        let src = Resolution::new(1280, 960); // 4:3
        let dst = Resolution::FHD_1080P; // 16:9 1920x1080
        let layout = ScalingLayout::compute(src, dst, ScalingMode::Fit);
        assert_eq!(layout.destination_rect.height, 1080.0);
        assert!((layout.destination_rect.width - 1440.0).abs() < 0.1);
        assert!((layout.destination_rect.x - 240.0).abs() < 0.1);
        assert_eq!(layout.destination_rect.y, 0.0);
    }

    #[test]
    fn test_stretch_scaling() {
        let src = Resolution::HD_720P;
        let dst = Resolution::UHD_4K;
        let layout = ScalingLayout::compute(src, dst, ScalingMode::Stretch);
        assert_eq!(layout.destination_rect.width, 3840.0);
        assert_eq!(layout.destination_rect.height, 2160.0);
    }
}
