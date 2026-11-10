//! Micro-compositor frame-pacing scheduler and latency budget calculator.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Target display refresh rate and VRR / Adaptive-Sync configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RefreshRate {
    /// Nominal refresh rate in Hertz (e.g. 60.0, 120.0, 144.0, 165.0, 240.0).
    pub hz: f64,
    /// Whether Variable Refresh Rate (VRR / G-Sync / FreeSync) is enabled.
    pub vrr_enabled: bool,
    /// Minimum VRR refresh rate in Hertz (e.g. 40.0 or 48.0 Hz for LFC).
    pub vrr_min_hz: f64,
}

impl Default for RefreshRate {
    fn default() -> Self {
        Self {
            hz: 120.0,
            vrr_enabled: true,
            vrr_min_hz: 48.0,
        }
    }
}

impl RefreshRate {
    pub const HZ_60: Self = Self { hz: 60.0, vrr_enabled: false, vrr_min_hz: 60.0 };
    pub const HZ_120: Self = Self { hz: 120.0, vrr_enabled: true, vrr_min_hz: 48.0 };
    pub const HZ_144: Self = Self { hz: 144.0, vrr_enabled: true, vrr_min_hz: 48.0 };
    pub const HZ_165: Self = Self { hz: 165.0, vrr_enabled: true, vrr_min_hz: 48.0 };
    pub const HZ_240: Self = Self { hz: 240.0, vrr_enabled: true, vrr_min_hz: 48.0 };

    pub fn new(hz: f64, vrr: bool) -> Self {
        let min_hz = if vrr { (hz * 0.4).max(40.0) } else { hz };
        Self {
            hz: hz.max(1.0),
            vrr_enabled: vrr,
            vrr_min_hz: min_hz,
        }
    }

    /// Target frame interval duration in milliseconds.
    pub fn target_interval_ms(&self) -> f64 {
        1000.0 / self.hz
    }

    /// Target frame interval duration in microseconds.
    pub fn target_interval_us(&self) -> u64 {
        (1_000_000.0 / self.hz).round() as u64
    }

    /// Maximum frame interval allowed before LFC (Low Framerate Compensation) kicks in.
    pub fn max_interval_ms(&self) -> f64 {
        1000.0 / self.vrr_min_hz
    }
}

/// Rolling frame-time histogram supporting ring-buffer statistical sampling,
/// standard deviation, and 1% / 0.1% low FPS computation.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RollingFrameHistogram<const CAPACITY: usize = 512> {
    samples_ms: [f64; CAPACITY],
    cursor: usize,
    count: usize,
    total_frames: u64,
}

impl<const CAPACITY: usize> Default for RollingFrameHistogram<CAPACITY> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const CAPACITY: usize> RollingFrameHistogram<CAPACITY> {
    pub fn new() -> Self {
        assert!(CAPACITY > 0, "Capacity must be greater than zero");
        Self {
            samples_ms: [0.0; CAPACITY],
            cursor: 0,
            count: 0,
            total_frames: 0,
        }
    }

    /// Record a frame duration in milliseconds.
    pub fn record_sample(&mut self, frame_time_ms: f64) {
        if frame_time_ms > 0.0 {
            self.samples_ms[self.cursor] = frame_time_ms;
            self.cursor = (self.cursor + 1) % CAPACITY;
            if self.count < CAPACITY {
                self.count += 1;
            }
            self.total_frames += 1;
        }
    }

    /// Number of samples currently held in the window.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Total number of recorded frames since creation.
    pub fn total_frames(&self) -> u64 {
        self.total_frames
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Average frame duration across the current rolling window in milliseconds.
    pub fn average_frame_time_ms(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        let sum: f64 = self.samples_ms[..self.count].iter().sum();
        sum / (self.count as f64)
    }

    /// Mean FPS across the rolling window.
    pub fn average_fps(&self) -> f64 {
        let avg_ms = self.average_frame_time_ms();
        if avg_ms > 0.0 {
            1000.0 / avg_ms
        } else {
            0.0
        }
    }

    /// Minimum frame duration in the window in milliseconds.
    pub fn min_frame_time_ms(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        self.samples_ms[..self.count]
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min)
    }

    /// Maximum frame duration in the window in milliseconds.
    pub fn max_frame_time_ms(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        self.samples_ms[..self.count]
            .iter()
            .copied()
            .fold(0.0, f64::max)
    }

    /// Standard deviation of frame times in milliseconds (measures stutter).
    pub fn standard_deviation_ms(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        let avg = self.average_frame_time_ms();
        let variance: f64 = self.samples_ms[..self.count]
            .iter()
            .map(|&x| {
                let diff = x - avg;
                diff * diff
            })
            .sum::<f64>()
            / (self.count as f64);
        variance.sqrt()
    }

    /// Compute an arbitrary percentile of frame times in milliseconds.
    /// `percentile` is in range `[0.0, 100.0]`. For example, 99.0 corresponds to the slowest 1% frame times.
    pub fn percentile_frame_time_ms(&self, percentile: f64) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        let mut sorted = [0.0; CAPACITY];
        sorted[..self.count].copy_from_slice(&self.samples_ms[..self.count]);
        let slice = &mut sorted[..self.count];
        slice.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let rank = (percentile / 100.0) * (self.count as f64 - 1.0);
        let lower = rank.floor() as usize;
        let upper = rank.ceil() as usize;
        if lower == upper {
            slice[lower]
        } else {
            let t = rank - (lower as f64);
            slice[lower] * (1.0 - t) + slice[upper] * t
        }
    }

    /// 1% Low FPS metric: based on the 99th percentile slowest frame times.
    pub fn one_percent_low_fps(&self) -> f64 {
        let p99_ms = self.percentile_frame_time_ms(99.0);
        if p99_ms > 0.0 {
            1000.0 / p99_ms
        } else {
            0.0
        }
    }

    /// 0.1% Low FPS metric: based on the 99.9th percentile slowest frame times.
    pub fn point_one_percent_low_fps(&self) -> f64 {
        let p99_9_ms = self.percentile_frame_time_ms(99.9);
        if p99_9_ms > 0.0 {
            1000.0 / p99_9_ms
        } else {
            0.0
        }
    }

    /// Frame-to-frame jitter (mean absolute delta between consecutive frames) in milliseconds.
    pub fn jitter_ms(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        let mut diff_sum = 0.0;
        for i in 1..self.count {
            diff_sum += (self.samples_ms[i] - self.samples_ms[i - 1]).abs();
        }
        diff_sum / ((self.count - 1) as f64)
    }
}

/// Comprehensive Input-to-Photon Latency Budget Calculator for Gamescope micro-compositing.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LatencyBudgetCalculator {
    /// Input polling delay (time from OS input event to game loop read).
    pub input_poll_ms: f64,
    /// Game simulation & logic update time.
    pub cpu_game_loop_ms: f64,
    /// Wayland client buffer commit / command recording time.
    pub cpu_render_record_ms: f64,
    /// Gamescope compositor GPU pass execution time (EASU + RCAS + HDR Tonemap).
    pub gpu_execution_ms: f64,
    /// Swapchain present queue delay before hardware vblank flip.
    pub present_queue_ms: f64,
    /// Display scanout duration (half-refresh scanout average).
    pub display_scanout_ms: f64,
}

impl Default for LatencyBudgetCalculator {
    fn default() -> Self {
        Self {
            input_poll_ms: 1.0,
            cpu_game_loop_ms: 3.5,
            cpu_render_record_ms: 1.2,
            gpu_execution_ms: 2.1,
            present_queue_ms: 0.8,
            display_scanout_ms: 4.16, // at 120Hz, scanout is ~8.33ms / 2
        }
    }
}

impl LatencyBudgetCalculator {
    /// Calculate total end-to-end input-to-photon latency in milliseconds.
    pub fn total_latency_ms(&self) -> f64 {
        self.input_poll_ms
            + self.cpu_game_loop_ms
            + self.cpu_render_record_ms
            + self.gpu_execution_ms
            + self.present_queue_ms
            + self.display_scanout_ms
    }

    /// Calculate processing time before scanout.
    pub fn pipeline_latency_ms(&self) -> f64 {
        self.input_poll_ms
            + self.cpu_game_loop_ms
            + self.cpu_render_record_ms
            + self.gpu_execution_ms
            + self.present_queue_ms
    }

    /// Remaining latency headroom against a given refresh rate target interval.
    pub fn headroom_ms(&self, refresh_rate: &RefreshRate) -> f64 {
        refresh_rate.target_interval_ms() - self.pipeline_latency_ms()
    }

    /// Whether the frame completed within the frame interval budget.
    pub fn is_within_budget(&self, refresh_rate: &RefreshRate) -> bool {
        self.headroom_ms(refresh_rate) >= 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_refresh_rate_intervals() {
        let r60 = RefreshRate::HZ_60;
        assert!((r60.target_interval_ms() - 16.66666).abs() < 0.01);
        assert_eq!(r60.target_interval_us(), 16667);

        let r120 = RefreshRate::HZ_120;
        assert!((r120.target_interval_ms() - 8.33333).abs() < 0.01);
    }

    #[test]
    fn test_rolling_frame_histogram_stats() {
        let mut hist = RollingFrameHistogram::<64>::new();
        assert_eq!(hist.len(), 0);

        // Feed constant 8.33 ms (120 FPS)
        for _ in 0..50 {
            hist.record_sample(8.33);
        }
        assert_eq!(hist.len(), 50);
        assert!((hist.average_fps() - 120.0).abs() < 0.5);
        assert!((hist.standard_deviation_ms() - 0.0).abs() < 0.001);

        // Introduce a frame drop / hitch
        hist.record_sample(33.33); // 30 FPS drop
        assert!(hist.max_frame_time_ms() > 30.0);
        assert!(hist.standard_deviation_ms() > 0.0);
        assert!(hist.one_percent_low_fps() < 120.0);
    }

    #[test]
    fn test_latency_budget() {
        let budget = LatencyBudgetCalculator {
            input_poll_ms: 0.5,
            cpu_game_loop_ms: 2.0,
            cpu_render_record_ms: 0.8,
            gpu_execution_ms: 1.2,
            present_queue_ms: 0.5,
            display_scanout_ms: 4.16,
        };
        assert!((budget.pipeline_latency_ms() - 5.0).abs() < 0.001);
        assert!((budget.total_latency_ms() - 9.16).abs() < 0.001);
        let r120 = RefreshRate::HZ_120;
        assert!(budget.is_within_budget(&r120));
        assert!((budget.headroom_ms(&r120) - 3.3333).abs() < 0.01);
    }
}
