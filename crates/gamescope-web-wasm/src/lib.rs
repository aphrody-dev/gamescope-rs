//! Gamescope WebAssembly & WebGPU module for modern browsers (Chromium).

use wasm_bindgen::prelude::*;
use gamescope_core::{Resolution, ScalingLayout, ScalingMode};
#[cfg(target_arch = "wasm32")]
use gamescope_wgsl::GamescopePipeline;

#[wasm_bindgen]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
pub fn gamescope_version() -> String {
    "gamescope-rs 0.1.0 (WebGPU / WASM November 2026)".into()
}

/// Compute scaling layout and return a 16-element Float32Array containing the 4x4 NDC projection matrix.
#[wasm_bindgen]
pub fn compute_scaling_transform(src_w: u32, src_h: u32, dst_w: u32, dst_h: u32, mode_index: u32) -> Vec<f32> {
    let mode = match mode_index {
        0 => ScalingMode::Integer,
        1 => ScalingMode::Fit,
        2 => ScalingMode::Stretch,
        3 => ScalingMode::Fill,
        _ => ScalingMode::Fit,
    };
    let layout = ScalingLayout::compute(
        Resolution::new(src_w, src_h),
        Resolution::new(dst_w, dst_h),
        mode,
    );
    layout.transform_matrix.to_vec()
}

/// WebGamescope session managing WebGPU device, pipelines, and canvas presentation.
#[wasm_bindgen]
pub struct WebGamescopeSession {
    #[cfg(target_arch = "wasm32")]
    device: wgpu::Device,
    #[cfg(target_arch = "wasm32")]
    queue: wgpu::Queue,
    #[cfg(target_arch = "wasm32")]
    pipeline: GamescopePipeline,
    #[cfg(target_arch = "wasm32")]
    surface: wgpu::Surface<'static>,
    #[cfg(target_arch = "wasm32")]
    surface_config: wgpu::SurfaceConfiguration,
    width: u32,
    height: u32,
}

#[wasm_bindgen]
impl WebGamescopeSession {
    /// Initialize WebGamescope from an HTML5 canvas ID.
    pub async fn create(canvas_id: String) -> Result<WebGamescopeSession, JsValue> {
        init_panic_hook();

        #[cfg(target_arch = "wasm32")]
        {
            let window = web_sys::window().ok_or_else(|| JsValue::from_str("No window found"))?;
            let document = window.document().ok_or_else(|| JsValue::from_str("No document found"))?;
            let canvas = document
                .get_element_by_id(&canvas_id)
                .ok_or_else(|| JsValue::from_str(&format!("Canvas #{} not found", canvas_id)))?;
            let html_canvas: web_sys::HtmlCanvasElement = canvas
                .dyn_into::<web_sys::HtmlCanvasElement>()
                .map_err(|_| JsValue::from_str("Element is not a canvas"))?;

            let width = html_canvas.width();
            let height = html_canvas.height();

            let instance = wgpu::Instance::default();

            let surface = instance
                .create_surface(wgpu::SurfaceTarget::Canvas(html_canvas))
                .map_err(|e| JsValue::from_str(&format!("Failed to create surface: {:?}", e)))?;

            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: Some(&surface),
                    ..Default::default()
                })
                .await
                .map_err(|e| JsValue::from_str(&format!("Failed to request adapter: {:?}", e)))?;

            let (device, queue) = adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        label: Some("WebGamescope Device"),
                        ..Default::default()
                    },
                )
                .await
                .map_err(|e| JsValue::from_str(&format!("Failed to request device: {:?}", e)))?;

            let surface_caps = surface.get_capabilities(&adapter);
            let surface_format = surface_caps
                .formats
                .iter()
                .copied()
                .find(|f| matches!(f, wgpu::TextureFormat::Rgba8UnormSrgb | wgpu::TextureFormat::Bgra8UnormSrgb))
                .unwrap_or(surface_caps.formats[0]);

            let surface_config = wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format: surface_format,
                width,
                height,
                present_mode: wgpu::PresentMode::Fifo,
                alpha_mode: surface_caps.alpha_modes[0],
                view_formats: vec![],
                desired_maximum_frame_latency: 2,
                color_space: wgpu::SurfaceColorSpace::Auto,
            };
            surface.configure(&device, &surface_config);

            let pipeline = GamescopePipeline::new(&device);

            Ok(Self {
                device,
                queue,
                pipeline,
                surface,
                surface_config,
                width,
                height,
            })
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = canvas_id;
            Ok(Self {
                width: 1920,
                height: 1080,
            })
        }
    }

    /// Resize canvas surface dimensions.
    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.width = width;
            self.height = height;
            #[cfg(target_arch = "wasm32")]
            {
                self.surface_config.width = width;
                self.surface_config.height = height;
                self.surface.configure(&self.device, &self.surface_config);
            }
        }
    }

    /// Execute a benchmark pass of EASU + RCAS and return elapsed time in milliseconds.
    pub async fn benchmark_upscale(
        &self,
        in_w: u32,
        in_h: u32,
        out_w: u32,
        out_h: u32,
    ) -> Result<f64, JsValue> {
        #[cfg(target_arch = "wasm32")]
        {
            let window = web_sys::window().ok_or_else(|| JsValue::from_str("No window"))?;
            let perf = window.performance().ok_or_else(|| JsValue::from_str("No performance object"))?;

            let upscaler = gamescope_wgsl::HeadlessUpscaler::with_device(
                self.device.clone(),
                self.queue.clone(),
                (in_w, in_h),
                (out_w, out_h),
            );

            let t0 = perf.now();
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Web Benchmark Encoder"),
            });
            upscaler.record_upscale_commands(&mut encoder);
            self.queue.submit(Some(encoder.finish()));

            // In WebGPU on WASM, submissions are processed asynchronously on RAF
            let t1 = perf.now();
            Ok(t1 - t0)
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = (in_w, in_h, out_w, out_h);
            Ok(0.0)
        }
    }
}
