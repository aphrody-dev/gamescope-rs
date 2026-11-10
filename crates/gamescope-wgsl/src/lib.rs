//! Gamescope WGSL: Compute pipelines and shaders for FSR 1.0 EASU, RCAS, and Multi-Plane Compositing.

pub mod pipeline;

pub use pipeline::{
    BenchmarkMetrics, CompositeUniforms, EasuUniforms, GamescopePipeline, HeadlessUpscaler,
    RcasUniforms, COMPOSITE_HDR_WGSL, EASU_WGSL, RCAS_WGSL,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wgsl_shader_compilation() {
        // Test that Naga parses and validates all WGSL shaders cleanly
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }));

        if let Ok(adapter) = adapter {
            let (device, _) = pollster::block_on(adapter.request_device(
                &wgpu::DeviceDescriptor::default(),
            ))
            .expect("Failed to create device");

            let pipeline = GamescopePipeline::new(&device);
            // If we reached here, EASU, RCAS, and Composite HDR shaders successfully parsed and compiled into compute pipelines!
            let _layout = pipeline.easu_pipeline.get_bind_group_layout(0);
        }
    }

    #[test]
    fn test_headless_upscale_rtx4070_execution() {
        // Run a real headless upscale pass on RTX 4070 (e.g. 64x64 -> 128x128)
        let (upscaler, info) = HeadlessUpscaler::new_native((64, 64), (128, 128));
        println!("Running headless upscale test on: {} ({:?})", info.name, info.backend);

        // Generate synthetic test checkerboard pattern
        let mut input_data = vec![0u8; 64 * 64 * 4];
        for y in 0..64 {
            for x in 0..64 {
                let idx = ((y * 64 + x) * 4) as usize;
                let checker = ((x / 8) + (y / 8)) % 2 == 0;
                let val = if checker { 240 } else { 40 };
                input_data[idx] = val;     // R
                input_data[idx + 1] = val; // G
                input_data[idx + 2] = val; // B
                input_data[idx + 3] = 255; // A
            }
        }

        let output_pixels = upscaler.execute_and_readback(&input_data);
        assert_eq!(output_pixels.len(), 128 * 128 * 4);

        // Verify output pixels have non-zero luminance
        let non_zero_count = output_pixels.iter().filter(|&&p| p > 0).count();
        assert!(non_zero_count > 0, "Upscaled output should not be black");
    }
}
