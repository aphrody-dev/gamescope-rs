//! Gamescope wgpu compute pipeline builder and execution runner.

use std::borrow::Cow;
use gamescope_core::Resolution;

pub const EASU_WGSL: &str = include_str!("shaders/easu.wgsl");
pub const RCAS_WGSL: &str = include_str!("shaders/rcas.wgsl");
pub const COMPOSITE_HDR_WGSL: &str = include_str!("shaders/composite_hdr.wgsl");

/// EASU shader uniform buffer payload.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct EasuUniforms {
    pub input_size: [f32; 4],
    pub output_size: [f32; 4],
}

/// RCAS shader uniform buffer payload.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RcasUniforms {
    pub output_size: [f32; 2],
    pub inv_output_size: [f32; 2],
    pub sharpness: f32,
    pub pad0: f32,
    pub pad1: f32,
    pub pad2: f32,
}

/// Composite HDR shader uniform buffer payload.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CompositeUniforms {
    pub output_size: [f32; 2],
    pub tonemap_operator: u32,
    pub display_peak_nits: f32,
    pub content_peak_nits: f32,
    pub paper_white_nits: f32,
    pub hdr_output_mode: u32,
    pub pad: u32,
}

/// Gamescope Compute Pipelines for EASU, RCAS, and Compositor.
pub struct GamescopePipeline {
    pub easu_pipeline: wgpu::ComputePipeline,
    pub easu_bgl: wgpu::BindGroupLayout,
    pub rcas_pipeline: wgpu::ComputePipeline,
    pub rcas_bgl: wgpu::BindGroupLayout,
    pub composite_pipeline: wgpu::ComputePipeline,
    pub composite_bgl: wgpu::BindGroupLayout,
}

impl GamescopePipeline {
    /// Compile WGSL shaders and create compute pipelines on the target device.
    pub fn new(device: &wgpu::Device) -> Self {
        // EASU Bind Group Layout
        let easu_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Gamescope EASU BindGroupLayout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let easu_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("EASU Shader Module"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(EASU_WGSL)),
        });

        let easu_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("EASU Pipeline Layout"),
            bind_group_layouts: &[Some(&easu_bgl)],
            immediate_size: 0,
        });

        let easu_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("EASU Compute Pipeline"),
            layout: Some(&easu_layout),
            module: &easu_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // RCAS Bind Group Layout
        let rcas_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Gamescope RCAS BindGroupLayout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let rcas_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("RCAS Shader Module"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(RCAS_WGSL)),
        });

        let rcas_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("RCAS Pipeline Layout"),
            bind_group_layouts: &[Some(&rcas_bgl)],
            immediate_size: 0,
        });

        let rcas_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("RCAS Compute Pipeline"),
            layout: Some(&rcas_layout),
            module: &rcas_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Composite HDR Bind Group Layout
        let composite_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Gamescope Composite HDR BindGroupLayout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let composite_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Composite HDR Shader Module"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(COMPOSITE_HDR_WGSL)),
        });

        let composite_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Composite HDR Pipeline Layout"),
            bind_group_layouts: &[Some(&composite_bgl)],
            immediate_size: 0,
        });

        let composite_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Composite HDR Compute Pipeline"),
            layout: Some(&composite_layout),
            module: &composite_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            easu_pipeline,
            easu_bgl,
            rcas_pipeline,
            rcas_bgl,
            composite_pipeline,
            composite_bgl,
        }
    }
}

/// Headless GPU Upscaler & Benchmarking runner for EASU + RCAS pipeline.
pub struct HeadlessUpscaler {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub pipeline: GamescopePipeline,
    pub input_res: (u32, u32),
    pub output_res: (u32, u32),
    pub input_texture: wgpu::Texture,
    pub intermediate_texture: wgpu::Texture,
    pub output_texture: wgpu::Texture,
    pub easu_uniform_buffer: wgpu::Buffer,
    pub rcas_uniform_buffer: wgpu::Buffer,
    pub easu_bind_group: wgpu::BindGroup,
    pub rcas_bind_group: wgpu::BindGroup,
}

#[derive(Debug, Clone)]
pub struct BenchmarkMetrics {
    pub input_resolution: (u32, u32),
    pub output_resolution: (u32, u32),
    pub iterations: usize,
    pub total_time_ms: f64,
    pub avg_time_ms: f64,
    pub min_time_ms: f64,
    pub max_time_ms: f64,
    pub throughput_gigapixels_per_sec: f64,
    pub adapter_name: String,
    pub backend: String,
}

impl HeadlessUpscaler {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new_native(input_res: (u32, u32), output_res: (u32, u32)) -> (Self, wgpu::AdapterInfo) {
        let instance = wgpu::Instance::default();

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .expect("Failed to find suitable GPU adapter");

        let adapter_info = adapter.get_info();

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Gamescope Headless Device"),
                ..Default::default()
            },
        ))
        .expect("Failed to create wgpu device");

        let upscaler = Self::with_device(device, queue, input_res, output_res);
        (upscaler, adapter_info)
    }

    pub fn with_resolutions(
        device: wgpu::Device,
        queue: wgpu::Queue,
        input: Resolution,
        output: Resolution,
    ) -> Self {
        Self::with_device(device, queue, (input.width, input.height), (output.width, output.height))
    }

    pub fn with_device(
        device: wgpu::Device,
        queue: wgpu::Queue,
        input_res: (u32, u32),
        output_res: (u32, u32),
    ) -> Self {
        let pipeline = GamescopePipeline::new(&device);

        // Input Texture
        let input_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Gamescope Input Texture"),
            size: wgpu::Extent3d {
                width: input_res.0,
                height: input_res.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let input_view = input_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Intermediate Texture (EASU output -> RCAS input)
        let intermediate_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Gamescope Intermediate Texture"),
            size: wgpu::Extent3d {
                width: output_res.0,
                height: output_res.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let intermediate_view = intermediate_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Final Output Texture
        let output_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Gamescope Output Texture"),
            size: wgpu::Extent3d {
                width: output_res.0,
                height: output_res.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let output_view = output_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // EASU Uniform Buffer
        let easu_uniforms = EasuUniforms {
            input_size: [
                input_res.0 as f32,
                input_res.1 as f32,
                1.0 / input_res.0 as f32,
                1.0 / input_res.1 as f32,
            ],
            output_size: [
                output_res.0 as f32,
                output_res.1 as f32,
                1.0 / output_res.0 as f32,
                1.0 / output_res.1 as f32,
            ],
        };

        use wgpu::util::DeviceExt;
        let easu_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("EASU Uniform Buffer"),
            contents: bytemuck::bytes_of(&easu_uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let easu_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("EASU BindGroup"),
            layout: &pipeline.easu_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: easu_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&input_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&intermediate_view),
                },
            ],
        });

        // RCAS Uniform Buffer
        let rcas_uniforms = RcasUniforms {
            output_size: [output_res.0 as f32, output_res.1 as f32],
            inv_output_size: [1.0 / output_res.0 as f32, 1.0 / output_res.1 as f32],
            sharpness: 0.2, // standard sharp setting
            pad0: 0.0,
            pad1: 0.0,
            pad2: 0.0,
        };

        let rcas_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("RCAS Uniform Buffer"),
            contents: bytemuck::bytes_of(&rcas_uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let rcas_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RCAS BindGroup"),
            layout: &pipeline.rcas_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: rcas_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&intermediate_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&output_view),
                },
            ],
        });

        Self {
            device,
            queue,
            pipeline,
            input_res,
            output_res,
            input_texture,
            intermediate_texture,
            output_texture,
            easu_uniform_buffer,
            rcas_uniform_buffer,
            easu_bind_group,
            rcas_bind_group,
        }
    }

    /// Upload an RGBA image buffer to the input texture.
    pub fn upload_input(&self, rgba_data: &[u8]) {
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.input_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            rgba_data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(self.input_res.0 * 4),
                rows_per_image: Some(self.input_res.1),
            },
            wgpu::Extent3d {
                width: self.input_res.0,
                height: self.input_res.1,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Dispatch both EASU and RCAS compute passes in a single command buffer.
    pub fn record_upscale_commands(&self, encoder: &mut wgpu::CommandEncoder) {
        let workgroups_x = (self.output_res.0 + 15) / 16;
        let workgroups_y = (self.output_res.1 + 15) / 16;

        // Pass 1: EASU Compute Pass
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("EASU Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipeline.easu_pipeline);
            compute_pass.set_bind_group(0, &self.easu_bind_group, &[]);
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        // Pass 2: RCAS Compute Pass
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("RCAS Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipeline.rcas_pipeline);
            compute_pass.set_bind_group(0, &self.rcas_bind_group, &[]);
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }
    }

    /// Run full headless upscale pass and read back output pixel buffer.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn execute_and_readback(&self, input_rgba: &[u8]) -> Vec<u8> {
        self.upload_input(input_rgba);

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Headless Upscale Encoder"),
        });

        self.record_upscale_commands(&mut encoder);

        // Copy output texture to staging buffer
        let bytes_per_pixel = 4u32;
        let unaligned_bytes_per_row = self.output_res.0 * bytes_per_pixel;
        // Row pitch must be aligned to 256 bytes for wgpu buffer copies
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bytes_per_row = (unaligned_bytes_per_row + align - 1) & !(align - 1);
        let buffer_size = (padded_bytes_per_row * self.output_res.1) as u64;

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Staging Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.output_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(self.output_res.1),
                },
            },
            wgpu::Extent3d {
                width: self.output_res.0,
                height: self.output_res.1,
                depth_or_array_layers: 1,
            },
        );

        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |res| {
            tx.send(res).unwrap();
        });

        self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        rx.recv().unwrap().expect("Failed to map staging buffer");

        let mapped = buffer_slice.get_mapped_range().expect("Failed to get mapped range");
        let mut result = Vec::with_capacity((self.output_res.0 * self.output_res.1 * 4) as usize);

        for row in 0..self.output_res.1 {
            let row_start = (row * padded_bytes_per_row) as usize;
            let row_end = row_start + (self.output_res.0 * 4) as usize;
            result.extend_from_slice(&mapped[row_start..row_end]);
        }

        drop(mapped);
        staging_buffer.unmap();

        result
    }

    /// Run benchmark iterations measuring real GPU execution time.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn benchmark(
        &self,
        iterations: usize,
        adapter_info: &wgpu::AdapterInfo,
    ) -> BenchmarkMetrics {
        // Warmup passes
        for _ in 0..3 {
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Warmup Encoder"),
            });
            self.record_upscale_commands(&mut encoder);
            self.queue.submit(Some(encoder.finish()));
            self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        }

        let mut samples_ms = Vec::with_capacity(iterations);

        for _ in 0..iterations {
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Benchmark Encoder"),
            });
            self.record_upscale_commands(&mut encoder);

            let start = std::time::Instant::now();
            self.queue.submit(Some(encoder.finish()));
            self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            let elapsed = start.elapsed();

            samples_ms.push(elapsed.as_secs_f64() * 1000.0);
        }

        let total_time_ms: f64 = samples_ms.iter().sum();
        let avg_time_ms = total_time_ms / (iterations as f64);
        let min_time_ms = samples_ms.iter().copied().fold(f64::INFINITY, f64::min);
        let max_time_ms = samples_ms.iter().copied().fold(0.0, f64::max);

        let pixels_per_frame = (self.output_res.0 as f64) * (self.output_res.1 as f64);
        let total_pixels = pixels_per_frame * (iterations as f64);
        let throughput_gigapixels_per_sec = (total_pixels / (total_time_ms / 1000.0)) / 1_000_000_000.0;

        BenchmarkMetrics {
            input_resolution: self.input_res,
            output_resolution: self.output_res,
            iterations,
            total_time_ms,
            avg_time_ms,
            min_time_ms,
            max_time_ms,
            throughput_gigapixels_per_sec,
            adapter_name: adapter_info.name.clone(),
            backend: format!("{:?}", adapter_info.backend),
        }
    }
}
