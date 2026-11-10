// AMD FidelityFX Super Resolution 1.0 (FSR 1.0)
// EASU: Edge-Adaptive Spatial Upsampling compute pass in pure WGSL.

struct EasuUniforms {
    // x: input_width, y: input_height, z: 1.0/input_width, w: 1.0/input_height
    input_size: vec4<f32>,
    // x: output_width, y: output_height, z: 1.0/output_width, w: 1.0/output_height
    output_size: vec4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: EasuUniforms;
@group(0) @binding(1) var input_texture: texture_2d<f32>;
@group(0) @binding(2) var output_texture: texture_storage_2d<rgba8unorm, write>;

fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn sample_clamped(coord: vec2<i32>) -> vec3<f32> {
    let max_x = i32(uniforms.input_size.x) - 1;
    let max_y = i32(uniforms.input_size.y) - 1;
    let clamped_coord = vec2<i32>(
        clamp(coord.x, 0, max_x),
        clamp(coord.y, 0, max_y)
    );
    return textureLoad(input_texture, clamped_coord, 0).rgb;
}

// 2-lobed Lanczos approximation: sinc(x) * sinc(x / 2.0)
fn lanczos2_approx(x: f32) -> f32 {
    let d = abs(x);
    if (d >= 2.0) {
        return 0.0;
    }
    // High-precision polynomial approximation of Lanczos2 window
    let d2 = d * d;
    let w = (1.0 - 0.5 * d2) * (1.0 - 0.25 * d2);
    return max(0.0, w);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_pos = global_id.xy;
    if (out_pos.x >= u32(uniforms.output_size.x) || out_pos.y >= u32(uniforms.output_size.y)) {
        return;
    }

    // Map output pixel center to input coordinate space
    let pp = (vec2<f32>(out_pos) + vec2<f32>(0.5, 0.5)) * uniforms.input_size.xy * uniforms.output_size.zw - vec2<f32>(0.5, 0.5);
    let fp = floor(pp);
    let base = vec2<i32>(fp);
    let f = pp - fp; // Subpixel phase in [0, 1]

    // 12-tap sampling kernel
    // Top row
    let c_a0 = sample_clamped(base + vec2<i32>(0, -1));
    let c_a1 = sample_clamped(base + vec2<i32>(1, -1));

    // Mid-upper row
    let c_d0 = sample_clamped(base + vec2<i32>(-1, 0));
    let c_b  = sample_clamped(base + vec2<i32>(0, 0));
    let c_c  = sample_clamped(base + vec2<i32>(1, 0));
    let c_e0 = sample_clamped(base + vec2<i32>(2, 0));

    // Mid-lower row
    let c_d1 = sample_clamped(base + vec2<i32>(-1, 1));
    let c_f  = sample_clamped(base + vec2<i32>(0, 1));
    let c_g  = sample_clamped(base + vec2<i32>(1, 1));
    let c_e1 = sample_clamped(base + vec2<i32>(2, 1));

    // Bottom row
    let c_h0 = sample_clamped(base + vec2<i32>(0, 2));
    let c_h1 = sample_clamped(base + vec2<i32>(1, 2));

    // Luminance computation for edge analysis
    let l_b  = luma(c_b);
    let l_c  = luma(c_c);
    let l_f  = luma(c_f);
    let l_g  = luma(c_g);
    let l_a0 = luma(c_a0);
    let l_a1 = luma(c_a1);
    let l_d0 = luma(c_d0);
    let l_e0 = luma(c_e0);
    let l_d1 = luma(c_d1);
    let l_e1 = luma(c_e1);
    let l_h0 = luma(c_h0);
    let l_h1 = luma(c_h1);

    // Directional gradients
    let gx = (l_c - l_b) + (l_g - l_f) + 0.5 * ((l_e0 - l_d0) + (l_e1 - l_d1));
    let gy = (l_f - l_b) + (l_g - l_c) + 0.5 * ((l_h0 - l_a0) + (l_h1 - l_a1));
    let len = sqrt(gx * gx + gy * gy);

    var dir = vec2<f32>(1.0, 0.0);
    if (len > 0.0001) {
        dir = vec2<f32>(gx, gy) / len;
    }
    let normal = vec2<f32>(-dir.y, dir.x);

    // Edge-adaptive stretch factor
    let edge_strength = clamp(len * 3.0, 0.0, 1.0);
    let stretch_cross = 1.0 + edge_strength * 1.5;

    // Weight and accumulate 12 taps
    var total_color = vec3<f32>(0.0);
    var total_weight = 0.0;

    let taps_offset = array<vec2<f32>, 12>(
        vec2<f32>(0.0, -1.0), vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, 0.0), vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(2.0, 0.0),
        vec2<f32>(-1.0, 1.0), vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 1.0), vec2<f32>(2.0, 1.0),
        vec2<f32>(0.0, 2.0), vec2<f32>(1.0, 2.0)
    );

    let taps_color = array<vec3<f32>, 12>(
        c_a0, c_a1,
        c_d0, c_b, c_c, c_e0,
        c_d1, c_f, c_g, c_e1,
        c_h0, c_h1
    );

    for (var i = 0u; i < 12u; i = i + 1u) {
        let rel = taps_offset[i] - f;
        let proj_along = dot(rel, dir);
        let proj_cross = dot(rel, normal) * stretch_cross;
        let dist = sqrt(proj_along * proj_along + proj_cross * proj_cross);
        let w = lanczos2_approx(dist);

        total_color = total_color + taps_color[i] * w;
        total_weight = total_weight + w;
    }

    var upscaled = total_color / max(total_weight, 0.0001);

    // Anti-ringing bounding box clamping to central 4 taps
    let box_min = min(min(c_b, c_c), min(c_f, c_g));
    let box_max = max(max(c_b, c_c), max(c_f, c_g));
    upscaled = clamp(upscaled, box_min, box_max);

    textureStore(output_texture, out_pos, vec4<f32>(upscaled, 1.0));
}
