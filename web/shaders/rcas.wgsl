// AMD FidelityFX Super Resolution 1.0 (FSR 1.0)
// RCAS: Robust Contrast-Adaptive Sharpening compute pass in pure WGSL.

struct RcasUniforms {
    output_size: vec2<f32>,
    inv_output_size: vec2<f32>,
    sharpness: f32, // Attenuation factor: 0.0 (maximum sharpness) to 1.0 (subtle)
    pad0: f32,
    pad1: f32,
    pad2: f32,
};

@group(0) @binding(0) var<uniform> uniforms: RcasUniforms;
@group(0) @binding(1) var input_texture: texture_2d<f32>;
@group(0) @binding(2) var output_texture: texture_storage_2d<rgba8unorm, write>;

fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn sample_clamped(coord: vec2<i32>) -> vec3<f32> {
    let max_x = i32(uniforms.output_size.x) - 1;
    let max_y = i32(uniforms.output_size.y) - 1;
    let clamped_coord = vec2<i32>(
        clamp(coord.x, 0, max_x),
        clamp(coord.y, 0, max_y)
    );
    return textureLoad(input_texture, clamped_coord, 0).rgb;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pos = global_id.xy;
    if (pos.x >= u32(uniforms.output_size.x) || pos.y >= u32(uniforms.output_size.y)) {
        return;
    }

    let ipos = vec2<i32>(pos);

    // 5-tap cross neighborhood
    let c_m = sample_clamped(ipos);
    let c_u = sample_clamped(ipos + vec2<i32>(0, -1));
    let c_d = sample_clamped(ipos + vec2<i32>(0, 1));
    let c_l = sample_clamped(ipos + vec2<i32>(-1, 0));
    let c_r = sample_clamped(ipos + vec2<i32>(1, 0));

    // Per-channel and luminance contrast analysis
    let l_m = luma(c_m);
    let l_u = luma(c_u);
    let l_d = luma(c_d);
    let l_l = luma(c_l);
    let l_r = luma(c_r);

    let min_l = min(l_m, min(min(l_u, l_d), min(l_l, l_r)));
    let max_l = max(l_m, max(max(l_u, l_d), max(l_l, l_r)));

    // Contrast calculation
    let contrast = max_l - min_l;

    // Soft-limit distance to peak / black
    let hit_min = min_l;
    let hit_max = 1.0 - max_l;
    let lobe_limit = min(hit_min, hit_max);

    // Dynamic negative lobe weight inversely proportional to contrast
    // Peak sharpening when contrast is high, smoothly clamped near boundaries
    let base_sharpness = clamp(1.0 - uniforms.sharpness * 0.5, 0.1, 1.0);
    var lobe = -base_sharpness * (0.25 * lobe_limit / max(contrast, 0.05));
    lobe = clamp(lobe, -0.22, 0.0);

    // Filtered result
    let cross_sum = c_u + c_d + c_l + c_r;
    var sharpened = (c_m + lobe * cross_sum) / (1.0 + 4.0 * lobe);

    // Anti-ringing clamp to local min/max
    let box_min = min(c_m, min(min(c_u, c_d), min(c_l, c_r)));
    let box_max = max(c_m, max(max(c_u, c_d), max(c_l, c_r)));
    sharpened = clamp(sharpened, box_min, box_max);

    textureStore(output_texture, pos, vec4<f32>(sharpened, 1.0));
}
