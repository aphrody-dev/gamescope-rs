// Gamescope Multi-Plane Compositor & HDR Tonemapping compute pass in pure WGSL.

struct CompositeUniforms {
    output_size: vec2<f32>,
    tonemap_operator: u32, // 0: passthrough, 1: reinhard, 2: aces, 3: gamescope_max_rgb
    display_peak_nits: f32,
    content_peak_nits: f32,
    paper_white_nits: f32,
    hdr_output_mode: u32,  // 0: sRGB gamma, 1: ST.2084 PQ HDR10, 2: scRGB linear
    pad: u32,
};

@group(0) @binding(0) var<uniform> uniforms: CompositeUniforms;
@group(0) @binding(1) var game_texture: texture_2d<f32>;
@group(0) @binding(2) var overlay_texture: texture_2d<f32>;
@group(0) @binding(3) var output_texture: texture_storage_2d<rgba8unorm, write>;

// SMPTE ST.2084 PQ constants
const PQ_M1: f32 = 0.1593017578125;
const PQ_M2: f32 = 78.84375;
const PQ_C1: f32 = 0.8359375;
const PQ_C2: f32 = 18.8515625;
const PQ_C3: f32 = 18.6875;
const PQ_MAX_NITS: f32 = 10000.0;

fn linear_to_pq(nits: f32) -> f32 {
    let y = clamp(nits / PQ_MAX_NITS, 0.0, 1.0);
    if (y == 0.0) {
        return 0.0;
    }
    let y_m1 = pow(y, PQ_M1);
    let num = PQ_C1 + PQ_C2 * y_m1;
    let den = 1.0 + PQ_C3 * y_m1;
    return pow(num / den, PQ_M2);
}

fn tonemap_aces(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn tonemap_gamescope_max_rgb(c: vec3<f32>) -> vec3<f32> {
    let max_c = max(c.r, max(c.g, c.b));
    if (max_c <= 0.0) {
        return vec3<f32>(0.0);
    }
    let scale = uniforms.display_peak_nits / max(uniforms.content_peak_nits, 1.0);
    let mapped_max = (max_c * scale) / (1.0 + max_c * scale);
    let ratio = mapped_max / max_c;
    return clamp(c * ratio, vec3<f32>(0.0), vec3<f32>(1.0));
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pos = global_id.xy;
    if (pos.x >= u32(uniforms.output_size.x) || pos.y >= u32(uniforms.output_size.y)) {
        return;
    }

    let ipos = vec2<i32>(pos);

    // Fetch game plane
    let game_sample = textureLoad(game_texture, ipos, 0);

    // Fetch overlay plane (MangoHud / Steam UI)
    let overlay_dims = textureDimensions(overlay_texture);
    var overlay_sample = vec4<f32>(0.0);
    if (pos.x < overlay_dims.x && pos.y < overlay_dims.y) {
        overlay_sample = textureLoad(overlay_texture, ipos, 0);
    }

    // Alpha blend overlay over game plane (premultiplied alpha)
    let blended_rgb = game_sample.rgb * (1.0 - overlay_sample.a) + overlay_sample.rgb;

    // Apply HDR Tonemapping
    var mapped_rgb = blended_rgb;
    if (uniforms.tonemap_operator == 1u) {
        // Standard Reinhard
        mapped_rgb = blended_rgb / (vec3<f32>(1.0) + blended_rgb);
    } else if (uniforms.tonemap_operator == 2u) {
        // ACES Film
        mapped_rgb = tonemap_aces(blended_rgb);
    } else if (uniforms.tonemap_operator == 3u) {
        // Gamescope Max-RGB
        mapped_rgb = tonemap_gamescope_max_rgb(blended_rgb);
    }

    // Encoding stage
    var final_rgb = mapped_rgb;
    if (uniforms.hdr_output_mode == 1u) {
        // ST.2084 PQ output
        let nits = mapped_rgb * uniforms.display_peak_nits;
        final_rgb = vec3<f32>(
            linear_to_pq(nits.r),
            linear_to_pq(nits.g),
            linear_to_pq(nits.b)
        );
    } else if (uniforms.hdr_output_mode == 0u) {
        // sRGB gamma 2.2 approximation
        final_rgb = pow(clamp(mapped_rgb, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / 2.2));
    }

    textureStore(output_texture, pos, vec4<f32>(final_rgb, 1.0));
}
