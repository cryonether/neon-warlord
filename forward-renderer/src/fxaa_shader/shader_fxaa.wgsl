// Vertex shader
struct CameraUniform {
    view_pos: vec4<f32>,
    view_proj: mat4x4<f32>,
    proj: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
}

struct InstanceInput {
    @location(5) position: vec3<f32>,
    @location(6) light_color: vec3<f32>,
    @location(7) radius: f32,
    @location(8) linear: f32,
    @location(9) quadratic: f32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
};

@vertex 
fn vs_main(
    model: VertexInput,
    instance: InstanceInput,
) -> VertexOutput {

    let position = instance.position + model.position;

    var out: VertexOutput;
    out.clip_position = vec4<f32>(position, 1.0);
    return out;
}

// Fragment shader

@group(1) @binding(0)
var post_processing_texture: texture_2d<f32>;

struct FragmentOutput {
    @location(0) surface: vec4<f32>,
};

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
    var out: FragmentOutput;

    // read post processing input texture
    let index = vec2<u32>(u32(in.clip_position.x), u32(in.clip_position.y));
    let vertex_color: vec4<f32> = textureLoad(post_processing_texture, index, 0);

    // https://catlikecoding.com/unity/tutorials/custom-srp/fxaa/

    let luma = get_luma_neighborhood(index);

    if can_skip_fxaa(luma) {
        return FragmentOutput(vertex_color);
    }

    let edge: FxaaEdge = get_fxaa_edge(luma);

    let blend_factor: f32 = get_subpixel_blend_factor(luma);
    var uv = index;
    if edge.is_horizontal {
        uv.y += u32(i32(uv.y) + edge.pixel_step);
    }
    else {
        uv.x += u32(i32(uv.x) + edge.pixel_step);
    }

    let val_1: vec4<f32> = textureLoad(post_processing_texture, uv, 0);

    var interpolated = linear_interpolate(vertex_color, val_1, blend_factor);

    // interpolated[1] = 0.0;
    // interpolated[2] = 0.0;
    return FragmentOutput(interpolated);
}

fn linear_interpolate(val_0: vec4<f32>, val_1: vec4<f32>, amount: f32) -> vec4<f32> {
    return val_0 * (1.0 - amount) + val_1 * amount;
}

struct FxaaEdge {
    is_horizontal: bool,
    pixel_step: i32,
}

fn get_fxaa_edge(luma: LumaNeighborhood) -> FxaaEdge {
    let is_horizontal = is_horizontal_edge(luma);
    
    var pixel_step: i32 = 1;
    var luma_p = luma.e;
    var luma_n = luma.w;
    if is_horizontal {
        luma_p = luma.n;
        luma_n = luma.s;
    }

    let gradient_p = abs(luma_p - luma.m);
    let gradient_n = abs(luma_n - luma.m);

    if gradient_p < gradient_n {
        pixel_step = - pixel_step;
    }

    return FxaaEdge(is_horizontal, pixel_step);
}

fn is_horizontal_edge(luma: LumaNeighborhood) -> bool {
    let horizontal = abs(luma.n + luma.s - 2.0 * luma.m) +
                     abs(luma.ne + luma.se - 2.0 * luma.e) +
                     abs(luma.nw + luma.sw - 2.0 * luma.w);
    let vertical = abs(luma.e + luma.w - 2.0 * luma.m) +
                   abs(luma.ne + luma.nw - 2.0 * luma.n) +
                   abs(luma.se + luma.sw - 2.0 * luma.s);
    return horizontal >= vertical;
}

fn get_subpixel_blend_factor(luma: LumaNeighborhood) -> f32 {
    // low pass filtering
    let low_pass: f32 = (2.0 * (luma.n + luma.e + luma.s + luma.w) + luma.ne + luma.nw + luma.se + luma.sw) / 12.0;

    // high pass filtering
    let high_pass: f32 = abs(low_pass - luma.m);

    // normalize
    let normalized: f32 = saturate(high_pass / luma.range);

    // smoothing
    let smoothed: f32 = smoothstep(0.0, 1.0, normalized);


    return smoothed * smoothed;
}

fn can_skip_fxaa(luma: LumaNeighborhood) -> bool {
    // Trims the algorithm from processing darks.
    //   0.0833 - upper limit (default, the start of visible unfiltered edges)
    //   0.0625 - high quality (faster)
    //   0.0312 - visible limit (slower)
    const fixed_threshold: f32 = 0.0833;

    // The minimum amount of local contrast required to apply algorithm.
    //   0.333 - too little (faster)
    //   0.250 - low quality
    //   0.166 - default
    //   0.125 - high quality 
    //   0.063 - overkill (slower)
    const relative_threshold: f32 = 0.166;

    return luma.range < max(fixed_threshold, relative_threshold * luma.highest);
}

struct LumaNeighborhood {
    m: f32,
    n: f32,
    e: f32,
    s: f32,
    w: f32,

    ne: f32,
    se: f32,
    sw: f32,
    nw: f32,

    highest: f32,
    lowest: f32,
    range: f32,
}

fn get_luma_neighborhood(uv: vec2<u32>) -> LumaNeighborhood
{
    let m = get_luma(uv, 0, 0);
    let n = get_luma(uv, 0, 1);
    let e = get_luma(uv, 1, 0);
    let s = get_luma(uv, 0, -1);
    let w = get_luma(uv, -1, 0);

    let ne = get_luma(uv, 1, 1);
    let se = get_luma(uv, 1, -1);
    let sw = get_luma(uv, -1, -1);
    let nw = get_luma(uv, -1, 1);

    let highest = max(max(max(max(m, n), e), s), w);
    let lowest = min(min(min(min(m, n), e), s), w);
    let range = highest - lowest;

    return LumaNeighborhood(
        m, 
        n, 
        e, 
        s, 
        w,

        ne, 
        se, 
        sw, 
        nw, 

        highest,
        lowest,
        range,
    );
}

fn get_luma(uv: vec2<u32>, u_offset: i32, v_offset: i32) -> f32
{
    let u: u32 = u32(i32(uv.x) + u_offset);
    let v: u32 = u32(i32(uv.y) + v_offset);

    // return sqrt(luminance(textureLoad(post_processing_texture, vec2(u, v), 0))); // accurate
    return textureLoad(post_processing_texture, vec2(u, v), 0).g; // fast
}

// https://en.wikipedia.org/wiki/Relative_luminance
fn luminance(rgb: vec4<f32>) -> f32 {
    return rgb.r *0.2126 + rgb.g * 0.7152 + rgb.b * 0.0722;
}
