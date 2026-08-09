// Vertex shader
struct CameraUniform {
    view_pos: vec4<f32>,
    view_proj: mat4x4<f32>,
    proj: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@group(0) @binding(1)
var<uniform> camera_light: CameraUniform;

@group(2) @binding(0)
var t_heightmap: texture_2d<f32>;

struct VertexInput {
    @location(0) position: vec3<f32>,
}

struct InstanceInput {
    @location(5) position: vec3<f32>,
    @location(6) color: vec3<f32>, 
    @location(7) distance: f32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) position: vec3<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) tex_coords: vec2<f32>,
    @location(4) position_light_space: vec4<f32>,
};

// Vertex shader without lighting
@vertex 
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    model: VertexInput,
    instance: InstanceInput,
) -> VertexOutput {

    let info = get_vertex_info(vertex_index, model, instance);

    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(info.position, 1.0);
    out.color = instance.color;
    out.position = info.position;
    out.normal = info.normal;
    out.tex_coords = info.tex_coords;

    return out;
}

// Vertex shader with Gouraud shading
@vertex 
fn vs_main_gouraud(
    @builtin(vertex_index) vertex_index: u32,
    model: VertexInput,
    instance: InstanceInput,
) -> VertexOutput {

    let info = get_vertex_info(vertex_index, model, instance);
    let position = info.position;
    let normal = info.normal;
    let tex_coords = info.tex_coords;
    let color = instance.color;
    let view_position = camera.view_pos.xyz;
    
    // calculate lighting
    let light_color = vec3<f32>(1.0, 1.0, 1.0);
    let ambient_strength = 0.2;
    let diffuse_strength = 0.2;
    let specular_strength = 0.8;
    
    // diffuse lighting
    let light_direction = normalize(vec3<f32>(0.0, 1000.0, 140.0));
    let diffuse_lighting_strength = max(dot(normal, light_direction) * diffuse_strength, 0.0);
    
    // specular lighting
    let view_dir = normalize(view_position - position);
    
    // pong model
    let reflect_dir = reflect(-light_direction, normal);
    let specular_lighting_strength = pow(max(dot(view_dir, reflect_dir), 0.0), 32.0) * specular_strength;

    // bling-pong model
    // let halfway_dir = normalize(light_direction + view_dir);
    // let specular_lighting_strength = pow(max(dot(normal, halfway_dir), 0.0), 32.0) * specular_strength;


    // pong shading
    // let pong_lighting = light_color * (ambient_strength + diffuse_lighting_strength + specular_lighting_strength);
    // let pong_lighting = light_color * (specular_lighting_strength);
    // let pong_lighting = light_color * ((ambient_strength + diffuse_lighting_strength) * 0.4 +  (ambient_strength+ diffuse_lighting_strength + specular_lighting_strength) * specular_strength);
    // let pong_lighting = light_color * ((ambient_strength + diffuse_lighting_strength));
    // let pong_light: vec3<f32> = pong_lighting * color;

    // blend with intensity
    // let intensity = vertex_color[3];
    // let out_color: vec3<f32> = vertex_color.xyz * intensity + pong_light * (1.0 -intensity);
    let out_color: vec3<f32> = color * (ambient_strength + diffuse_lighting_strength + specular_lighting_strength);

    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(position, 1.0);
    out.color = out_color;
    out.position = position;
    out.normal = normal;
    out.tex_coords = tex_coords;
    out.position_light_space = camera_light.view_proj * vec4<f32>(position, 1.0);

    return out;
}

// Fragment shader

// texture
@group(1) @binding(0)
var t_texture: texture_2d<f32>;
@group(1) @binding(1)
var s_texture: sampler;

// shadow map
@group(3) @binding(0)
var shadow_map: texture_depth_2d;
@group(3) @binding(1)
var shadow_sampler: sampler_comparison;

struct FragmentOutput {
    @location(0) surface: vec4<f32>,
};

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {

    // read shadow map
    var ndc = in.position_light_space.xyz / in.position_light_space.w;
    ndc.y = -ndc.y;
    let texSize = vec2<f32>(textureDimensions(shadow_map));
    let pixel = vec2<i32>(
        (ndc.xy * 0.5 + vec2(0.5)) * texSize
    );

    let current_depth = ndc.z;

    let bias = 0.00001;

    let uv = ndc.xy * 0.5 + vec2<f32>(0.5);
    let visibility = textureSampleCompare(
        shadow_map,
        shadow_sampler,
        uv,
        current_depth - bias,
    );

    var shadow = visibility + 0.2;

    // load texture
    let color = textureSample(t_texture, s_texture, in.tex_coords);
    let color_out = vec4<f32>(in.color.xyz * color[3], color[3]);

    // apply shadow
    let lighting = shadow * color_out;    
    

    var out: FragmentOutput;
    out.surface = lighting;

    return out;
}

// structure for getting the neighboring values of the height texture
struct Neighborhood {
    m: f32,
    n: f32,
    e: f32,
    s: f32,
    w: f32,
}

fn get_neighborhood(uv: vec2<u32>) -> Neighborhood
{
    let m = get_height(uv, 0, 0);
    let n = get_height(uv, 0, 1);
    let e = get_height(uv, 1, 0);
    let s = get_height(uv, 0, -1);
    let w = get_height(uv, -1, 0);

    return Neighborhood(
        m,
        n,
        e,
        s,
        w,
    );
}

fn get_height(uv: vec2<u32>, u_offset: i32, v_offset: i32) -> f32
{
    let u: u32 = u32(i32(uv.x) + u_offset);
    let v: u32 = u32(i32(uv.y) + v_offset);

    return textureLoad(t_heightmap, vec2(u, v), 0).r; 
}

struct VertexInfo {
    position: vec3<f32>,
    normal: vec3<f32>,
    tex_coords: vec2<f32>
}

fn get_vertex_info(vertex_index: u32,
    model: VertexInput,
    instance: InstanceInput,) -> VertexInfo
{
    let dim: vec2<u32> = textureDimensions(t_heightmap);
    // let index = vec2<u32>(vertex_index % dim.x, vertex_index / dim.y);
    let index = vec2<u32>(u32(model.position.x), u32(model.position.y));
    let distance = instance.distance;
    let tex_coords = vec2<f32>(model.position.x * distance, model.position.y * distance);
    // let tex_coords = vec2<f32>(model.position.x, model.position.y);

    let heights = get_neighborhood(index);

    let pos_rgb: vec4<f32> = textureLoad(t_heightmap, index, 0);
    let posz = pos_rgb.r;

    // calculate position
    let vertex_position = vec3<f32>(model.position.x * distance, model.position.y *distance, posz);
    let position = instance.position + vertex_position;

    // normal calculation, use negative derivatives
    let normal_x = (heights.w - heights.e) / 2.0;
    let normal_y = (heights.s - heights.n) / 2.0;
    let normal = normalize(vec3<f32>(normal_x, normal_y, distance));

    // return
    return VertexInfo (
        position,
        normal,
        tex_coords,
    );
}
