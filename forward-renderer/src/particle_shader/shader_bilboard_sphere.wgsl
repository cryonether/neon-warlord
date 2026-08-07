// Shader to draw a 2D sphere in 3D

// Vertex shader
struct CameraUniform {
    view_pos: vec4<f32>,
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
}

struct InstanceInput {
    @location(5) position: vec3<f32>,
    @location(6) color: vec3<f32>, 
    @location(7) time: f32, 
    @location(8) size: f32, 
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv_coords: vec2<f32>,
};

@vertex 
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    model: VertexInput,
    instance: InstanceInput,
) -> VertexOutput {
    // let size = 0.1 + 0.1 * instance.time;
    let size = instance.size;

    // constants
    const pi2 = radians(90.0);
    const nr_vertices_per_object = 4; // must match the used objects
    
    let object_index= vertex_index/nr_vertices_per_object;
    let time =  instance.time;

    // billboard
    let billboard_center_position = instance.position;
    // Rotation matrix to face billboard to camera
    let look_to = normalize(camera.view_pos.xyz - billboard_center_position);
    let sideways = normalize(cross(vec3(0.,0.,1.), look_to));
    let new_up = cross(look_to, sideways);
    
    let rotated_model_pos = mat3x3<f32>(sideways, new_up, look_to) * model.position;
    let global_position = instance.position + rotated_model_pos;

    // final result
    var out: VertexOutput;
    out.color = vec4(instance.color, time);
    out.clip_position = camera.view_proj * vec4<f32>(global_position, 1.0);
    // billboards are [-0.5, 0.5]^2, uv-coords are [0, 1]^2
    out.uv_coords = model.position.xy / size + 0.5;
    return out;
}

// Fragment shader
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    const pi = radians(180.);
    let centered_uv = in.uv_coords - 0.5;
    // let radius = min(2 * length(centered_uv), 1.);
    let radius = 2 * length(centered_uv);
    // this function is flat at 0 and 1
    var alpha = 0.5 + 0.5 * cos(radius * pi);

    // var alpha = 0.0;
    if(radius <= 1) {
        alpha = 1;
    }
    alpha = 1.0;

    if(radius > 1) {
        discard;
    }

    return vec4(in.color.xyz, in.color.w * alpha);
}

