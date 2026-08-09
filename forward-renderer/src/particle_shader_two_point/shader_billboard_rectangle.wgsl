// Shader to draw a 2D sphere in 3D

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
    @location(5) position_0: vec3<f32>,
    @location(6) position_1: vec3<f32>,
    @location(7) color: vec3<f32>,
    @location(8) time: f32,
    @location(9) size: f32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv_coords: vec2<f32>,

    // Billboard basis in world space.
    @location(2) billboard_right: vec3<f32>,
    @location(3) billboard_up: vec3<f32>,
    @location(4) billboard_forward: vec3<f32>,
};

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    model: VertexInput,
    instance: InstanceInput,
) -> VertexOutput {
    let size = instance.size;

    const nr_vertices_per_object = 4u;

    let object_index = vertex_index / nr_vertices_per_object;
    let rectangle_index = vertex_index % nr_vertices_per_object;
    let time = instance.time;

    // Billboard
    let center = (instance.position_0 + instance.position_1) * 0.5;
    let center_pos_1 = instance.position_1 - center;

    // Direction from sphere to camera.
    let look_to = normalize(
        camera.view_pos.xyz - center
    );
        
    let sideways = normalize(
        cross(vec3<f32>(0.0, 0.0, 1.0), look_to)
    );

    let new_up = cross(look_to, sideways);

    // 
    let object_up = normalize(cross(look_to, center_pos_1)) * size;


    var model_position = vec3(0.0, 0.0, 0.0);
    if(rectangle_index == 0) {
        model_position = center - center_pos_1 - object_up;
    }
    else if(rectangle_index == 1) {
        model_position = center + center_pos_1 - object_up;
    }
    else if(rectangle_index == 2) {
        model_position = center + center_pos_1 + object_up;
    }
    else {
        model_position = center - center_pos_1 + object_up;
    }

    let global_position = model_position;

    var out: VertexOutput;

    out.color = vec4<f32>(instance.color, time);

    out.clip_position =
        camera.view_proj *
        vec4<f32>(global_position, 1.0);

    // Billboards are [-0.5, 0.5]^2,
    // UV coordinates are [0, 1]^2.
    out.uv_coords =
        model.position.xy / size + 0.5;

    // Pass the billboard coordinate system to the fragment shader.
    out.billboard_right = sideways;
    out.billboard_up = new_up;
    out.billboard_forward = look_to;

    return out;
}


// Fragment shader
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}