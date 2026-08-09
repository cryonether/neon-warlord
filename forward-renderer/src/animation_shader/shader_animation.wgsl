
const MAX_JOINTS: u32 = 16u;
const MAX_JOINT_WEIGHTS: u32 = 4u;

// Vertex shader
struct CameraUniform {
    view_pos: vec4<f32>,
    view_proj: mat4x4<f32>,
    proj: mat4x4<f32>,
};

struct JointUniform {
    joint_transform: array<mat4x4<f32>, 16>,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@group(1) @binding(0)
var<uniform> joints: JointUniform;

struct VertexInput {
    @location(0) position: vec4<f32>,
    @location(1) normal: vec4<f32>,

    // animation data
    @location(2) joint_indices: vec4<u32>,
    @location(3) joint_weights: vec4<f32>,
}

struct InstanceInput {
    @location(5) model_matrix_0: vec4<f32>,
    @location(6) model_matrix_1: vec4<f32>,
    @location(7) model_matrix_2: vec4<f32>,
    @location(8) model_matrix_3: vec4<f32>,

    @location(9) color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) position: vec3<f32>,
    @location(2) normal: vec3<f32>,
};

@vertex 
fn vs_main(
    model: VertexInput,
    instance: InstanceInput,
) -> VertexOutput {
    let info = get_vertex_info(model, instance);
    
    let position = info.position;
    let normal = info.normal;
    let color = instance.color;
    let view_position = camera.view_pos.xyz;
    
    // calculate output
    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(position, 1.0);
    out.color = color.xyz;
    out.position = position;
    out.normal = normal;

    return out;
}

// Vertex shader with Gouraud shading
@vertex 
fn vs_main_gouraud(
    model: VertexInput,
    instance: InstanceInput,
) -> VertexOutput 
{
    let info = get_vertex_info( model, instance);
    let position = info.position;
    let normal = info.normal;
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

    let out_color: vec3<f32> = color.xyz * (ambient_strength + diffuse_lighting_strength + specular_lighting_strength);

    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(position, 1.0);
    out.color = out_color;
    out.position = position;
    out.normal = normal;

    return out;
}


// Fragment shader
struct FragmentOutput {
    @location(0) surface: vec4<f32>,
};

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {

    let color_out = vec4<f32>(in.color, 1.0);

    var out: FragmentOutput;
    out.surface = color_out;

    return out;
}

struct VertexInfo {
    position: vec3<f32>,
    normal: vec3<f32>,
}

fn get_vertex_info(
    model: VertexInput,
    instance: InstanceInput,
) -> VertexInfo
{
    let model_matrix = mat4x4<f32>(
        instance.model_matrix_0,
        instance.model_matrix_1,
        instance.model_matrix_2,
        instance.model_matrix_3,
    );

    let scale = 0.1;

    // calculate the animation
    var total_local_pos = vec4<f32>(0.0);
    var total_local_normal = vec4<f32>(0.0);

    for (var i: u32 = 0u; i < MAX_JOINT_WEIGHTS; i++) {
        // get vertex data
        let joint_index = model.joint_indices[i];
        let joint_weight = model.joint_weights[i];

        // get transform matrix
        let joint_transform = joints.joint_transform[joint_index];

        // move position
        let local_position = joint_transform * model.position;
        total_local_pos += local_position * joint_weight;

        // move normal
        let local_normal = joint_transform * model.normal;
        total_local_normal += local_normal * joint_weight;
    }

    // move to the instance position
    let world_position = model_matrix * vec4<f32>((total_local_pos * scale).xyz, 1.0);

    // return
    return VertexInfo (
        world_position.xyz,
        total_local_normal.xyz,
    );
}