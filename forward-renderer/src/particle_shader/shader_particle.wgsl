// Shader to draw a view repeating particles

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

    let model_position = model.position;

    // constants
    const pi2 = radians(90.0);
    const nr_vertices_per_object = 4; // must match the used objects
    const distance = 4.0;

    // start time
    let object_index= vertex_index/nr_vertices_per_object;
    let rand_time = random2(object_index);
    let time =  instance.time + rand_time;

    // position rands
    let rand0 = 1.0 -2 * random(object_index + 0 + u32(time));
    let rand1 = 1.0 -2 * random(object_index + 1 + u32(time));
    let rand2 = 1.0 -2 * random(object_index + 2 + u32(time));
    let position_offset = vec3(rand0, rand1, rand2) * distance;

    // Percentage of progress to center
    let time_fn = time % 1.0;
    // Relative distance from the center
    let distance_fn = cos(pi2 * time_fn);

    let billboard_center_position = distance_fn * position_offset + instance.position;
    // Rotation matrix to face billboard to camera
    let look_to = normalize(camera.view_pos.xyz - billboard_center_position);
    let sideways = normalize(cross(vec3(0.,0.,1.), look_to));
    let new_up = cross(look_to, sideways);
    
    let rotated_model_pos = mat3x3<f32>(sideways, new_up, look_to) * model.position;
    let global_position = instance.position + distance_fn * (rotated_model_pos + position_offset);

    // final result
    var out: VertexOutput;
    out.color = vec4(instance.color, time_fn);
    out.clip_position = camera.view_proj * vec4<f32>(global_position, 1.0);
    // billboards are [-0.5, 0.5]^2, uv-coords are [0, 1]^2
    out.uv_coords = model.position.xy + 0.5;
    return out;
}

// Fragment shader
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    const pi = radians(180.);
    let centered_uv = in.uv_coords - 0.5;
    let radius = min(2 * length(centered_uv), 1.);
    // this function is flat at 0 and 1
    let alpha = 0.5 + 0.5 * cos(radius * pi);

    return vec4(in.color.xyz, in.color.w * alpha);
}


fn rand_xorshift(rng_state: u32) -> u32
{
    let rng_state1 = rng_state ^ (rng_state << 13);
    let rng_state2 =  rng_state1 ^ (rng_state1 >> 17);
    let rng_state3 = rng_state2 ^ (rng_state2 << 5);
    return rng_state3;
}

// “Wang hash", s a general-purpose 32-bit-to-32-bit integer hash function.
fn wang_hash(seed: u32) -> u32
{
    let seed1 = (seed ^ 61) ^ (seed >> 16);
    let seed2 = seed1 * 9;
    let seed3 = seed2 ^ (seed2 >> 4);
    let seed4 = seed3 * 0x27d4eb2d;
    let seed5 = seed4 ^ (seed4 >> 15);
    return seed5;
}

// 32-bit PCG hash used by Jarzynski and Olano, one of the best balanced choices between 
// performance and quality overall.
fn pcg_hash(input: u32) -> u32
{
    let state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// returns a random number between 0 and 1
fn random(input: u32) -> f32 {
    
    let val2 = rand_xorshift(input);
    let val3 = wang_hash(val2);
    let val4 = f32(val3) * (1.0 / 4294967296.0);
    
    return val4;
}

// returns a random number between 0 and 1
fn random2(input: u32) -> f32 {
    
    let val2 = rand_xorshift(input);
    let val3 = pcg_hash(val2);
    let val4 = f32(val3) * (1.0 / 4294967296.0);
    
    return val4;
}