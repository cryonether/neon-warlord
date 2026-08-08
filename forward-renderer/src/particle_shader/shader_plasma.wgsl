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
    @location(1) time: f32,
    @location(2) model_position: vec3<f32>,
    @location(3) size: f32, 
};

@vertex 
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    model: VertexInput,
    instance: InstanceInput,
) -> VertexOutput {
    // let size = 0.001 * instance.time;
    let size = instance.size;

    let model_position = model.position * size;
    let position = model_position + instance.position;

    // final result
    var out: VertexOutput;
    out.color = vec4(instance.color, 1.0);
    out.time = instance.time;
    out.model_position = model_position;
    out.size = size;
    out.clip_position = camera.view_proj * vec4<f32>(position, 1.0);
    return out;
}

// Fragment shader
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {

    // growth function
    let size = 1.0 - 1.0/(in.size +1.0);

    let v_pos = in.model_position.xyz;
    let u_time = in.time;

    let res = voronoi(v_pos / size * 2 , u_time);

    let color0 = vec4( vec3(res.x), 1.0 );
    let color1 =  vec3(pow(res.x, 1.5));

    let r = color1.x * 0.4 + 0.6 * smoothstep(0.4,1.0,color1.x);
    // let r = res.x;
    let g = color1.y * 0.4 + 0.6 *  smoothstep(0.4,1.0,color1.y);
    // let g = res.x;
    let b = color1.z;

    let color = vec4( vec3(r, g, b), 0.98 );

    return color;
}

/// cellular noise function
fn voronoi(x: vec3<f32>, time: f32) -> vec2<f32>
{
    // current cell coordinates
    let n = floor(x);
    // pixel coordinates in current cell
    let f = fract(x);

    // initialize m with a large number
    // (which will be get replaced very soon with smaller distances below)
    var m = vec4(8.0);

    // in 2D voronoi, we only have 2 dimensions to loop over
    // in 3D, we would naturally have one more dimension to loop over
    for (var k: i32 = -1; k <= 1; k = k + 1) {
        for (var j: i32 = -1; j <= 1; j = j + 1) {
            for (var i: i32 = -1; i <= 1; i = i + 1) {

                // coordinates for the relative cell  within the 3x3x3 3D grid
                let g = vec3(f32(i), f32(j), f32(k));
                // calculate a random point within the cell relative to 'n'(current cell coordinates)
                let o = hash3d( n + g );
                // calculate the distance vector between the current pixel and the moving random point 'o'
                let r = g + (0.5+0.5*sin(vec3(time)+6.2831*o)) - f;
                // calculate the scalar distance of r
                let d = dot(r,r);

                // find the minimum distance
                // it is most important to save the minimum distance into the result 'm'
                // saving other information into 'm' is optional and up to your liking
                // e.g. displaying different colors according to various cell coordinates
                if( d<m.x )
                {
                    m = vec4( d, o );
                }
            }
        }
    }

    return vec2(m.x, m.y+m.z+m.w);
}

// hash function from https://github.com/MaxBittker/glsl-voronoi-noise
fn hash3d(p: vec3<f32>) -> vec3<f32> {
  return fract(
      sin(vec3(dot(p, vec3(1.0, 57.0, 113.0)), dot(p, vec3(57.0, 113.0, 1.0)),
               dot(p, vec3(113.0, 1.0, 57.0)))) *
      43758.5453);
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