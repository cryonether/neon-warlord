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
    @location(5) position: vec3<f32>,
    @location(6) color: vec3<f32>,
    @location(7) time: f32,
    @location(8) size: f32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv_coords: vec2<f32>,
    
    // Billboard basis in world space.
    @location(2) billboard_right: vec3<f32>,
    @location(3) billboard_up: vec3<f32>,
    @location(4) billboard_forward: vec3<f32>,
    
    @location(5) size: f32,
    @location(6) clip_position_no_division: vec4<f32>,
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
    let time = instance.time;

    // Billboard
    let billboard_center_position = instance.position;

    // Direction from sphere to camera.
    let look_to = normalize(
        camera.view_pos.xyz - billboard_center_position
    );

    let sideways = normalize(
        cross(vec3<f32>(0.0, 0.0, 1.0), look_to)
    );

    let new_up = cross(look_to, sideways);

    let rotated_model_pos =
        mat3x3<f32>(
            sideways,
            new_up,
            look_to
        ) * model.position;

    let global_position =
        instance.position + rotated_model_pos;

    var out: VertexOutput;

    out.color = vec4<f32>(instance.color, time);

    out.clip_position_no_division =
        camera.view_proj *
        vec4<f32>(global_position, 1.0);
    out.clip_position = out.clip_position_no_division;

    // Billboards are [-0.5, 0.5]^2,
    // UV coordinates are [0, 1]^2.
    out.uv_coords =
        model.position.xy / size + 0.5;

    // Pass the billboard coordinate system to the fragment shader.
    out.billboard_right = sideways;
    out.billboard_up = new_up;
    out.billboard_forward = look_to;
    out.size = size;

    return out;
}

struct FragmentOutput {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
};

// Fragment shader
@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
    const pi = radians(180.0);

    let centered_uv = in.uv_coords - 0.5;

    // Convert UV to [-1, 1].
    let sphere_xy = centered_uv * 2.0;

    // Squared distance from the center of the sphere.
    let radius_squared =
        dot(sphere_xy, sphere_xy);

    // Outside the circle -> transparent.
    if (radius_squared > 1.0) {
        discard;
    }

    // Reconstruct the Z component of the sphere normal.
    //
    // x² + y² + z² = 1
    let sphere_z =
        sqrt(max(0.0, 1.0 - radius_squared));

    // Normal in billboard-local coordinates.
    let local_normal = normalize(
        vec3<f32>(
            sphere_xy.x,
            sphere_xy.y,
            sphere_z
        )
    );

    // Transform billboard-local normal into world space.
    let world_normal = normalize(
        local_normal.x * in.billboard_right +
        local_normal.y * in.billboard_up +
        local_normal.z * in.billboard_forward
    );

    // ------------------------------------------------------------
    // Lighting
    // ------------------------------------------------------------

    // Global directional light.
    //
    // This is the direction the light travels.
    // Therefore the direction from the surface toward
    // the light is the opposite direction.
    let light_direction =
        normalize(vec3<f32>(-1.0, -1.0, 0.0));

    let to_light = -light_direction;

    // Simple Lambert diffuse lighting.
    let diffuse =
        max(dot(world_normal, to_light), 0.0);

    // Small ambient component so the dark side isn't black.
    let ambient = 0.15;

    let lighting =
        ambient + (1.0 - ambient) * diffuse;

    // Optional soft edge.
    let radius = sqrt(radius_squared);


    // ------------------------------------------------------------
    // Alpha
    // ------------------------------------------------------------

    var alpha =
        0.5 + 0.5 * cos(radius * pi);

    // Keep the original hard circular boundary.
    if (radius <= 1.0) {
        alpha = 1.0;
    }

    // ------------------------------------------------------------
    // Sphere depth
    // ------------------------------------------------------------

    // efficient cone geometry
    // let depth_offset = (1.0 - radius) * in.size * 0.5;

    // actual sphere version
    let depth_offset = sphere_z * in.size * 0.5;
    
    // insert third column of projection matrix here
    let mod_clip_position = in.clip_position_no_division + camera.proj * vec4(0.0,0.0,depth_offset,0.0);

    // let sphere_depth = mod_clip_position.z / mod_clip_position.w;

    let sphere_depth =  in.clip_position.z;

    // ------------------------------------------------------------
    // Output
    // ------------------------------------------------------------

    var out: FragmentOutput;

    out.color = vec4<f32>(
        in.color.xyz * lighting,
        in.color.w * alpha
    );

    
    out.depth = sphere_depth;

    return out;
}