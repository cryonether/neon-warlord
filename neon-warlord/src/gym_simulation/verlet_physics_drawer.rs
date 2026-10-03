//! Draws AdvancedCompositions

use forward_renderer::{particle_shader, particle_shader_two_point, to_rgb};

use crate::verlet_physics_simd::VerletPhysicsSimd;

type Vec3 = cgmath::Vector3<f32>;

/// Draws AdvancedCompositions
pub struct VerletPhysicsDrawer {
    nodes_color_0: Vec3,
    links_color_0: Vec3,

    _nr_nodes: usize,
    _nr_edges: usize,

    radius: f32,
    position: Vec3,
}

impl VerletPhysicsDrawer {
    pub fn new(verlet_physics: &VerletPhysicsSimd, radius: f32, position: Vec3) -> Self {
        let _nr_nodes = verlet_physics.particles.len();
        let _nr_edges = verlet_physics.distance_constraints.len();

        let nodes_color_0: Vec3 = to_rgb("#ce51ff").into();
        let _nodes_color_1: Vec3 = to_rgb("#a72ebc").into();
        let links_color_0: Vec3 = to_rgb("#131922").into();
        let _links_color_1: Vec3 = to_rgb("#2d2e27").into();

        Self {
            nodes_color_0,
            links_color_0,
            _nr_nodes,
            _nr_edges,
            radius,
            position,
        }
    }

    pub fn update(
        &mut self,
        verlet_physics: &VerletPhysicsSimd,
        producer_nodes: &mut Vec<particle_shader::Instance>,
        producer_edges: &mut Vec<particle_shader_two_point::Instance>,
    ) {
        let particles = &verlet_physics.particles;
        let constrains = &verlet_physics.distance_constraints;

        for (&x, &y, &z, &radius) in
            itertools::izip!(&particles.x, &particles.y, &particles.z, &particles.radius)
        {
            let position = Vec3::new(x, y, z) + self.position;

            producer_nodes.push(particle_shader::Instance {
                position: position.into(),
                color: self.nodes_color_0.into(),
                time: 1.0,
                size: radius * 2.0,
            });
        }

        for (&a, &b) in itertools::izip!(&constrains.a, &constrains.b,) {
            let pos_0 = particles.position(a as usize) + self.position;
            let pos_1 = particles.position(b as usize) + self.position;

            producer_edges.push(particle_shader_two_point::Instance {
                position_0: pos_0.into(),
                position_1: pos_1.into(),
                color: self.links_color_0.into(),
                time: 1.0,
                size: self.radius * 0.1,
            });
        }
    }
}
