//! Draws AdvancedCompositions

use forward_renderer::{particle_shader, particle_shader_two_point, to_rgb};

use crate::{advanced_composition::{AdvancedComposition, Vec3}, advanced_composition_simd::AdvancedCompositionSimd};

/// Draws AdvancedCompositions
pub struct AdvancedCompositionDrawer {
    nodes_instances: Vec<particle_shader::Instance>,

    edges_instances: Vec<particle_shader_two_point::Instance>,

    nodes_color_0: Vec3,
    _nodes_color_1: Vec3,
    links_color_0: Vec3,
    _links_color_1: Vec3,

    _nr_nodes: usize,
    _nr_edges: usize,

    radius: f32,
}

impl AdvancedCompositionDrawer {
    pub fn new(composition: &AdvancedCompositionSimd, radius: f32) -> Self {
        let _nr_nodes = composition.verlet_physics.particles.len();
        let _nr_edges = composition.verlet_physics.distance_constraints.len();

        let nodes_color_0: Vec3 = to_rgb("#ce51ff").into();
        let _nodes_color_1: Vec3 = to_rgb("#a72ebc").into();
        let links_color_0: Vec3 = to_rgb("#131922").into();
        let _links_color_1: Vec3 = to_rgb("#2d2e27").into();

        let mut nodes_instances = Vec::with_capacity(_nr_nodes);
        for _i in 0.._nr_nodes {
            nodes_instances.push(particle_shader::Instance::new());
        }

        let mut edges_instances = Vec::with_capacity(_nr_edges);
        for _i in 0.._nr_edges {
            edges_instances.push(particle_shader_two_point::Instance::new());
        }

        Self {
            nodes_instances,
            edges_instances,
            nodes_color_0,
            _nodes_color_1,
            links_color_0,
            _links_color_1,
            _nr_nodes,
            _nr_edges,
            radius,
        }
    }

    pub fn update(
        &mut self,
        composition: &AdvancedCompositionSimd,
        producer_nodes: &mut Vec<particle_shader::Instance>,
        producer_edges: &mut Vec<particle_shader_two_point::Instance>,
    ) {
        let particles = &composition.verlet_physics.particles;
        let constrains = &composition.verlet_physics.distance_constraints;

        for (&x, &y, &z, &radius) in itertools::izip!(
            &particles.x, 
            &particles.y,
            &particles.z,
            &particles.radius
        ) {
            producer_nodes.push(particle_shader::Instance { 
                position: [x, y, z], 
                color: self.nodes_color_0.into(), 
                time: 1.0, 
                size: radius * 2.0 
            });
        }

        for (&a, &b) in itertools::izip!(
            &constrains.a,
            &constrains.b,
        ) {
            let pos_0 = particles.position(a as usize);
            let pos_1 = particles.position(b as usize);

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
