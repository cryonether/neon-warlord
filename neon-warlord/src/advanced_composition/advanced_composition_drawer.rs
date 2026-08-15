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
        // let particles_len = std::cmp::min(self.nodes_instances.len(), composition.verlet_physics.particles.len());
        let particles = &composition.verlet_physics.particles;
        let constrains = &composition.verlet_physics.distance_constraints;

        // copy from physics to model
        for i in 0..particles.len() {
            let instance = &mut self.nodes_instances[i];

            instance.position = particles.position(i).into();
            instance.color = self.nodes_color_0.into();
            instance.size = self.radius * 2.0;
            instance.time = 1.0;
        }

        for i in 0..constrains.len() {
            let index_0 = constrains.a[i];
            let index_1 = constrains.b[i];
            let pos_0 = particles.position(index_0 as usize);
            let pos_1 = particles.position(index_1 as usize);

            let instance = &mut self.edges_instances[i];
            instance.position_0 = pos_0.into();
            instance.position_1 = pos_1.into();
            instance.color = self.links_color_0.into();
            instance.size = self.radius * 0.1;
            instance.time = 1.0;
        }

        // copy from model to device

        producer_nodes.extend_from_slice(&self.nodes_instances);
        producer_edges.extend_from_slice(&self.edges_instances);
    }
}
