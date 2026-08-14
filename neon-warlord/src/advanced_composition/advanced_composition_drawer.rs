//! Draws AdvancedCompositions

use forward_renderer::{particle_shader, particle_shader_two_point, to_rgb};

use crate::advanced_composition::{AdvancedComposition, Vec3};

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
    pub fn new(composition: &AdvancedComposition, radius: f32) -> Self {
        let _nr_nodes = composition.verlet_objects.len();
        let _nr_edges = composition.links.len();

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
        composition: &AdvancedComposition,
        producer_nodes: &mut Vec<particle_shader::Instance>,
        producer_edges: &mut Vec<particle_shader_two_point::Instance>,
    ) {
        let size = std::cmp::min(self.nodes_instances.len(), composition.verlet_objects.len());

        // copy from physics to model
        for i in 0..size {
            let instance = &mut self.nodes_instances[i];
            let verlet_object = &composition.verlet_objects[i];

            instance.position = verlet_object.position().into();
            instance.color = self.nodes_color_0.into();
            instance.size = self.radius * 2.0;
            instance.time = 1.0;
        }

        for (i, link) in composition.links.iter().enumerate() {
            match link {
                super::Link::Fixed(elem) => {
                    let index_0 = elem.node_id_1;
                    let index_1 = elem.node_id_2;

                    let pos_0 = composition.verlet_objects[index_0].position();
                    let pos_1 = composition.verlet_objects[index_1].position();

                    let instance = &mut self.edges_instances[i];
                    instance.position_0 = pos_0.into();
                    instance.position_1 = pos_1.into();
                    instance.color = self.links_color_0.into();
                    instance.size = self.radius * 0.1;
                    instance.time = 1.0;
                }
                super::Link::FixedDistance(elem) => {
                    let index_0 = elem.node_id_1;
                    let index_1 = elem.node_id_2;

                    let pos_0 = composition.verlet_objects[index_0].position();
                    let pos_1 = composition.verlet_objects[index_1].position();

                    let instance = &mut self.edges_instances[i];
                    instance.position_0 = pos_0.into();
                    instance.position_1 = pos_1.into();
                    instance.color = self.links_color_0.into();
                    instance.size = self.radius * 0.1;
                    instance.time = 1.0;
                }
                super::Link::Loose(elem) => {
                    let index_0 = elem.node_id_1;
                    let index_1 = elem.node_id_2;

                    let pos_0 = composition.verlet_objects[index_0].position();
                    let pos_1 = composition.verlet_objects[index_1].position();

                    let instance = &mut self.edges_instances[i];
                    instance.position_0 = pos_0.into();
                    instance.position_1 = pos_1.into();
                    instance.color = self.links_color_0.into();
                    instance.size = self.radius * 0.1;
                    instance.time = 1.0;
                }
            }
        }

        // copy from model to device

        producer_nodes.extend_from_slice(&self.nodes_instances);
        producer_edges.extend_from_slice(&self.edges_instances);
    }
}
