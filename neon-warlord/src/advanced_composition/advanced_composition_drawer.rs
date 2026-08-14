//! Draws AdvancedCompositions

use cgmath::Rotation3;
use forward_renderer::{geometry, particle_shader, particle_shader_two_point, to_rgb};
use wgpu_renderer::{
    vertex_color_shader::{
        self, VertexColorShaderDraw, vertex_color_shader_draw::VertexColorShaderDrawLines,
    },
    wgpu_renderer::WgpuRendererInterface,
};

use crate::advanced_composition::{AdvancedComposition, Vec3};

/// Draws AdvancedCompositions
pub struct AdvancedCompositionDrawer {
    nodes_instances: Vec<particle_shader::Instance>,

    edges_instances: Vec<particle_shader_two_point::Instance>,

    nodes_color_0: Vec3,
    nodes_color_1: Vec3,
    links_color_0: Vec3,
    links_color_1: Vec3,

    nr_nodes: usize,
    nr_edges: usize,
}

impl AdvancedCompositionDrawer {
    pub fn new(
        composition: &AdvancedComposition,
        radius: f32,
    ) -> Self {
        let nr_nodes = composition.verlet_objects.len();
        let nr_edges = composition.links.len();

        let nodes_color_0: Vec3 = to_rgb("#d8b0e8").into();
        let nodes_color_1: Vec3 = to_rgb("#300c36").into();
        let links_color_0: Vec3 = to_rgb("#131922").into();
        let links_color_1: Vec3 = to_rgb("#2d2e27").into();

        let mut nodes_instances = Vec::with_capacity(nr_nodes);
        for _i in 0..nr_nodes {
            nodes_instances.push(particle_shader::Instance::new());
        }

        let mut edges_instances = Vec::with_capacity(nr_edges);
        for _i in 0..nr_edges {
            edges_instances.push(particle_shader_two_point::Instance::new());
        }

        // let nodes_circle =
        //     geometry::Circle::new_color_fade(radius, 32, nodes_color_0, nodes_color_1);
        // let links_instances = geometry::Lines::new_color_fade(nr_links, links_color_0, links_color_1);

        // let instance = vertex_color_shader::Instance {
        //     position: cgmath::Vector3::new(0.0, 0.0, 0.0),
        //     rotation: cgmath::Quaternion::from_angle_x(cgmath::Deg(90.0)),
        // };

        // let mut nodes_instances = Vec::with_capacity(nr_nodes);
        // for _i in 0..nr_nodes {
        //     nodes_instances.push(instance);
        // }

        // let edges_instances = vec![vertex_color_shader::Instance::zero()];

        // let nodes_mesh = vertex_color_shader::Mesh::new(
        //     wgpu_renderer.device(),
        //     &nodes_circle.vertices,
        //     &nodes_circle.colors,
        //     &nodes_circle.indices,
        //     &nodes_instances,
        // );

        // let links_mesh = vertex_color_shader::Mesh::new(
        //     wgpu_renderer.device(),
        //     &edges_instances.vertices,
        //     &edges_instances.colors,
        //     &edges_instances.indices,
        //     &edges_instances,
        // );

        Self {
            nodes_instances,
            edges_instances,
            nodes_color_0,
            nodes_color_1,
            links_color_0,
            links_color_1,
            nr_nodes,
            nr_edges,
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
        }

        for (i, link) in composition.links.iter().enumerate() {
            match link {
                super::Link::Fixed(elem) => {
                    let index_0 = elem.node_id_1;
                    let index_1 = elem.node_id_2;

                    let pos_0 = composition.verlet_objects[index_0].position();
                    let pos_1 = composition.verlet_objects[index_1].position();

                    // self.edges_instances.set_line_position(i, pos_0, pos_1);
                    self.edges_instances[i].position_0 = pos_0.into();
                    self.edges_instances[i].position_1 = pos_1.into();
                }
                super::Link::FixedDistance(elem) => {
                    let index_0 = elem.node_id_1;
                    let index_1 = elem.node_id_2;

                    let pos_0 = composition.verlet_objects[index_0].position();
                    let pos_1 = composition.verlet_objects[index_1].position();

                    // self.edges_instances.set_line_position(i, pos_0, pos_1);
                    self.edges_instances[i].position_0 = pos_0.into();
                    self.edges_instances[i].position_1 = pos_1.into();
                }
                super::Link::Loose(elem) => {
                    let index_0 = elem.node_id_1;
                    let index_1 = elem.node_id_2;

                    let pos_0 = composition.verlet_objects[index_0].position();
                    let pos_1 = composition.verlet_objects[index_1].position();

                    // self.edges_instances.set_line_position(i, pos_0, pos_1);
                    self.edges_instances[i].position_0 = pos_0.into();
                    self.edges_instances[i].position_1 = pos_1.into();
                }
            }
        }

        // copy from model to device

        producer_nodes.extend_from_slice(&self.nodes_instances);
        producer_edges.extend_from_slice(&self.edges_instances);


        // self.nodes_mesh
        //     .update_instance_buffer(wgpu_renderer.queue(), &self.nodes_instances);

        // self.links_mesh
        //     .update_vertex_buffer(wgpu_renderer.queue(), &self.edges_instances.vertices);
    }
}

// impl VertexColorShaderDraw for AdvancedCompositionDrawer {
//     fn draw<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
//         self.nodes_mesh.draw(render_pass);
//     }
// }

// impl VertexColorShaderDrawLines for AdvancedCompositionDrawer {
//     fn draw_lines<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
//         self.links_mesh.draw(render_pass);
//     }
// }
