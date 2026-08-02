//! Draws AdvancedCompositions

use forward_renderer::{geometry, to_rgb};
use wgpu_renderer::{vertex_color_shader::{self, VertexColorShaderDraw, vertex_color_shader_draw::VertexColorShaderDrawLines}, wgpu_renderer::WgpuRendererInterface};
use cgmath::Rotation3;

use crate::advanced_composition::AdvancedComposition;


/// Draws AdvancedCompositions
pub struct AdvancedCompositionDrawer {
    nodes_mesh: vertex_color_shader::Mesh,
    nodes_instances: Vec<vertex_color_shader::Instance>,

    links_lines: geometry::Lines,
    links_mesh: vertex_color_shader::Mesh,
}

impl AdvancedCompositionDrawer {
    pub fn new(
        wgpu_renderer: &mut dyn WgpuRendererInterface,
        composition: &AdvancedComposition,
        radius: f32,
    ) -> Self {
        let nr_nodes = composition.verlet_objects.len();
        let nr_links = composition.links.len();

        let nodes_color_0 = to_rgb("#d8b0e8");
        let nodes_color_1 = to_rgb("#300c36");
        let links_color_0 = to_rgb("#131922");
        let links_color_1 = to_rgb("#2d2e27");

        let nodes_circle =
            geometry::Circle::new_color_fade(radius, 32, nodes_color_0, nodes_color_1);
        let links_lines = geometry::Lines::new_color_fade(nr_links, links_color_0, links_color_1);

        let instance = vertex_color_shader::Instance {
            position: cgmath::Vector3::new(0.0, 0.0, 0.0),
            rotation: cgmath::Quaternion::from_angle_x(cgmath::Deg(90.0)),
        };

        let mut nodes_instances = Vec::with_capacity(nr_nodes);
        for _i in 0..nr_nodes {
            nodes_instances.push(instance);
        }

        let links_instances = vec![vertex_color_shader::Instance::zero()];

        let nodes_mesh = vertex_color_shader::Mesh::new(
            wgpu_renderer.device(),
            &nodes_circle.vertices,
            &nodes_circle.colors,
            &nodes_circle.indices,
            &nodes_instances,
        );

        let links_mesh = vertex_color_shader::Mesh::new(
            wgpu_renderer.device(),
            &links_lines.vertices,
            &links_lines.colors,
            &links_lines.indices,
            &links_instances,
        );

        Self {
            nodes_mesh,
            nodes_instances,
            links_lines,
            links_mesh,
        }
    }

    pub fn update(
        &mut self,
        wgpu_renderer: &mut dyn WgpuRendererInterface,
        composition: &AdvancedComposition,
    ) {
        let size = std::cmp::min(self.nodes_instances.len(), composition.verlet_objects.len());

        // copy from physics to model
        for i in 0..size {
            let instance = &mut self.nodes_instances[i];
            let verlet_object = &composition.verlet_objects[i];

            instance.position = verlet_object.position();
        }

        for (i, link) in composition.links.iter().enumerate() {
            match link {
                super::Link::Fixed(elem) => {
                    let index_0 = elem.node_id_1;
                    let index_1 = elem.node_id_2;

                    let pos_0 = composition.verlet_objects[index_0].position();
                    let pos_1 = composition.verlet_objects[index_1].position();

                    self.links_lines.set_line_position(i, pos_0, pos_1);
                },
                super::Link::FixedDistance(elem) => {
                    let index_0 = elem.node_id_1;
                    let index_1 = elem.node_id_2;

                    let pos_0 = composition.verlet_objects[index_0].position();
                    let pos_1 = composition.verlet_objects[index_1].position();

                    self.links_lines.set_line_position(i, pos_0, pos_1);
                },
                super::Link::Loose(elem) => {
                    let index_0 = elem.node_id_1;
                    let index_1 = elem.node_id_2;

                    let pos_0 = composition.verlet_objects[index_0].position();
                    let pos_1 = composition.verlet_objects[index_1].position();

                    self.links_lines.set_line_position(i, pos_0, pos_1);
                },
            }
        }

        // copy from model to device

        self.nodes_mesh
            .update_instance_buffer(wgpu_renderer.queue(), &self.nodes_instances);

        self.links_mesh
            .update_vertex_buffer(wgpu_renderer.queue(), &self.links_lines.vertices);
    }
}

impl VertexColorShaderDraw for AdvancedCompositionDrawer {
    fn draw<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        self.nodes_mesh.draw(render_pass);
    }
}

impl VertexColorShaderDrawLines for AdvancedCompositionDrawer {
    fn draw_lines<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        self.links_mesh.draw(render_pass);
    }
}
