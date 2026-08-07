//! Draws a Neat network

use std::alloc::LayoutError;

use cgmath::Rotation3;
use forward_renderer::{geometry, particle_shader::{self, ParticleShaderDrawRange}, to_rgb};
use wgpu_renderer::{
    vertex_color_shader::{
        self, VertexColorShaderDraw, vertex_color_shader_draw::{VertexColorShaderDrawLines, VertexColorShaderDrawLinesRange},
    }, wgpu_renderer::WgpuRendererInterface,
};

use crate::{advanced_composition::AdvancedComposition, reinforcement_learning::neat::{self, Neat}};
use cgmath::VectorSpace;

type Vec3 = cgmath::Vector3<f32>;


/// Draws AdvancedCompositions
pub struct GenomeDrawer {
    nodes_instances: Vec<particle_shader::Instance>,
    nodes_mesh: particle_shader::Mesh,

    edges_lines: geometry::Lines,
    edges_mesh: vertex_color_shader::Mesh,

    size: f32,
    color_negative: Vec3,
    color_zero: Vec3,
    color_positive: Vec3,
    color_edge: Vec3,

    nr_nodes: usize,
    nr_edges: usize,
}

impl GenomeDrawer {
    pub fn new(
        wgpu_renderer: &mut dyn WgpuRendererInterface,
        genome: &neat::Genome,
        radius: f32,
    ) -> Self {
        let nr_nodes = genome.nodes.len();
        let nr_edges = genome.edges.len();

        let color_negative = to_rgb("#6164cc");
        let color_zero = to_rgb("#1d0b20");
        let color_positive = to_rgb("#ca6868");
        let color_edge = to_rgb("#8c6993");

        let size = radius*2.0;

        let (nodes_instances, nodes_mesh) = Self::create_nodes(
            wgpu_renderer, 
            color_zero.into(), 
            size, 
            nr_nodes
        );

        let (edges_lines, edges_mesh)= Self::create_edges(
            wgpu_renderer, 
            color_edge.into(), 
            size, 
            nr_edges
        );

        Self {
            nodes_mesh,
            nodes_instances,
            edges_lines,
            edges_mesh,
            size,
            color_negative: color_negative.into(),
            color_zero: color_zero.into(),
            color_positive: color_positive.into(),
            color_edge: color_edge.into(),
            nr_nodes,
            nr_edges,
        }
    }

    pub fn update(
        &mut self,
        wgpu_renderer: &mut dyn WgpuRendererInterface,
        genome: &neat::Genome,
    ) {
        self.update_nodes(wgpu_renderer, genome);
        self.update_edges(wgpu_renderer, genome);
    }

    fn update_nodes(
        &mut self,
        wgpu_renderer: &mut dyn WgpuRendererInterface,
        genome: &neat::Genome,
    ) {
        // check if there is enough space
        let nodes = &genome.nodes;
        if nodes.len() > self.nodes_instances.len() {
            self.grow_nodes(
                wgpu_renderer,
                nodes.len() * 2
            );
        }

        // update data
        let mut previous_layer = 0;
        let mut index = 0;
        for (node, instance) in std::iter::zip(&genome.nodes, &mut self.nodes_instances) {
            
            // position
            let layer = node.layer;
            
            if layer > previous_layer {
                previous_layer = layer;
                index = 0;
            }

            let x = layer as f32 * self.size * 2.0;
            let y = index as f32;
            index += 1;
            
            let pos = Vec3::new(x, y, 0.0);
            instance.position = pos.into();

            // color
            let value = node.value;
            let color = gradient(value, self.color_negative, self.color_zero, self.color_positive);
            instance.color = color.into();
        }

        // update device
        self.nr_nodes = nodes.len();
        self.nodes_mesh.update_instance_buffer(
            wgpu_renderer.queue(), 
            &self.nodes_instances[0..nodes.len()],
        );
    }


    fn update_edges(
        &mut self,
        wgpu_renderer: &mut dyn WgpuRendererInterface,
        genome: &neat::Genome,
    ) {
        // check if there is enough space
        let edges = &genome.edges;
        if edges.len() > self.edges_lines.vertices.len() {
            self.grow_edges(
                wgpu_renderer,
                edges.len() * 2);
        }

        // update data
        for (i, edge) in genome.edges.iter().enumerate() {
            let index_from = edge.index_from;
            let index_to = edge.index_to;
            let pos_0: Vec3 = self.nodes_instances[index_from].position.into();
            let pos_1: Vec3 = self.nodes_instances[index_to].position.into();

            self.edges_lines.set_line_position(i, pos_0, pos_1);
        }

        // update device
        self.nr_edges = edges.len();
        self.edges_mesh.update_vertex_buffer(wgpu_renderer.queue(), &self.edges_lines.vertices[0..self.nr_edges*2]);

    }

    fn grow_nodes(
        &mut self,
        wgpu_renderer: &mut dyn WgpuRendererInterface,
        nr_nodes: usize,
    ) {
        let (nodes_instances, nodes_mesh) = Self::create_nodes(
            wgpu_renderer, 
            self.color_zero, 
            self.size, 
            nr_nodes
        );

        self.nodes_instances = nodes_instances;
        self.nodes_mesh = nodes_mesh;
    }

    fn grow_edges(
        &mut self,
        wgpu_renderer: &mut dyn WgpuRendererInterface,
        nr_edges: usize,
    ) {
        let (edges_lines, edges_mesh) = Self::create_edges(
            wgpu_renderer, 
            self.color_edge, 
            self.size, 
            nr_edges
        );

        self.edges_lines = edges_lines;
        self.edges_mesh = edges_mesh;
    }

    fn create_nodes(
        wgpu_renderer: &mut dyn WgpuRendererInterface,
        color_zero: Vec3,
        size: f32,
        nr_nodes:usize,
    ) -> (Vec<particle_shader::Instance>, particle_shader::Mesh) 
    {
        let node_quad = geometry::Quad::new(size); // 4 positions
        let mut nodes_quads = geometry::Mesh::new();
        for _i in 0..nr_nodes {
            nodes_quads.add(&node_quad);
        }

        let instance = particle_shader::Instance {
                position: [0.0, 0.0, 0.0],
                color: color_zero.into(),
                time: 1.0,
                size,
            };

        let mut nodes_instances = Vec::with_capacity(nr_nodes);
        for _i in 0..nr_nodes {
            nodes_instances.push(instance);
        }

        let nodes_mesh = particle_shader::Mesh::from_geometry(
            wgpu_renderer.device(),
            &nodes_quads,
            &nodes_instances,
        );

        (nodes_instances, nodes_mesh)
    }

    fn create_edges(
        wgpu_renderer: &mut dyn WgpuRendererInterface,
        color_edge: Vec3,
        size: f32,
        nr_edges:usize,
    ) -> (geometry::Lines, vertex_color_shader::Mesh) {
        let edges_lines = geometry::Lines::new_color_fade(
            nr_edges, 
            color_edge.into(), 
            color_edge.into()
        );

        let edges_instances = vec![vertex_color_shader::Instance::zero()];

        let edges_mesh = vertex_color_shader::Mesh::new(
            wgpu_renderer.device(),
            &edges_lines.vertices,
            &edges_lines.colors,
            &edges_lines.indices,
            &edges_instances,
        );

        (edges_lines, edges_mesh)
    }

}


impl particle_shader::ParticleShaderDraw for GenomeDrawer {
    fn draw<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        self.nodes_mesh.draw_range(render_pass, self.nr_nodes);
    }
}


impl VertexColorShaderDrawLines for GenomeDrawer {
    fn draw_lines<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        self.edges_mesh.draw_lines_range(render_pass, self.nr_edges);
    }
}


fn gradient(
    t: f32,
    negative: Vec3,
    zero: Vec3,
    positive: Vec3,
) -> Vec3 {
    let t = t.clamp(-1.0, 1.0);

    if t < 0.0 {
        // Map [-1, 0] -> [0, 1]
        negative.lerp(zero, t + 1.0)
    } else {
        // Map [0, 1] -> [0, 1]
        zero.lerp(positive, t)
    }
}