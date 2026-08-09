//! Draws a Neat network

use std::alloc::LayoutError;

use cgmath::Rotation3;
use forward_renderer::{geometry, particle_shader::{self, ParticleShaderDrawRange}, particle_shader_two_point::{self, ParticleShaderTwoPointDrawRange}, to_rgb};
use wgpu_renderer::{
    vertex_color_shader::{
        self, VertexColorShaderDraw, vertex_color_shader_draw::{VertexColorShaderDrawLines},
    }, wgpu_renderer::WgpuRendererInterface,
};

use crate::{advanced_composition::AdvancedComposition, reinforcement_learning::neat::{self, Neat}};
use cgmath::VectorSpace;

type Vec3 = cgmath::Vector3<f32>;


/// Draws AdvancedCompositions
pub struct GenomeDrawer {
    nodes_instances: Vec<particle_shader::Instance>,
    nodes_mesh: particle_shader::Mesh,

    edges_instances: Vec<particle_shader_two_point::Instance>,
    edges_mesh: particle_shader_two_point::Mesh,

    size: f32,
    color_negative: Vec3,
    color_zero: Vec3,
    color_positive: Vec3,

    nr_nodes: usize,
    nr_edges: usize,

    position: Vec3,
}

impl GenomeDrawer {
    pub fn new(
        wgpu_renderer: &mut dyn WgpuRendererInterface,
        genome: &neat::Genome,
        radius: f32,
        position: Vec3,
    ) -> Self {
        let nr_nodes = genome.nodes.len();
        let nr_edges = genome.edges.len();

        let color_negative = to_rgb("#0911ff");
        let color_zero = to_rgb("#282428");
        let color_positive = to_rgb("#ff0d0d");

        // let size = radius*2.0;
        let size = radius;

        let (nodes_instances, nodes_mesh) = Self::create_nodes(
            wgpu_renderer, 
            color_zero.into(), 
            size, 
            nr_nodes, 
        );

        let (edges_instances, edges_mesh)= Self::create_edges(
            wgpu_renderer, 
            color_zero.into(), 
            size*0.5,
            nr_edges
        );

        Self {
            nodes_mesh,
            nodes_instances,
            edges_instances,
            edges_mesh,
            size,
            color_negative: color_negative.into(),
            color_zero: color_zero.into(),
            color_positive: color_positive.into(),
            nr_nodes,
            nr_edges,
            position,
        }
    }

    pub fn update(
        &mut self,
        wgpu_renderer: &mut dyn WgpuRendererInterface,
        genome: &neat::Genome,
    ) {
        self.position = genome.world_position;

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
            let z = index as f32 * self.size * 2.0;
            index += 1;
            
            let pos = Vec3::new(x, 0.0, z) + self.position;
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
        if edges.len() > self.edges_instances.len() {
            self.grow_edges(
                wgpu_renderer,
                edges.len() * 2);
        }

        // update data
        let mut nr_edges = 0;
        for (edge, instance) in std::iter::zip(&genome.edges, &mut self.edges_instances) {
            if !edge.enabled {
                continue;
            }

            let index_from = edge.index_from;
            let index_to = edge.index_to;
            let pos_0: Vec3 = self.nodes_instances[index_from].position.into();
            let color_0: Vec3 = self.nodes_instances[index_from].color.into();
            let pos_1: Vec3 = self.nodes_instances[index_to].position.into();
            let weight = edge.weight;

            instance.position_0 = pos_0.into();
            instance.position_1 = pos_1.into();
            instance.color = color_0.into();
            instance.size = self.size * 0.1 * weight;

            nr_edges += 1;
        }

        // update device
        self.nr_edges = nr_edges;
        self.edges_mesh.update_instance_buffer(
            wgpu_renderer.queue(), 
            &self.edges_instances[0..nr_edges],
        );

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
        let (edges_instances, edges_mesh) = Self::create_edges(
            wgpu_renderer, 
            self.color_zero,
            self.size * 0.5, 
            nr_edges
        );

        self.edges_instances = edges_instances;
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
                size: size,
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
        color_zero: Vec3,
        size: f32,
        nr_edges:usize,
    ) -> (Vec<particle_shader_two_point::Instance>, particle_shader_two_point::Mesh)  {
        let node_quad = geometry::Quad::new(size); // 4 positions
        let mut nodes_quads = geometry::Mesh::new();
        for _i in 0..nr_edges {
            nodes_quads.add(&node_quad);
        }

        let instance = particle_shader_two_point::Instance {
                position_0: [0.0, 0.0, 0.0],
                position_1: [0.0, 0.0, 0.0],
                color: color_zero.into(),
                time: 1.0,
                size: size,
            };

        let mut edge_instances = Vec::with_capacity(nr_edges);
        for _i in 0..nr_edges {
            edge_instances.push(instance);
        }

        let edges_mesh = particle_shader_two_point::Mesh::from_geometry(
            wgpu_renderer.device(),
            &nodes_quads,
            &edge_instances,
        );

        (edge_instances, edges_mesh)
    }

}


impl particle_shader::ParticleShaderDraw for GenomeDrawer {
    fn draw<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        self.nodes_mesh.draw_range(render_pass, self.nr_nodes);
    }
}

impl particle_shader_two_point::ParticleShaderTwoPointDraw for GenomeDrawer {
    fn draw<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        if self.nr_edges > 0 {
            self.edges_mesh.draw_range(render_pass, self.nr_edges);
        }
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
