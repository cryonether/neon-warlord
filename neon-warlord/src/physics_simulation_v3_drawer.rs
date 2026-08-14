//! Draws Objects from the physics simulation

use forward_renderer::{
    geometry,
    particle_shader::{self, ParticleShaderDrawRange},
    particle_shader_two_point::{self, ParticleShaderTwoPointDrawRange},
};
use wgpu_renderer::performance_monitor::watch::{self};
use wgpu_renderer::wgpu_renderer::WgpuRendererInterface;

pub struct PhysicsSimulationV3Drawer {
    genome_nodes_mesh: particle_shader::Mesh,
    genome_edges_mesh: particle_shader_two_point::Mesh,

    verlet_object_nodes_mesh: particle_shader::Mesh,
    verlet_object_edges_mesh: particle_shader_two_point::Mesh,
    nr_genome_nodes: usize,
    nr_verlet_object_nodes: usize,
    nr_genome_edges: usize,
    nr_verlet_object_edges: usize,
}

impl PhysicsSimulationV3Drawer {
    pub fn new(wgpu_renderer: &mut dyn WgpuRendererInterface) -> Self {
        // Genome nodes
        let genome_node_size = 1.0;
        let genome_node_quad = geometry::Quad::new(genome_node_size); // 4 positions
        let genome_nodes_mesh =
            particle_shader::Mesh::from_geometry(wgpu_renderer.device(), &genome_node_quad, &[]);

        // Genome edges
        let genome_edge_size = 1.0;
        let genome_edge_quad = geometry::Quad::new(genome_edge_size); // 4 positions
        let genome_edges_mesh = particle_shader_two_point::Mesh::from_geometry(
            wgpu_renderer.device(),
            &genome_edge_quad,
            &[],
        );

        // Verlet object nodes
        let verlet_object_nodes_size = 1.0;
        let verlet_object_nodes_quad = geometry::Quad::new(verlet_object_nodes_size); // 4 positions
        let verlet_object_nodes_mesh = particle_shader::Mesh::from_geometry(
            wgpu_renderer.device(),
            &verlet_object_nodes_quad,
            &[],
        );

        // Verlet object edges
        let verlet_object_edges_size = 1.0;
        let verlet_object_edges_quad = geometry::Quad::new(verlet_object_edges_size); // 4 positions
        let verlet_object_edges_mesh = particle_shader_two_point::Mesh::from_geometry(
            wgpu_renderer.device(),
            &verlet_object_edges_quad,
            &[],
        );

        Self {
            genome_nodes_mesh,
            genome_edges_mesh,
            verlet_object_nodes_mesh,
            verlet_object_edges_mesh,
            nr_genome_nodes: 0,
            nr_verlet_object_nodes: 0,
            nr_genome_edges: 0,
            nr_verlet_object_edges: 0,
        }
    }

    pub fn update(
        &mut self,
        wgpu_renderer: &mut dyn WgpuRendererInterface,
        consumer: &DrawerObjects,
    ) {
        // Genome nodes
        {
            let instances = &consumer.genome_nodes;
            let mesh = &mut self.genome_nodes_mesh;
            self.nr_genome_nodes = instances.len();
            if mesh.max_instances() < instances.len() {
                let new_len = instances.len() * 2;

                let mut new_instances = Vec::with_capacity(new_len);
                for _i in 0..new_len {
                    new_instances.push(particle_shader::Instance::new());
                }
                mesh.resize_instance_buffer(wgpu_renderer.device(), &new_instances);
            }
            mesh.update_instance_buffer(wgpu_renderer.queue(), instances);
        }

        // Genome edges
        {
            let instances = &consumer.genome_edges;
            let mesh = &mut self.genome_edges_mesh;
            self.nr_genome_edges = instances.len();
            if mesh.max_instances() < instances.len() {
                let new_len = instances.len() * 2;

                let mut new_instances = Vec::with_capacity(new_len);
                for _i in 0..new_len {
                    new_instances.push(particle_shader_two_point::Instance::new());
                }
                mesh.resize_instance_buffer(wgpu_renderer.device(), &new_instances);
            }
            mesh.update_instance_buffer(wgpu_renderer.queue(), instances);
        }

        // Verlet object nodes
        {
            let instances = &consumer.verlet_object_nodes;
            let mesh = &mut self.verlet_object_nodes_mesh;
            self.nr_verlet_object_nodes = instances.len();
            if mesh.max_instances() < instances.len() {
                let new_len = instances.len() * 2;

                let mut new_instances = Vec::with_capacity(new_len);
                for _i in 0..new_len {
                    new_instances.push(particle_shader::Instance::new());
                }
                mesh.resize_instance_buffer(wgpu_renderer.device(), &new_instances);
            }
            mesh.update_instance_buffer(wgpu_renderer.queue(), instances);
        }

        // Verlet object edges
        {
            let instances = &consumer.verlet_object_edges;
            let mesh = &mut self.verlet_object_edges_mesh;
            self.nr_verlet_object_edges = instances.len();
            if mesh.max_instances() < instances.len() {
                let new_len = instances.len() * 2;

                let mut new_instances = Vec::with_capacity(new_len);
                for _i in 0..new_len {
                    new_instances.push(particle_shader_two_point::Instance::new());
                }
                mesh.resize_instance_buffer(wgpu_renderer.device(), &new_instances);
            }
            mesh.update_instance_buffer(wgpu_renderer.queue(), instances);
        }
    }
}

impl particle_shader::ParticleShaderDraw for PhysicsSimulationV3Drawer {
    fn draw<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        if self.nr_genome_nodes > 0 {
            self.genome_nodes_mesh
                .draw_range(render_pass, self.nr_genome_nodes);
        }
        if self.nr_verlet_object_nodes > 0 {
            self.verlet_object_nodes_mesh
                .draw_range(render_pass, self.nr_verlet_object_nodes);
        }
    }
}

impl particle_shader_two_point::ParticleShaderTwoPointDraw for PhysicsSimulationV3Drawer {
    fn draw<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        if self.nr_genome_edges > 0 {
            self.genome_edges_mesh
                .draw_range(render_pass, self.nr_genome_edges);
        }
        if self.nr_verlet_object_edges > 0 {
            self.verlet_object_edges_mesh
                .draw_range(render_pass, self.nr_verlet_object_edges);
        }
    }
}

pub const WATCH_POINTS_SIZE: usize = 10;

#[derive(Clone)]
pub struct DrawerObjects {
    pub genome_nodes: Vec<particle_shader::Instance>,
    pub genome_edges: Vec<particle_shader_two_point::Instance>,

    pub verlet_object_nodes: Vec<particle_shader::Instance>,
    pub verlet_object_edges: Vec<particle_shader_two_point::Instance>,

    pub ups: u32,
    pub watch_ups: watch::WatchViewerData<WATCH_POINTS_SIZE>,
}

impl DrawerObjects {
    pub fn clear(&mut self) {
        self.genome_nodes.clear();
        self.genome_edges.clear();
        self.verlet_object_nodes.clear();
        self.verlet_object_edges.clear();
    }

    pub fn new() -> Self {
        let genome_nodes = Vec::new();
        let genome_edges = Vec::new();
        let verlet_object_nodes = Vec::new();
        let verlet_object_edges = Vec::new();

        let watch_ups = watch::WatchViewerData::new();

        Self {
            genome_nodes,
            genome_edges,
            verlet_object_nodes,
            verlet_object_edges,
            ups: 0,
            watch_ups,
        }
    }
}
