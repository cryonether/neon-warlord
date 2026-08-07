//! Contains the device buffers to render an object with this shader
//!

use crate::geometry::MeshInterface;
use crate::particle_shader::ParticleShaderDraw;
use crate::particle_shader::particle_shader_draw::ParticleShaderDrawRange;

use super::IndexBuffer;
use super::Instance;
use super::InstanceBuffer;
use super::Vertex;
use super::VertexBuffer;

/// A general purpose shader using vertices, colors and an instance matrix
pub struct Mesh {
    vertex_buffer: VertexBuffer<Vertex>,
    index_buffer: IndexBuffer<u32>,
    instance_buffer: InstanceBuffer<Instance>,
}

#[allow(dead_code)]
impl Mesh {
    pub fn new(
        device: &wgpu::Device,
        vertices: &[Vertex],
        indices: &[u32],
        instances: &[Instance],
    ) -> Self {
        let vertex_buffer = VertexBuffer::new(device, vertices);
        let index_buffer = IndexBuffer::new(device, indices);
        let instance_buffer = InstanceBuffer::new(device, instances);

        Self {
            vertex_buffer,
            index_buffer,
            instance_buffer,
        }
    }

    pub fn from_geometry(
        device: &wgpu::Device,
        data: &dyn MeshInterface,
        instances: &[Instance],
    ) -> Self {
        let vertices = data.vertices();
        let normals = data.normals();
        let indices = data.indices();

        assert_eq!(vertices.len(), normals.len());

        let len = vertices.iter().len();
        let mut mesh_vertices = Vec::with_capacity(len);

        for elem in vertices {
            mesh_vertices.push(Vertex {
                position: (*elem).into(),
            });
        }

        Self::new(device, &mesh_vertices, indices, instances)
    }

    pub fn update_vertex_buffer(&mut self, queue: &wgpu::Queue, vertices: &[Vertex]) {
        self.vertex_buffer.update(queue, vertices);
    }

    pub fn update_instance_buffer(&mut self, queue: &wgpu::Queue, instances: &[Instance]) {
        self.instance_buffer.update(queue, instances);
    }

    fn do_draw<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        self.vertex_buffer.bind(render_pass);
        self.index_buffer.bind(render_pass);
        self.instance_buffer.bind(render_pass);

        render_pass.draw_indexed(
            0..self.index_buffer.size(),
            0,
            0..self.instance_buffer.size(),
        );
    }
}

impl ParticleShaderDraw for Mesh {
    fn draw<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        self.draw_range(render_pass, self.instance_buffer.size() as usize);
    }
}

impl ParticleShaderDrawRange for Mesh {
    fn draw_range<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>, nr_instances: usize) {
        self.vertex_buffer.bind(render_pass);
        self.index_buffer.bind(render_pass);
        self.instance_buffer.bind(render_pass);

        render_pass.draw_indexed(
            0..self.index_buffer.size(),
            0,
            0..nr_instances as u32,
        );
    }
}
