//! Deferred shader drawing a terrain height map
//!

// mod vertex;
mod instance;
mod mesh;
mod particle_shader_draw;
mod pipeline_particle;

pub use instance::Instance;
pub use mesh::Mesh;
pub use particle_shader_draw::ParticleShaderDraw;
pub use particle_shader_draw::ParticleShaderDrawRange;
pub use pipeline_particle::ParticleKind;
pub use pipeline_particle::PipelineParticle;

pub use wgpu_renderer::vertex_color_shader::Vertex;
// pub use wgpu_renderer::vertex_color_shader::Instance;
pub use wgpu_renderer::vertex_color_shader::CameraBindGroupLayout;
pub use wgpu_renderer::vertex_color_shader::IndexBuffer;
pub use wgpu_renderer::vertex_color_shader::InstanceBuffer;
pub use wgpu_renderer::vertex_color_shader::VertexBuffer;
