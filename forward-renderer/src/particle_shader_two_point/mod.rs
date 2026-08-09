//! Deferred shader drawing a terrain height map
//!

// mod vertex;
mod instance;
mod mesh;
mod particle_two_point_shader_draw;
mod pipeline_particle_two_point;

pub use instance::Instance;
pub use mesh::Mesh;
pub use particle_two_point_shader_draw::ParticleShaderTwoPointDraw;
pub use particle_two_point_shader_draw::ParticleShaderTwoPointDrawRange;
pub use pipeline_particle_two_point::ParticleKind;
pub use pipeline_particle_two_point::PipelineParticleTwoPoint;

pub use wgpu_renderer::vertex_color_shader::CameraBindGroupLayout;
pub use wgpu_renderer::vertex_color_shader::IndexBuffer;
pub use wgpu_renderer::vertex_color_shader::InstanceBuffer;
pub use wgpu_renderer::vertex_color_shader::Vertex;
pub use wgpu_renderer::vertex_color_shader::VertexBuffer;
