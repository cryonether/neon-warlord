//! Interface to draw objects of this shader
//!

pub trait ParticleShaderDraw {
    fn draw<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>);
}

pub trait ParticleShaderDrawRange {
    fn draw_range<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>, nr_instances: usize);
}
