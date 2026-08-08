//! Interface to draw objects of this shader
//!

pub trait ParticleShaderTwoPointDraw {
    fn draw<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>);
}

pub trait ParticleShaderTwoPointDrawRange {
    fn draw_range<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>, nr_instances: usize);
}
