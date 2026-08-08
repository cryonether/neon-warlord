//! Next iteration of the verlet physics simulation

use forward_renderer::{height_map::HeightMapInterface, particle_shader::ParticleShaderDraw, particle_shader_two_point::ParticleShaderTwoPointDraw};
use wgpu_renderer::{
    vertex_color_shader::{
        VertexColorShaderDraw, vertex_color_shader_draw::VertexColorShaderDrawLines,
    },
    wgpu_renderer::WgpuRendererInterface,
};

use crate::{
    advanced_composition::{
        definition::{
            ParsedDefinition, get_pendulum_definition, get_pendulum_definition_fitness_function,
        },
        swarm::Swarm,
    },
    verlet_physics::solver::Solver,
};

type Vec3 = cgmath::Vector3<f32>;

pub struct PhysicsSimulationV3 {
    // Physics
    swarm: Swarm,
    solver: Solver,
    ticks: u64,
}

impl PhysicsSimulationV3 {
    pub fn new(wgpu_renderer: &mut dyn WgpuRendererInterface) -> Self {
        // agent 0
        let pos = Vec3::new(0.0, 0.0, 2.0);
        let scale = 0.1;
        let definition = get_pendulum_definition();
        let fitness_function = get_pendulum_definition_fitness_function();
        let parsed_definition = ParsedDefinition::parse(&definition, pos, scale);

        let swarm = Swarm::new(wgpu_renderer, &parsed_definition, 25)
            .set_fitness_functions(&[fitness_function]);

        // solver
        let solver = Solver::new();

        Self {
            swarm,
            solver,
            ticks: 0,
        }
    }

    // Update

    pub fn update_physics(&mut self, height_map: &impl HeightMapInterface) {
        let dt = 1.0 / 60.0;
        self.ticks += 1;

        self.swarm.update_physics(dt);

        self.solver.update_advanced_composites(
            &mut self.swarm.advanced_composition,
            height_map,
            dt,
        );
    }

    pub fn update_device(&mut self, wgpu_renderer: &mut dyn WgpuRendererInterface) {
        self.swarm.update_device(wgpu_renderer);
    }
}

impl VertexColorShaderDraw for PhysicsSimulationV3 {
    fn draw<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        VertexColorShaderDraw::draw(&self.swarm, render_pass);

    }
}

impl VertexColorShaderDrawLines for PhysicsSimulationV3 {
    fn draw_lines<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        self.swarm.draw_lines(render_pass);
    }
}

impl ParticleShaderDraw for PhysicsSimulationV3 {
    fn draw<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        ParticleShaderDraw::draw(&self.swarm, render_pass);
    }
}

impl ParticleShaderTwoPointDraw for PhysicsSimulationV3 {
    fn draw<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        
    }
}