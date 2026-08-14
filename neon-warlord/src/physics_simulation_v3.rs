//! Next iteration of the verlet physics simulation

use forward_renderer::{
    height_map::HeightMapInterface, particle_shader::ParticleShaderDraw,
    particle_shader_two_point::ParticleShaderTwoPointDraw,
};
use wgpu_renderer::{
    performance_monitor::Fps, vertex_color_shader::{
        VertexColorShaderDraw, vertex_color_shader_draw::VertexColorShaderDrawLines,
    }, wgpu_renderer::WgpuRendererInterface,
};

use crate::{
    advanced_composition::{
        definition::{
            ParsedDefinition, get_pendulum_definition, get_pendulum_definition_fitness_function,
        },
        swarm::Swarm,
    }, physics_simulation_v3_drawer::DrawerObjects, triple_buffer, verlet_physics::solver::Solver, worker_thread,
};

type Vec3 = cgmath::Vector3<f32>;

pub struct PhysicsSimulationV3 {
    producer: triple_buffer::Producer<DrawerObjects>,

    // Physics
    swarm: Swarm,
    solver: Solver,
    ticks: u64,

    // Debug
    ups: Fps,
    last_render_time: instant::Instant,
}

impl PhysicsSimulationV3 {
    pub fn new(
        producer: triple_buffer::Producer<DrawerObjects>
    ) -> Self {
        // agent 0
        let pos = Vec3::new(0.0, 0.0, 2.0);
        let scale = 0.1;
        let definition = get_pendulum_definition();
        let fitness_function = get_pendulum_definition_fitness_function();
        let parsed_definition = ParsedDefinition::parse(&definition, pos, scale);

        let swarm = Swarm::new(&parsed_definition, 1000)
            .set_fitness_functions(&[fitness_function]);

        // solver
        let solver = Solver::new();

        // Debug
        let ups = Fps::new();

        Self {
            producer,
        
            swarm,
            solver,
            ticks: 0,

            ups,
            last_render_time: instant::Instant::now(),
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

        // ups
        let now = instant::Instant::now();
        let dt = now - self.last_render_time;
        self.last_render_time = now;
        self.ups.update(dt);
    }

    pub fn update_drawer(&mut self) {
        let data = self.producer.buffer();
        data.clear();

        self.swarm.update_drawer(data);
        data.ups = self.ups.get(); 

        self.producer.publish();
    }
}

// impl VertexColorShaderDraw for PhysicsSimulationV3 {
//     fn draw<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
//         VertexColorShaderDraw::draw(&self.swarm, render_pass);
//     }
// }

// impl VertexColorShaderDrawLines for PhysicsSimulationV3 {
//     fn draw_lines<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
//         self.swarm.draw_lines(render_pass);
//     }
// }



pub struct PhysicSimThread<T> {
    pub sim: PhysicsSimulationV3,
    pub height_map: T,
}

impl<T> worker_thread::Update for PhysicSimThread<T>
where T: HeightMapInterface
{
    fn update(&mut self) {
        self.sim.update_physics(&self.height_map);
        self.sim.update_drawer();
    }
}