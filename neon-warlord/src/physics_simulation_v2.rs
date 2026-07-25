//! Next iteration of the verlet physics simulation

use std::ops::DerefMut;

use wgpu_renderer::{vertex_color_shader::{VertexColorShaderDraw, vertex_color_shader_draw::VertexColorShaderDrawLines}, wgpu_renderer::WgpuRendererInterface};

use crate::{agents::{agent_definitions::get_agent_0_definition, agent_drawer::AgentDrawer, agent_factory::{self, AgentFactory}}, game_board::Agent, verlet_physics::{solver::Solver, verlet_composition::VerletComposition}};

type Vec3 = cgmath::Vector3<f32>;

pub struct PhysicsSimulationV2 {
    // Physics
    verlet_compositions: Vec<VerletComposition>,
    solver: Solver,
    ticks: u64,

    // Draw
    drawer: Vec<Drawer>,

    // Factories
    agent_factory: AgentFactory,
}

impl PhysicsSimulationV2 {
    
    pub fn new() -> Self {
        let verlet_compositions = Vec::new();
        let solver = Solver::new();
        let drawer = Vec::new();
        let agent_factory = AgentFactory::new();
        
        Self {
            verlet_compositions,
            solver,
            ticks: 0,
            drawer,
            agent_factory,
        }
    }
    
    // Creation

    pub fn create_agent_0(
        &mut self,
        wgpu_renderer: &mut dyn WgpuRendererInterface,
        
    ) {
        let layers = get_agent_0_definition();
        let scale = 0.1;
        let nodes = self.agent_factory.create_agent(&layers, Vec3::new(0.0, 0.0, 1.0), scale);
        let composition = VerletComposition::create(&nodes, scale / 2.0);
        let drawer = AgentDrawer::new(
            wgpu_renderer, 
            &composition,
            scale / 2.0,
        );

        self.verlet_compositions.push(composition);
        self.drawer.push(Drawer::AgentDrawer(drawer));
    }  

    // Update

    pub fn update_physics(&mut self) 
    {
        let dt = 1.0 / 60.0;
        self.ticks += 1;

        self.solver.update_composites(
            &mut self.verlet_compositions, 
            dt,
        );
    }

    pub fn update_device(
        &mut self, 
        wgpu_renderer: &mut dyn WgpuRendererInterface
    ) {
        assert!(self.drawer.len() == self.verlet_compositions.len());

        let size = self.drawer.len();
        for i in 0..size {
            match &mut self.drawer[i] {
                Drawer::AgentDrawer(agent_drawer) => {
                    agent_drawer.update(
                        wgpu_renderer, 
                        &self.verlet_compositions[i]
                    );
                },
            }
        }

    }
}

impl VertexColorShaderDraw for PhysicsSimulationV2 {
    fn draw<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        for elem in &self.drawer {
            match elem {
                Drawer::AgentDrawer(agent_drawer) => {
                    agent_drawer.draw(render_pass);
                },
            }
        }
    }
}

impl VertexColorShaderDrawLines for PhysicsSimulationV2 {
    fn draw_lines<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        for elem in &self.drawer {
            match elem {
                Drawer::AgentDrawer(agent_drawer) => {
                    agent_drawer.draw_lines(render_pass);
                },
            }
        }
    }
}


enum Drawer {
    AgentDrawer(AgentDrawer)
}