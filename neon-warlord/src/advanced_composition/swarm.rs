//! Multiple Advanced composition together evolving neural networks

use std::iter::zip;

use cgmath::Zero;
use forward_renderer::particle_shader::ParticleShaderDraw;
use wgpu_renderer::{
    vertex_color_shader::{
        VertexColorShaderDraw, vertex_color_shader_draw::VertexColorShaderDrawLines,
    },
    wgpu_renderer::WgpuRendererInterface,
};

use crate::{
    advanced_composition::{
        AdvancedComposition, advanced_composition_drawer::AdvancedCompositionDrawer, definition::ParsedDefinition, genome_drawer::GenomeDrawer, neural_network::FitnessFunction,
    }, reinforcement_learning::neat::Neat,
};

type Vec3 = cgmath::Vector3<f32>;

/// Multiple Advanced composition together evolving neural networks
pub struct Swarm {
    /// Structure
    /// 1 element per entity
    pub advanced_composition: Vec<AdvancedComposition>,
    composition_drawer: Vec<AdvancedCompositionDrawer>,

    /// Reinforcement learning
    /// multiple elements per entity
    neats: Vec<Neat>,
    neat_drawers: Vec<Vec<GenomeDrawer>>,


    /// Draw
    /// 1 element per entity

    // Some random variables
    phase: f32,
    omega: f32, // radians/sec
}

impl Swarm {
    pub fn new(
        wgpu_renderer: &mut dyn WgpuRendererInterface,
        definition: &ParsedDefinition,
        size: usize,
    ) -> Self {
        let radius = definition.scale / 2.0;

        // create neural networks
        let nr_neural_networks = definition.count_nr_neural_networks();
        let neural_network_inputs = definition.count_nr_neural_network_inputs();
        let neural_network_outputs = definition.count_nr_neural_network_outputs();

        let mut neats = Vec::new();
        for _i in 0..nr_neural_networks {
            neats.push(Neat::new(
                neural_network_inputs,
                neural_network_outputs,
                size,
            ));
        }

        // create neat drawers
        let mut neat_drawers = Vec::new();
        for neat in &neats {
            let mut genome_drawers = Vec::new();
            for (i, genome) in neat.genomes.iter().enumerate() {
                let pos = Vec3::new(0.0, i as f32 * definition.scale as f32, definition.scale);
                let genome_drawer = GenomeDrawer::new(wgpu_renderer, genome, radius, pos);
                genome_drawers.push(genome_drawer);
            }
            neat_drawers.push(genome_drawers);
        }

        // create advanced compositions
        let pos = Vec3::zero();
        let mut advanced_composition = Vec::new();

        let a = f32::sqrt(size as f32) as usize;
        for i in 0..size {
            let pos = pos + Vec3::new((i % a) as f32, (i / a) as f32, 0.0);

            advanced_composition.push(AdvancedComposition::new(definition, pos, radius));
        }

        // drawer
        let mut composition_drawer = Vec::new();
        for advanced_composition in &advanced_composition {
            composition_drawer.push(AdvancedCompositionDrawer::new(
                wgpu_renderer,
                advanced_composition,
                radius,
            ));
        }

        Self {
            advanced_composition,
            neats,
            neat_drawers,
            composition_drawer,
            phase: 0.0,
            omega: 1.0,
        }
    }

    pub fn update_physics(&mut self, dt: f32) {
        // input
        // sensors -> neural_network inputs
        self.update_sensors();

        // NEAT evolve
        self.update_neuron_fitness();
        self.evolve_neurons();

        // NEAT calculate
        self.update_neuron_inputs();
        self.evaluate_neurons();
        self.update_neuron_outputs();

        // output
        // neural_network outputs -> actors
        self.update_actors(dt);

        // run verlet physics step
    }

    pub fn update_device(&mut self, wgpu_renderer: &mut dyn WgpuRendererInterface) {
        assert!(self.composition_drawer.len() == self.advanced_composition.len());

        // update composites
        let size = self.composition_drawer.len();
        for i in 0..size {
            self.composition_drawer[i].update(wgpu_renderer, &self.advanced_composition[i]);
        }

        // update neats
        for (neat_drawer, neat) in zip(&mut self.neat_drawers, &self.neats) {
            for (genome_drawer, genome) in zip(neat_drawer, &neat.genomes) {
                genome_drawer.update(wgpu_renderer, genome);
            }
        }
    }

    fn update_neuron_inputs(&mut self) {
        for i in 0..self.advanced_composition.len() {
            for j in 0..self.neats.len() {
                assert!(self.neats.len() == self.advanced_composition[i].neural_networks.len());
                assert!(self.advanced_composition.len() == self.neats[j].genomes.len());

                let neural_network = &self.advanced_composition[i].neural_networks[j];
                let genome = &mut self.neats[j].genomes[i];

                // update inputs
                for k in 0..neural_network.inputs.len() {
                    genome.sensors()[k].value = neural_network.inputs[k];
                }
            }
        }
    }

    fn update_neuron_fitness(&mut self) {
        for i in 0..self.advanced_composition.len() {
            for j in 0..self.neats.len() {
                assert!(self.neats.len() == self.advanced_composition[i].neural_networks.len());
                assert!(self.advanced_composition.len() == self.neats[j].genomes.len());

                let neural_network = &self.advanced_composition[i].neural_networks[j];
                let genome = &mut self.neats[j].genomes[i];

                assert!(neural_network.inputs.len() == genome.sensors().len());

                // update fitness
                genome.fitness = neural_network.fitness;
            }
        }
    }

    fn update_neuron_outputs(&mut self) {
        for i in 0..self.advanced_composition.len() {
            for j in 0..self.neats.len() {
                assert!(self.neats.len() == self.advanced_composition[i].neural_networks.len());
                assert!(self.advanced_composition.len() == self.neats[j].genomes.len());

                let neural_network = &mut self.advanced_composition[i].neural_networks[j];
                let genome = &self.neats[j].genomes[i];

                assert!(neural_network.outputs.len() == genome.outputs().len());

                // update outputs
                for k in 0..neural_network.outputs.len() {
                    neural_network.outputs[k] = genome.outputs()[k].value;
                }

                // update fitness
                neural_network.calculate_fitness();
            }
        }
    }

    fn evolve_neurons(&mut self) {
        for elem in &mut self.neats {
            elem.rank();
            elem.survival_selection();
            elem.evolve();
        }
    }

    fn evaluate_neurons(&mut self) {
        for elem in &mut self.neats {
            for genome in &mut elem.genomes {
                genome.evaluate();
            }
        }
    }

    fn update_sensors(&mut self) {
        for composition in &mut self.advanced_composition {
            let mut index = 0;
            for (_i, sensor) in &mut composition.sensors.iter_mut().enumerate() {
                match sensor {
                    super::Sensor::RelativePosition(elem) => {
                        // update sensor
                        elem.update(&composition.verlet_objects);

                        let val = elem.get_val();

                        // update connected neural network
                        if !composition.neural_networks.is_empty() {
                            composition.neural_networks[0].inputs[index] = val.x;
                            composition.neural_networks[0].inputs[index + 1] = val.y;
                            composition.neural_networks[0].inputs[index + 2] = val.z;
                            index += 3;
                        }
                    }
                }
            }
        }
    }

    fn update_actors(&mut self, dt: f32) {
        self.phase += self.omega * dt;

        // Keep phase small
        self.phase = self.phase.rem_euclid(std::f32::consts::TAU);

        let _sin = self.phase.sin();

        for composition in &mut self.advanced_composition {
            let mut index = 0;
            for actor in &mut composition.actors {
                match actor {
                    super::Actor::MotorLinear(motor_linear) => {
                        // get output from neural network
                        if !composition.neural_networks.is_empty() {
                            let val = composition.neural_networks[0].outputs[index];
                            index += 1;

                            motor_linear.accelerate(val, &mut composition.verlet_objects);
                        }

                        // update actor
                        motor_linear.update(&mut composition.verlet_objects);
                    }
                }
            }
        }
    }

    pub fn set_fitness_functions(
        mut self,
        fitness_functions: &[Box<dyn FitnessFunction>],
    ) -> Swarm {
        for elem in &mut self.advanced_composition {
            elem.set_fitnesss_functions(fitness_functions);
        }

        self
    }
}

impl VertexColorShaderDraw for Swarm {
    fn draw<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        for drawer in &self.composition_drawer {
            drawer.draw(render_pass);
        }
    }
}

impl VertexColorShaderDrawLines for Swarm {
    fn draw_lines<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        // composites
        for drawer in &self.composition_drawer {
            drawer.draw_lines(render_pass);
        }

        // neats
        for neat_drawer in &self.neat_drawers {
            for genome_drawer in neat_drawer {
                genome_drawer.draw_lines(render_pass);
            }
        }
    }
}

impl ParticleShaderDraw for Swarm {
    fn draw<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        for neat_drawer in &self.neat_drawers {
            for genome_drawer in neat_drawer {
                genome_drawer.draw(render_pass);
            }
        }
    }
}
