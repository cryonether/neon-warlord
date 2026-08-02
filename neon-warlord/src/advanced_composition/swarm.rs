//! Multiple Advanced composition together evolving neural networks

use cgmath::Zero;

use crate::{advanced_composition::definition::{self, NodeKind}, reinforcement_learning::neat::Neat};

type Vec3 = cgmath::Vector3<f32>;

/// Multiple Advanced composition together evolving neural networks
pub struct Swarm {
    // 1 element per entity
    advanced_composition: Vec<super::AdvancedComposition>,
    // multiple elements per entity
    neat: Vec<Neat>,
}

impl Swarm {
    pub fn new(definition: &[definition::LocatedNode], size: usize) -> Self {

        // create neural networks
        let nr_neural_networks = Self::count_nr_neural_networks(definition);
        let neural_network_inputs = 2;
        let neural_network_outputs = 2;

        let mut neat = Vec::new();
        for _i in 0..nr_neural_networks {
            neat.push(Neat::new(
                neural_network_inputs,
                neural_network_outputs, 
                size
            ));
        }

        // create advanced compositions
        let pos = Vec3::zero();
        let mut advanced_composition = Vec::new();
        for _i in 0..size {
            advanced_composition.push(super::AdvancedComposition::new(definition, pos));
        }

        Self { advanced_composition, neat }
    }

    fn update_neuron_inputs(&mut self) {
        for i in 0..self.advanced_composition.len() {
            for j in 0..self.neat.len() {
                assert!(self.neat.len() == self.advanced_composition[i].neural_networks.len());
                assert!(self.advanced_composition.len() == self.neat[j].genomes.len());

                let neural_network = &self.advanced_composition[i].neural_networks[j];
                let genome = &mut self.neat[j].genomes[i];

                // update inputs
                for k in 0..neural_network.inputs.len() {
                    genome.sensors()[k].value = neural_network.inputs[k];
                }
            }
        }
    }

    fn update_neuron_fitness(&mut self) {
        for i in 0..self.advanced_composition.len() {
            for j in 0..self.neat.len() {
                assert!(self.neat.len() == self.advanced_composition[i].neural_networks.len());
                assert!(self.advanced_composition.len() == self.neat[j].genomes.len());

                let neural_network = &self.advanced_composition[i].neural_networks[j];
                let genome = &mut self.neat[j].genomes[i];

                assert!(neural_network.inputs.len() == genome.sensors().len());

                // update fitness
                genome.fitness = neural_network.fitness;
            }
        }
    }

    fn update_neuron_outputs(&mut self) {
        for i in 0..self.advanced_composition.len() {
            for j in 0..self.neat.len() {
                assert!(self.neat.len() == self.advanced_composition[i].neural_networks.len());
                assert!(self.advanced_composition.len() == self.neat[j].genomes.len());

                let neural_network = &mut self.advanced_composition[i].neural_networks[j];
                let genome = &self.neat[j].genomes[i];

                assert!(neural_network.outputs.len() == genome.outputs().len());

                // update outputs
                for k in 0..neural_network.outputs.len() {
                    neural_network.outputs[k] = genome.outputs()[k].value;
                }
            }
        }
    }

    pub fn evolve_neurons(&mut self) {
        for elem in &mut self.neat {
            elem.rank();
            elem.survival_selection();
            elem.evolve();
        }
    }

    pub fn evaluate_neurons(&mut self) {
        for elem in &mut self.neat {
            for genome in &mut elem.genomes {
                genome.evaluate();
            }
        }
    }

    pub fn update(&mut self) {
        // input
        self.update_sensors();

        // evolve
        self.update_neuron_fitness();
        self.evolve_neurons();

        // update neurons
        self.update_neuron_inputs();
        self.evaluate_neurons();
        self.update_neuron_outputs();

        // output
        self.update_actors();

        // physics
        self.update_verlet_physics();
    }

    fn update_sensors(&mut self) {

    }

    fn update_actors(&mut self) {

    }

    fn update_verlet_physics(&mut self) {

    }

    fn count_nr_neural_networks(definition: &[definition::LocatedNode]) -> usize {
        let mut sum = 0;
        for elem in definition {
            if elem.node.kind == NodeKind::NeuralNetwork {
                sum += 1;
            }
        }

        sum
    }
}
