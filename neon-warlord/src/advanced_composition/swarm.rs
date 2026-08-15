//! Multiple Advanced composition together evolving neural networks

use std::iter::zip;

use cgmath::Zero;
use wgpu_renderer::performance_monitor::watch::Watch;

use crate::{
    advanced_composition::{
        advanced_composition_drawer::AdvancedCompositionDrawer, definition::ParsedDefinition,
        genome_drawer::GenomeDrawer, neural_network::FitnessFunction,
    },
    advanced_composition_simd::AdvancedCompositionSimd,
    physics_simulation_v3_drawer::DrawerObjects,
    reinforcement_learning::neat::Neat,
};

type Vec3 = cgmath::Vector3<f32>;

/// Multiple Advanced composition together evolving neural networks
pub struct Swarm {
    /// Structure
    /// 1 element per entity
    pub advanced_composition: AdvancedCompositionSimd,
    advanced_composition_original: AdvancedCompositionSimd,
    composition_drawer: AdvancedCompositionDrawer,

    /// Reinforcement learning
    /// multiple elements per entity
    neat: Neat,
    genome_drawers: Vec<GenomeDrawer>,

    /// Draw
    /// 1 element per entity
    // Some random variables
    // phase: f32,
    // omega: f32, // radians/sec
    ticks: u64,
}

impl Swarm {
    pub fn new(definition: &ParsedDefinition, size: usize) -> Self {
        let radius = definition.scale / 2.0;

        // create neural networks
        let _nr_neural_networks = definition.count_nr_neural_networks();
        let neural_network_inputs = definition.count_nr_neural_network_inputs();
        let neural_network_outputs = definition.count_nr_neural_network_outputs();

        let neat = Neat::new(neural_network_inputs, neural_network_outputs, size);

        // create neat drawers
        let mut genome_drawers = Vec::new();
        for (i, genome) in neat.genomes.iter().enumerate() {
            let pos = Vec3::new(0.0, i as f32 * definition.scale, definition.scale);
            let genome_drawer = GenomeDrawer::new(genome, radius * 0.5, pos);
            genome_drawers.push(genome_drawer);
        }

        // create advanced compositions
        let pos = Vec3::zero();
        let mut advanced_composition = AdvancedCompositionSimd::new();

        let a = f32::sqrt(size as f32) as usize;
        for i in 0..size {
            let pos = pos + Vec3::new((i % a) as f32, (i / a) as f32, 0.0);

            advanced_composition.push(definition, pos, radius);
        }

        let advanced_composition_original = advanced_composition.clone();

        // drawer
        let composition_drawer = AdvancedCompositionDrawer::new(&advanced_composition, radius);

        Self {
            advanced_composition,
            advanced_composition_original,
            neat,
            genome_drawers,
            composition_drawer,
            // phase: 0.0,
            // omega: 1.0,
            ticks: 0,
        }
    }

    pub fn update_physics(&mut self, dt: f32, watch_ups: &mut Watch<10>) {
        // input
        // sensors -> neural_network inputs
        watch_ups.start("swarm input");
        self.update_sensors();
        self.update_neural_network_inputs();
        self.calculate_neural_network_fitness();

        // NEAT evolve
        watch_ups.start("swarm evolve");
        self.update_genome_fitness();
        if self.ticks.is_multiple_of(3000) {
            self.evolve_genomes();
            self.advanced_composition = self.advanced_composition_original.clone();
        }

        // NEAT calculate
        watch_ups.start("swarm neat");
        self.update_genome_inputs();
        self.evaluate_genomes();
        self.update_genome_outputs();

        // output
        // neural_network outputs -> actors
        watch_ups.start("swarm output");
        self.update_neural_network_outputs();
        self.update_actors(dt);

        // run verlet physics step
        self.ticks += 1
    }

    pub fn update_drawer(&mut self, producer: &mut DrawerObjects, watch_ups: &mut Watch<10>) {
        // update composites
        watch_ups.start("draw composition");
        self.composition_drawer.update(
            &self.advanced_composition,
            &mut producer.verlet_object_nodes,
            &mut producer.verlet_object_edges,
        );

        // update neats
        watch_ups.start("draw genomes");
        for (genome_drawer, genome) in zip(&mut self.genome_drawers, &self.neat.genomes) {
            genome_drawer.update(
                genome,
                &mut producer.genome_nodes,
                &mut producer.genome_edges,
            );
        }
    }

    fn evolve_genomes(&mut self) {
        self.neat.rank();
        self.neat.survival_selection();
        self.neat.evolve();
    }

    fn evaluate_genomes(&mut self) {
        for genome in &mut self.neat.genomes {
            genome.evaluate();
        }
    }

    fn update_genome_inputs(&mut self) {
        let neat = &mut self.neat;

        assert!(neat.genomes.len() == self.advanced_composition.len());
        for i in 0..neat.genomes.len() {
            let neural_network = &self.advanced_composition.neural_networks[i];
            let genome = &mut neat.genomes[i];

            assert!(genome.nr_sensors == neural_network.inputs.len());
            for (sensor, input) in zip(genome.sensors(), &neural_network.inputs) {
                sensor.value = *input;
            }

            genome.world_position = neural_network.position;
        }
    }

    fn update_genome_outputs(&mut self) {
        let neat = &mut self.neat;

        assert!(neat.genomes.len() == self.advanced_composition.len());
        for i in 0..neat.genomes.len() {
            let neural_network = &mut self.advanced_composition.neural_networks[i];
            let genome = &neat.genomes[i];

            assert!(genome.nr_outputs == neural_network.outputs.len());
            for (output_genome, output_neural_network) in
                zip(genome.outputs(), &mut neural_network.outputs)
            {
                *output_neural_network = output_genome.value;
            }
        }
    }

    fn update_genome_fitness(&mut self) {
        let neat = &mut self.neat;

        assert!(neat.genomes.len() == self.advanced_composition.len());
        for i in 0..neat.genomes.len() {
            let neural_network = &self.advanced_composition.neural_networks[i];
            let genome = &mut neat.genomes[i];

            genome.fitness = neural_network.fitness;
        }
    }

    fn update_neural_network_inputs(&mut self) {
        self.advanced_composition.update_neural_network_inputs();
    }

    fn calculate_neural_network_fitness(&mut self) {
        self.advanced_composition.calculate_neural_network_fitness();
    }

    fn update_neural_network_outputs(&mut self) {
        self.advanced_composition.update_neural_network_outputs();
    }

    fn update_actors(&mut self, _dt: f32) {
        self.advanced_composition.update_actors();
    }

    fn update_sensors(&mut self) {
        self.advanced_composition.update_sensors();
    }

    pub fn set_fitness_functions(
        mut self,
        fitness_functions: &[Box<dyn FitnessFunction + Send>],
    ) -> Swarm {
        self.advanced_composition
            .set_fitness_functions(fitness_functions);

        self
    }
}
