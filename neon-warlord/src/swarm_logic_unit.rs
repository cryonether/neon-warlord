//! Training a neat reinforcement model to find a solution for a logic unit

use std::iter::zip;

use crate::{advanced_composition::genome_drawer::GenomeDrawer, reinforcement_learning::neat::Neat};
type Vec3 = cgmath::Vector3<f32>;

struct SwarmLogicUnit {
    neat: Neat,
    genome_drawers: Vec<GenomeDrawer>,

    logic_unit: LogicUnit,
}

impl SwarmLogicUnit {
    fn new() -> Self {
        let size = 1_000;

        let logic_unit = LogicUnit::new();

        let nr_inputs = logic_unit.get_nr_inputs();
        let nr_outputs = logic_unit.get_nr_outputs();

        let neat = Neat::new(nr_inputs, nr_outputs, size);

        let mut genome_drawers = Vec::new();
        for (i, genome) in neat.genomes.iter().enumerate() {
            let pos = Vec3::new(0.0, i as f32 * 2.0, 1.0);
            let genome_drawer = GenomeDrawer::new(genome, 0.2, pos);
            genome_drawers.push(genome_drawer);
        }

        Self { neat, genome_drawers, logic_unit }
    }

    fn update_physics(&mut self) {

        // evolve genomes
        self.neat.rank();
        self.neat.survival_selection();
        self.neat.evolve();

        // evaluate genomes
        for genome in &mut self.neat.genomes {
            let mut fitness = 0.0;
            for row in &self.logic_unit.logic_table {

                // update genome inputs
                for (row, node) in zip(row, genome.sensors()) {
                    node.value = *row as f32;
                }

                // evaluate
                genome.evaluate();
                
                // update genome fitness
                let res = genome.outputs()[0].value;
                let expected = *row.last().unwrap() as f32;
                let error = expected - res;

                // Max reward is 1.0 per case.
                fitness += 1.0 - error * error;
            }

            genome.fitness = fitness;
        }
    }
}


struct LogicUnit {
    pub logic_table: [[u8; 4]; 8]
}

impl LogicUnit {
    fn new() -> Self {

        let logic_table: [[u8; 4]; 8] = [
            // OR
            [0, 0, 0, 0],
            [0, 0, 1, 1],
            [0, 1, 0, 1],
            [0, 1, 1, 1],

            // AND
            [1, 0, 0, 0],
            [1, 0, 1, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 1],
        ];

        Self { logic_table  }
    }

    fn get_nr_inputs(&self) -> usize {
        self.logic_table[0].len() -1
    }

    fn get_nr_outputs(&self) -> usize {
        1
    }
}