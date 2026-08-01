//! Implementation of a NEAT algorithm
//!
//! Details:
//! 
//! Evolving Neural Networks through Augmenting Topologies (2002)
//! Kenneth O. Stanley, Risto Miikulainen 
//! 
//! Abstract:
//! An important question in neuroevolution is how to gain an advantage from evolving
//! neural network topologies along with weights. We present a method, NeuroEvolu-
//! tion of Augmenting Topologies (NEAT), which outperforms the best fixed-topology
//! method on a challenging benchmark reinforcement learning task. We claim that the
//! increased efficiency is due to (1) employing a principled method of crossover of differ-
//! ent topologies, (2) protecting structural innovation using speciation, and (3) incremen-
//! tally growing from minimal structure. We test this claim through a series of ablation
//! studies that demonstrate that each component is necessary to the system as a whole
//! and to each other. What results is significantly faster learning. NEAT is also an im-
//! portant contribution to GAs because it shows how it is possible for evolution to both
//! optimize and complexify solutions simultaneously, offering the possibility of evolving
//! increasingly complex solutions over generations, and strengthening the analogy with
//! biological evolution.
//! 

pub mod node;
pub mod edge;
pub mod genome;
mod tests;

pub use node::Node;
pub use edge::Edge;
pub use genome::Genome;

use fastrand::Rng;

pub struct Neat {
    genomes: Vec<Genome>,
    rank: Vec<usize>,

    rng: Rng,
}

impl Neat {
    pub fn new(
        nr_sensors: usize,
        nr_outputs: usize,
        size: usize
    ) -> Self {
        let mut genomes = Vec::with_capacity(size);
        let mut rank = Vec::with_capacity(size);

        for i in 0..size {
            genomes.push(Genome::new(nr_sensors, nr_outputs));
            rank.push(i);
        } 

        let seed: u64 = 0;
        let rng = Rng::with_seed(seed);

        Self { genomes, rank, rng }
    }

    /// Ranks all genomes
    pub fn rank(&mut self) {
        self.rank.sort_unstable_by(|&a, &b| {
            self.genomes[b]
                .fitness
                .total_cmp(&self.genomes[a].fitness)
        });
    }

    /// Returns the genome with rank 0
    pub fn get_rank_0(&self) -> Option<&Genome> {
        self.rank.first().map(|&index| &self.genomes[index])
    }

    /// Picks the fittest survivors and replaces the bottom with it
    pub fn survival_selection(&mut self) {
        let proportion = 0.3;
        
        // get to genome
        let best = self.get_rank_0();
        let best = match best {
            Some(best) => best.clone(),
            None => return,
        };

        // clone best genome
        let size = self.rank.len();
        let survivor = (size as f32 * proportion) as usize;

        for i in survivor..size {
            let index = self.rank[i];
            self.genomes[index] = best.clone();
        }  
    }

    /// Modifies the existing genomes
    pub fn evolve(&mut self) {        
        for genome in &mut self.genomes {
            let val = self.rng.f32();
            if val < 0.1 {
                Self::add_edge(genome, &mut self.rng);
            }
            else if val < 0.2 {
                Self::modify_bias(genome, &mut self.rng);
            }
            else if val < 0.3 {
                Self::modify_weight(genome, &mut self.rng);
            }
        }
    }

    fn add_edge(genome: &mut Genome, rng: &mut Rng) {

        // try insert element if it isn't duplicated or in the same layer
        for _i in 0..10 {
            let size = genome.nodes.len();
            let id_0 = rng.usize(0..size);
            let id_1 = rng.usize(id_0..size);

            let res = genome.add_edge(id_0, id_1);
            if res {
                // successfully inserted
                break;
            }
        }
    }

    fn modify_bias(genome: &mut Genome, rng: &mut Rng) {
        let size = genome.nodes.len();
        let index = rng.usize(0..size);
        let bias = rng.f32();
        genome.nodes[index].bias = bias;
    }

    fn modify_weight(genome: &mut Genome, rng: &mut Rng) {
        let size = genome.edges.len();
        if size == 0 {
            return;
        }

        let index = rng.usize(0..size);
        let weight = rng.f32();
        genome.edges[index].weight = weight;
    }

    fn evaluate() {

    } 

    fn growth() {

    }

    fn spicate() {

    }

    fn mate() {

    }
    

}







