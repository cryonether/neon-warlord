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

pub struct Neat {
    genomes: Vec<Genome>,
    rank: Vec<usize>,
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

        Self { genomes, rank }
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


    pub fn evolve(&mut self) {        

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







