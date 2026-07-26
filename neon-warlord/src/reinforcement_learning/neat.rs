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
pub mod connection;
pub mod genome;

pub use node::Node;
pub use connection::Edge;
pub use genome::Genome;

pub struct Neat {

}

impl Neat {
    pub fn new() -> Self {
        Self {  }
    }

    fn evaluate(genome: &mut Genome) {

    } 

    fn growth() {

    }

    fn spicate() {

    }

    fn mate() {

    }
}







