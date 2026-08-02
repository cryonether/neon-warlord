//! Advanced objects with actors and sensors using verlet physics and reinforcement learning

pub mod definition;
pub mod swarm;
pub mod advanced_composition_drawer;

use cgmath::Zero;

use crate::{advanced_composition::{self, definition::NodeKind}, reinforcement_learning::neat::Neat, verlet_physics::{self, VerletObject}};

type Vec3 = cgmath::Vector3<f32>;

/// Advanced objects with actors and sensors using verlet physics and reinforcement learning
pub struct AdvancedComposition {
    pub neural_networks: Vec<NeuralNetwork>,
    pub sensors: Vec<Sensor>,
    pub actors: Vec<Actor>,

    pub verlet_objects: Vec<VerletObject>,
    pub links: Vec<Link>,
}

impl AdvancedComposition {
    fn new(definition: &[definition::LocatedNode], pos: Vec3, radius: f32) -> Self {
        let neural_networks = Vec::new();
        let sensors = Vec::new();
        let actors = Vec::new();

        let mut verlet_objects = Vec::new();
        let mut links = Vec::new();

        // Create a verlet object for every node
        for elem in definition {
            let position_current = elem.pos + pos;

            match elem.node.kind{
                NodeKind::None => {
                    
                },
                NodeKind::Regular => {
                    verlet_objects.push(VerletObject::new(position_current, radius));
                },
                NodeKind::Static => {
                    
                },
                NodeKind::LinearMotor(_, _) => {
                    
                },
                NodeKind::NeuralNetwork => {
                    
                },
            }
        } 

        // Create all links
        for elem in definition {
            let id_0 = elem.node.id;
            let pos_0 = verlet_objects[id_0].position();


            match elem.node.edge {
                definition::EdgeKind::None => {
                    // nothing to do
                },
                definition::EdgeKind::Fixed(target) => {
                    let id_1 = target;
                    let pos_1 = verlet_objects[target].position();
                    links.push(
                        Link::Fixed(verlet_physics::fixed_link::FixedLink::new(id_0, id_1, pos_1 - pos_0)
                            .damping(0.9)
                            .force_split(0.45),)
                        );
                },
            }
        }

        Self { neural_networks, sensors, actors, verlet_objects, links  }
    }
}


struct NeuralNetwork {
    inputs: Vec<f32>,
    outputs: Vec<f32>,
    fitness: f32,
}

enum Sensor {

}

enum Actor {

}

// struct Link {
//     node_id_0: usize,
//     node_id_1: usize,
//     link_kind: LinkKind,
// }

pub enum Link {
    Fixed(verlet_physics::fixed_link::FixedLink)
}