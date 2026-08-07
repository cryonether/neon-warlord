//! Advanced objects with actors and sensors using verlet physics and reinforcement learning

pub mod definition;
pub mod swarm;
pub mod advanced_composition_drawer;
pub mod motor_linear;
pub mod sensor_relative_position;
pub mod neural_network;

use cgmath::{InnerSpace, MetricSpace, Zero};

use crate::{advanced_composition::{self, definition::{NodeKind, ParsedDefinition}, motor_linear::MotorLinear, neural_network::{FitnessFunction, NeuralNetwork}, sensor_relative_position::SensorRelativePosition}, reinforcement_learning::neat::Neat, verlet_physics::{self, VerletObject}};

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
    fn new(definition: &ParsedDefinition, pos: Vec3, radius: f32) -> Self {
        let neural_network_inputs = definition.count_nr_neural_network_inputs();
        let neural_network_outputs = definition.count_nr_neural_network_outputs();

        let mut neural_networks = Vec::new();
        let mut sensors = Vec::new();
        let mut actors = Vec::new();

        let mut verlet_objects = Vec::new();
        let mut links = Vec::new();

        // Create a verlet object for every node
        for elem in &definition.nodes {
            let position_current = elem.pos + pos;

            match elem.node.kind{
                NodeKind::None => {
                    
                },
                NodeKind::Regular => {
                    verlet_objects.push(VerletObject::new(position_current, radius));
                },
                NodeKind::Static => {
                    let mut verlet_object = VerletObject::new(position_current, radius);
                    verlet_object.is_static = true;
                    verlet_objects.push(verlet_object);
                },
                NodeKind::MotorLinear(a, b) => {
                    let node_id = verlet_objects.len();
                    verlet_objects.push(VerletObject::new(position_current, radius));
                    actors.push(Actor::MotorLinear(MotorLinear{
                        node_id,
                        node_a_id: a,
                        node_b_id: b,
                    }));
                },
                NodeKind::SensorRelativePosition(a) => {
                    let node_id = verlet_objects.len();
                    verlet_objects.push(VerletObject::new(position_current, radius));

                    sensors.push(Sensor::RelativePosition(SensorRelativePosition::new(
                        node_id,
                        a,
                    )));
                },
                NodeKind::NeuralNetwork => {
                    verlet_objects.push(VerletObject::new(position_current, radius));

                    neural_networks.push(NeuralNetwork::new(neural_network_inputs, neural_network_outputs));
                },
            }
        } 

        // Create all links
        for elem in &definition.nodes {
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
                definition::EdgeKind::FixedDistance(target) => {
                    let id_1 = target;
                    let pos_1 = verlet_objects[target].position();
                    links.push(
                        Link::FixedDistance(verlet_physics::link::Link::new(id_0, id_1, pos_0.distance(pos_1)))
                        );
                },
                definition::EdgeKind::Loose(target) => {
                    let id_1 = target;
                    links.push(
                        Link::Loose(verlet_physics::loose_link::LooseLink::new(id_0, id_1))
                        );
                },
            }
        }

        Self { neural_networks, sensors, actors, verlet_objects, links  }
    }

    pub fn set_fitnesss_functions(&mut self, fitness_functions: &[Box<dyn FitnessFunction>]) {
        assert!(self.neural_networks.len() == fitness_functions.len());
        for (neural_network, fitness_function) in std::iter::zip(&mut self.neural_networks, fitness_functions) {
            neural_network.set_fitness_function(fitness_function.clone());
        }
    }
}


enum Sensor {
    RelativePosition(SensorRelativePosition)
}

enum Actor {
    MotorLinear(MotorLinear)
}

pub enum Link {
    Fixed(verlet_physics::fixed_link::FixedLink),
    FixedDistance(verlet_physics::link::Link),
    Loose(verlet_physics::loose_link::LooseLink),
}