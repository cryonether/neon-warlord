//! Advanced objects with actors and sensors using verlet physics and reinforcement learning

pub mod advanced_composition_drawer;
pub mod definition;
pub mod motor_linear;
pub mod sensor_linear;
pub mod neural_network;
pub mod sensor_relative_position;
pub mod swarm;
pub mod genome_drawer;

use cgmath::MetricSpace;

use crate::{
    advanced_composition::{
        definition::{NodeKind, ParsedDefinition}, motor_linear::MotorLinear, neural_network::{FitnessFunction, NeuralNetwork}, sensor_linear::SensorLinear, sensor_relative_position::SensorRelativePosition,
    }, verlet_physics::{self, VerletObject},
};

type Vec3 = cgmath::Vector3<f32>;

/// Advanced objects with actors and sensors using verlet physics and reinforcement learning
#[derive(Clone)]
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

            match elem.node.kind {
                NodeKind::None => {}
                NodeKind::Regular => {
                    verlet_objects.push(VerletObject::new(position_current, radius));
                }
                NodeKind::Static => {
                    let mut verlet_object = VerletObject::new(position_current, radius);
                    verlet_object.is_static = true;
                    verlet_objects.push(verlet_object);
                }
                NodeKind::MotorLinear(a, b) => {
                    let node_id = verlet_objects.len();
                    let node_id_a = a;
                    let node_id_b = b;

                    verlet_objects.push(VerletObject::new(position_current, radius));
                    actors.push(Actor::MotorLinear(MotorLinear::new(
                        node_id,
                        node_id_a,
                        node_id_b,
                    )));
                    sensors.push(Sensor::SenorLinear(SensorLinear::new(
                        node_id, 
                        node_id_a, 
                        node_id_b,
                    )));
                }
                NodeKind::SensorRelativePosition(a) => {
                    let node_id = verlet_objects.len();
                    verlet_objects.push(VerletObject::new(position_current, radius));

                    sensors.push(Sensor::RelativePosition(SensorRelativePosition::new(
                        node_id, a,
                    )));
                }
                NodeKind::NeuralNetwork => {
                    let node_id = verlet_objects.len();
                    verlet_objects.push(VerletObject::new(position_current, radius));

                    neural_networks.push(NeuralNetwork::new(
                        node_id, 
                        neural_network_inputs,
                        neural_network_outputs,
                    ));
                }
            }
        }

        // Create all links
        for elem in &definition.nodes {
            let id_0 = elem.node.id;
            let pos_0 = verlet_objects[id_0].position();

            match elem.node.edge {
                definition::EdgeKind::None => {
                    // nothing to do
                }
                definition::EdgeKind::Fixed(target) => {
                    let id_1 = target;
                    let pos_1 = verlet_objects[target].position();
                    links.push(Link::Fixed(
                        verlet_physics::fixed_link::FixedLink::new(id_0, id_1, pos_1 - pos_0)
                            .damping(0.9)
                            .force_split(0.45),
                    ));
                }
                definition::EdgeKind::FixedDistance(target) => {
                    let id_1 = target;
                    let pos_1 = verlet_objects[target].position();
                    links.push(Link::FixedDistance(verlet_physics::link::Link::new(
                        id_0,
                        id_1,
                        pos_0.distance(pos_1),
                    )));
                }
                definition::EdgeKind::Loose(target) => {
                    let id_1 = target;
                    links.push(Link::Loose(verlet_physics::loose_link::LooseLink::new(
                        id_0, id_1,
                    )));
                }
            }
        }

        Self {
            neural_networks,
            sensors,
            actors,
            verlet_objects,
            links,
        }
    }

    pub fn set_fitnesss_functions(&mut self, fitness_functions: &[Box<dyn FitnessFunction>]) {
        assert!(self.neural_networks.len() == fitness_functions.len());
        for (neural_network, fitness_function) in
            std::iter::zip(&mut self.neural_networks, fitness_functions)
        {
            neural_network.set_fitness_function(fitness_function.clone());
        }
    }


    pub fn update_neural_network_inputs(&mut self) {
        for neural_network in &mut self.neural_networks {
            // set position
            let node_id = neural_network.node_id;
            let pos =  self.verlet_objects[node_id].position();
            
            neural_network.position = pos + Vec3::new(0.0, 0.0, -0.2);

            // set inputs
            let mut i = 0;
            for sensors in &self.sensors {
                match sensors {
                    Sensor::RelativePosition(sensor_relative_position) => {
                        let val = sensor_relative_position.get_val();
                        let val_0 = val.x;
                        let val_1 = val.y;
                        let val_2 = val.z;

                        neural_network.inputs[i] = val_0;
                        neural_network.inputs[i+1] = val_1;
                        neural_network.inputs[i+2] = val_2;
                        i += 3;
                    },
                    Sensor::SenorLinear(sensor_linear) => {
                        let val = sensor_linear.value();
                        neural_network.inputs[i] = val;
                        i += 1;
                    },
                }
            }
        }
    }

    pub fn update_neural_network_outputs(&mut self) {
        for neural_network in &self.neural_networks {
            // set outputs
            let mut i = 0;
            for actor in &mut self.actors {
                match actor {
                    Actor::MotorLinear(motor_linear) => {
                        let val = neural_network.outputs[i];

                        motor_linear.accelerate(val);                        
                        i += 1;
                    },
                }
            }
        }
    }

    fn calculate_neural_network_fitness(&mut self) {
        for neural_network in &mut self.neural_networks {
            neural_network.calculate_fitness();
        }
    }

    pub fn update_sensors(&mut self) {
        for sensor in &mut self.sensors {
            match sensor {
                Sensor::RelativePosition(sensor_relative_position) => {
                    sensor_relative_position.update(&self.verlet_objects);
                },
                Sensor::SenorLinear(sensor_linear) => {
                    sensor_linear.update(&self.verlet_objects);
                },
            }
        }
    }

    pub fn update_actors(&mut self) {
        for actor in &mut self.actors {
            match actor {
                Actor::MotorLinear(motor_linear) => {
                    motor_linear.update(&mut self.verlet_objects);
                },
            }
        }
    }
    

}

#[derive(Clone)]
pub enum Sensor {
    RelativePosition(SensorRelativePosition),
    SenorLinear(SensorLinear)
}

#[derive(Clone)]
pub enum Actor {
    MotorLinear(MotorLinear),
}

#[derive(Clone)]
pub enum Link {
    Fixed(verlet_physics::fixed_link::FixedLink),
    FixedDistance(verlet_physics::link::Link),
    Loose(verlet_physics::loose_link::LooseLink),
}
