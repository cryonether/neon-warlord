//! Advanced objects with actors and sensors using verlet physics and reinforcement learning

use cgmath::MetricSpace;

use crate::{
    advanced_composition::{
        definition::{self, NodeKind, ParsedDefinition},
        motor_linear::MotorLinear,
        neural_network::{FitnessFunction, NeuralNetwork},
        sensor_linear::SensorLinear,
        sensor_relative_position::SensorRelativePosition,
    }, verlet_physics_simd::VerletPhysicsSimd,
};

type Vec3 = cgmath::Vector3<f32>;

/// Advanced objects with actors and sensors using verlet physics and reinforcement learning
#[derive(Clone)]
pub struct AdvancedCompositionSimd {
    pub neural_networks: Vec<NeuralNetwork>,
    pub sensors: Vec<Sensor>,
    pub actors: Vec<Actor>,

    pub verlet_physics: VerletPhysicsSimd,

    // Tracks the indices for every distinct object
    objects: Vec<CompositeObjects>,
}

impl AdvancedCompositionSimd {
    pub fn new() -> Self {
        let neural_networks = Vec::new();
        let sensors = Vec::new();
        let actors = Vec::new();
        let verlet_physics = VerletPhysicsSimd::new();
        let objects = Vec::new();

        Self {
            neural_networks,
            sensors,
            actors,
            verlet_physics,
            objects,
        }
    }

    pub fn push(&mut self, definition: &ParsedDefinition, pos: Vec3, radius: f32) -> usize {
        let neural_network_inputs = definition.count_nr_neural_network_inputs();
        let neural_network_outputs = definition.count_nr_neural_network_outputs();

        let index = self.objects.len();
        assert!(self.objects.len() == self.neural_networks.len());

        let index_neural_network = self.objects.len();

        let range_sensors_start = self.sensors.len();
        let mut range_sensors_end = self.sensors.len();

        let range_actors_start = self.actors.len();
        let mut range_actors_end = self.actors.len();

        let range_particles_start = self.verlet_physics.particles.len();
        let mut range_particles_end = self.verlet_physics.particles.len();

        let range_distance_constraints_start = self.verlet_physics.distance_constraints.len();
        let mut range_distance_constraints_end = self.verlet_physics.distance_constraints.len();

        // Create a verlet object for every node
        for elem in &definition.nodes {
            let position_current = elem.pos + pos;
            let node_id = self.verlet_physics.particles.len();

            match elem.node.kind {
                NodeKind::None => {}
                NodeKind::Regular => {
                    self.verlet_physics.push_particle(position_current, radius, 1.0);
                    range_particles_end += 1;
                }
                NodeKind::Static => {
                    self.verlet_physics.push_particle(position_current, radius, 0.0);
                    range_particles_end += 1;
                }
                NodeKind::MotorLinear(a, b) => {
                    let node_id_a = range_particles_start + a;
                    let node_id_b = range_particles_start + b;

                    self.verlet_physics.push_particle(position_current, radius, 1.0);
                    range_particles_end += 1;
                    

                    self.actors.push(Actor::MotorLinear(MotorLinear::new(
                        node_id, node_id_a, node_id_b,
                    )));
                    range_actors_end += 1;

                    self.sensors.push(Sensor::SenorLinear(SensorLinear::new(
                        node_id, node_id_a, node_id_b,
                    )));
                    range_sensors_end += 1;
                }
                NodeKind::SensorRelativePosition(a) => {
                    let node_id_a = range_particles_start + a;

                    self.verlet_physics.push_particle(position_current, radius, 1.0);
                    range_particles_end += 1;

                    self.sensors.push(Sensor::RelativePosition(SensorRelativePosition::new(
                        node_id, node_id_a,
                    )));
                    range_sensors_end += 1;
                }
                NodeKind::NeuralNetwork => {
                    self.verlet_physics.push_particle(position_current, radius, 1.0);
                    range_particles_end += 1;

                    self.neural_networks.push(NeuralNetwork::new(
                        node_id,
                        neural_network_inputs,
                        neural_network_outputs,
                    ));
                }
            }
        }

        // Create all links
        for elem in &definition.nodes {
            let node_id_0 = range_particles_start + elem.node.id;
            let pos_0 = self.verlet_physics.get_particle_position(node_id_0);

            match elem.node.edge {
                definition::EdgeKind::None => {
                    // nothing to do
                }
                definition::EdgeKind::Fixed(target) => {
                    let node_id_1 =  range_particles_start + target;
                    let pos_1 = self.verlet_physics.get_particle_position(node_id_1);

                    let rest = pos_1 - pos_0;

                    self.verlet_physics.push_constraint_position(
                        node_id_0, 
                        node_id_1, 
                        rest,
                        0.8,
                    );
                    range_distance_constraints_end += 1;
                }
                definition::EdgeKind::FixedDistance(target) => {
                    let node_id_1 =  range_particles_start + target;
                    let pos_1 = self.verlet_physics.get_particle_position(node_id_1);

                    let rest = pos_0.distance(pos_1);

                    self.verlet_physics.push_constraint_distance(
                        node_id_0, 
                        node_id_1, 
                        rest,
                        0.8,
                    );
                    range_distance_constraints_end += 1;
                }
                definition::EdgeKind::Loose(target) => {
                    let node_id_1 =  range_particles_start + target;

                    self.verlet_physics.push_constraint_none(
                        node_id_0, 
                        node_id_1, 
                    );
                    range_distance_constraints_end += 1;
                }
            }
        }

        self.objects.push(CompositeObjects{
            index_neural_network,
            range_sensors: range_sensors_start..range_sensors_end,
            range_actors: range_actors_start..range_actors_end,
            _range_particles: range_particles_start..range_particles_end,
            _range_distance_constraints: range_distance_constraints_start..range_distance_constraints_end,
        });

        index
    }

    /// Sets the custom fitness function for each neural network
    pub fn set_fitness_functions(
        &mut self,
        fitness_functions: &[Box<dyn FitnessFunction + Send>],
    ) {
        assert!(fitness_functions.len() == 1);
        for neural_network in &mut self.neural_networks {
            neural_network.set_fitness_function(fitness_functions[0].clone());
        }
    }

    /// Sets the state of the neural network inputs based on the sensors
    pub fn update_neural_network_inputs(&mut self) {
        for object in &self.objects{
            let neural_network = &mut self.neural_networks[object.index_neural_network];

            // set position
            let node_id = neural_network.node_id;
            let pos = self.verlet_physics.get_particle_position(node_id);

            neural_network.position = pos + Vec3::new(0.0, 0.0, -0.2);

            // set inputs
            let mut k = 0;
            for sensor in &self.sensors[object.range_sensors.clone()] {
                match sensor {
                    Sensor::RelativePosition(sensor_relative_position) => {
                        let val = sensor_relative_position.get_val();
                        let val_0 = val.x;
                        let val_1 = val.y;
                        let val_2 = val.z;

                        neural_network.inputs[k] = val_0;
                        neural_network.inputs[k + 1] = val_1;
                        neural_network.inputs[k + 2] = val_2;
                        k += 3;
                    }
                    Sensor::SenorLinear(sensor_linear) => {
                        let val = sensor_linear.value();
                        neural_network.inputs[k] = val;
                        k += 1;
                    }
                }
            }
        }
    }

    /// Sets the state of the actors based on the neural network outputs
    pub fn update_neural_network_outputs(&mut self) {
        for object in &self.objects {
            let neural_network = &self.neural_networks[object.index_neural_network];

            // set outputs
            let mut i = 0;
            for actor in &mut self.actors[object.range_actors.clone()] {
                match actor {
                    Actor::MotorLinear(motor_linear) => {
                        let val = neural_network.outputs[i];

                        motor_linear.accelerate(val);
                        i += 1;
                    }
                }
            }
        }
    }

    /// Runs the custom neural network fitness function
    pub fn calculate_neural_network_fitness(&mut self) {
        for neural_network in &mut self.neural_networks {
            neural_network.calculate_fitness();
        }
    }

    /// Updates the internal state based on verlet physics
    pub fn update_sensors(&mut self) {
        for sensor in &mut self.sensors {
            match sensor {
                Sensor::RelativePosition(sensor_relative_position) => {
                    sensor_relative_position.update_simd(&self.verlet_physics.particles);
                }
                Sensor::SenorLinear(sensor_linear) => {
                    sensor_linear.update_simd(&self.verlet_physics.particles);
                }
            }
        }
    }

    /// Updates the verlet physics of the actors
    pub fn update_actors(&mut self) {
        for actor in &mut self.actors {
            match actor {
                Actor::MotorLinear(motor_linear) => {
                    motor_linear.update_simd(&mut self.verlet_physics.particles);
                }
            }
        }
    }

    pub fn len(&self) -> usize {
        self.objects.len()
    }
}

#[derive(Clone)]
struct CompositeObjects {
    pub index_neural_network: usize,
    pub range_sensors: std::ops::Range<usize>,
    pub range_actors: std::ops::Range<usize>,
    pub _range_particles: std::ops::Range<usize>,
    pub _range_distance_constraints: std::ops::Range<usize>,
}

#[derive(Clone)]
pub enum Sensor {
    RelativePosition(SensorRelativePosition),
    SenorLinear(SensorLinear),
}

#[derive(Clone)]
pub enum Actor {
    MotorLinear(MotorLinear),
}
