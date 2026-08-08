//! Representing a neural network
//! 

use cgmath::Zero;
type Vec3 = cgmath::Vector3<f32>;

/// Representing a neural network
#[derive(Clone)]
pub struct NeuralNetwork {
    pub node_id: usize,
    pub position: Vec3,

    pub inputs: Vec<f32>,
    pub outputs: Vec<f32>,
    pub fitness: f32,

    pub fitness_function: Option<Box<dyn FitnessFunction>>,
}

impl NeuralNetwork {
    pub fn new(node_id: usize, inputs: usize, outputs: usize) -> Self {
        let inputs = vec![0.0; inputs];
        let outputs = vec![0.0; outputs];
        let fitness = 0.0;
        let position = Vec3::zero();

        Self {
            node_id,
            position,

            inputs,
            outputs,
            fitness,

            fitness_function: None,
        }
    }

    pub fn set_fitness_function(&mut self, fitness_function: Box<dyn FitnessFunction>) {
        self.fitness_function = Some(fitness_function);
    }

    pub fn calculate_fitness(&mut self) {
        if let Some(fitness_function) = &mut self.fitness_function {
            self.fitness = fitness_function.calculate_fitness(&self.inputs);
        }
    }
}

pub trait FitnessFunction {
    fn calculate_fitness(&mut self, outputs: &[f32]) -> f32;
    fn clone_box(&self) -> Box<dyn FitnessFunction>;
}

impl Clone for Box<dyn FitnessFunction> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}
