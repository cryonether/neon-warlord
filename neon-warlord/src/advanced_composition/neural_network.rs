//! Representing a neural network

/// Representing a neural network
pub struct NeuralNetwork {
    pub inputs: Vec<f32>,
    pub outputs: Vec<f32>,
    pub fitness: f32,
}

impl NeuralNetwork {
    pub fn new(inputs: usize, outputs: usize) -> Self {
        let inputs = vec![0.0; inputs];
        let outputs = vec![0.0; outputs];
        let fitness = 0.0;

        Self {
            inputs,
            outputs,
            fitness,
        }
    }
}