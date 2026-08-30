//! Deep Q Network (DQN)

use std::iter::zip;

use crate::reinforcement_learning::neural_network_simd::{self, NeuralNetworkSimd, gradients::{self, GradientsSimd}};

const LAYERS: usize = 3;

pub struct Dqn<const INPUTS: usize, const OUTPUTS: usize> {
    model: NeuralNetworkSimd<LAYERS>,

    gradients: Vec<GradientsSimd<LAYERS>>
}

impl<const INPUTS: usize, const OUTPUTS: usize> Dqn<INPUTS, OUTPUTS> {
    pub fn new() -> Self {
        let model = NeuralNetworkSimd::new_zero_one();
        let gradients = Vec::new();

        Self { model, gradients }
    }

    pub fn step(&mut self, inputs: &[f32], outputs: &mut [f32]) {
        assert!(inputs.len() == INPUTS);
        assert!(inputs.len() <= self.model.x.len());

        for (x, input) in zip(&mut self.model.x, inputs) {
            *x = *input;
        }

        let y = self.model.forward();
        let gradients = self.model.backward(0);

    } 
}