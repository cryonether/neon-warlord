//! An epoch of the neural network

use std::iter::zip;


use crate::reinforcement_learning::neural_network_simd::{Gradient16, NeuralNetwork16, gradients_sum::GradientsSum};


pub struct EpochSimd<const SIZE: usize> {
    pub model: NeuralNetwork16<16, 16, SIZE, false>,

    pub loss: f32,
}

impl<const SIZE: usize> EpochSimd<SIZE> {
    pub fn new() -> Self {
        let model = NeuralNetwork16::new_rand();
        let loss = 0.0;

        Self { model, loss }
    }

    pub fn learn<const INPUT_SIZE: usize, const BATCH_SIZE: usize>(
        &mut self,
        input: [[f32; INPUT_SIZE]; BATCH_SIZE],
        output: [[f32; 1]; BATCH_SIZE],
    ) -> [f32; BATCH_SIZE] {
        self.learn_output(input, output, 0)
    }

    pub fn learn_output<const INPUT_SIZE: usize, const BATCH_SIZE: usize>(
        &mut self,
        input: [[f32; INPUT_SIZE]; BATCH_SIZE],
        output: [[f32; 1]; BATCH_SIZE],
        output_index: usize,
    ) -> [f32; BATCH_SIZE] 
    {
        let mut y_pred_vec: Vec<f32> = Vec::new();

        let n = BATCH_SIZE as f32;

        // accumulate gradients
        let mut gradients_loss_sum: Gradient16<SIZE> = GradientsSum::new();
        let mut sum = 0.0;
        // evaluate
        for (input, output) in zip(input, output) {
            for (x, input) in zip(self.model.x.as_mut_array(), input) {
                *x = input;
            }

            let x = self.model.x;
            let y_pred_ = self.model.forward(x.as_array());
            let gradients = self.model.backward(output_index);

            let y_pred = y_pred_[output_index];
            y_pred_vec.push(y_pred);

            // Loss function
            // mean square error
            //      1    N-1
            // L = --- * ∑ (y_pred_i − y_i)²
            //      N    i=0

            let y = output[0];

            let diff = y_pred - y;
            sum += diff * diff;

            // Derivative loss function
            // derivative mean square error
            // ∂L           2
            // --------- = --- * (y_pred_i − y_i)
            // ∂L_pred_i    N
            let y = output[0];

            let diff = y_pred - y;
            let d_loss_dy = 2.0 / n * diff;

            // sum loss
            gradients_loss_sum.add_loss_gradients(&gradients, d_loss_dy);
        }

        let loss = sum / n;
        self.loss = loss;

        // optimizer
        /// plain gradient descent
        /// w_new = w_old - eta * dw
        const LEARNING_RATE: f32 = 0.1;
        self.model
            .subtract_gradients(&(&gradients_loss_sum * LEARNING_RATE));

        let res: [f32; BATCH_SIZE] = y_pred_vec.try_into().unwrap();

        res
    }
}
