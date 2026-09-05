//! An epoch of the neural network

use std::iter::zip;

use itertools::izip;

use crate::reinforcement_learning::neural_network_simd::gradients::GradientsSimd;

use super::NeuralNetworkSimd;

pub struct EpochSimd<const SIZE: usize> {
    pub model: NeuralNetworkSimd<16, 16, SIZE, false>,

    pub loss: f32,
}

impl<const SIZE: usize> EpochSimd<SIZE> {
    pub fn new() -> Self {
        let model = NeuralNetworkSimd::new_rand();
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
        output_indx: usize,
    ) -> [f32; BATCH_SIZE] {
        let mut y_pred = Vec::new();
        let mut history = Vec::new();

        // evaluate
        for input in input {
            for (x, input) in zip(&mut self.model.x, input) {
                *x = input;
            }

            let x = self.model.x;
            let y_pred_ = self.model.forward(&x);
            let gradients = self.model.backward(output_indx);

            y_pred.push(y_pred_[output_indx]);
            history.push(gradients);
        }

        let n = history.len();
        assert_eq!(n, y_pred.len());
        assert_eq!(n, output.len());
        let n = n as f32;

        // Loss function
        // mean square error
        //      1    N-1
        // L = --- * ∑ (y_pred_i − y_i)²
        //      N    i=0
        let mut sum = 0.0;
        for (y_pred, output) in izip!(&y_pred, output) {
            let y = output[0];

            let diff = y_pred - y;
            sum += diff * diff;
        }
        let loss = sum / n;
        self.loss = loss;

        // accumulate gradients
        let mut gradients_loss_sum: GradientsSimd<SIZE> = GradientsSimd::new();

        // Derivative loss function
        // derivative mean square error
        // ∂L           2
        // --------- = --- * (y_pred_i − y_i)
        // ∂L_pred_i    N
        for (y_pred, output, gradients) in izip!(&y_pred, output, history) {
            let y = output[0];

            let diff = y_pred - y;
            let d_loss_dy = 2.0 / n * diff;

            // sum loss
            gradients_loss_sum += &gradients * d_loss_dy;
        }

        // optimizer
        /// plain gradient descent
        /// w_new = w_old - eta * dw
        const LEARNING_RATE: f32 = 0.1;
        self.model
            .subtract_gradients(&(&gradients_loss_sum * LEARNING_RATE));

        let res: [f32; BATCH_SIZE] = y_pred.try_into().unwrap();

        res
    }
}
