//! An epoch of the neural network

use std::iter::zip;

use crate::reinforcement_learning::neural_network_simd::{
    NeuralNetworkSimd, activation_function::activation_none::ActivationNone,
};

pub struct EpochSimd<
    const INPUTS: usize,
    const OUTPUTS: usize,
    const NEURONS: usize,
    const NR_LAYERS: usize,
    const RESIDUAL: bool,
> {
    pub model: NeuralNetworkSimd<INPUTS, OUTPUTS, NEURONS, NR_LAYERS, RESIDUAL, ActivationNone>,

    pub loss: f32,
}

impl<
    const INPUTS: usize,
    const OUTPUTS: usize,
    const NEURONS: usize,
    const NR_LAYERS: usize,
    const RESIDUAL: bool,
> EpochSimd<INPUTS, OUTPUTS, NEURONS, NR_LAYERS, RESIDUAL>
{
    pub fn new() -> Self {
        let model = NeuralNetworkSimd::new_rand(0);
        let loss = 0.0;

        Self { model, loss }
    }

    pub fn learn<const BATCH_SIZE: usize>(
        &mut self,
        input: [[f32; INPUTS]; BATCH_SIZE],
        output: [[f32; 1]; BATCH_SIZE],
    ) -> [f32; BATCH_SIZE] {
        self.learn_output(input, output, 0)
    }

    pub fn learn_output<const BATCH_SIZE: usize>(
        &mut self,
        input: [[f32; INPUTS]; BATCH_SIZE],
        output: [[f32; 1]; BATCH_SIZE],
        output_index: usize,
    ) -> [f32; BATCH_SIZE] {
        let mut y_pred_vec: Vec<f32> = Vec::new();

        let n = BATCH_SIZE as f32;

        // accumulate gradients
        let mut sum = 0.0;
        // evaluate
        for (input, output) in zip(input, output) {
            let y_pred_ = self.model.forward(&input);

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
            let mut backward_vec = [0.0; OUTPUTS];
            backward_vec[output_index] = 1.0 * d_loss_dy;
            let _dx = self.model.backward(&backward_vec);
        }

        let loss = sum / n;
        self.loss = loss;

        // optimizer
        /// plain gradient descent
        /// w_new = w_old - eta * dw
        const LEARNING_RATE: f32 = 0.1;
        self.model.subtract_gradients(LEARNING_RATE);

        let res: [f32; BATCH_SIZE] = y_pred_vec.try_into().unwrap();

        res
    }
}
