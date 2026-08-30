//! Deep Q Network (DQN)

use std::iter::zip;

use crate::reinforcement_learning::neural_network_simd::{NeuralNetworkSimd, gradients::GradientsSimd};

const LAYERS: usize = 3;

pub struct Dqn<const INPUTS: usize, const OUTPUTS: usize> {
    model: NeuralNetworkSimd<LAYERS>,
    index: usize,

    y_pred: Vec<f32>,
    gradients: Vec<GradientsSimd<LAYERS>>,

    loss: f32,
}

impl<const INPUTS: usize, const OUTPUTS: usize> Dqn<INPUTS, OUTPUTS> {
    pub fn new() -> Self {
        let model = NeuralNetworkSimd::new_zero_one();
        let index = 0;
        let y_pred = Vec::new();
        let gradients = Vec::new();
        let loss = 0.0;

        Self { index, model, y_pred, gradients, loss }
    }

    /// 1) Modifies the network's input buffer,
    /// 2) Performs inference,
    /// 3) Selects an action,
    pub fn predict(&mut self, inputs: &[f32]) -> (usize, [f32; OUTPUTS]) {
        assert!(inputs.len() == INPUTS);
        assert!(inputs.len() <= self.model.x.len());

        for (x, input) in zip(&mut self.model.x, inputs) {
            *x = *input;
        }

        let y_pred = self.model.forward();
        self.index = Self::arg_max(&y_pred);

        let res: [f32; OUTPUTS] = y_pred[..OUTPUTS].try_into().unwrap();
        (self.index, res)
    } 

    /// 4) Computes gradients,
    /// 5) Stores training state.
    pub fn remeber(&mut self) {
        let y_pred = self.model.y;
        let index = self.index;
        let gradients = self.model.backward(index);
        
        self.y_pred.push(y_pred[index]);
        self.gradients.push(gradients);
    }

    /// 6) Adjusts the training model
    pub fn adjust(&mut self, reward: f32) 
    {
        assert_eq!(self.y_pred.len(), self.gradients.len());
        if self.y_pred.len() == 0 {
            return;
        }

        let n = self.gradients.len();
        assert_eq!(n, self.y_pred.len());
        let n = n as f32;

        // Loss function
        // mean square error
        //      1    N-1
        // L = --- * ∑ (y_pred_i − y_i)²
        //      N    i=0
        let mut sum = 0.0;
        for y_pred in &self.y_pred {
            let y = reward;

            let diff = y_pred - y;
            sum += diff * diff;
        }
        let loss = sum / n;
        self.loss = loss;

        // accumulate gradients
        let mut gradients_loss_sum: GradientsSimd<LAYERS> = GradientsSimd::new();

        // Derivative loss function
        // derivative mean square error
        // ∂L           2
        // --------- = --- * (y_pred_i − y_i)
        // ∂L_pred_i    N
        for (y_pred, gradients) in zip(&self.y_pred, &self.gradients) {
            let y = reward;

            let diff = y_pred - y;
            let d_loss_dy = 2.0 / n * diff;

            // sum loss
            gradients_loss_sum += gradients * d_loss_dy;
        }

        // optimizer
        /// plain gradient descent
        /// w_new = w_old - eta * dw
        const LEARNING_RATE: f32 = 0.1;
        self.model
            .subtract_gradients(&(&gradients_loss_sum * LEARNING_RATE));


        // cleanup history
        self.y_pred.clear();
        self.gradients.clear();
    }


    fn arg_max(val: &[f32]) -> usize {
        assert!(val.len() >= OUTPUTS);

        let mut max_idx = 0;
        let mut max_val = val[0];

        for (i, &val) in zip(0..OUTPUTS, val) {
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }

        max_idx
    }

    fn epsilon_greedy(val: &[f32]) -> usize 
    {
        const EPSILON: f32 = 0.1;

        if fastrand::f32() > EPSILON {
            Self::arg_max(val)
        }
        else {
            fastrand::usize(0..OUTPUTS)
        }
    }
}


struct Transition<const INPUTS: usize> {
    state: [f32; INPUTS],
    action: usize,
    reward: f32,
    next_state: [f32; INPUTS],
    done: bool,
}