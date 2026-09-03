//! Deep Q Network (DQN)

#[cfg(test)]
mod test_maze;

use std::iter::zip;

use crate::reinforcement_learning::neural_network_simd::{
    NeuralNetworkSimd, gradients::GradientsSimd,
};

const LAYERS: usize = 3;

pub struct Dqn<const INPUTS: usize, const OUTPUTS: usize> {
    model: NeuralNetworkSimd<LAYERS>,
    index: usize,
    index_max: usize,

    transitions: Vec<Transition>,
    gradients: Vec<GradientsSimd<LAYERS>>,

    loss: f32,
}

impl<const INPUTS: usize, const OUTPUTS: usize> Dqn<INPUTS, OUTPUTS> {
    pub fn new() -> Self {
        let model = NeuralNetworkSimd::new_rand();
        let index = 0;
        let index_max = 0;
        let transitions = Vec::new();
        let gradients = Vec::new();
        let loss = 0.0;

        Self {
            index,
            index_max,
            model,
            transitions,
            gradients,
            loss,
        }
    }

    ///
    /// 1) Modifies the network's input buffer,
    /// 2) Performs inference,
    /// 3) Selects an action,
    ///
    /// Usage:
    /// ```rust
    /// let mut physics = PhysicsModel::new();
    /// let mut dqn = Dqn::new();
    ///
    /// for episode in 0..1000
    /// {
    ///     for epoch in 0..1000 {
    ///         let output = dqn.predict(physics.state());
    ///         physics.simulate(output.action);
    ///         dqn.remember(physics.reward());
    ///     }
    ///
    ///     dqn.adjust();
    /// }
    /// ```
    ///
    pub fn predict(&mut self, inputs: &[f32]) -> Prediction<OUTPUTS> {
        assert_eq!(inputs.len(), INPUTS);
        assert!(inputs.len() <= self.model.x.len());

        for (x, input) in zip(&mut self.model.x, inputs) {
            *x = *input;
        }

        let y_pred = self.model.forward();
        self.index_max = Self::arg_max(&y_pred);

        self.index = Self::epsilon_greedy(&y_pred);

        let res: [f32; OUTPUTS] = y_pred[..OUTPUTS].try_into().unwrap();

        Prediction {
            action: self.index,
            q_values: res,
        }
    }

    /// 4) Computes gradients,
    /// 5) Stores training state.
    pub fn remember(&mut self, reward: f32) {
        let y_pred = self.model.y;
        let index = self.index;
        let gradients = self.model.backward(index);

        self.transitions.push(Transition {
            y_prd: y_pred[index],
            y_prd_max: y_pred[self.index_max],
            reward,
        });
        self.gradients.push(gradients);
    }

    /// 6) Adjusts the training model
    pub fn adjust(&mut self) -> f32 {
        const GAMMA: f32 = 0.99;

        assert_eq!(self.transitions.len(), self.gradients.len());
        if self.transitions.len() == 0 {
            return 0.0;
        }

        let n = self.gradients.len();
        assert_eq!(n, self.transitions.len());
        let n = n as f32;

        let mut sum = 0.0;
        let mut gradients_loss_sum: GradientsSimd<LAYERS> = GradientsSimd::new();
        let mut transition = self.transitions.iter().peekable();
        let mut gradients = self.gradients.iter();
        while let Some(current) = transition.next() {
            let next = transition.peek();
            let gradients = gradients.next().unwrap();

            let target = match next {
                Some(next) => {
                    let max_q_next = next.y_prd_max;
                    current.reward + GAMMA * max_q_next
                }
                None => current.reward,
            };

            let y_pred = current.y_prd;
            let y = target;
            let diff = y_pred - y;

            // Loss function
            // mean square error
            //      1    N-1
            // L = --- * ∑ (y_pred_i − y_i)²
            //      N    i=0
            sum += diff * diff;

            // Derivative loss function
            // derivative mean square error
            // ∂L           2
            // --------- = --- * (y_pred_i − y_i)
            // ∂L_pred_i    N
            let d_loss_dy = 2.0 / n * diff;
            gradients_loss_sum += gradients * d_loss_dy;
        }

        // loss
        let loss = sum / n;
        self.loss = loss;

        // optimizer
        /// plain gradient descent
        /// w_new = w_old - eta * dw
        const LEARNING_RATE: f32 = 0.1;
        self.model
            .subtract_gradients(&(&gradients_loss_sum * LEARNING_RATE));

        // cleanup history
        self.transitions.clear();
        self.gradients.clear();

        self.loss
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

    fn epsilon_greedy(val: &[f32]) -> usize {
        const EPSILON: f32 = 0.5;

        // let epsilon = epsilon_min + (epsilon_max - epsilon_min) * exp(-step / decay);

        if fastrand::f32() > EPSILON {
            Self::arg_max(val)
        } else {
            fastrand::usize(0..OUTPUTS)
        }
    }
}

struct Transition {
    y_prd: f32,     // y_pred
    y_prd_max: f32, // y_pred max
    reward: f32,    // y
}

pub struct Prediction<const OUTPUTS: usize> {
    pub action: usize,
    pub q_values: [f32; OUTPUTS],
}
