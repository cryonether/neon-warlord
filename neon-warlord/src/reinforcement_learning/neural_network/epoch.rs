//! A sequence of predictions to calculate a loss
//!
//! "something vaguely resembling backpropagation."
//! Explicitly implementing the mathematical structure of automatic differentiation.
//!

use crate::reinforcement_learning::neural_network::{Mat2, NeuralNetwork, RowVec2, RowVec4, Vec2};

pub struct NeuralNetworkEpoch {
    pub model: NeuralNetwork,

    pub loss: f32,
}

impl NeuralNetworkEpoch {
    pub fn new() -> Self {
        let mut model = NeuralNetwork::new();

        let mut rng = fastrand::Rng::with_seed(fastrand::u64(..));
        // let mut rand = || (rng.f32() * 2.0 - 1.0);

        // Kaiming/He-style initialization
        let fan_in: f32 = 2.0; // fan_in is the number of inputs to the neuron/filter.
        let bound = 1.0 / (fan_in).sqrt();
        let mut rand = || (rng.f32() * 2.0 - 1.0) * bound;

        model.w_0 = Mat2::new(rand(), rand(), rand(), rand());
        model.w_1 = Mat2::new(rand(), rand(), rand(), rand());
        model.w_2 = Mat2::new(rand(), rand(), rand(), rand());
        model.w_3 = Mat2::new(rand(), rand(), rand(), rand());

        model.b_0 = Vec2::new(rand(), rand());
        model.b_1 = Vec2::new(rand(), rand());
        model.b_2 = Vec2::new(rand(), rand());
        model.b_3 = Vec2::new(rand(), rand());

        // model.b_0 = Vec2::zeros();
        // model.b_1 = Vec2::zeros();
        // model.b_2 = Vec2::zeros();
        // model.b_3 =  0.0;

        Self { model, loss: 0.0 }
    }

    pub fn learn(&mut self, input: [[f32; 2]; 4], output: [[f32; 1]; 4]) -> [f32; 4] {
        let mut history = Vec::new();

        // prediction
        let mut res = Vec::new();
        for elem in input {
            self.model.x = elem.into();
            self.model.forward();
            res.push(self.model.y[0]);

            history.push(self.model.clone());
        }

        // mean square error
        //      1    N-1
        // L = --- * ∑ (y_pred_i − y_i)²
        //      N    i=0
        assert_eq!(history.len(), output.len());
        let mut sum = 0.0;
        let size = history.len() as f32;
        for (model, output) in std::iter::zip(&history, output) {
            let y_pred = model.y[0];
            let y = output[0];
            let diff = y_pred - y;

            sum += diff * diff;
        }

        let loss = sum / size;
        self.loss = loss;

        // gradients

        // accumulated loss gradients
        let mut d_loss_dw0 = RowVec4::zeros();
        let mut d_loss_dw1 = RowVec4::zeros();
        let mut d_loss_dw2 = RowVec4::zeros();
        let mut d_loss_dw3 = RowVec4::zeros();

        let mut d_loss_db0 = RowVec2::zeros();
        let mut d_loss_db1 = RowVec2::zeros();
        let mut d_loss_db2 = RowVec2::zeros();
        let mut d_loss_db3 = RowVec2::zeros();

        // derivative mean square error
        // ∂L           2
        // --------- = --- * (y_pred_i − y_i)
        // ∂L_pred_i    N
        for (model, output) in std::iter::zip(&mut history, output) {
            let y_pred = model.y[0];
            let y = output[0];
            let diff = y_pred - y;

            let d_loss_dy = 2.0 / size * diff;

            // Creates Jacobian of the network output with respect to its parameters.
            model.backward(0);

            d_loss_dw0 += d_loss_dy * model.dy_dw0;
            d_loss_dw1 += d_loss_dy * model.dy_dw1;
            d_loss_dw2 += d_loss_dy * model.dy_dw2;
            d_loss_dw3 += d_loss_dy * model.dy_dw3;

            d_loss_db0 += d_loss_dy * model.dy_db0;
            d_loss_db1 += d_loss_dy * model.dy_db1;
            d_loss_db2 += d_loss_dy * model.dy_db2;
            d_loss_db3 += d_loss_dy * model.dy_db3;
        }

        // optimizer
        /// plain gradient descent
        /// w_new = w_old - eta * dw
        const LEARNING_RATE: f32 = 0.1;

        self.model.w_0 -= Self::to_2x2(d_loss_dw0) * LEARNING_RATE;
        self.model.w_1 -= Self::to_2x2(d_loss_dw1) * LEARNING_RATE;
        self.model.w_2 -= Self::to_2x2(d_loss_dw2) * LEARNING_RATE;
        self.model.w_3 -= Self::to_2x2(d_loss_dw3) * LEARNING_RATE;

        self.model.b_0 -= Self::to_2x1(d_loss_db0) * LEARNING_RATE;
        self.model.b_1 -= Self::to_2x1(d_loss_db1) * LEARNING_RATE;
        self.model.b_2 -= Self::to_2x1(d_loss_db2) * LEARNING_RATE;
        self.model.b_3 -= Self::to_2x1(d_loss_db3) * LEARNING_RATE;

        [res[0], res[1], res[2], res[3]]
    }

    fn to_2x2(val: RowVec4) -> Mat2 {
        Mat2::new(val[0], val[1], val[2], val[3])
    }

    fn to_2x1(val: RowVec2) -> Vec2 {
        Vec2::new(val[0], val[1])
    }
}
