//! A sequence of predictions to calculate a loss
//! 
//! "something vaguely resembling backpropagation." 
//! Explicitly implementing the mathematical structure of automatic differentiation.
//! 

use crate::reinforcement_learning::neural_network::{Mat2, NeuralNetwork, RowVec2, RowVec4, Vec2};

pub struct NeuralNetworkEpoch {
    model: NeuralNetwork,
}

impl NeuralNetworkEpoch {
    pub fn new( history: Vec<NeuralNetwork>) -> Self {

        let model = NeuralNetwork::new();

        Self { model }
    }

    pub fn learn(&mut self, input: [[f32; 2]; 4], output: [[f32; 1]; 4]) -> [f32; 4]
    {
        let mut history = Vec::new();
        
        // prediction
        for elem in input {
            
            self.model.x = elem.into();
            self.model.forward();
            
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
            
            let y_pred = model.y;
            let y = output[0];
            let diff = y_pred - y;

            sum += diff * diff;
        }

        let loss = sum / size;

        // gradients

        // accumulated loss gradients
        let mut d_loss_dw0 = RowVec4::zeros();
        let mut d_loss_dw1 = RowVec4::zeros();
        let mut d_loss_dw2 = RowVec4::zeros();
        let mut d_loss_dw3 = RowVec2::zeros();

        let mut d_loss_db0 = RowVec2::zeros();
        let mut d_loss_db1 = RowVec2::zeros();
        let mut d_loss_db2 = RowVec2::zeros();
        let mut d_loss_db3 = 0.0;

        // derivative mean square error
        // ∂L           2    
        // --------- = --- * (y_pred_i − y_i)
        // ∂L_pred_i    N    
        for (model, output) in std::iter::zip(&mut history, output) {
            
            let y_pred = model.y;
            let y = output[0];
            let diff = y_pred - y;

            let d_loss_dy = 2.0 / size * diff;

            // Creates Jacobian of the network output with respect to its parameters.
            model.backward();

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
        self.model.w_3 -= d_loss_dw3 * LEARNING_RATE;

        self.model.b_0 -= Self::to_2x1(d_loss_db0) * LEARNING_RATE;
        self.model.b_1 -= Self::to_2x1(d_loss_db1) * LEARNING_RATE;
        self.model.b_2 -= Self::to_2x1(d_loss_db2) * LEARNING_RATE;
        self.model.b_3 -= d_loss_db3 * LEARNING_RATE;
        

        [0.0, 0.0, 0.0, 0.0]
    }

    fn to_2x2(val: RowVec4) -> Mat2 {
        Mat2::new(val[0], val[1], val[2], val[3])
    }

    fn to_2x1(val: RowVec2) -> Vec2 {
        Vec2::new(val[0], val[1], )
    }
}
