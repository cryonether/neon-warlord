//! Deep Q Network (DQN)

#[cfg(test)]
mod test_maze;

use crate::reinforcement_learning::neural_network_simd::{
    NeuralNetworkSimd, gradients::GradientsSimd,
};

const LAYERS: usize = 3;

pub struct Dqn<const INPUTS: usize, const OUTPUTS: usize> {
    model: NeuralNetworkSimd<INPUTS, OUTPUTS, LAYERS>,
    steps: Vec<Transition<INPUTS>>,

    epsilon: f32,
}

impl<const INPUTS: usize, const OUTPUTS: usize> Dqn<INPUTS, OUTPUTS> {
    pub fn new() -> Self {
        let model = NeuralNetworkSimd::new_zero_one();
        let steps = Vec::new();
        let epsilon = 0.6;

        Self {
            model,
            steps,
            epsilon,
        }
    }


    pub fn choose_action_u8(&mut self, inputs: &[u8; INPUTS]) -> (usize, [f32; OUTPUTS])
    {
        let inputs_f32 = inputs.map(|x| x as f32);
        self.choose_action(&inputs_f32)
    }

    pub fn choose_action(&mut self, inputs: &[f32; INPUTS]) -> (usize, [f32; OUTPUTS])
    {
        let q_values: [f32; OUTPUTS] = self.model.forward(inputs);

        let mut action = Self::pick_action(q_values);

        if fastrand::f32() < self.epsilon {
            action = fastrand::usize(0..OUTPUTS);
        }

        (action, q_values)
    }

    fn pick_action(q_values: [f32; OUTPUTS]) -> usize {

        let mut q_value_max = f32::NEG_INFINITY;
        let mut max_index = 0;
        for (i, &q_value) in q_values.iter().enumerate() {
            if q_value > q_value_max {
                q_value_max = q_value;
                max_index = i;
            }
        }

        max_index
    }

    pub fn set_reward_u8(
        &mut self, 
        inputs: [u8; INPUTS], 
        action: usize, 
        reward: f32,
        next_inputs: [u8; INPUTS], 
    ) 
    {
        let inputs_f32 = inputs.map(|x| x as f32);
        let next_inputs_f32 = next_inputs.map(|x| x as f32);
        self.set_reward(inputs_f32, action, reward, next_inputs_f32);
    }

    pub fn set_reward(
        &mut self, 
        inputs: [f32; INPUTS], 
        action: usize, 
        reward: f32,
        next_inputs: [f32; INPUTS], 
    ) 
    {
        self.steps.push(Transition{
            inputs,
            action,
            reward,
            inputs_next: next_inputs,
        });
    }

    pub fn learn(&mut self) -> f32 {
        const GAMMA: f32 = 0.99;
        
        if self.steps.len() == 0 {
            return 0.0;
        }

        let n = self.steps.len() as f32;

        let mut sum = 0.0;
        let mut gradients_loss_sum: GradientsSimd<LAYERS> = GradientsSimd::new();
        for step in self.steps.iter().rev() {
            let inputs = step.inputs;
            let action = step.action;
            let reward = step.reward;
            let inputs_next = step.inputs_next;

            let q_values = self.model.forward(&inputs);
            let gradients = self.model.backward(action);
            let q_values_next =  self.model.forward(&inputs_next);

            let mut q_value_max_next = f32::NEG_INFINITY;
            for q_value in q_values_next {
                if q_value > q_value_max_next {
                    q_value_max_next = q_value
                }
            }

            let reward_2 = reward + GAMMA * (q_value_max_next - reward);


            let y_pred = q_values[action];
            let y = reward_2;
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
            gradients_loss_sum += &gradients * d_loss_dy;
        }

        // loss
        let loss = sum / n;

        // optimizer
        /// plain gradient descent
        /// w_new = w_old - eta * dw
        const LEARNING_RATE: f32 = 0.1;
        self.model
            .subtract_gradients(&(&gradients_loss_sum * LEARNING_RATE));


        self.steps.clear();
        self.epsilon = f32::max(self.epsilon * 0.95, 0.1); 

        loss
    }
}

pub struct Transition<const INPUTS: usize> {
    pub inputs: [f32; INPUTS],
    pub action: usize,
    pub reward: f32,
    pub inputs_next: [f32; INPUTS],
}
