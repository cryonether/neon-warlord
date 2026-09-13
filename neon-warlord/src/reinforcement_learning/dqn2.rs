//! A dqn implementation based on the output of gemini

use std::{collections::VecDeque, iter::zip};



use crate::reinforcement_learning::neural_network_simd::{Gradient128, NeuralNetwork128};

const INPUTS: usize = 4;
const OUTPUTS: usize = 2;
const LAYERS: usize = 2;


struct Transition {
    state: [f32; INPUTS],
    action: usize,
    reward: f32,
    next_state: [f32; INPUTS],
    done: bool,
}

pub struct Dqn2 {
    q_net: Box<NeuralNetwork128<INPUTS, OUTPUTS, LAYERS, false>>,
    pub target_net: Box<NeuralNetwork128<INPUTS, OUTPUTS, LAYERS, false>>,

    epsilon: f32,
    epsilon_decay: f32,
    epsilon_min: f32,
    gamma: f32,
    replay_buffer_capacity: usize,

    replay_buffer: VecDeque<Transition>,

    total_reward: f32,
    loss: f32,
}

impl Dqn2 {
    pub fn new(

    ) -> Self {
        let q_net = Box::new(NeuralNetwork128::new());
        let target_net = q_net.clone();

        let epsilon: f32 = 1.0f32;
        let epsilon_decay: f32 = 0.998; 
        let epsilon_min: f32 = 0.02;
        let gamma: f32 = 0.99f32;
        let replay_buffer_capacity: usize = 20000;
        let replay_buffer: VecDeque<Transition> = VecDeque::with_capacity(replay_buffer_capacity);

        let total_reward: f32 = 0.0;
        let loss: f32 = 0.0;

        Self {
            q_net,
            target_net,
            epsilon,
            epsilon_decay,
            epsilon_min,
            gamma,
            replay_buffer_capacity,
            replay_buffer,
            total_reward,
            loss,
        }
    }

    pub fn choose_action(&mut self, state: &[f32; INPUTS]) -> (usize, [f32; OUTPUTS]) {

            // Epsilon-Greedy mit fastrand
            let mut q_values_res = [0.0; 2];
            let action = if fastrand::f32() < self.epsilon {
                fastrand::usize(0..2)
            } else {
                let q_values = self.q_net.forward(state);
                let q_arr: [f32; 2] = q_values;
                q_values_res = q_arr;
                if q_arr[0] > q_arr[1] { 0 } else { 1 }
            };

        (action, q_values_res)
    }

    pub fn set_reward_learn(
        &mut self,
        state: [f32; INPUTS],
        action: usize,
        reward: f32,
        next_state: [f32; INPUTS],
        done: bool,
    ) -> f32 {
        self.total_reward += reward;

        self.replay_buffer.push_back(Transition { state, action, reward, next_state, done });
        if self.replay_buffer.len() > self.replay_buffer_capacity {
            self.replay_buffer.pop_front();
        }

        let mut sum: f32 = 0.0;
        let mut gradients_loss_sum = Gradient128::new();
        const BATCH_SIZE: usize = 64;
        if self.replay_buffer.len() >= BATCH_SIZE {

            for _i in 0..BATCH_SIZE {
                let idx = fastrand::usize(0..self.replay_buffer.len());
                let transition = &self.replay_buffer[idx];
                
                let state = transition.state;
                let action = transition.action;
                let reward = transition.reward;
                let next_state = transition.next_state;
                let done = if transition.done { 1.0f32 } else { 0.0f32 };

                // Double dqn implementation
                // Online network gives best action
                // Target network gives q-value
                let next_online_q = self.q_net.forward(&next_state);
                let next_target_q = self.target_net.forward(&next_state);

                let best_action_next = if next_online_q[0] > next_online_q[1] { 0 } else { 1 };
                let max_next_q = next_target_q[best_action_next];

                // reward function
                let target_qs = reward + self.gamma * max_next_q * (1.0 - done);

                // get current q-value
                let pred_q_values = self.q_net.forward(&state);
                let gradients = self.q_net.backward(action);

                let y_pred = pred_q_values[action];
                let y = target_qs;
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
                let d_loss_dy = 2.0 / BATCH_SIZE as f32 * diff;
                gradients_loss_sum.add_loss_gradients(&gradients, d_loss_dy);
            }

            // loss
            self.loss = sum / BATCH_SIZE as f32;

            // optimizer
            /// plain gradient descent
            const LEARNING_RATE: f32 = 0.01;
            self.q_net.subtract_gradients(&(&gradients_loss_sum * LEARNING_RATE));

            return self.total_reward;
        }

        return 0.0;
    }

    pub fn update_target_net(&mut self) {
        self.target_net = self.q_net.clone();
    }
    
    pub fn epsilon_decay(&mut self) {
        if self.epsilon > self.epsilon_min {
            self.epsilon *= self.epsilon_decay;
        }

        println!("Episode: {:4}, Accum. Reward: {:7.1}, Epsilon: {:.3}", 0.0, self.total_reward, self.epsilon);

        self.total_reward = 0.0;
    }
}
