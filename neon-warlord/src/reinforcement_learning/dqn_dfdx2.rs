//! A dqn implementation based on the output of gemini

use std::collections::VecDeque;

use dfdx::{
    losses::mse_loss,
    nn::modules::modules::modules::{DeviceBuildExt, Module, ReLU, ZeroGrads, builders, modules},
    optim::Optimizer,
    tensor::{AsArray, Cpu, TensorFrom, Trace},
    tensor_ops::Backward,
};

const INPUTS: usize = 4;
const OUTPUTS: usize = 2;

type QNetworkBuilder = (
    (builders::Linear<INPUTS, 128>, ReLU),
    (builders::Linear<128, 128>, ReLU),
    (builders::Linear<128, 128>, ReLU),
    builders::Linear<128, OUTPUTS>,
);

type QNetworkModule = (
    (modules::Linear<INPUTS, 128, f32, Cpu>, ReLU),
    (modules::Linear<128, 128, f32, Cpu>, ReLU),
    (modules::Linear<128, 128, f32, Cpu>, ReLU),
    modules::Linear<128, OUTPUTS, f32, Cpu>,
);

struct Transition {
    state: [f32; 4],
    action: usize,
    reward: f32,
    next_state: [f32; 4],
    done: bool,
}

pub struct DqnDfdx2 {
    dev: Cpu,
    q_net: QNetworkModule,
    target_net: QNetworkModule,
    adam: dfdx::optim::Adam<QNetworkModule, f32, Cpu>,

    epsilon: f32,
    epsilon_decay: f32,
    epsilon_min: f32,
    gamma: f32,
    replay_buffer_capacity: usize,

    replay_buffer: VecDeque<Transition>,

    total_reward: f32,
}

impl DqnDfdx2 {
    pub fn new() -> Self {
        let dev = Cpu::default();

        let q_net: QNetworkModule = dev.build_module::<QNetworkBuilder, f32>();
        let target_net: QNetworkModule = q_net.clone();
        let adam: dfdx::optim::Adam<QNetworkModule, f32, Cpu> = dfdx::optim::Adam::new(
            &q_net,
            dfdx::optim::AdamConfig {
                lr: 1e-3,
                betas: [0.9, 0.999],
                eps: 1e-8,
                weight_decay: None,
            },
        );
        let epsilon: f32 = 1.0f32;
        let epsilon_decay: f32 = 0.998;
        let epsilon_min: f32 = 0.02;
        let gamma: f32 = 0.99f32;
        let replay_buffer_capacity: usize = 20000;
        let replay_buffer: VecDeque<Transition> = VecDeque::with_capacity(replay_buffer_capacity);

        let total_reward: f32 = 0.0;

        Self {
            dev,
            q_net,
            target_net,
            adam,
            epsilon,
            epsilon_decay,
            epsilon_min,
            gamma,
            replay_buffer_capacity,
            replay_buffer,
            total_reward,
        }
    }

    pub fn choose_action(&mut self, state: &[f32; INPUTS]) -> (usize, [f32; OUTPUTS]) {
        // Epsilon-Greedy mit fastrand
        let mut q_values_res = [0.0; 2];
        let action = if fastrand::f32() < self.epsilon {
            fastrand::usize(0..2)
        } else {
            let state_tensor = self.dev.tensor(state);
            let q_values = self.q_net.forward(state_tensor);
            let q_arr: [f32; 2] = q_values.array();
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

        self.replay_buffer.push_back(Transition {
            state,
            action,
            reward,
            next_state,
            done,
        });
        if self.replay_buffer.len() > self.replay_buffer_capacity {
            self.replay_buffer.pop_front();
        }

        const BATCH_SIZE: usize = 64;
        if self.replay_buffer.len() >= BATCH_SIZE {
            let mut states_batch = [[0.0f32; 4]; BATCH_SIZE];
            let mut actions_batch = [0usize; BATCH_SIZE];
            let mut rewards_batch = [0.0f32; BATCH_SIZE];
            let mut next_states_batch = [[0.0f32; 4]; BATCH_SIZE];
            let mut dones_batch = [0.0f32; BATCH_SIZE];

            for i in 0..BATCH_SIZE {
                let idx = fastrand::usize(0..self.replay_buffer.len());
                let t = &self.replay_buffer[idx];

                states_batch[i] = t.state;
                actions_batch[i] = t.action;
                rewards_batch[i] = t.reward;
                next_states_batch[i] = t.next_state;
                dones_batch[i] = if t.done { 1.0f32 } else { 0.0f32 };
            }

            let states_t = self.dev.tensor(states_batch);
            let next_states_t = self.dev.tensor(next_states_batch);

            // --- DOUBLE DQN IMPLEMENTIERUNG ---
            // 1. Das ONLINE-Netzwerk bestimmt die beste Aktion für den Folgezustand
            let next_online_q = self.q_net.forward(next_states_t.clone());
            let next_online_arr = next_online_q.array();

            // 2. Das TARGET-Netzwerk berechnet den stabilen Q-Wert für ebendiese Aktion
            let next_target_q = self.target_net.forward(next_states_t);
            let next_target_arr = next_target_q.array();

            let mut target_qs = [0.0f32; 64];
            for i in 0..BATCH_SIZE {
                // Wähle Aktion mit dem Online-Netz aus
                let best_action_next = if next_online_arr[i][0] > next_online_arr[i][1] {
                    0
                } else {
                    1
                };
                // Bewerte sie über das Target-Netz (Verhindert Überschätzungs-Bias)
                let max_next_q = next_target_arr[i][best_action_next];

                target_qs[i] = rewards_batch[i] + self.gamma * max_next_q * (1.0 - dones_batch[i]);
            }

            let gradients = self.q_net.alloc_grads();
            let pred_q_values = self.q_net.forward(states_t.trace(gradients));
            let pred_q_arr = pred_q_values.array();

            let mut targets_array = pred_q_arr;
            for i in 0..BATCH_SIZE {
                targets_array[i][actions_batch[i]] = target_qs[i];
            }
            let targets_t = self.dev.tensor(targets_array);

            let loss = mse_loss(pred_q_values, targets_t);
            let _loss_res = loss.as_vec()[0];
            let grads = loss.backward();
            self.adam
                .update(&mut self.q_net, &grads)
                .expect("Fehler beim Optimizer-Update");

            return self.total_reward;
        }

        0.0
    }

    pub fn update_target_net(&mut self) {
        self.target_net = self.q_net.clone();
    }

    pub fn epsilon_decay(&mut self) {
        if self.epsilon > self.epsilon_min {
            self.epsilon *= self.epsilon_decay;
        }

        println!(
            "Episode: {:4}, Accum. Reward: {:7.1}, Epsilon: {:.3}",
            0.0, self.total_reward, self.epsilon
        );

        self.total_reward = 0.0;
    }
}
