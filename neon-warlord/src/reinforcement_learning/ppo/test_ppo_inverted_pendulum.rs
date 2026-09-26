pub struct CartPoleEnv {
    pub x: f32,
    pub x_dot: f32,
    pub theta: f32,
    pub theta_dot: f32,
    pub steps: usize,
}

impl CartPoleEnv {
    pub fn new() -> Self {
        Self { x: 0.0, x_dot: 0.0, theta: 0.0, theta_dot: 0.0, steps: 0 }
    }

    // Step physics forward using simple Euler integration
    pub fn step(&mut self, action: usize) -> ( [f32; 4], f32, bool ) {
        self.steps += 1;
        
        // Force applied: 0 = Push Left (-10N), 1 = Push Right (+10N)
        let force = if action == 0 { -10.0 } else { 10.0 };
        
        // Simplified physical constants
        let gravity = 9.8;
        let mass_cart = 1.0;
        let mass_pole = 0.1;
        let total_mass = mass_cart + mass_pole;
        let pole_length = 0.5; // half-length
        let pole_mass_length = mass_pole * pole_length;
        let dt = 0.02;

        let temp = (force + pole_mass_length * self.theta_dot.powi(2) * self.theta.sin()) / total_mass;
        let theta_acc = (gravity * self.theta.sin() - self.theta.cos() * temp) 
            / (pole_length * (4.0/3.0 - mass_pole * self.theta.cos().powi(2) / total_mass));
        let x_acc = temp - pole_mass_length * theta_acc * self.theta.cos() / total_mass;

        // Update state variables
        self.x += self.x_dot * dt;
        self.x_dot += x_acc * dt;
        self.theta += self.theta_dot * dt;
        self.theta_dot += theta_acc * dt;

        let next_state = [self.x, self.x_dot, self.theta, self.theta_dot];

        // Terminal conditions (15 degrees or 2.4 units out of bounds)
        let done = self.x.abs() > 2.4 || self.theta.abs() > (15.0 * std::f32::consts::PI / 180.0) || self.steps >= 200;
        let reward = 1.0; // Standard +1 reward for keeping it upright

        (next_state, reward, done)
    }
}

use dfdx::prelude::*;

// 4 Inputs (State Space) -> 64 Hidden -> 2 Outputs (Action Logits)
type ActorNet = (
    Linear<4, 64>, ReLU,
    Linear<64, 2>,
);

// 4 Inputs (State Space) -> 64 Hidden -> 1 Output (State Value V(s))
type CriticNet = (
    Linear<4, 64>, ReLU,
    Linear<64, 1>,
);

fn sample_action(logits: [f32; 2]) -> (usize, f32) {
    // Numerically stable Softmax calculation
    let max = logits[0].max(logits[1]);
    let exp0 = (logits[0] - max).exp();
    let exp1 = (logits[1] - max).exp();
    let sum = exp0 + exp1;
    
    let prob0 = exp0 / sum;
    let prob1 = exp1 / sum;

    // Roll a float between 0.0 and 1.0 using fastrand
    let roll = fastrand::f32();
    let action = if roll < prob0 { 0 } else { 1 };
    let log_prob = if action == 0 { prob0.ln() } else { prob1.ln() };

    (action, log_prob)
}

#[test]
fn main() {
    // Instantiate execution device
    let dev = Cpu::default();

    // Build modules and allocate networks
    let mut actor = dev.build_module::<ActorNet, f32>();
    let mut critic = dev.build_module::<CriticNet, f32>();

    let mut actor_opt = dfdx::optim::Adam::new(&actor, AdamConfig { lr: 3e-4, ..Default::default() });
    let mut critic_opt = dfdx::optim::Adam::new(&critic, AdamConfig { lr: 1e-3, ..Default::default() });

    for episode in 0..500 {
        let mut env = CartPoleEnv::new();
        let mut state = [0.0, 0.0, 0.0, 0.0];
        
        let mut states = Vec::new();
        let mut actions = Vec::new();
        let mut log_probs = Vec::new();
        let mut rewards = Vec::new();

        // 1. Rollout Phase: Collect trajectory
        loop {
            // Forward pass to get logits (without gradient tracking during execution)
            let state_tensor: Tensor<Rank1<4>, f32, _> = dev.tensor(state);
            let logits = actor.forward(state_tensor).array();
            
            let (action, log_prob) = sample_action(logits);
            let (next_state, reward, done) = env.step(action);

            states.push(state);
            actions.push(action);
            log_probs.push(log_prob);
            rewards.push(reward);

            state = next_state;
            if done { break; }
        }

        // 2. Optimization Phase
        let batch_size = states.len();
        
        // Compute state values via Critic without tracking tape history
        let mut values = Vec::with_capacity(batch_size);
        for s in &states {
            let s_tensor: Tensor<Rank1<4>, f32, _> = dev.tensor(*s);
            values.push(critic.forward(s_tensor).array()[0]); 
        }

        // Compute rewards-to-go (Returns) and Advantages
        let mut returns = vec![0.0; batch_size];
        let mut advantages = vec![0.0; batch_size];
        let gamma = 0.99;
        let mut running_return = 0.0;

        for t in (0..batch_size).rev() {
            running_return = rewards[t] + gamma * running_return;
            returns[t] = running_return;
            advantages[t] = returns[t] - values[t];
        }

        // Advantage Normalization
        let adv_sum: f32 = advantages.iter().sum();
        let adv_mean = adv_sum / batch_size as f32;
        let adv_var: f32 = advantages.iter().map(|&x| (x - adv_mean).powi(2)).sum::<f32>() / batch_size as f32;
        let adv_std = (adv_var + 1e-8).sqrt();
        for adv in advantages.iter_mut() {
            *adv = (*adv - adv_mean) / adv_std;
        }

        // Hyperparameters for PPO Optimization
        let ppo_epochs = 4;
        let eps_clip = 0.2;
        let entropy_coef = 0.01;

        for _epoch in 0..ppo_epochs {
            // Allocate a single gradient container for the entire epoch batch
            let mut critic_grads = critic.alloc_grads();
            let mut actor_grads = actor.alloc_grads();

            for t in 0..batch_size {
                // --- CRITIC GRADIENT ACCUMULATION ---
                let s_tensor = dev.tensor(states[t]).traced(critic_grads);
                let val_pred = critic.forward(s_tensor);
                let target_val_tensor: Tensor<Rank1<1>, f32, _> = dev.tensor([returns[t]]);
                
                let critic_loss = (val_pred - target_val_tensor).square().mean() / (batch_size as f32);
                critic_grads = critic_loss.backward();

                // --- ACTOR GRADIENT ACCUMULATION ---
                let s_tensor_actor = dev.tensor(states[t]).traced(actor_grads);
                let logits = actor.forward(s_tensor_actor);
                
                let a = actions[t];
                let old_log_prob = log_probs[t];
                let adv = advantages[t];

                let log_probs_tensor = logits.log_softmax();
                let (log_probs_no_tape, tape) = log_probs_tensor.split_tape();

                let probs_tensor = log_probs_no_tape.clone().exp();
                let entropy = -(probs_tensor * log_probs_no_tape.clone()).sum();

                let new_log_prob = log_probs_no_tape.select(dev.tensor(a));
                let ratio = (new_log_prob - old_log_prob).exp();
                
                let surr1 = ratio.clone() * adv;
                let ratio_clipped = ratio.clamp(1.0 - eps_clip, 1.0 + eps_clip);
                let surr2 = ratio_clipped * adv;

                let actor_loss_no_tape = (-minimum(surr1, surr2) - (entropy * entropy_coef)) / (batch_size as f32);
                let actor_loss = actor_loss_no_tape.put_tape(tape);

                actor_grads = actor_loss.backward();
            }

            // Only update the network weights ONCE per epoch after accumulating the entire batch
            critic_opt.update(&mut critic, &critic_grads).expect("Critic opt failed");
            actor_opt.update(&mut actor, &actor_grads).expect("Actor opt failed");
        }

        println!("Episode {} finished in {} steps.", episode, batch_size);
    }
}
