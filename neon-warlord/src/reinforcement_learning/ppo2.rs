//! Implements a beginner version of policy gradient decent
//! https://github.com/ericyangyu/PPO-for-Beginners/tree/master
//! PPO was published in 2017

mod test_ppo2;

use std::{collections::VecDeque, iter::zip};

use cgmath::num_traits::clamp;
use itertools::izip;

use crate::reinforcement_learning::neural_network_simd::{NeuralNetwork64, loss_function::{GaussianLogProbability, MeanSquareError, PpoActorRatio, PpoSurrogateLossClipped}, neural_network_layered::NeuralNetworkLayered};


// const INPUTS: usize = 4;
// const OUTPUTS: usize = 2;
// const NEURONS: usize = 64;
// const LAYERS: usize = 1;

pub struct Ppo2<
    const INPUTS: usize,
    const OUTPUTS: usize,
    const NEURONS: usize,
    const LAYERS: usize,
    const RESIDUAL: bool,
>
{
    actor: NeuralNetworkLayered<INPUTS, OUTPUTS, NEURONS, LAYERS, false>,
    critic: NeuralNetworkLayered<INPUTS, 1, NEURONS, LAYERS, false>,
    
    transitions: Vec<Transition<INPUTS, OUTPUTS>>,
    
    variance: f32,
    std_dev: f32,
    gamma: f32,
    clip: f32,
    nr_updates_per_iteration: usize,
}

impl<
    const INPUTS: usize,
    const OUTPUTS: usize,
    const NEURONS: usize,
    const LAYERS: usize,
    const RESIDUAL: bool,
> Ppo2<INPUTS, OUTPUTS, NEURONS, LAYERS, RESIDUAL> {
    pub fn new(seed: u64) -> Self {
        // For choosing an action
        const VARIANCE: f32 = 0.5;
        const STD_DEV: f32 = 0.70710677; // sqrt(0.5)

        // Discount factor, for calculating the discounted reward
        const GAMMA: f32 = 0.95;     

        // Threshold to clip the ratio
        const CLIP: f32 = 0.2;

        // Number of times to update the network from the same batch of data
        const NR_UPDATES_PER_ITERATION: usize = 5;  

        let actor = NeuralNetworkLayered::new_rand(seed);
        let critic = NeuralNetworkLayered::new_rand(seed);

        let transitions = Vec::new();

        Self { 
            actor, 
            critic,
            transitions,
            variance: VARIANCE,
            std_dev: STD_DEV,
            gamma: GAMMA, 
            clip: CLIP,
            nr_updates_per_iteration: NR_UPDATES_PER_ITERATION,
        }
    }


    //
    //   Queries an action from the actor network
    //
    //   Parameters:
    //       observations - the environment observation at the current timestep
    //
    //   Return:
    //       action - the action to take
    //       log_prob - the log probability of the selected action in the distribution
    //
    pub fn get_action(&mut self, observation: &[f32; INPUTS])
-> ([f32; OUTPUTS], [f32; OUTPUTS], f32)     {
        // Query the actor network for a mean action.
        let mean_action = self.actor.forward(&observation);

        let action = mean_action.map(|mu| {
            mu + self.std_dev * box_mueller_standard_normal()
        });

        // Calculate the log probability over the sampled action
        let mut glp = GaussianLogProbability::new();
        let log_probability = glp.calc(&action, &mean_action, self.std_dev);

        (action, mean_action, log_probability)
    }

    pub fn save_reward(&mut self, 
        observation: [f32; INPUTS], 
        action: [f32; OUTPUTS], 
        log_probability: f32,
        reward: f32,
        done: bool,
    ) {
        self.transitions.push(
            Transition {
                observation,
                action,
                log_probability,
                reward,
                done,
            }
        )
    }

    pub fn learn(&mut self) -> (f32, f32) {

        let mut last_discounted_reward = 0.0;

        let mut discounted_rewards = VecDeque::new();
        let mut advantages = VecDeque::new();

        for transition in self.transitions.iter().rev() {
            let observation = transition.observation;
            let action = transition.action;
            let log_probability = transition.log_probability;
            let reward = transition.reward;

            // Calculate the discounted reward
            // let discounted_reward = reward + self.gamma * last_discounted_reward;
            let discounted_reward = reward + self.gamma * last_discounted_reward;

            // Query critic network for the quality value
            let baseline_estimate = self.critic.forward(&observation)[0];
        
            // Calculate the advantage
            // A = Q - V
            let advantage = discounted_reward - baseline_estimate;

            discounted_rewards.push_front(discounted_reward);
            advantages.push_front(advantage);

            last_discounted_reward = discounted_reward;
        }

        // Normalizing advantages
        // isn't theoretically necessary, but in practice it decreases the variance of 
        // our advantages and makes convergence much more stable and faster.
        let advantages_mean = advantages.iter().sum::<f32>() / advantages.len() as f32;
        let advantages_variance = advantages
                .iter()
                .map(|x| {
                    let diff = x - advantages_mean;
                    diff * diff
                })
                .sum::<f32>()
                / advantages.len() as f32;
        let advantages_std_dev = advantages_variance.sqrt();

        for advantage in &mut advantages {
            *advantage = (*advantage - advantages_mean) / (advantages_std_dev + 1e-10);
        } 

        let n = self.transitions.len();
        assert_eq!(advantages.len(), n);
        assert_eq!(discounted_rewards.len(), n);
        let n = n as f32;

        // Update the neural networks for n epochs
        let mut critic_loss = 0.0;
        let mut actor_loss = 0.0;
        for _i in 0..self.nr_updates_per_iteration {
            let mut critic_loss_sum = 0.0;
            let mut actor_loss_sum = 0.0;

            for (transition, advantage, discounted_reward) in izip!(&self.transitions, &advantages, &discounted_rewards) {
                let observation = transition.observation;
                let action = transition.action;
                let log_probability = transition.log_probability;
                let reward = transition.reward;

                // Calculate V_phi and pi_theta(a_t | s_t)
                // Estimate the values of each observation, and the log probs of
                // each action in the most recent batch with the most recent
                // iteration of the actor network.
                let curr_estimate = self.critic.forward(&observation);
                let cur_mean_action = self.actor.forward(&observation);

                // Critic mean square error
                let mut mse = MeanSquareError::new();
                let critic_square_error = mse.calc(curr_estimate, *discounted_reward);
                let critic_square_error_derivative = mse.derivative();

                critic_loss_sum += critic_square_error;
                
                // Calculate the log probability over the sampled action
                let mut glp = GaussianLogProbability::new();
                let curr_log_probability = glp.calc(&action, &cur_mean_action, self.std_dev);
                let curr_log_probability_derivative = glp.derivative();

                // Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
                let mut ppo_actor_ratio = PpoActorRatio::new();
                let ratio = ppo_actor_ratio.calc(curr_log_probability, log_probability);
                let ratio_derivative = ppo_actor_ratio.derivative();

                // Calculate surrogate loss
                let mut ppo_surrogate_loss_clipped = PpoSurrogateLossClipped::new();
                let surrogate_loss_clipped = ppo_surrogate_loss_clipped.calc(ratio, *advantage, self.clip);
                let surrogate_loss_clipped_derivative = ppo_surrogate_loss_clipped.derivative();

                actor_loss_sum += surrogate_loss_clipped;

                // dL / dy
                //
                // ∂L_t                      a_t − μ_t
                // ----- = - − A_t * r_t * --------------
                // ∂μt                          σ2
                //
                let mut loss_derivative = [0.0; OUTPUTS];
                for (curr_log_probability_derivative, loss_derivative) in zip(curr_log_probability_derivative, &mut loss_derivative) {
                    *loss_derivative = surrogate_loss_clipped_derivative * ratio_derivative * curr_log_probability_derivative;
                }

                // Calculate gradients
                //
                //  ∂Lt        ∂L_t
                // ----- = Jᵀ -----
                //  ∂θ         ∂μ_t
                //
                let _critic_dx = self.critic.backward(&critic_square_error_derivative);
                let _actor_dx = self.actor.backward(&loss_derivative);

            }

            critic_loss = critic_loss_sum / n;
            actor_loss = actor_loss_sum / n;

            const LEARNING_RATE: f32 = 0.001;
            self.critic.subtract_gradients(LEARNING_RATE / n);
            self.actor.subtract_gradients(LEARNING_RATE);

        }
        
        self.transitions.clear();

        (actor_loss, critic_loss)
    }

}


struct Transition<
    const INPUTS: usize,
    const OUTPUTS: usize,
> {
    observation: [f32; INPUTS],
    action: [f32; OUTPUTS],
    log_probability: f32,
    reward: f32,
    done: bool,
}



///
/// Generates a single random sample from a Standard Normal Distribution (mean = 0, std_dev = 1)
/// using the Box-Muller transform.
///
/// ### Mathematical Formula:
/// Given two independent uniformly distributed random variables u1, u2 in [0, 1),
/// the transformation produces a standard normal variable Z:
///
/// Z = √[ -2 * ln(u1) ] * cos( 2 * π * u2 )
///
/// Multi-line representation:
///              ________________
///        Z = \/ -2 * ln(u_1)   *  cos( 2 * π * u_2 )
///
/// Where:
/// - u1 = `u1` (clamped to prevent ln(0) -> NaN/Infinity)
/// - u2 = `u2`
/// - Z  = Returned standard normal sample
///
fn box_mueller_standard_normal() -> f32 {
    const PI: f32 = std::f32::consts::PI;

    // Box-Muller transform
    let u1 = fastrand::f32().max(f32::MIN_POSITIVE);
    let u2 = fastrand::f32();

    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}
