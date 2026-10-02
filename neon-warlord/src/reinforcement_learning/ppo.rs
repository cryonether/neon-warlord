//! Proximal Policy Optimization (PPO) algorithm
//!
//! Details:
//!
//! PPO (Proximal Policy Optimization) is a reinforcement learning algorithm that
//! improves a policy by rewarding good actions while limiting how much the policy
//! can change in each update, making learning more stable and reliable.
//!

mod test_ppo_inverted_pendulum;

use std::{char::MAX, collections::VecDeque, iter::zip};

use crate::reinforcement_learning::neural_network_simd::{NeuralNetwork64, neural_network_layered::NeuralNetworkLayered};

const INPUTS: usize = 4;
const OUTPUTS: usize = 1;
const NEURONS: usize = 64;
const LAYERS: usize = 1;

pub struct Ppo {
    actor: NeuralNetworkLayered<INPUTS, OUTPUTS, NEURONS, LAYERS, false>,
    critic: NeuralNetworkLayered<INPUTS, 1, NEURONS, LAYERS, false>,
}

impl Ppo {
    pub fn learn(&mut self) {
        let total_timesteps = 100;
        
        


        let t_so_far = 0; // Timesteps simulated so far
        while t_so_far < total_timesteps {              // ALG STEP 2
           // Increment t_so_far somewhere below

            let (batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens) = self.rollout();


            // Calculate V_{phi, k}
            let (v, v_log_probs) = self.evaluate(&batch_obs, &batch_acts);

            // Calculate advantage
            let mut advantage = Vec::new();
            for (batch_rtgs, v) in zip(batch_rtgs, v) {
                advantage.push(batch_rtgs - v);
            }

            let mean = advantage.iter().sum::<f32>() / advantage.len() as f32;

            let variance = advantage
                    .iter()
                    .map(|x| {
                        let diff = x - mean;
                        diff * diff
                    })
                    .sum::<f32>()
                    / advantage.len() as f32;

            let std_dev = variance.sqrt();

            // Normalize advantages
            for elem in &mut advantage {
                *elem = (*elem - mean) / std_dev;
            } 


        }
    }

    pub fn rollout(&mut self) -> (Vec<[f32; 4]>, Vec<[f32; 1]>, Vec<f32>, Vec<f32>, Vec<usize>) {
        const TIMESTEPS_PER_BATCH: usize = 4800;            // timesteps per batch
        const MAX_TIMESTEPS_PER_EPISODE: usize = 1600;      // timesteps per episode
        const NR_EPISODES: usize = 100;


        // Batch data
        let mut batch_observations: Vec<[f32; 4]> = Vec::new();             // batch observations
        let mut batch_actions: Vec<[f32; 1]> =  Vec::new();             // batch actions
        let mut batch_log_probabilities =  Vec::new();        // log probs of each action
        let mut batch_rewards: Vec<Vec<f32>> =  Vec::new();            // batch rewards
        let mut batch_reward_to_goes =  Vec::new();             // batch rewards-to-go
        let mut batch_lengths: Vec<usize> =  Vec::new();             // episodic lengths in batch



        let mut t = 0;
        while t < TIMESTEPS_PER_BATCH {

            let mut episode_rewards: Vec<f32> = Vec::new();
            let mut observation: [f32; INPUTS] = Self::get_observations();
            let mut done;

            let mut epidode = 0;
            for episode_ in 0..MAX_TIMESTEPS_PER_EPISODE {
                epidode = episode_;

                // Increment timesteps ran this batch so far
                t += 1;

                // Collect observation
                batch_observations.push(observation);

                let (action, log_probability) = self.get_action(observation);
                let (observation_, reward, done_) = Self::step(action);
                observation = observation_;
                done = done_;


                // Collect reward, action, and log prob
                episode_rewards.push(reward);
                batch_actions.push(action);
                batch_log_probabilities.push(log_probability);

                if done {
                    break;
                }


            }


            // Collect episodic length and rewards
            batch_lengths.push(epidode + 1); // plus 1 because timestep starts at 0
            batch_rewards.push(episode_rewards);

        }

        // Calculate rewards to go
        batch_reward_to_goes = self.compute_rewards_to_go(&batch_rewards);

        // Return the batch data
        (batch_observations, batch_actions, batch_log_probabilities, batch_reward_to_goes, batch_lengths)

    }

    fn get_observations() -> [f32; INPUTS] {
        [0.0; INPUTS]
    }

    fn get_action(&mut self, observations: [f32; INPUTS]) -> ([f32; OUTPUTS], f32) {

        const VARIANCE: f32 = 0.5;
        const STD_DEV: f32 = 0.70710677; // sqrt(0.5)

        // Query the actor network for a mean action.
        let mean = self.actor.forward(&observations);

        let mut action = [0.0; OUTPUTS];

        // Create our Multivariate Normal Distribution
        for (&mu, action) in zip(&mean, &mut action) {
            let noise = box_mueller_standard_normal();
            *action = mu + STD_DEV * noise;
        }

        // Calculate log probability of the sampled action.
        let log_prob = gaussian_log_prob(
            &action,
            &mean,
            STD_DEV,
        );

        (action, log_prob)
    }

    fn compute_rewards_to_go(&self, rewards: &Vec<Vec<f32>>) -> Vec<f32> {
        const GAMMA: f32 = 0.95;

        // The rewards-to-go (rtg) per episode per batch to return.
        // The shape will be (num timesteps per episode)
        let mut batch_reward_to_goes = VecDeque::new();

        // Iterate through each episode backwards to maintain same order
        // in batch_rtgs

        for episode_reverse in rewards.iter().rev() {
            let mut discounted_reward = 0.0; // The discounted reward so far

            for reward in episode_reverse.iter().rev() {
                discounted_reward = reward + discounted_reward * GAMMA;
                batch_reward_to_goes.push_front(discounted_reward)
            }
        }

        batch_reward_to_goes.into()
    }

    fn evaluate(&mut self, batch_observations: &[[f32; INPUTS]], batch_actions: &Vec<[f32; OUTPUTS]>) -> (Vec<f32>, Vec<f32>)  {
        
        let mut v = Vec::new();
        let mut log_probs = Vec::new();

        for (batch_obs, batch_acts) in zip(batch_observations, batch_actions) {
            // Query critic network for a value V for each obs in batch_obs.
            let val = self.critic.forward(batch_obs);
            v.push(val[0]);
        

            // Calculate the log probabilities of batch actions using most 
            // recent actor network.
            // This segment of code is similar to that in get_action()
            const VARIANCE: f32 = 0.5;
            const STD_DEV: f32 = 0.70710677; // sqrt(0.5)

            // Query the actor network for a mean action.
            let mean = self.actor.forward(&batch_obs);

            let mut action = [0.0; OUTPUTS];

            // Create our Multivariate Normal Distribution
            for (&mu, action) in zip(&mean, &mut action) {
                let noise = box_mueller_standard_normal();
                *action = mu + STD_DEV * noise;
            }

            // Calculate log probability of the sampled action.
            let log_prob = gaussian_log_prob(
                batch_acts,
                &mean,
                STD_DEV,
            );

            log_probs.push(log_prob);

        }


        (v, log_probs)
    }

    fn step(action: [f32; OUTPUTS]) -> ([f32; INPUTS], f32, bool) {
        ([0.0; INPUTS], 0.0, false)
    }




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

    (-2.0 * u1.ln()).sqrt()
        * (2.0 * PI * u2).cos()
}

///
/// Calculates the log probability density of a multi-dimensional continuous action 
/// under an isotropic (spherical) Gaussian distribution.
///
/// ### Mathematical Formula:
///
/// ```text
///              1   /  Σ (x_i - μ_i)²                  \
/// log p(x) = - — * |  ——————————————  + D * ln(2πσ²)  |
///              2   \        σ²                        /
/// ```
/// 
/// Where:
/// - x_i = Elements of the `action` vector
/// - μ_i = Elements of the `mean` vector
/// - σ (sigma) = `std_dev` (scalar shared across all dimensions)
/// - σ² (variance) = `std_dev * std_dev`
/// - D = `num_dimensions` (length of the action vector)
///
fn gaussian_log_prob(
    action: &[f32],
    mean: &[f32],
    std_dev: f32,
) -> f32 {
    const PI: f32 = std::f32::consts::PI;

    let variance = std_dev * std_dev;
    
    // 1. Calculate the squared differences sum
    let sum_squared_diffs: f32 = action
        .iter()
        .zip(mean.iter())
        .map(|(&a, &mu)| {
            let diff = a - mu;
            diff * diff
        })
        .sum();

    // 2. Compute the total log probability
    let num_dimensions = action.len() as f32;
    let log_normalization = num_dimensions * (2.0 * PI * variance).ln();

    -0.5 * (sum_squared_diffs / variance + log_normalization)
}

