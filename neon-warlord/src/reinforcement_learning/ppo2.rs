//! Implements a beginner version of policy gradient decent
//! https://github.com/ericyangyu/PPO-for-Beginners/tree/master

use crate::reinforcement_learning::neural_network_simd::NeuralNetwork64;


const INPUTS: usize = 4;
const OUTPUTS: usize = 2;
const LAYERS: usize = 1;

pub struct Ppo2 {
    actor: NeuralNetwork64<INPUTS, OUTPUTS, LAYERS, false>,
    critic: NeuralNetwork64<INPUTS, 1, LAYERS, false>,
    
    transitions: Vec<Transition>,
    
    variance: f32,
    std_dev: f32,
    gamma: f32,
}

impl Ppo2 {
    pub fn new() -> Self {
        const VARIANCE: f32 = 0.5;
        const STD_DEV: f32 = 0.70710677; // sqrt(0.5)
        const GAMMA: f32 = 0.99;

        let actor = NeuralNetwork64::new();
        let critic = NeuralNetwork64::new();

        let transitions = Vec::new();

        Self { 
            actor, 
            critic,
            transitions,
            variance: VARIANCE,
            std_dev: STD_DEV,
            gamma: GAMMA, 
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
    pub fn get_action(&mut self, observation: &[f32; INPUTS]) -> ([f32; OUTPUTS], f32)
    {
        let std_dev = self.std_dev;

        // Query the actor network for a mean action.
        let mean = self.actor.forward(&observation);

        // Create multivariate normal distributed action
        let action = mean.map(|mu| {
            mu + std_dev * box_mueller_standard_normal()
        });

        // Calculate the log probability over the sampled action
        let log_probability = gaussian_log_prob(
            &action,
            &mean,
            std_dev,
        );

        (action, log_probability)
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

    pub fn learn(&mut self) {

        let gamma = self.gamma;

        let mut discounted_reward = 0.0;

        let mut advantages = Vec::new();

        for transition in &self.transitions {
            let observation = transition.observation;
            let action = transition.action;
            let log_probability = transition.log_probability;
            let reward = transition.reward;

            discounted_reward = reward + self.gamma * discounted_reward;

            // Query critic network for the quality value
            let v = self.critic.forward(&observation)[0];
        
            // Calculate the advantage
            let advantage = discounted_reward - v;

            advantages.push(advantage)
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
            *advantage = (*advantage - advantages_mean) / advantages_std_dev + 1e-10;
        } 

    }
}


struct Transition {
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

///
/// Calculates the log probability density of a multi-dimensional continuous action
/// under an isotropic (spherical) Gaussian distribution.
///
/// ### Mathematical Formula:
///
///              1   /  Σ (x_i - μ_i)²                  \
/// log p(x) = - — * |  ——————————————  + D * ln(2πσ²)  |
///              2   \        σ²                        /
///
/// Where:
/// - x_i = Elements of the `action` vector
/// - μ_i = Elements of the `mean` vector
/// - σ (sigma) = `std_dev` (scalar shared across all dimensions)
/// - σ² (variance) = `std_dev * std_dev`
/// - D = `num_dimensions` (length of the action vector)
///
fn gaussian_log_prob(action: &[f32], mean: &[f32], std_dev: f32) -> f32 {
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


