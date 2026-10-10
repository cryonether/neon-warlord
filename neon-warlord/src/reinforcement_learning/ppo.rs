//! Implements the Proximal Policy Optimization algorithm
//! https://github.com/ericyangyu/PPO-for-Beginners/tree/master
//! PPO was published in 2017

pub mod loss_function;
mod ppo_worker;
#[cfg(test)]
mod test_ppo;

use std::{collections::VecDeque, iter::zip};

use itertools::izip;

use crate::{
    gym_simulation::worker_thread_2::WorkerThread2,
    reinforcement_learning::{
        neural_network_simd::{
            NeuralNetworkSimd,
            activation_function::{ActivationFunction, activation_none::ActivationNone},
        },
        ppo::{
            loss_function::{
                GaussianLogProbability, MeanSquareError, PpoActorRatio, PpoSurrogateLossClipped,
            },
            ppo_worker::{PpoWorkerData},
        },
    },
};

// const NR_THREADS: usize = 8;

/// Implements the Proximal Policy Optimization algorithm
pub struct Ppo<
    const INPUTS: usize,
    const OUTPUTS: usize,
    const NEURONS: usize,
    const LAYERS: usize,
    const RESIDUAL: bool,
    OutputActivationActor: ActivationFunction<OUTPUTS>,
> where
    OutputActivationActor: std::clone::Clone + Send + 'static,
{
    pub actor: NeuralNetworkSimd<INPUTS, OUTPUTS, NEURONS, LAYERS, RESIDUAL, OutputActivationActor>,
    pub critic: NeuralNetworkSimd<INPUTS, 1, NEURONS, LAYERS, RESIDUAL, ActivationNone>,

    pub transitions: Vec<Transition<INPUTS, OUTPUTS>>,

    _variance: f32,
    std_dev: f32,
    gamma: f32,
    gae_lambda: f32,
    _clip: f32,
    nr_updates_per_iteration: usize,

    // parallel
    ppo_worker_data: Vec<
        Option<
            Box<PpoWorkerData<INPUTS, OUTPUTS, NEURONS, LAYERS, RESIDUAL, OutputActivationActor>>,
        >,
    >,
    ppo_worker_thread: Vec<
        WorkerThread2<
            PpoWorkerData<INPUTS, OUTPUTS, NEURONS, LAYERS, RESIDUAL, OutputActivationActor>,
        >,
    >,
}

impl<
    const INPUTS: usize,
    const OUTPUTS: usize,
    const NEURONS: usize,
    const LAYERS: usize,
    const RESIDUAL: bool,
    OutputActivationActor: ActivationFunction<OUTPUTS>,
> Ppo<INPUTS, OUTPUTS, NEURONS, LAYERS, RESIDUAL, OutputActivationActor>
where
    OutputActivationActor: std::clone::Clone + Send + 'static,
{
    pub fn new(seed: u64, nr_threads: usize) -> Self {
        // let nr_threads = 8;

        // For choosing an action
        const VARIANCE: f32 = 0.5;
        const STD_DEV: f32 = 0.70710677; // sqrt(0.5)

        // Discount factor, for calculating the discounted reward
        const GAMMA: f32 = 0.95;
        const GAE_LAMBDA: f32 = 0.90;

        // Threshold to clip the ratio
        const CLIP: f32 = 0.2;

        // Number of times to update the network from the same batch of data
        const NR_UPDATES_PER_ITERATION: usize = 5;

        let actor = NeuralNetworkSimd::new_rand(seed);
        let critic = NeuralNetworkSimd::new_rand(seed);

        let transitions = Vec::new();

        let mut ppo_worker_data = Vec::with_capacity(nr_threads);
        let mut ppo_worker_thread = Vec::with_capacity(nr_threads);
        for i in 0..nr_threads {
            ppo_worker_data.push(Some(Box::new(PpoWorkerData {
                actor: actor.clone(),
                critic: critic.clone(),
                transitions: Vec::new(),
                advantages: Vec::new(),
                value_targets: Vec::new(),
                actor_loss_sum: 0.0,
                critic_loss_sum: 0.0,
                std_dev: STD_DEV,
                clip: CLIP,
            })));

            ppo_worker_thread.push(WorkerThread2::new(format!("Ppo Worker {}", i)));
        }        

        Self {
            actor,
            critic,
            transitions,
            _variance: VARIANCE,
            std_dev: STD_DEV,
            gamma: GAMMA,
            gae_lambda: GAE_LAMBDA,
            _clip: CLIP,
            nr_updates_per_iteration: NR_UPDATES_PER_ITERATION,
            ppo_worker_thread,
            ppo_worker_data,
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
    pub fn get_action(
        &mut self,
        observation: &[f32; INPUTS],
    ) -> ([f32; OUTPUTS], [f32; OUTPUTS], f32) {
        // Query the actor network for a mean action.
        let mean_action = self.actor.forward(observation);

        let action = mean_action.map(|mu| mu + self.std_dev * box_mueller_standard_normal());

        // Calculate the log probability over the sampled action
        let mut glp = GaussianLogProbability::new();
        let log_probability = glp.calc(&action, &mean_action, self.std_dev);

        (action, mean_action, log_probability)
    }

    pub fn save_reward(
        &mut self,
        observation: [f32; INPUTS],
        action: [f32; OUTPUTS],
        log_probability: f32,
        reward: f32,
        done: bool,
    ) {
        let value = self.critic.forward(&observation);
        let value = value[0];

        self.transitions.push(Transition {
            observation,
            action,
            log_probability,
            reward,
            done,
            value,
        })
    }

    // pub fn create_transition(
    //     &mut self,
    //     observation: [f32; INPUTS],
    //     action: [f32; OUTPUTS],
    //     log_probability: f32,
    //     reward: f32,
    //     done: bool,
    // ) -> Transition<INPUTS, OUTPUTS> {
    //     let value = self.critic.forward(&observation);
    //     let value = value[0];

    //     Transition {
    //         observation,
    //         action,
    //         log_probability,
    //         reward,
    //         done,
    //         value,
    //     }
    // }

    fn calculate_gae(&self) -> (VecDeque<f32>, VecDeque<f32>) {
        let mut advantages = VecDeque::new();
        let mut returns = VecDeque::new();

        let mut next_value = 0.0;
        let mut last_gae = 0.0;

        for transition in self.transitions.iter().rev() {
            let reward = transition.reward;
            let value = transition.value;

            // Terminal states have no bootstrap value.
            // let bootstrap_value = if transition.done { 0.0 } else { next_value };
            let bootstrap_value = if transition.done { value } else { next_value };

            let delta = reward + self.gamma * bootstrap_value - value;

            // Generalized Advantage Estimate:
            //
            // A_t = δ_t + γ λ A_{t+1}
            //
            // Do not propagate GAE across an episode boundary.
            last_gae = if transition.done {
                delta
            } else {
                delta + self.gamma * self.gae_lambda * last_gae
            };

            advantages.push_front(last_gae);

            // PPO's value target is usually:
            //
            // return_t = A_t + V(s_t)
            //
            // rather than the recursively calculated discounted reward.
            let value_target = value + last_gae;
            returns.push_front(value_target);

            next_value = value;
        }

        (advantages, returns)
    }

    pub fn learn_parallel(&mut self) -> (f32, f32) {
        let transitions = &self.transitions;
        let (mut advantages, mut value_targets) = self.calculate_gae();
        let advantages = advantages.make_contiguous();
        let value_targets = value_targets.make_contiguous();

        assert_eq!(transitions.len(), advantages.len());
        assert_eq!(transitions.len(), value_targets.len());

        let nr_threads = self.ppo_worker_thread.len();
        assert_eq!(self.ppo_worker_data.len(), nr_threads);
        // assert!(transitions.len() >= nr_threads);
        let chunk_size = transitions.len().div_ceil(nr_threads);

        let (mut actor_loss, mut critic_loss) = (0.0, 0.0);
        for _i in 0..self.nr_updates_per_iteration {
            // calculate gradients
            for (worker, worker_data, transitions, advantages, value_targets) in izip!(
                &mut self.ppo_worker_thread,
                &mut self.ppo_worker_data,
                transitions.chunks(chunk_size),
                advantages.chunks(chunk_size),
                value_targets.chunks(chunk_size),
            ) {
                let mut worker_data = worker_data.take().unwrap();

                worker_data.actor.zero_grad();
                worker_data.actor.copy_weights(&self.actor);

                worker_data.critic.zero_grad();
                worker_data.critic.copy_weights(&self.critic);

                worker_data.transitions.clear();
                worker_data.transitions.extend_from_slice(transitions);

                worker_data.advantages.clear();
                worker_data.advantages.extend_from_slice(advantages);

                worker_data.value_targets.clear();
                worker_data.value_targets.extend_from_slice(value_targets);

                worker_data.actor_loss_sum = 0.0;
                worker_data.critic_loss_sum = 0.0;

                // send worker request
                worker.send(worker_data);
            }

            let mut actor_loss_sum = 0.0;
            let mut critic_loss_sum = 0.0;
            for (worker, worker_data, _transitions, _advantages, _value_targets) in izip!(
                &mut self.ppo_worker_thread,
                &mut self.ppo_worker_data,
                transitions.chunks(chunk_size),
                advantages.chunks(chunk_size),
                value_targets.chunks(chunk_size),
            )
            {
                // get worker result
                let data = worker.receive();

                // sum loss
                actor_loss_sum += data.actor_loss_sum;
                critic_loss_sum += data.critic_loss_sum;

                // sum gradients
                self.actor.add_gradients(&data.actor);
                self.critic.add_gradients(&data.critic);

                *worker_data = Some(data); // store chunk for later reuse
            }

            let inv_n = 1.0 / transitions.len() as f32;

            // calculate loss
            actor_loss = actor_loss_sum * inv_n;
            critic_loss = critic_loss_sum * inv_n;

            // subtract gradients
            self.actor.multiply_gradients_const(inv_n);
            self.actor.subtract_gradients();

            self.critic.multiply_gradients_const(inv_n);
            self.critic.subtract_gradients();

            // reset gradients
            self.actor.zero_grad();
            self.critic.zero_grad();
        }

        self.transitions.clear();

        (actor_loss, critic_loss)
    }

    pub fn learn_sequential(&mut self) -> (f32, f32) {
        let transitions = &self.transitions;
        let (mut advantages, mut value_targets) = self.calculate_gae();
        let advantages = advantages.make_contiguous();
        let value_targets = value_targets.make_contiguous();

        let (mut actor_loss, mut critic_loss) = (0.0, 0.0);
        for _i in 0..self.nr_updates_per_iteration {
            // calculate gradients
            (actor_loss, critic_loss) = Self::calculate_gradients(
                &mut self.actor,
                &mut self.critic,
                transitions,
                advantages,
                value_targets,
                self.std_dev,
                self._clip,
            );

            let inv_n = 1.0 / transitions.len() as f32;

            // calculate loss
            actor_loss *= inv_n;
            critic_loss *= inv_n;

            // subtract gradients
            self.actor.multiply_gradients_const(inv_n);
            self.actor.subtract_gradients();

            self.critic.multiply_gradients_const(inv_n);
            self.critic.subtract_gradients();

            // reset gradients
            self.actor.zero_grad();
            self.critic.zero_grad();
        }

        self.transitions.clear();

        (actor_loss, critic_loss)
    }

    pub fn learn(&mut self) -> (f32, f32) {
        #[allow(unused)]
        let mut single_threaded = self.ppo_worker_thread.len() == 0;
        #[cfg(target_arch = "wasm32")]
        {
            single_threaded = true;
        }

        if single_threaded {
            self.learn_sequential()
        } else {
            // self.learn_sequential()
            self.learn_parallel()
        }
    }

    pub fn calculate_gradients(
        actor: &mut NeuralNetworkSimd<
            INPUTS,
            OUTPUTS,
            NEURONS,
            LAYERS,
            RESIDUAL,
            OutputActivationActor,
        >,
        critic: &mut NeuralNetworkSimd<INPUTS, 1, NEURONS, LAYERS, RESIDUAL, ActivationNone>,
        transitions: &[Transition<INPUTS, OUTPUTS>],
        advantages: &[f32],
        value_targets: &[f32],
        std_dev: f32,
        clip: f32,
    ) -> (f32, f32) {
        let n = transitions.len();
        assert_eq!(advantages.len(), n);
        assert_eq!(value_targets.len(), n);
        let _n = n as f32;

        let mut critic_loss_sum = 0.0;
        let mut actor_loss_sum = 0.0;

        for (transition, advantage, returns) in izip!(transitions, advantages, value_targets) {
            let observation = transition.observation;
            let action = transition.action;
            let log_probability = transition.log_probability;
            let _reward = transition.reward;

            // Calculate V_phi and pi_theta(a_t | s_t)
            // Estimate the values of each observation, and the log probs of
            // each action in the most recent batch with the most recent
            // iteration of the actor network.
            let curr_estimate = critic.forward(&observation);
            let cur_mean_action = actor.forward(&observation);

            // Critic mean square error
            let mut mse = MeanSquareError::new();
            let critic_square_error = mse.calc(curr_estimate, *returns);
            let critic_square_error_derivative = mse.derivative();

            critic_loss_sum += critic_square_error;

            // Calculate the log probability over the sampled action
            let mut glp = GaussianLogProbability::new();
            let curr_log_probability = glp.calc(&action, &cur_mean_action, std_dev);
            let curr_log_probability_derivative = glp.derivative();

            // Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
            let mut ppo_actor_ratio = PpoActorRatio::new();
            let ratio = ppo_actor_ratio.calc(curr_log_probability, log_probability);
            let ratio_derivative = ppo_actor_ratio.derivative();

            // Calculate surrogate loss
            let mut ppo_surrogate_loss_clipped = PpoSurrogateLossClipped::new();
            let surrogate_loss_clipped = ppo_surrogate_loss_clipped.calc(ratio, *advantage, clip);
            let surrogate_loss_clipped_derivative = ppo_surrogate_loss_clipped.derivative();

            actor_loss_sum += surrogate_loss_clipped;

            // dL / dy
            //
            // ∂L_t                      a_t − μ_t
            // ----- = - − A_t * r_t * --------------
            // ∂μt                          σ2
            //
            let mut loss_derivative = [0.0; OUTPUTS];
            for (curr_log_probability_derivative, loss_derivative) in
                zip(curr_log_probability_derivative, &mut loss_derivative)
            {
                *loss_derivative = surrogate_loss_clipped_derivative
                    * ratio_derivative
                    * curr_log_probability_derivative;
            }

            // Calculate gradients
            //
            //  ∂Lt        ∂L_t
            // ----- = Jᵀ -----
            //  ∂θ         ∂μ_t
            //
            let _critic_dx = critic.backward(&critic_square_error_derivative);
            let _actor_dx = actor.backward(&loss_derivative);
        }

        (actor_loss_sum, critic_loss_sum)
    }
}

#[derive(Clone, Debug)]
pub struct Transition<const INPUTS: usize, const OUTPUTS: usize> {
    observation: [f32; INPUTS],
    action: [f32; OUTPUTS],
    log_probability: f32,
    reward: f32,
    done: bool,
    value: f32,
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

// Normalizing advantages
// isn't theoretically necessary, but in practice it decreases the variance of
// our advantages and makes convergence much more stable and faster.
// let advantages_mean = advantages.iter().sum::<f32>() / advantages.len() as f32;
// let advantages_variance = advantages
//     .iter()
//     .map(|x| {
//         let diff = x - advantages_mean;
//         diff * diff
//     })
//     .sum::<f32>()
//     / advantages.len() as f32;
// let advantages_std_dev = advantages_variance.sqrt();

// for advantage in &mut advantages {
//     *advantage = (*advantage - advantages_mean) / (advantages_std_dev + 1e-10);
// }
