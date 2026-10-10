//! Worker thread for parallel learning

use crate::{
    gym_simulation::worker_thread_2::WorkerThread2Run,
    reinforcement_learning::{
        neural_network_simd::{
            NeuralNetworkSimd,
            activation_function::{ActivationFunction, activation_none::ActivationNone},
        },
        ppo::Transition,
    },
};

/// Data to send from and to the thread
#[derive(Clone)]
pub struct PpoWorkerData<
    const INPUTS: usize,
    const OUTPUTS: usize,
    const NEURONS: usize,
    const LAYERS: usize,
    const RESIDUAL: bool,
    OutputActivationActor: ActivationFunction<OUTPUTS>,
> where
    OutputActivationActor: std::clone::Clone + Send,
{
    pub actor: NeuralNetworkSimd<INPUTS, OUTPUTS, NEURONS, LAYERS, RESIDUAL, OutputActivationActor>,
    pub critic: NeuralNetworkSimd<INPUTS, 1, NEURONS, LAYERS, RESIDUAL, ActivationNone>,
    pub transitions: Vec<Transition<INPUTS, OUTPUTS>>,
    pub advantages: Vec<f32>,
    pub value_targets: Vec<f32>,
    pub actor_loss_sum: f32,
    pub critic_loss_sum: f32,
    pub std_dev: f32,
    pub clip: f32,
}

impl<
    const INPUTS: usize,
    const OUTPUTS: usize,
    const NEURONS: usize,
    const LAYERS: usize,
    const RESIDUAL: bool,
    OutputActivationActor: ActivationFunction<OUTPUTS>,
> WorkerThread2Run
    for PpoWorkerData<INPUTS, OUTPUTS, NEURONS, LAYERS, RESIDUAL, OutputActivationActor>
where
    OutputActivationActor: std::clone::Clone + Send + 'static,
{
    fn run(&mut self) {
        use crate::reinforcement_learning::ppo::Ppo;

        (self.actor_loss_sum, self.critic_loss_sum) = Ppo::calculate_gradients(
            &mut self.actor,
            &mut self.critic,
            &self.transitions,
            &self.advantages,
            &self.value_targets,
            self.std_dev,
            self.clip,
        );
    }
}
