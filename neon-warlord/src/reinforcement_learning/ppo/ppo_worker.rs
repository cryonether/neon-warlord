//! Worker thread for parallel learning

use std::{sync::{Arc, mpsc::{Receiver, SyncSender, sync_channel}}, thread};

use crate::reinforcement_learning::{neural_network_simd::{NeuralNetworkSimd, activation_function::{ActivationFunction, activation_none::ActivationNone}}, ppo::{Ppo, Transition}};

pub struct PpoWorker <
    const INPUTS: usize,
    const OUTPUTS: usize,
    const NEURONS: usize,
    const LAYERS: usize,
    const RESIDUAL: bool,
    OutputActivationActor: ActivationFunction<OUTPUTS>,
> where
    OutputActivationActor: std::clone::Clone + Send,
{
    pub request_tx: SyncSender<Box<PpoWorkerData<INPUTS, OUTPUTS, NEURONS, LAYERS, RESIDUAL, OutputActivationActor>>>,
    pub result_rx: Receiver<Box<PpoWorkerData<INPUTS, OUTPUTS, NEURONS, LAYERS, RESIDUAL, OutputActivationActor>>>,

    pub thread: Option<thread::JoinHandle<()>>,
}

impl<const INPUTS: usize, const OUTPUTS: usize, const NEURONS: usize, const LAYERS: usize, const RESIDUAL: bool, OutputActivationActor: ActivationFunction<OUTPUTS>> PpoWorker<INPUTS, OUTPUTS, NEURONS, LAYERS, RESIDUAL, OutputActivationActor>
where
    OutputActivationActor: std::clone::Clone + Send + 'static,
{

    pub fn new(
        std_dev: f32,
        clip: f32,
    ) -> Self {
        let (request_tx, request_rx) = sync_channel::<Box<PpoWorkerData<INPUTS, OUTPUTS, NEURONS, LAYERS, RESIDUAL, OutputActivationActor>>>(1);
        let (result_tx, result_rx) = sync_channel::<Box<PpoWorkerData<INPUTS, OUTPUTS, NEURONS, LAYERS, RESIDUAL, OutputActivationActor>>>(1);

        let builder = thread::Builder::new()
            .name("PpoWorker".into());

        let thread;
        #[cfg(target_arch = "wasm32")]
        {
            thread = None;
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            thread = Some(builder.spawn(move || {
            while let Ok(mut data) = request_rx.recv() {
                
                    (data.actor_loss_sum, data.critic_loss_sum) = Ppo::calculate_gradients(
                        &mut data.actor,
                        &mut data.critic,
                        &data.transitions,
                        &data.advantages,
                        &data.value_targets,
                        std_dev,
                        clip,
                    );

                    result_tx.send(data).unwrap();
                }
            }).unwrap());
        }


        Self {
            request_tx: request_tx,
            result_rx: result_rx,
            thread: thread,
        }

    }

}

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
}
