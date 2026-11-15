//! Exports and imports a neural network as SafeTensor

use std::collections::HashMap;

use safetensors::tensor::{Dtype, TensorView};
use safetensors::{SafeTensorError};

use crate::reinforcement_learning::neural_network_simd::{NeuralNetworkSimd, activation_function::ActivationFunction};



impl<
    const INPUTS: usize,
    const OUTPUTS: usize,
    const NEURONS: usize,
    const NR_LAYERS: usize,
    const RESIDUAL: bool,
    OutputActivation: ActivationFunction<OUTPUTS>,
> NeuralNetworkSimd<
    INPUTS,
    OUTPUTS,
    NEURONS,
    NR_LAYERS,
    RESIDUAL,
    OutputActivation,
> {
    pub fn export(
        &self,
        neural_network: &NeuralNetworkSimd<INPUTS, OUTPUTS, NEURONS, NR_LAYERS, RESIDUAL, OutputActivation>,
        path: &std::path::Path,
    ) -> Result<(), Box<dyn std::error::Error>> {

        let input = &neural_network.input;
        let w0 = &input.w;
        let data = w0;
        let b0 = &input.b;

        for layer in &neural_network.layers {
            let w = &layer.w;
            let b = &layer.b;
        }

        let output = &neural_network.output;
        let wn = &output.w;
        let bn = &output.b;


        Ok(())
    }
}


fn push_tensor(
    tensors: &mut Vec<(String, Vec<usize>, Vec<u8>)>,
    name: String,
    data: &[f32],
    shape: Vec<usize>,
) {
    let bytes = data
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();

    tensors.push((name, shape, bytes));
}