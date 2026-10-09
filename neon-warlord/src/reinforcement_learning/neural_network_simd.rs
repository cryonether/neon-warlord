//! A universal function approximator implemented using simd operations

use std::iter::zip;

use crate::reinforcement_learning::neural_network_simd::{
    activation_function::{ActivationFunction, activation_leaky_relu::ActivationLeakyReLu}, layer_simd::LayerSimd, simd_math::{simd_mat::SMat16, simd_vec::SVec16},
};

pub mod activation_function;
pub mod epoch;
pub mod layer_simd;
pub mod optimizer;
pub mod simd_math;

#[cfg(test)]
mod test_neural_network_simd;

#[cfg(test)]
mod test_logic_functions;

#[cfg(test)]
mod test_logic_functions3;

#[cfg(test)]
mod test_predict_maze;

#[cfg(test)]
mod test_fit_function;

/// A universal function approximator implemented using simd operations
#[derive(Clone)]
pub struct NeuralNetworkSimd<
    const INPUTS: usize,
    const OUTPUTS: usize,
    const NEURONS: usize,
    const NR_LAYERS: usize,
    const RESIDUAL: bool,
    OutputActivation: ActivationFunction<OUTPUTS>,
> 

{
    pub input: LayerSimd<INPUTS, NEURONS, RESIDUAL, ActivationLeakyReLu>,
    pub layers: [LayerSimd<NEURONS, NEURONS, RESIDUAL, ActivationLeakyReLu>; NR_LAYERS],
    pub output: LayerSimd<NEURONS, OUTPUTS, RESIDUAL, OutputActivation>,
}

impl<
    const INPUTS: usize,
    const OUTPUTS: usize,
    const NEURONS: usize,
    const NR_LAYERS: usize,
    const RESIDUAL: bool,
    OutputActivation: ActivationFunction<OUTPUTS>,
> NeuralNetworkSimd<INPUTS, OUTPUTS, NEURONS, NR_LAYERS, RESIDUAL, OutputActivation>
{
    pub fn new() -> Self {
        let input = LayerSimd::new();
        let layers = std::array::from_fn(|_| LayerSimd::new());
        let output = LayerSimd::new();

        Self {
            input,
            layers,
            output,
        }
    }

    pub fn new_rand(mut seed: u64) -> Self {
        if seed == 0 {
            seed = fastrand::u64(..);
        }
        let mut rng = fastrand::Rng::with_seed(seed);

        let input = LayerSimd::new_rand(&mut rng);
        let layers = std::array::from_fn(|_| LayerSimd::new_rand(&mut rng));
        let output = LayerSimd::new_rand(&mut rng);

        Self {
            input,
            layers,
            output,
        }
    }

    pub fn new_zero_one() -> Self {
        let input = LayerSimd::new_zero_one();
        let layers = std::array::from_fn(|_| LayerSimd::new_zero_one());
        let output = LayerSimd::new_zero_one();

        Self {
            input,
            layers,
            output,
        }
    }

    pub fn forward(&mut self, input: &[f32; INPUTS]) -> [f32; OUTPUTS] {
        let x = SVec16::new(*input);

        let mut x = self.input.forward(&x);

        for layer in &mut self.layers {
            x = layer.forward(&x);
        }

        let y = self.output.forward(&x);

        y.0.0
    }

    pub fn backward(&mut self, output: &[f32; OUTPUTS]) -> [f32; INPUTS] {
        let dy = SVec16::new(*output);

        let mut dy = self.output.backward(&dy);

        for layer in &mut self.layers.iter_mut().rev() {
            dy = layer.backward(&dy);
        }

        let dx = self.input.backward(&dy);

        dx.0.0
    }

    pub fn subtract_gradients(&mut self) {
        self.input.subtract_gradients();
        for layer in &mut self.layers.iter_mut().rev() {
            layer.subtract_gradients();
        }
        self.output.subtract_gradients();
    }

    pub fn copy_weights(&mut self,
        other: &Self
    ) {
        self.input.w = other.input.w.clone();
        self.input.b = other.input.b.clone();

        for (layer_self, layer) in zip(&mut self.layers, &other.layers) {
            layer_self.w = layer.w.clone();
            layer_self.b = layer.b.clone();
        }

        self.output.w = other.output.w.clone();
        self.output.b = other.output.b.clone();
    }

    pub fn add_gradients(&mut self, other: &Self) {
        self.input.dl_dw += &other.input.dl_dw;
        self.input.dl_db += &other.input.dl_db;

        for (layer_self, layer) in zip(&mut self.layers, &other.layers) {
            layer_self.dl_dw += &layer.dl_dw;
            layer_self.dl_db += &layer.dl_db;
        }

        self.output.dl_dw += &other.output.dl_dw;
        self.output.dl_db += &other.output.dl_db;
    }

    pub fn zero_grad(&mut self) {
        self.input.dl_dw = SMat16::zero();
        self.input.dl_db = SVec16::zero();

        for layer_self in &mut self.layers {
            layer_self.dl_dw = SMat16::zero();
            layer_self.dl_db = SVec16::zero();
        }

        self.output.dl_dw = SMat16::zero();
        self.output.dl_db = SVec16::zero();
    }

}

impl<
    const INPUTS: usize,
    const OUTPUTS: usize,
    const NEURONS: usize,
    const NR_LAYERS: usize,
    const RESIDUAL: bool,
    OutputActivation: ActivationFunction<OUTPUTS>,
> std::fmt::Display
    for NeuralNetworkSimd<INPUTS, OUTPUTS, NEURONS, NR_LAYERS, RESIDUAL, OutputActivation>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "NeuralNetworkLayered {{")?;

        writeln!(f, "input.x: {}", self.input.x)?;
        writeln!(f,)?;

        writeln!(f, "input.w:     {}", self.input.w)?;
        for (i, layer) in self.layers.iter().enumerate() {
            writeln!(f, "layers[{}].w: {}", i, layer.w)?;
        }
        writeln!(f, "output.w:    {}", self.output.w)?;
        writeln!(f,)?;

        writeln!(f, "input.b:     {}", self.input.b)?;
        for (i, layer) in self.layers.iter().enumerate() {
            writeln!(f, "layers[{}].b: {}", i, layer.b)?;
        }
        writeln!(f, "output.b:    {}", self.output.b)?;
        writeln!(f,)?;

        writeln!(f, "input.z:     {}", self.input.z)?;
        for (i, layer) in self.layers.iter().enumerate() {
            writeln!(f, "layers[{}].z: {}", i, layer.z)?;
        }
        writeln!(f, "output.z:    {}", self.output.z)?;
        writeln!(f,)?;

        writeln!(f, "input.a:     {}", self.input.a)?;
        for (i, layer) in self.layers.iter().enumerate() {
            writeln!(f, "layers[{}].a: {}", i, layer.a)?;
        }
        writeln!(f, "output.a:    {}", self.output.a)?;
        writeln!(f,)?;

        writeln!(f, "input.dl_dw:     {}", self.input.dl_dw)?;
        for (i, layer) in self.layers.iter().enumerate() {
            writeln!(f, "layers[{}].dl_dw: {}", i, layer.dl_dw)?;
        }
        writeln!(f, "output.dl_dw:    {}", self.output.dl_dw)?;
        writeln!(f,)?;

        writeln!(f, "input.dl_db:     {}", self.input.dl_db)?;
        for (i, layer) in self.layers.iter().enumerate() {
            writeln!(f, "layers[{}].dl_db: {}", i, layer.dl_db)?;
        }
        writeln!(f, "output.dl_db:    {}", self.output.dl_db)?;
        writeln!(f,)?;

        writeln!(f,)?;

        write!(f, "}}")
    }
}
