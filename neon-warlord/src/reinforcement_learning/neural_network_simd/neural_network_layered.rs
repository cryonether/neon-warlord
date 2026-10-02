//! Implementation of a neural network composed of single neuron layers

use crate::reinforcement_learning::neural_network_simd::{layer_simd::LayerSimd, simd_math::simd_vec::SVec16};

#[derive(Clone)]
pub struct NeuralNetworkLayered<
    const INPUTS: usize,
    const OUTPUTS: usize,
    const NEURONS: usize,
    const NR_LAYERS: usize,
    const RESIDUAL: bool,
> {
    pub input: LayerSimd<INPUTS, NEURONS, true, false>,
    pub layers: [LayerSimd<NEURONS, NEURONS, true, RESIDUAL>; NR_LAYERS],
    pub output: LayerSimd<NEURONS, OUTPUTS, false, false>,
}

impl<
    const INPUTS: usize,
    const OUTPUTS: usize,
    const NEURONS: usize,
    const NR_LAYERS: usize,
    const RESIDUAL: bool,
> NeuralNetworkLayered<INPUTS, OUTPUTS, NEURONS, NR_LAYERS, RESIDUAL>
{
    pub fn new() -> Self {
        let input: LayerSimd<INPUTS, NEURONS, true, false> = LayerSimd::new();
        let layers: [LayerSimd<NEURONS, NEURONS, true, RESIDUAL>; NR_LAYERS] =
            std::array::from_fn(|_| LayerSimd::new()
        );
        let output: LayerSimd<NEURONS, OUTPUTS, false, false> = LayerSimd::new();

        Self {
            input,
            layers,
            output,
        }
    }

    pub fn new_rand() -> Self {
        let input: LayerSimd<INPUTS, NEURONS, true, false> = LayerSimd::new_rand();
        let layers: [LayerSimd<NEURONS, NEURONS, true, RESIDUAL>; NR_LAYERS] =
            std::array::from_fn(|_| LayerSimd::new_rand()
        );
        let output: LayerSimd<NEURONS, OUTPUTS, false, false> = LayerSimd::new_rand();

        Self {
            input,
            layers,
            output,
        }
    }

    pub fn new_zero_one() -> Self {
        let input: LayerSimd<INPUTS, NEURONS, true, false> = LayerSimd::new_zero_one();
        let layers: [LayerSimd<NEURONS, NEURONS, true, RESIDUAL>; NR_LAYERS] =
            std::array::from_fn(|_| LayerSimd::new_zero_one()
        );
        let output: LayerSimd<NEURONS, OUTPUTS, false, false> = LayerSimd::new_zero_one();

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

    pub fn subtract_gradients(&mut self, learning_rate: f32) {
        self.input.subtract_gradients(learning_rate);
        for layer in &mut self.layers.iter_mut().rev() {
            layer.subtract_gradients(learning_rate);
        }
        self.output.subtract_gradients(learning_rate);
    }

}


impl<
    const INPUTS: usize,
    const OUTPUTS: usize,
    const NEURONS: usize,
    const NR_LAYERS: usize,
    const RESIDUAL: bool,
> std::fmt::Display
    for NeuralNetworkLayered<INPUTS, OUTPUTS, NEURONS, NR_LAYERS, RESIDUAL>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                writeln!(f, "NeuralNetworkLayered {{")?;

        writeln!(f, "input.x: {}", self.input.x)?;
        writeln!(f,)?;

        writeln!(f, "input.w:     {}", self.input.w)?;
        for (i, layer) in self.layers.iter().enumerate() {
            writeln!(f, "layers[{}].w: {}", i,layer.w)?;
        }
        writeln!(f, "output.w:    {}", self.output.w)?;
        writeln!(f,)?;

        writeln!(f, "input.b:     {}", self.input.b)?;
        for (i, layer) in self.layers.iter().enumerate() {
            writeln!(f, "layers[{}].b: {}", i,layer.b)?;
        }
        writeln!(f, "output.b:    {}", self.output.b)?;
        writeln!(f,)?;

        writeln!(f, "input.z:     {}", self.input.z)?;
        for (i, layer) in self.layers.iter().enumerate() {
            writeln!(f, "layers[{}].z: {}", i,layer.z)?;
        }
        writeln!(f, "output.z:    {}", self.output.z)?;
        writeln!(f,)?;

        writeln!(f, "input.a:     {}", self.input.a)?;
        for (i, layer) in self.layers.iter().enumerate() {
            writeln!(f, "layers[{}].a: {}", i,layer.a)?;
        }
        writeln!(f, "output.a:    {}", self.output.a)?;
        writeln!(f,)?;

        writeln!(f, "input.dl_dw:     {}", self.input.dl_dw)?;
        for (i, layer) in self.layers.iter().enumerate() {
            writeln!(f, "layers[{}].dl_dw: {}", i,layer.dl_dw)?;
        }
        writeln!(f, "output.dl_dw:    {}", self.output.dl_dw)?;
        writeln!(f,)?;

        writeln!(f, "input.dl_db:     {}", self.input.dl_db)?;
        for (i, layer) in self.layers.iter().enumerate() {
            writeln!(f, "layers[{}].dl_db: {}", i,layer.dl_db)?;
        }
        writeln!(f, "output.dl_db:    {}", self.output.dl_db)?;
        writeln!(f,)?;

        writeln!(f,)?;

        write!(f, "}}")
    }
}