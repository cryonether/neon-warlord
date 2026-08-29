//! An epoch of the neural network

use super::NeuralNetworkSimd;
use super::LANES;

pub struct EpochSimd<const SIZE: usize> {
    model: NeuralNetworkSimd<SIZE>,

    pub loss: f32,
}