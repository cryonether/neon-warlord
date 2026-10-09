//! Activation functions for the neural network

use crate::reinforcement_learning::neural_network_simd::simd_math::simd_vec::SVec16;

pub mod activation_leaky_relu;
pub mod activation_none;
pub mod activation_tan_h;

/// Activation Function
pub trait ActivationFunction<const OUTPUTS: usize> {
    fn activation(x: &SVec16<OUTPUTS>) -> SVec16<OUTPUTS>;
    fn derivative(x: &SVec16<OUTPUTS>) -> SVec16<OUTPUTS>;
}
