//! Activation Function None

use crate::reinforcement_learning::neural_network_simd::{activation_function::ActivationFunction, simd_math::simd_vec::SVec16};

/// Activation Function None
#[derive(Clone)]
pub struct ActivationNone {}

impl<const OUTPUTS: usize> ActivationFunction<OUTPUTS> for ActivationNone {
    fn activation(x: &SVec16<OUTPUTS>) -> SVec16<OUTPUTS> {
        x.clone()
    }

    fn derivative(_x: &SVec16<OUTPUTS>) -> SVec16<OUTPUTS> {
        SVec16::one()
    }
}
