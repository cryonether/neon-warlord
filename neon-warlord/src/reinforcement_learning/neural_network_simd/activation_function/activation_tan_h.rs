//! Activation Function TanH

use std::iter::zip;

use wide::f32x16;

use crate::reinforcement_learning::neural_network_simd::{
    activation_function::ActivationFunction, simd_math::simd_vec::SVec16,
};

/// Activation Function TanH
#[derive(Clone)]
pub struct ActivationTanH {}

impl<const OUTPUTS: usize> ActivationFunction<OUTPUTS> for ActivationTanH {
    fn activation(x: &SVec16<OUTPUTS>) -> SVec16<OUTPUTS> {
        let mut res = SVec16::zero();

        for (x, res) in zip(x.simd_iter(), res.simd_iter_mut()) {
            *res = x.tanh();
        }

        for (x, res) in zip(x.remainder(), res.remainder_mut()) {
            *res = x.tanh();
        }

        res
    }

    fn derivative(x: &SVec16<OUTPUTS>) -> SVec16<OUTPUTS> {
        let mut res = SVec16::zero();
        let one = f32x16::splat(1.0);

        for (x, res) in zip(x.simd_iter(), res.simd_iter_mut()) {
            let tanh_x = x.tanh();
            *res = one - tanh_x * tanh_x;
        }

        for (x, res) in zip(x.remainder(), res.remainder_mut()) {
            let tanh_x = x.tanh();
            *res = 1.0 - tanh_x * tanh_x;
        }

        res
    }
}
