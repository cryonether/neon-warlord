// Activation Function Leaky ReLu

use std::iter::zip;

use wide::f32x16;

use crate::reinforcement_learning::neural_network_simd::{activation_function::ActivationFunction, simd_math::simd_vec::SVec16};

#[derive(Clone)]
pub struct ActivationLeakyReLu {}

impl<const OUTPUTS: usize> ActivationFunction<OUTPUTS> for ActivationLeakyReLu {
    fn activation(x: &SVec16<OUTPUTS>) -> SVec16<OUTPUTS> {
        const LEAKY_RELU_ALPHA: f32 = 0.01;

        let mut res = SVec16::zero();
        let zero = f32x16::ZERO;
        let alpha = f32x16::splat(LEAKY_RELU_ALPHA);

        for (x, res) in zip(x.simd_iter(), res.simd_iter_mut()) {
            *res = x.simd_gt(zero).select(*x, x * alpha);
        }

        for (x, res) in zip(x.remainder(), res.remainder_mut()) {
            *res = if *x > 0.0 { *x } else { x * LEAKY_RELU_ALPHA };
        }

        res
    }

    fn derivative(x: &SVec16<OUTPUTS>) -> SVec16<OUTPUTS> {
        const LEAKY_RELU_ALPHA: f32 = 0.01;

        let mut res = SVec16::zero();
        let zero = f32x16::ZERO;
        let alpha = f32x16::splat(LEAKY_RELU_ALPHA);
        let one = f32x16::splat(1.0);

        for (x, res) in zip(x.simd_iter(), res.simd_iter_mut()) {
            *res = x.simd_gt(zero).select(one, alpha);
        }

        for (x, res) in zip(x.remainder(), res.remainder_mut()) {
            *res = if *x > 0.0 { 1.0 } else { LEAKY_RELU_ALPHA };
        }

        res
    }
}


// Activation Function ReLu

// #[inline]
// fn activation_re_lu(value: f32) -> f32 {
//     if value > 0.0 {
//         value
//     } else {
//         Self::LEAKY_RELU_ALPHA * value
//     }
// }

// #[inline]
// fn derivative_re_lu(value: f32) -> f32 {
//     if value > 0.0 {
//         1.0
//     } else {
//         Self::LEAKY_RELU_ALPHA
//     }
// }

