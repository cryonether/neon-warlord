//! A layer of a neural network

use std::iter::zip;

use wide::f32x16;

use crate::reinforcement_learning::neural_network_simd::simd_math::{
    simd_mat::SMat16, simd_vec::SVec16,
};

const LANES: usize = 16;

/// A simd layer
/// Inputs and outputs are multiple of 16
pub struct LayerSimd<
    const INPUTS: usize,
    const OUTPUTS: usize,
    const ACTIVATION: bool,
    const RESIDUAL: bool,
> {

    x: SVec16<INPUTS>,

    w: SMat16<OUTPUTS, INPUTS>,
    b: SVec16<OUTPUTS>,

    // intermediate products

    // z = W * a + b
    z: SVec16<OUTPUTS>,

    // a = f(z)
    a: SVec16<OUTPUTS>,

    // back propagation
    dy_dw: SMat16<OUTPUTS, INPUTS>,
    dy_db: SVec16<OUTPUTS>,
}

impl<const INPUTS: usize, const OUTPUTS: usize, const ACTIVATION: bool, const RESIDUAL: bool>
    LayerSimd<INPUTS, OUTPUTS, ACTIVATION, RESIDUAL>
{
    pub fn new() -> Self {
        let x = SVec16::zero();
        let w = SMat16::zero();
        let b = SVec16::zero();
        let z = SVec16::zero();
        let a = SVec16::zero();
        let dy_dw = SMat16::zero();
        let dy_db = SVec16::zero();

        Self {
            x,
            w,
            b,
            z,
            a,
            dy_dw,
            dy_db,
        }
    }

    pub fn forward(&mut self, x: &SVec16<INPUTS>) -> SVec16<OUTPUTS> {
        
        self.x = *x;

        // z = W * x + b
        self.z = &(&self.w * x) + &self.b;

        if RESIDUAL {
            // Note: residual requires INPUTS == OUTPUTS.
            assert_eq!(INPUTS, OUTPUTS);
            for (z, x) in zip(self.z.simd_iter_mut(), x.simd_iter()){
                *z += x;
            }
        }

        // a = f(z)
        self.a = Self::activation_re_lu_vec(&self.z);

        self.a
    }

    pub fn backward(&mut self, delta: &SVec16<OUTPUTS>) -> SVec16<INPUTS> {
        //
        // Gradient through activation
        //
        // dz = delta ⊙ f'(z)
        //

        let dz_activation = Self::derivative_re_lu_vec(&self.z);
        let dz = delta * &dz_activation;

        // W^T * dz
        //
        // dz^T * W gives the same vector as W^T * dz.
        let dz_row = dz.as_row_vec();
        let mut dx = (&dz_row * &self.w).as_column_vec();

        //
        // Gradient through residual connection
        //
        // If:
        //
        //     z = W*x + b + x
        //
        // then:
        //
        //     dx = W^T * dz + dz
        //
        if RESIDUAL {
            // Note: residual requires INPUTS == OUTPUTS.
            assert_eq!(INPUTS, OUTPUTS);

            for (dx, dz) in zip(dx.simd_iter_mut(), dz.simd_iter()) {
                *dx += *dz;
            }
        }

        // Store gradients
        // dL/db = dz
        self.dy_db = dz.clone();

        // dL/dW = dz * x^T
        self.dy_dw = &dz * &self.x.as_row_vec();

        dx
    }

    const LEAKY_RELU_ALPHA: f32 = 0.01;



    #[inline]
    fn activation_re_lu_vec(x: &SVec16<OUTPUTS>) -> SVec16<OUTPUTS> {
        let mut res = SVec16::zero();
        let zero = f32x16::ZERO;
        let alpha = f32x16::splat(Self::LEAKY_RELU_ALPHA);

        for (x, res) in zip(x.simd_iter(), res.simd_iter_mut()) {
            *res = x.simd_gt(zero).select(*x, x * alpha);
        }

        res
    }

    #[inline]
    fn derivative_re_lu_vec(x: &SVec16<OUTPUTS>) -> SVec16<OUTPUTS> {
        let mut res = SVec16::zero();
        let zero = f32x16::ZERO;
        let alpha = f32x16::splat(Self::LEAKY_RELU_ALPHA);
        let one = f32x16::splat(1.0);

        for (x, res) in zip(x.simd_iter(), res.simd_iter_mut()) {
            *res = x.simd_gt(zero).select(one, alpha);
        }

        res
    }

    #[inline]
    fn activation_re_lu(value: f32) -> f32 {
        if value > 0.0 {
            value
        } else {
            Self::LEAKY_RELU_ALPHA * value
        }
    }

    #[inline]
    fn derivative_re_lu(value: f32) -> f32 {
        if value > 0.0 {
            1.0
        } else {
            Self::LEAKY_RELU_ALPHA
        }
    }
}
