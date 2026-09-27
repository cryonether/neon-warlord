//! A layer of a neural network

use std::iter::zip;

use wide::f32x16;

use crate::reinforcement_learning::neural_network_simd::simd_math::{
    simd_mat::SMat16, simd_vec::SVec16,
};

const LANES: usize = 16;

/// A simd layer
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
    dl_dw: SMat16<OUTPUTS, INPUTS>,
    dl_db: SVec16<OUTPUTS>,
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
        let dl_dw = SMat16::zero();
        let dl_db = SVec16::zero();

        Self {
            x,
            w,
            b,
            z,
            a,
            dl_dw,
            dl_db,
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

            for (z, x) in zip(self.z.remainder_mut(), x.remainder()){
                *z += x;
            }
        }

        // a = f(z)
        self.a = if ACTIVATION {
            Self::activation_re_lu_vec(&self.z)
        } else {
            self.z.clone()
        };

        self.a.clone()
    }

    pub fn backward(&mut self, delta: &SVec16<OUTPUTS>) -> SVec16<INPUTS> {
        //
        // Gradient through activation
        //
        // dz = delta ⊙ f'(z)
        //
        let dz = if ACTIVATION {
            delta * &Self::derivative_re_lu_vec(&self.z)
        } else {
            delta.clone()
        };

        // W^T * dz
        //
        // dz^T * W gives the same vector as W^T * dz.
        //
        // dz^T * W= (W^T * dz)^T
        //
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

            for (dx, dz) in zip(dx.remainder_mut(), dz.remainder()) {
                *dx += *dz;
            }
        }

        // Store gradients
        // dL/db = dz
        self.dl_db = dz.clone();

        // dL/dW = dz * x^T
        self.dl_dw = &dz * &self.x.as_row_vec();

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

        for (x, res) in zip(x.remainder(), res.remainder_mut()) {
            *res = if *x > 0.0 { *x } else { x * Self::LEAKY_RELU_ALPHA };
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

        for (x, res) in zip(x.remainder(), res.remainder_mut()) {
            *res = if *x > 0.0 { 1.0 } else { Self::LEAKY_RELU_ALPHA };
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
