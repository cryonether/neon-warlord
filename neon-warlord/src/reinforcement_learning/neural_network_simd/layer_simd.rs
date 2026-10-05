//! A layer of a neural network

use std::{iter::zip, marker::PhantomData};

use wide::f32x16;

use crate::reinforcement_learning::neural_network_simd::simd_math::{
    simd_mat::SMat16, simd_vec::SVec16,
};

/// A simd layer
#[derive(Clone)]
pub struct LayerSimd<
    const INPUTS: usize,
    const OUTPUTS: usize,
    const RESIDUAL: bool,
    ACTIVATION: ActivationFunction<OUTPUTS>,
> {
    pub x: SVec16<INPUTS>,

    pub w: SMat16<OUTPUTS, INPUTS>,
    pub b: SVec16<OUTPUTS>,

    // intermediate products

    // z = W * a + b
    pub z: SVec16<OUTPUTS>,

    // a = f(z)
    pub a: SVec16<OUTPUTS>,

    // back propagation
    pub dl_dw: SMat16<OUTPUTS, INPUTS>,
    pub dl_db: SVec16<OUTPUTS>,

    // intermediate products
    dx: SVec16<INPUTS>,

    phantom_data: PhantomData<ACTIVATION>
}

impl<const INPUTS: usize, const OUTPUTS: usize, const RESIDUAL: bool, ACTIVATION: ActivationFunction<OUTPUTS>>
    LayerSimd<INPUTS, OUTPUTS, RESIDUAL, ACTIVATION>
{
    pub fn new() -> Self {
        let x = SVec16::zero();
        let w = SMat16::zero();
        let b = SVec16::zero();
        let z = SVec16::zero();
        let a = SVec16::zero();
        let dl_dw = SMat16::zero();
        let dl_db = SVec16::zero();
        let dx = SVec16::zero();

        Self {
            x,
            w,
            b,
            z,
            a,
            dl_dw,
            dl_db,
            dx,
            
            phantom_data: PhantomData,
        }
    }

    pub fn new_rand(rng: &mut fastrand::Rng) -> Self {
        let x = SVec16::zero();
        let mut w = SMat16::zero();
        let mut b = SVec16::zero();
        let z = SVec16::zero();
        let a = SVec16::zero();
        let dl_dw = SMat16::zero();
        let dl_db = SVec16::zero();
        let dx = SVec16::zero();

        // Kaiming/He-style initialization
        let fan_in: f32 = INPUTS as f32; // fan_in is the number of inputs to the neuron/filter.
        let bound = 1.0 / (fan_in).sqrt();
        let mut rand = || (rng.f32() * 2.0 - 1.0) * bound;

        for w in &mut w {
            for w in w {
                *w = rand();
            }
        }

        for b in &mut b {
            *b = rand();
        }

        Self {
            x,
            w,
            b,
            z,
            a,
            dl_dw,
            dl_db,
            dx,
            
            phantom_data: PhantomData,
        }
    }

    pub fn new_zero_one() -> Self {
        let x = SVec16::zero();
        let mut w = SMat16::zero();
        let mut b = SVec16::zero();
        let z = SVec16::zero();
        let a = SVec16::zero();
        let dl_dw = SMat16::zero();
        let dl_db = SVec16::zero();
        let dx = SVec16::zero();

        for w in &mut w {
            for w in w {
                *w = 0.1;
            }
        }

        b.fill(0.1);

        Self {
            x,
            w,
            b,
            z,
            a,
            dl_dw,
            dl_db,
            dx,

            phantom_data: PhantomData,
        }
    }

    pub fn forward(&mut self, x: &SVec16<INPUTS>) -> SVec16<OUTPUTS> {
        // self.assert_finite();

        self.x = x.clone();

        // z = W * x + b
        self.z = &(&self.w * x) + &self.b;

        if RESIDUAL {
            // Note: residual requires INPUTS == OUTPUTS.
            assert_eq!(INPUTS, OUTPUTS);
            for (z, x) in zip(self.z.simd_iter_mut(), x.simd_iter()) {
                *z += x;
            }

            for (z, x) in zip(self.z.remainder_mut(), x.remainder()) {
                *z += x;
            }
        }

        // a = f(z)
        self.a = ACTIVATION::activation(&self.z);

        // self.assert_finite();

        self.a.clone()
    }

    pub fn backward(&mut self, _delta: &SVec16<OUTPUTS>) -> SVec16<INPUTS> {
        // self.assert_finite();

        //
        // Gradient through activation
        //
        // dz = delta ⊙ f'(z)
        //
        let dz = ACTIVATION::derivative(&self.z);

        // W^T * dz
        //
        // dz^T * W gives the same vector as W^T * dz.
        //
        // dz^T * W= (W^T * dz)^T
        //
        let dz_row = dz.clone().as_row_vec();
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
                *dx += dz;
            }

            for (dx, dz) in zip(dx.remainder_mut(), dz.remainder()) {
                *dx += dz;
            }
        }

        // Store gradients
        // dL/db = dz
        let dl_db = &dz;

        // dL/dW = dz * x^T
        let dl_dw = &dz * &self.x.clone().as_row_vec();

        self.dx = dx.clone();

        // Update gradients
        self.dl_db += dl_db;
        self.dl_dw += &dl_dw;

        // self.assert_finite();

        dx
    }

    pub fn subtract_gradients(&mut self, learning_rate: f32) {
        // self.assert_finite();

        let learning_rate_ = f32x16::splat(learning_rate);
        let zero = f32x16::splat(0.0);

        // b
        for (b, dl_db) in zip(self.b.simd_iter_mut(), self.dl_db.simd_iter_mut()) {
            *b -= *dl_db * learning_rate_;
            *dl_db = zero;
        }

        for (b, dl_db) in zip(self.b.remainder_mut(), self.dl_db.remainder_mut()) {
            *b -= *dl_db * learning_rate;
            *dl_db = 0.0;
        }

        // w
        for (w, dl_dw) in zip(&mut self.w, &mut self.dl_dw) {
            for (w, dl_dw) in zip(w.simd_iter_mut(), dl_dw.simd_iter_mut()) {
                *w -= *dl_dw * learning_rate_;
                *dl_dw = zero;
            }

            for (w, dl_dw) in zip(w.remainder_mut(), dl_dw.remainder_mut()) {
                *w -= *dl_dw * learning_rate;
                *dl_dw = 0.0;
            }
        }

        self.assert_finite();
    }

    pub fn assert_not_nan(&self) {
        self.x.assert_not_nan("x");

        self.w.assert_not_nan("w");
        self.b.assert_not_nan("b");

        self.z.assert_not_nan("z");
        self.a.assert_not_nan("a");

        self.dl_dw.assert_not_nan("dl_dw");
        self.dl_db.assert_not_nan("dl_db");

        self.dx.assert_not_nan("dx");
    }

    pub fn assert_finite(&mut self) {
        self.x.assert_finite("x");
        self.w.assert_finite("w");
        self.b.assert_finite("b");
        self.z.assert_finite("z");
        self.a.assert_finite("a");
        self.dl_dw.assert_finite("dl_dw");
        self.dl_db.assert_finite("dl_db");
        self.dx.assert_finite("dx");
    }

    

 
}

// Activation Function

pub trait ActivationFunction<const OUTPUTS: usize> {
    fn activation(x: &SVec16<OUTPUTS>) -> SVec16<OUTPUTS>;
    fn derivative(x: &SVec16<OUTPUTS>) -> SVec16<OUTPUTS>;
}

// Activation Function None

#[derive(Clone)]
pub struct ActivationNone {}

impl<const OUTPUTS: usize> ActivationFunction<OUTPUTS> for ActivationNone {
    fn activation(x: &SVec16<OUTPUTS>) -> SVec16<OUTPUTS> {
        x.clone()
    }

    fn derivative(x: &SVec16<OUTPUTS>) -> SVec16<OUTPUTS> {
        x.clone()
    }
}

// Activation Function Leaky ReLu

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
            *res = if *x > 0.0 {
                *x
            } else {
                x * LEAKY_RELU_ALPHA
            };
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
            *res = if *x > 0.0 {
                1.0
            } else {
                LEAKY_RELU_ALPHA
            };
        }

        res
    }
}

// Activation Function TanH

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


