//! A layer of a neural network

use std::iter::zip;

use wide::f32x16;

use crate::reinforcement_learning::neural_network_simd::simd_math::{
    simd_mat::SMat16,
    simd_vec::SVec16,
};

const LANES: usize = 16;

/// A simd layer
#[derive(Clone)]
pub struct LayerSimd<
    const INPUTS: usize,
    const OUTPUTS: usize,
    const ACTIVATION: bool,
    const RESIDUAL: bool,
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
        }
    }

    pub fn new_rand() -> Self {
        let x = SVec16::zero();
        let mut w = SMat16::zero();
        let mut b = SVec16::zero();
        let z = SVec16::zero();
        let a = SVec16::zero();
        let dl_dw = SMat16::zero();
        let dl_db = SVec16::zero();
        let dx = SVec16::zero();

        let mut rng = fastrand::Rng::with_seed(fastrand::u64(..));

        // Kaiming/He-style initialization
        let fan_in: f32 = LANES as f32; // fan_in is the number of inputs to the neuron/filter.
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

        for b in &mut b {
            *b = 0.1;
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
        }
    }

    pub fn forward(&mut self, x: &SVec16<INPUTS>) -> SVec16<OUTPUTS> {
        self.x = *x;

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
        let dl_db = dz;

        // dL/dW = dz * x^T
        let dl_dw = &dz * &self.x.as_row_vec();

        self.dx = dx;

        // Update gradients
        self.dl_db += &dl_db;
        self.dl_dw += &dl_dw;

        dx
    }

    pub fn subtract_gradients(&mut self, learning_rate: f32) {
        let learning_rate_ = f32x16::splat(learning_rate);

        // b
        for (b, dl_db) in zip(self.b.simd_iter_mut(), self.dl_db.simd_iter()) {
            *b -= dl_db * learning_rate_;
        }

        for (b, dl_db) in zip(self.b.remainder_mut(), self.dl_db.remainder()) {
            *b -= dl_db * learning_rate;
        }

        // w
        for (w, dl_dw) in zip(&mut self.w, &self.dl_dw) {
            for (w, dl_dw) in zip(w.simd_iter_mut(), dl_dw.simd_iter()) {
                *w -= dl_dw * learning_rate_;
            }

            for (w, dl_dw) in zip(w.remainder_mut(), dl_dw.remainder()) {
                *w -= dl_dw * learning_rate;
            }
        }
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
            *res = if *x > 0.0 {
                *x
            } else {
                x * Self::LEAKY_RELU_ALPHA
            };
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
            *res = if *x > 0.0 {
                1.0
            } else {
                Self::LEAKY_RELU_ALPHA
            };
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


