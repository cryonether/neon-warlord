//! A layer of a neural network

use std::iter::zip;

use wide::f32x16;

use crate::reinforcement_learning::neural_network_simd::simd_math::{
    AlignedVecSlice, simd_mat::{SMat16, SMat16Slice}, simd_vec::{SVec16, SVec16Ref},
};

const LANES: usize = 16;

/// A simd layer
#[derive(Copy, Clone)]
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

    // intermediate products
    dx: SVec16<INPUTS>,
}

impl<const INPUTS: usize, const OUTPUTS: usize, const ACTIVATION: bool, const RESIDUAL: bool>
    LayerSimd<INPUTS, OUTPUTS, ACTIVATION, RESIDUAL>
{
    pub fn new_box() -> Box<dyn LayerRef> {
        Box::new(Self::new())
    }

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
            dx
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
            dx
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
            dx
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
        let learning_rate_= f32x16::splat(learning_rate);

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

impl<const A: usize, const B: usize, const C: bool, const D: bool> Default
    for LayerSimd<A, B, C, D>
{
    fn default() -> Self {
        Self::new()
    }
}

pub trait LayerRef {
    fn forward(&mut self, x: &dyn SVec16Ref) ->  &dyn SVec16Ref;
    fn backward(&mut self, x: &dyn SVec16Ref) ->  &dyn SVec16Ref;
    fn subtract_gradients(&mut self, learning_rate: f32);

    fn get_inputs(&self) -> usize;
    fn get_outputs(&self) -> usize;
    fn get_w(&self) ->  &dyn SMat16Slice;
    fn get_b(&self) ->  &[f32];
}

impl<const INPUTS: usize, const OUTPUTS: usize, const ACTIVATION: bool, const RESIDUAL: bool>
 LayerRef for LayerSimd<INPUTS, OUTPUTS, ACTIVATION, RESIDUAL> {
    fn forward(&mut self, x: &dyn SVec16Ref) -> &dyn SVec16Ref {

        let x = x.as_any().downcast_ref::<SVec16<INPUTS>>();
        match x {
            Some(x) => {
                self.forward(x)
            },
            None => panic!("layer does not match"),
        };

        &self.z        
    }
 
    fn backward(&mut self, y: &dyn SVec16Ref) ->  &dyn SVec16Ref {
        let y = y.as_any().downcast_ref::<SVec16<OUTPUTS>>();
        match y {
            Some(y) => {
                self.backward(y)
            },
            None => panic!("layer does not match"),
        };

        &self.dx   
    }

     fn subtract_gradients(&mut self, learning_rate: f32) {
        self.subtract_gradients(learning_rate);
     }

    fn get_inputs(&self) -> usize {
        INPUTS
    }

    fn get_outputs(&self) -> usize {
        OUTPUTS
    }


    fn get_w(&self) -> &dyn SMat16Slice {
        &self.w
    }

    fn get_b(&self) ->  &[f32] {
        self.b.as_slice()
    }
 }






type Model = (
    LayerSimd::<2, 4, true, false>,
    LayerSimd::<4, 4, true, false>,
    LayerSimd::<4, 2, false, false>,
);

fn create() {


    let model: Vec<Box<dyn LayerRef>>= vec![
        LayerSimd::<2, 4, true, false>::new_box(),
        LayerSimd::<4, 4, true, false>::new_box(),
        LayerSimd::<4, 2, false, false>::new_box(),
    ];


}


pub struct NeuralNetworkLayered {
    model: Vec<Box<dyn LayerRef>>
}

impl NeuralNetworkLayered {
    pub fn new(model: Vec<Box<dyn LayerRef>>) -> Self {
        assert!(!model.is_empty());

        // Check parameters
        let mut outputs = model[0].get_inputs();
        for layer in &model {
            let inputs = layer.get_inputs();
            assert_eq!(outputs, inputs);
            outputs = inputs;
        }

        Self { model }
    }

    pub fn forward<const INPUTS: usize, const OUTPUTS: usize>(&mut self, x: [f32; INPUTS]) {
        assert!(!self.model.is_empty());

        let x = SVec16::new(x);

        // forward
        let mut x: &dyn SVec16Ref = &x;
        for layer in &mut self.model {
            let y = layer.forward(x);
            x = y;
        }

        // return result
        let y = x.as_any().downcast_ref::<SVec16<OUTPUTS>>();
        match y {
            Some(y) => {
                y
            },
            None => panic!("layer does not match"),
        };

    }
}