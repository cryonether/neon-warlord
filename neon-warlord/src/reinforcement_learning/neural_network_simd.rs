//! A universal function approximator efficiently implemented using simd

pub mod epoch;
pub mod gradients;
pub mod simd_math;

#[cfg(test)]
mod test_neural_network_simd;

#[cfg(test)]
mod test_logic_functions;

#[cfg(test)]
mod test_logic_functions3;

#[cfg(test)]
mod test_predict_maze;

use std::iter::zip;

use itertools::izip;
use wide::f32x16;

use crate::reinforcement_learning::neural_network_simd::{gradients::GradientsSimd, simd_math::{SMat, SRowVec, SVec}};


const LANES: usize = 16;
const NR_NEURONS: usize = 128;
const NR_LANES: usize = NR_NEURONS / LANES;



pub type NeuralNetwork16<
    const INPUTS: usize,
    const OUTPUTS: usize,
    const NR_LAYERS: usize,
    const RESIDUAL: bool> 
= NeuralNetworkSimd<
    INPUTS, 
    OUTPUTS, 
    NR_LAYERS, 
    RESIDUAL,
    16, 
    1, 
>;

pub type Gradient16<const SIZE: usize> = GradientsSimd<SIZE, 16, 1>;

pub type NeuralNetwork32<
    const INPUTS: usize,
    const OUTPUTS: usize,
    const NR_LAYERS: usize,
    const RESIDUAL: bool> 
= NeuralNetworkSimd<
    INPUTS, 
    OUTPUTS, 
    NR_LAYERS, 
    RESIDUAL,
    32, 
    2, 
>;

pub type Gradient32<const SIZE: usize> = GradientsSimd<SIZE, 32, 2>;


pub type NeuralNetwork64<
    const INPUTS: usize,
    const OUTPUTS: usize,
    const NR_LAYERS: usize,
    const RESIDUAL: bool> 
= NeuralNetworkSimd<
    INPUTS, 
    OUTPUTS, 
    NR_LAYERS, 
    RESIDUAL,
    64, 
    4, 
>;

pub type Gradient64<const SIZE: usize> = GradientsSimd<SIZE, 164, 4>;


pub type NeuralNetwork128<
    const INPUTS: usize,
    const OUTPUTS: usize,
    const NR_LAYERS: usize,
    const RESIDUAL: bool> 
= NeuralNetworkSimd<
    INPUTS, 
    OUTPUTS, 
    NR_LAYERS, 
    RESIDUAL,
    128, 
    8, 
>;

pub type Gradient128<const SIZE: usize> = GradientsSimd<SIZE, 128, 8>;


#[derive(Clone)]
pub struct NeuralNetworkSimd<
    const INPUTS: usize,
    const OUTPUTS: usize,
    const NR_LAYERS: usize,
    const RESIDUAL: bool,
    const NR_NEURONS: usize,
    const NR_LANES: usize,
> {
    // input
    pub x: [f32; NR_NEURONS],

    // parameters
    pub w: [[[f32; NR_NEURONS]; NR_NEURONS]; NR_LAYERS],
    b: [[f32; NR_NEURONS]; NR_LAYERS],

    // output
    pub w_y: [[f32; NR_NEURONS]; NR_NEURONS],
    b_y: [f32; NR_NEURONS],
    pub y: [f32; NR_NEURONS],

    // intermediate products

    // a = f(z)
    a: [[f32; NR_NEURONS]; NR_LAYERS],

    // z = W*a + b
    z: [[f32; NR_NEURONS]; NR_LAYERS],

    // back propagation
    dy_dw: [[[f32; NR_NEURONS]; NR_NEURONS]; NR_LAYERS],
    dy_db: [[f32; NR_NEURONS]; NR_LAYERS],

    dy_dw_y: [[f32; NR_NEURONS]; NR_NEURONS],
    dy_db_y: [f32; NR_NEURONS],
}

impl<const INPUTS: usize, const OUTPUTS: usize, const NR_LAYERS: usize, const RESIDUAL: bool, const NR_NEURONS: usize, const NR_LANES: usize>
    NeuralNetworkSimd<INPUTS, OUTPUTS, NR_LAYERS, RESIDUAL, NR_NEURONS, NR_LANES>
{
    pub fn new() -> Self {
        let x = [0.0; NR_NEURONS];

        let w = [[[0.0; NR_NEURONS]; NR_NEURONS]; NR_LAYERS];
        let b = [[0.0; NR_NEURONS]; NR_LAYERS];
        let w_y = [[0.0; NR_NEURONS]; NR_NEURONS];
        let b_y = [0.0; NR_NEURONS];

        let y = [0.0; NR_NEURONS];

        let a = [[0.0; NR_NEURONS]; NR_LAYERS];
        let z = [[0.0; NR_NEURONS]; NR_LAYERS];

        let dy_dw = [[[0.0; NR_NEURONS]; NR_NEURONS]; NR_LAYERS];
        let dy_db = [[0.0; NR_NEURONS]; NR_LAYERS];

        let dy_dw_y = [[0.0; NR_NEURONS]; NR_NEURONS];
        let dy_db_y = [0.0; NR_NEURONS];

        Self {
            x,
            w,
            b,
            w_y,
            b_y,
            y,
            a,
            z,
            dy_dw,
            dy_db,
            dy_dw_y,
            dy_db_y,
        }
    }

    pub fn new_rand() -> Self {
        let mut rng = fastrand::Rng::with_seed(fastrand::u64(..));

        // Kaiming/He-style initialization
        let fan_in: f32 = LANES as f32; // fan_in is the number of inputs to the neuron/filter.
        let bound = 1.0 / (fan_in).sqrt();
        let mut rand = || (rng.f32() * 2.0 - 1.0) * bound;

        let mut model = Self::new();

        for w in &mut model.w {
            for w in w {
                for w in w {
                    *w = rand();
                }
            }
        }

        for b in &mut model.b {
            for b in b {
                *b = rand();
            }
        }

        for w in &mut model.w_y {
            for w in w {
                *w = rand();
            }
        }

        for b in &mut model.b_y {
            *b = rand();
        }

        model
    }

    pub fn new_zero_one() -> Self {
        let mut model = Self::new();

        for w in &mut model.w {
            for w in w {
                w.fill(0.1);
            }
        }

        for b in &mut model.b {
            b.fill(0.1);
        }

        for w in &mut model.w_y {
            w.fill(0.1);
        }

        model.b_y.fill(0.1);

        model
    }

    pub fn forward(&mut self, x: &[f32; INPUTS]) -> [f32; OUTPUTS] {
        assert!(INPUTS <= LANES);
        assert!(OUTPUTS <= LANES);

        let mut padded = [0.0; NR_NEURONS];
        padded[..INPUTS].copy_from_slice(x);

        let output = self.forward_16(padded);

        output[..OUTPUTS].try_into().unwrap()
    }



    pub fn forward_16(&mut self, x: [f32; NR_NEURONS]) -> [f32; NR_NEURONS] {
        self.x = x;

        let mut input_: SVec<NR_NEURONS, NR_LANES> = SVec::new(self.x);

        for (w, b, a, z) in izip!(self.w, self.b, &mut self.a, &mut self.z) {
            // z = W * input + b + input
            let mut z_ = SMat::new(w) * &input_;
            z_ += &SVec::new(b);

            if RESIDUAL {
                z_ += &input_;
            }

            // a = f(z)
            let a_ = Self::activation_re_lu_vec(&z_);

            *z = z_.into();
            *a = a_.clone().into();

            input_ = a_;
        }

        // y
        // z = W * a + b
        let mut y_ = SMat::new(self.w_y) * &input_;

        y_ += &SVec::new(self.b_y);
        self.y = y_.into();

        self.y
    }

    pub fn backward(&mut self, index: usize) -> GradientsSimd<NR_LAYERS, NR_NEURONS, NR_LANES> {
        assert!(index < NR_NEURONS);

        let mut z_iter = self.z.iter().rev();
        let mut a_iter = self.a.iter().rev().chain([&self.x]);
        let w_iter = self.w.iter().rev();
        let mut dy_db_iter = self.dy_db.iter_mut().rev();
        let mut dy_dw_iter = self.dy_dw.iter_mut().rev();

        // last element
        let mut dy_db_y= [0.0; NR_NEURONS];
        dy_db_y[index] = 1.0; // choose weight
        self.dy_db_y = dy_db_y;

        let mut dy_dw_y = [[0.0; NR_NEURONS]; NR_NEURONS];
        dy_dw_y[index] = *a_iter.next().unwrap(); // choose weight
        self.dy_dw_y = dy_dw_y;

        // last element -1
        let z = z_iter.next().unwrap();
        let a = a_iter.next().unwrap();
        let w = &self.w_y[index]; // choose weight
        let dy_db = dy_db_iter.next().unwrap();
        let dy_dw = dy_dw_iter.next().unwrap();
        let mut delta_previous_;
        {
            let dz_ = Self::derivative_re_lu_vec(&SVec::new(*z));
            let delta_ = SVec::new(*w) * &dz_;

            delta_previous_ = delta_.clone();

            let dy_dw_ = delta_.clone() * SRowVec::new(*a);

            *dy_db = delta_.clone().into();
            *dy_dw = dy_dw_.into();
        }

        // other elements
        for (&z, &a, w, dy_db, dy_dw) in izip!(z_iter, a_iter, w_iter, dy_db_iter, dy_dw_iter,) {
            // Gradient through W
            let delta_previous_row_vec: SRowVec<NR_NEURONS, NR_LANES> = SRowVec::new(delta_previous_.clone().into());
            let w_ = SMat::new(*w);
            let mut delta_ = SVec::new((delta_previous_row_vec * w_).into());

            // Gradient through residual connection
            if RESIDUAL {
                delta_ += &delta_previous_;
            }

            // Gradient through ReLU
            // let dz_ = Self::derivative_re_lu_f32x16(f32x16::from(z));
            let dz_ = Self::derivative_re_lu_vec(&SVec::new(z));
            let delta_ = delta_ * &dz_;

            delta_previous_ = delta_.clone();

            let dy_dw_ = delta_.clone() * SRowVec::new(a);

            *dy_db = delta_.into();
            *dy_dw = dy_dw_.into();
        }

        GradientsSimd {
            dy_dw: self.dy_dw,
            dy_db: self.dy_db,
            dy_dw_y: self.dy_dw_y,
            dy_db_y: self.dy_db_y,
        }
    }

    pub fn subtract_gradients(&mut self, gradients: &GradientsSimd<NR_LAYERS, NR_NEURONS, NR_LANES>) {
        // w
        for (w, dw) in zip(&mut self.w, &gradients.dy_dw) {
            let res_: SMat<NR_NEURONS, NR_LANES> = SMat::new(*w) - &SMat::new(*dw);
            *w = res_.into();
        }

        // b
        for (b, db) in zip(&mut self.b, &gradients.dy_db) {
            let res: SVec<NR_NEURONS, NR_LANES> = SVec::new(*b) - &SVec::new(*db);
            *b = res.into();
        }

        // w_y
        for (w, dw) in zip(&mut self.w_y, &gradients.dy_dw_y) {
            let res: SVec<NR_NEURONS, NR_LANES> =  SVec::new(*w) - &SVec::new(*dw);
            *w = res.into();
        }

        // b_y
        {
            let res: SVec<NR_NEURONS, NR_LANES> =  SVec::new(self.b_y) - &SVec::new(gradients.dy_db_y);
            self.b_y = res.into();
        }
    }

    const LEAKY_RELU_ALPHA: f32 = 0.01;

    #[inline]
    fn activation_re_lu_vec(x: &SVec<NR_NEURONS, NR_LANES>) -> SVec<NR_NEURONS, NR_LANES> {
        let mut res =  SVec::new([0.0; NR_NEURONS]);
        let zero = f32x16::ZERO;
        let alpha = f32x16::splat(Self::LEAKY_RELU_ALPHA);

        for (x, res) in zip(x.a.as_chunks::<LANES>().0,  res.a.as_chunks_mut::<LANES>().0) {
            let x_ = f32x16::from(*x);

            let res_ = x_.simd_gt(zero).select(x_, x_ * alpha);

            *res = res_.into();
        }

        res
    }

    #[inline]
    fn derivative_re_lu_vec(x: &SVec<NR_NEURONS, NR_LANES>) -> SVec<NR_NEURONS, NR_LANES> {
        let mut res =  SVec::new([0.0; NR_NEURONS]);
        let zero = f32x16::ZERO;
        let alpha = f32x16::splat(Self::LEAKY_RELU_ALPHA);
        let one = f32x16::splat(1.0);

        for (x, res) in zip(x.a.as_chunks::<LANES>().0,  res.a.as_chunks_mut::<LANES>().0) {
            let x_ = f32x16::from(*x);

            let res_ = x_.simd_gt(zero).select(one, alpha);

            *res = res_.into();
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

impl<const INPUTS: usize, const OUTPUTS: usize, const NR_LAYERS: usize, const RESIDUAL: bool, const NR_NEURONS: usize, const NR_LANES: usize>
    std::fmt::Display for NeuralNetworkSimd<INPUTS, OUTPUTS, NR_LAYERS, RESIDUAL, NR_NEURONS, NR_LANES>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "NeuralNetworkSimd {{")?;

        writeln!(f, "x: {:?}", self.x)?;
        writeln!(f)?;

        for (i, w) in self.w.iter().enumerate() {
            for (j, w) in w.iter().enumerate() {
                writeln!(f, "w_{}_{:02}: {:?}", i, j, w)?;
            }
        }
        writeln!(f)?;

        for (i, b) in self.b.iter().enumerate() {
            writeln!(f, "b_{}: {:?}", i, b)?;
        }
        writeln!(f)?;

        for (i, z) in self.z.iter().enumerate() {
            writeln!(f, "z_{}: {:?}", i, z)?;
        }
        writeln!(f)?;

        for (i, a) in self.a.iter().enumerate() {
            writeln!(f, "a_{}: {:?}", i, a)?;
        }
        writeln!(f)?;

        writeln!(f, "y: {:?}", self.y)?;
        writeln!(f)?;

        for (i, dy_dw) in self.dy_dw.iter().enumerate() {
            for (j, dy_dw) in dy_dw.iter().enumerate() {
                writeln!(f, "dy_dw_{}_{:02}: {:?}", i, j, dy_dw)?;
            }
        }
        writeln!(f, "dy_dw_y: {:?}", self.dy_dw_y)?;
        writeln!(f)?;

        for (i, dy_db) in self.dy_db.iter().enumerate() {
            writeln!(f, "dy_db_{}: {:?}", i, dy_db)?;
        }
        writeln!(f, "dy_db_y: {:?}", self.dy_db_y)?;
        writeln!(f)?;

        write!(f, "}}")
    }
}

