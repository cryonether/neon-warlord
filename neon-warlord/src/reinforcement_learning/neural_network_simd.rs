//! A universal function approximator efficiently implemented using simd

pub mod epoch;
pub mod gradients;

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

use crate::reinforcement_learning::neural_network_simd::gradients::GradientsSimd;

const LANES: usize = 16;

pub struct NeuralNetworkSimd<
    const INPUTS: usize,
    const OUTPUTS: usize,
    const NR_LAYERS: usize,
    const RESIDUAL: bool,
> {
    // input
    pub x: [f32; LANES],

    // parameters
    w: [[[f32; LANES]; LANES]; NR_LAYERS],
    b: [[f32; LANES]; NR_LAYERS],

    // output
    w_y: [[f32; LANES]; LANES],
    b_y: [f32; LANES],
    pub y: [f32; LANES],

    // intermediate products

    // a = f(z)
    a: [[f32; LANES]; NR_LAYERS],

    // z = W*a + b
    z: [[f32; LANES]; NR_LAYERS],

    // back propagation
    dy_dw: [[[f32; LANES]; LANES]; NR_LAYERS],
    dy_db: [[f32; LANES]; NR_LAYERS],

    dy_dw_y: [[f32; LANES]; LANES],
    dy_db_y: [f32; LANES],
}

impl<const INPUTS: usize, const OUTPUTS: usize, const NR_LAYERS: usize, const RESIDUAL: bool>
    NeuralNetworkSimd<INPUTS, OUTPUTS, NR_LAYERS, RESIDUAL>
{
    pub fn new() -> Self {
        let x = [0.0; LANES];

        let w = [[[0.0; LANES]; LANES]; NR_LAYERS];
        let b = [[0.0; LANES]; NR_LAYERS];
        let w_y = [[0.0; LANES]; LANES];
        let b_y = [0.0; LANES];

        let y = [0.0; LANES];

        let a = [[0.0; LANES]; NR_LAYERS];
        let z = [[0.0; LANES]; NR_LAYERS];

        let dy_dw = [[[0.0; LANES]; LANES]; NR_LAYERS];
        let dy_db = [[0.0; LANES]; NR_LAYERS];

        let dy_dw_y = [[0.0; LANES]; LANES];
        let dy_db_y = [0.0; LANES];

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

        let mut padded = [0.0; LANES];
        padded[..INPUTS].copy_from_slice(x);

        let output = self.forward_16(padded);

        output[..OUTPUTS].try_into().unwrap()
    }

    pub fn forward_16(&mut self, x: [f32; LANES]) -> [f32; LANES] {
        self.x = x;

        let mut input_ = f32x16::from(self.x);

        for (w, b, a, z) in izip!(self.w, self.b, &mut self.a, &mut self.z) {
            // z = W * input + b + input
            let mut z_ = Self::mul_16x16_16x1(&w, input_);
            z_ += f32x16::from(b);
            if RESIDUAL {
                z_ += input_;
            }

            // a = f(z)
            let a_ = Self::activation_re_lu_f32x16(z_);

            *z = z_.into();
            *a = a_.into();

            input_ = a_;
        }

        // y
        // z = W * a + b
        let mut y_ = Self::mul_16x16_16x1(&self.w_y, input_);
        y_ += f32x16::from(self.b_y);
        self.y = y_.into();

        self.y
    }

    pub fn backward(&mut self, index: usize) -> GradientsSimd<NR_LAYERS> {
        assert!(index < LANES);

        let mut z_iter = self.z.iter().rev();
        let mut a_iter = self.a.iter().rev().chain([&self.x]);
        let w_iter = self.w.iter().rev();
        let mut dy_db_iter = self.dy_db.iter_mut().rev();
        let mut dy_dw_iter = self.dy_dw.iter_mut().rev();

        // last element
        let mut dy_db_y = [0.0; LANES];
        dy_db_y[index] = 1.0; // choose weight
        self.dy_db_y = dy_db_y;

        let mut dy_dw_y = [[0.0; LANES]; LANES];
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
            let dz_ = Self::derivative_re_lu_f32x16(f32x16::from(*z));
            let delta_ = f32x16::from(*w) * dz_;

            delta_previous_ = delta_;

            let dy_dw_ = Self::mul_delta_a(delta_, f32x16::from(*a));

            *dy_db = delta_.into();
            *dy_dw = dy_dw_;
        }

        // other elements
        for (&z, &a, w, dy_db, dy_dw) in izip!(z_iter, a_iter, w_iter, dy_db_iter, dy_dw_iter,) {
            // Gradient through W
            let mut delta_ = Self::mul_1x16_16x16(delta_previous_, w);

            // Gradient through residual connection
            if RESIDUAL {
                delta_ += delta_previous_;
            }

            // Gradient through ReLU
            let dz_ = Self::derivative_re_lu_f32x16(f32x16::from(z));
            let delta_ = delta_ * dz_;

            delta_previous_ = delta_;

            let dy_dw_ = Self::mul_delta_a(delta_, f32x16::from(a));

            *dy_db = delta_.into();
            *dy_dw = dy_dw_;
        }

        GradientsSimd {
            dy_dw: self.dy_dw,
            dy_db: self.dy_db,
            dy_dw_y: self.dy_dw_y,
            dy_db_y: self.dy_db_y,
        }
    }

    pub fn subtract_gradients(&mut self, gradients: &GradientsSimd<NR_LAYERS>) {
        // w
        for (w, dw) in zip(&mut self.w, &gradients.dy_dw) {
            // println!("dw {:?}", dw);
            for (w, dw) in zip(w, dw) {
                let res = f32x16::from(*w) - f32x16::from(*dw);
                *w = res.into();
            }
        }

        // b
        for (b, db) in zip(&mut self.b, &gradients.dy_db) {
            // println!("db {:?}", db);
            let res = f32x16::from(*b) - f32x16::from(*db);
            *b = res.into();
        }

        // w_y
        for (w, dw) in zip(&mut self.w_y, &gradients.dy_dw_y) {
            let res = f32x16::from(*w) - f32x16::from(*dw);
            *w = res.into();
        }

        // b_y
        {
            let res = f32x16::from(self.b_y) - f32x16::from(gradients.dy_db_y);
            self.b_y = res.into();
        }
    }

    // #[inline]
    // fn mul_16x16_16x1(a: &[[f32;16]], b: f32x16) -> f32x16{
    //     let mut out_ = f32x16::splat(0.0);

    //     for &a in a {
    //         let a_ = f32x16::from(a);

    //         out_ += a_ * b;
    //     }

    //     out_.into()
    // }

    #[inline]
    fn mul_16x16_16x1(a: &[[f32; 16]], b: f32x16) -> f32x16 {
        let mut out = [0.0f32; 16];

        for (out, row) in zip(&mut out, a) {
            let row = f32x16::from(*row);
            let product = row * b;

            // horizontal sum of product
            *out = product.reduce_add();
        }

        f32x16::from(out)
    }

    #[inline]
    fn mul_1x16_16x16(a: f32x16, b: &[[f32; 16]]) -> f32x16 {
        let mut out_ = f32x16::splat(0.0);

        for (&a, &b) in zip(a.as_array(), b) {
            let a_ = f32x16::splat(a);
            let b_ = f32x16::from(b);

            out_ += a_ * b_;
        }

        out_
    }

    #[inline]
    pub fn mul_delta_a(delta_: f32x16, a_: f32x16) -> [[f32; 16]; 16] {
        let mut res: [[f32; 16]; 16] = [[0.0; 16]; 16];

        for (&delta, res) in zip(delta_.as_array(), &mut res) {
            let delta_ = f32x16::splat(delta);
            let row = delta_ * a_;

            *res = row.into();
        }

        res
    }

    // #[inline]
    // fn activation_re_lu_f32x16(x: f32x16) -> f32x16 {
    //     x.max(f32x16::splat(0.0))
    // }

    // #[inline]
    // fn derivative_re_lu_f32x16(x: f32x16) -> f32x16 {
    //     x.simd_gt(f32x16::splat(0.0))
    //         .select(f32x16::splat(1.0), f32x16::splat(0.0))
    // }

    // fn activation_re_lu(value: f32) -> f32 {
    //     value.max(0.0)
    // }

    // fn derivative_re_lu(value: f32) -> f32 {
    //     if value > 0.0 { 1.0 } else { 0.0 }
    // }

    const LEAKY_RELU_ALPHA: f32 = 0.01;

    #[inline]
    fn activation_re_lu_f32x16(x: f32x16) -> f32x16 {
        x.simd_gt(f32x16::splat(0.0))
            .select(x, x * f32x16::splat(Self::LEAKY_RELU_ALPHA))
    }

    #[inline]
    fn derivative_re_lu_f32x16(x: f32x16) -> f32x16 {
        x.simd_gt(f32x16::splat(0.0))
            .select(f32x16::splat(1.0), f32x16::splat(Self::LEAKY_RELU_ALPHA))
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

impl<const INPUTS: usize, const OUTPUTS: usize, const NR_LAYERS: usize, const RESIDUAL: bool>
    std::fmt::Display for NeuralNetworkSimd<INPUTS, OUTPUTS, NR_LAYERS, RESIDUAL>
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
