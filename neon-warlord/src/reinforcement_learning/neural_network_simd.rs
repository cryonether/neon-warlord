//! A universal function approximator efficiently implemented using simd

#[cfg(test)]
mod tests;

use std::iter::zip;

use itertools::izip;
use wide::f32x16;

const LANES: usize = 16;

pub struct NeuralNetworkSimd<const SIZE: usize> {
    // input
    x: [f32; LANES],

    // parameters
    w: [[[f32; LANES]; LANES]; SIZE],
    b: [[f32; LANES]; SIZE],

    // output
    w_y: [f32; LANES],
    b_y: f32,
    y: f32,

    // intermediate products

    // a = f(z)
    a: [[f32; LANES]; SIZE],

    // z = W*a + b
    z: [[f32; LANES]; SIZE],

    // back propagation

    // delta_i = delta_i+2 * w_i+1 * f⁻¹(z_i)
    delta: [[f32; LANES]; SIZE],

    // dy/dw
    dy_dw: [[[f32; LANES]; LANES]; SIZE],
    dy_db: [[f32; LANES]; SIZE],

    dy_dw_y: [f32; LANES],
    dy_db_y: f32,
}

impl<const SIZE: usize> NeuralNetworkSimd<SIZE> {
    pub fn new() -> Self {
        let x = [0.0; LANES];

        let w = [[[0.0; LANES]; LANES]; SIZE];
        let b = [[0.0; LANES]; SIZE];
        let w_y = [0.0; LANES];
        let b_y = 0.0;

        let y = 0.0;

        let a = [[0.0; LANES]; SIZE];
        let z = [[0.0; LANES]; SIZE];
        
        let delta = [[0.0; LANES]; SIZE];
        let dy_dw = [[[0.0; LANES]; LANES]; SIZE];
        let dy_db = [[0.0; LANES]; SIZE];

        let dy_dw_y = [0.0; LANES];
        let dy_db_y = 0.0;

        Self {
            x,
            w,
            b,
            w_y,
            b_y,
            y,
            a,
            z,
            delta,
            dy_dw,
            dy_db,
            dy_dw_y,
            dy_db_y,
        }
    }

    pub fn forward(&mut self) {
        let mut input_ = f32x16::from(self.x);

        for (w, b, a, z) in izip!(self.w, self.b, &mut self.a, &mut self.z) 
        {
            // z = W * a + b
            let mut z_ = Self::mul_16x16_16x1(&w, input_);
            z_ += f32x16::from(b);

            // a = f(z)
            let a_ = Self::activation_re_lu_f32x16(z_);

            *z = z_.into();
            *a = a_.into();

            input_ = a_;
        }

        // y
        let mut y = 0.0;
        for (&input, w) in zip(input_.as_array(), self.w_y) {
            y += w * input;
        }   
        y += self.b_y;
        self.y = y;
    }

    pub fn backward(&mut self) {

        let mut z_iter = self.z.iter().rev(); 
        let mut a_iter = self.a.iter().rev().chain([&self.x]); 
        let w_iter = self.w.iter().rev();
        let mut delta_iter = self.delta.iter_mut().rev();
        let mut dy_db_iter = self.dy_db.iter_mut().rev();
        let mut dy_dw_iter = self.dy_dw.iter_mut().rev();

        // last element
        self.dy_db_y = 1.0;
        self.dy_dw_y = *a_iter.next().unwrap();

        // last element -1
        let &z = z_iter.next().unwrap();
        let &a = a_iter.next().unwrap();
        let w = self.w_y;
        let delta = delta_iter.next().unwrap();
        let dy_db = dy_db_iter.next().unwrap();
        let dy_dw = dy_dw_iter.next().unwrap();
        let mut delta_previous_;
        {
            let dz_ = Self::derivative_re_lu_f32x16(f32x16::from(z));
            let delta_ = f32x16::from(w) * dz_;

            delta_previous_ = delta_;

            let dy_dw_ = Self::mul_delta_a(delta_, f32x16::from(a));

            *delta = delta_.into();
            *dy_db = delta_.into();
            *dy_dw = dy_dw_.into();
        }

        // other elements
        for (&z, &a, w, delta, dy_db, dy_dw) in izip!(
            z_iter,
            a_iter,
            w_iter,
            delta_iter,
            dy_db_iter,
            dy_dw_iter,
        ) {
            let delta_ = Self::mul_1x16_16x16(delta_previous_, w);
            let dz_ = Self::derivative_re_lu_f32x16(f32x16::from(z));
            let delta_ = delta_ * dz_;

            delta_previous_ = delta_;

            let dy_dw_ = Self::mul_delta_a(delta_, f32x16::from(a));

            *delta = delta_.into();
            *dy_db = delta_.into();
            *dy_dw = dy_dw_.into();
        }
    }


        #[inline]
    fn mul_16x16_16x1(a: &[[f32;16]], b: f32x16) -> f32x16{
        let mut out_ = f32x16::splat(0.0);

        for &a in a {
            let a_ = f32x16::from(a);
            
            out_ += a_ * b;
        }

        out_.into()
    }


    #[inline]
    fn mul_1x16_16x16(a: f32x16, b: &[[f32;16]]) -> f32x16{
        let mut out_ = f32x16::splat(0.0);

        for (&a, &b) in zip(a.as_array(), b){
            let a_ = f32x16::splat(a);
            let b_ = f32x16::from(b);

            out_ += a_ * b_;
        }

        out_.into()
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

    #[inline]
    fn activation_re_lu_f32x16(x: f32x16) -> f32x16 {
        x.max(f32x16::splat(0.0))
    }

    #[inline]
    fn derivative_re_lu_f32x16(x: f32x16) -> f32x16 {
        // 1 where x > 0, otherwise 0.
        x.simd_gt(f32x16::splat(0.0))
            .blend(
                f32x16::splat(1.0),
                f32x16::splat(0.0),
            )
    }

    fn activation_re_lu(value: f32) -> f32 {
        value.max(0.0)
    }

    fn derivative_re_lu(value: f32) -> f32 {
        if value > 0.0 { 1.0 } else { 0.0 }
    }
}

impl<const SIZE: usize>  std::fmt::Display for NeuralNetworkSimd<SIZE> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "NeuralNetworkSimd {{")?;
        
        writeln!(f, "x: {:?}", self.x)?;
        writeln!(f, "")?;

        for (i, w) in self.w.iter().enumerate() {
            for (j, w) in w.iter().enumerate() {
                writeln!(f, "w_{}_{:02}: {:?}", i, j, w)?;

            }
        }
        writeln!(f, "")?;

        for (i, b) in self.b.iter().enumerate() {
            writeln!(f, "b_{}: {:?}", i, b)?;
        }
        writeln!(f, "")?;

        for (i, z) in self.z.iter().enumerate() {
            writeln!(f, "z_{}: {:?}", i, z)?;
        }
        writeln!(f, "")?;

        for (i, a) in self.a.iter().enumerate() {
            writeln!(f, "a_{}: {:?}", i, a)?;
        }
        writeln!(f, "")?;

        writeln!(f, "y: {:?}", self.y)?;
        writeln!(f, "")?;

        for (i, dy_dw) in self.dy_dw.iter().enumerate() {
            for (j, dy_dw) in dy_dw.iter().enumerate() {
                writeln!(f, "dy_dw_{}_{:02}: {:?}", i, j, dy_dw)?;

            }
        }
        writeln!(f, "dy_dw_y: {:?}", self.dy_dw_y)?;
        writeln!(f, "")?;

        for (i, dy_db) in self.dy_db.iter().enumerate() {
            writeln!(f, "dy_db_{}: {:?}", i, dy_db)?;
        }
        writeln!(f, "dy_db_y: {:?}", self.dy_db_y)?;
        writeln!(f, "")?;

        write!(f, "}}")
    }
}