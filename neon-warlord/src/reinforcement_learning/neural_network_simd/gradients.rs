//! Gradients of NeuralNetworkSimd

use std::iter::zip;

use itertools::izip;
use wide::f32x16;

use super::LANES;

pub struct GradientsSimd<const SIZE: usize>  {
    dy_dw: [[[f32; LANES]; LANES]; SIZE],
    dy_db: [[f32; LANES]; SIZE],

    dy_dw_y: [f32; LANES],
    dy_db_y: f32,
}

impl<const SIZE: usize> GradientsSimd<SIZE> {

    pub fn new() -> Self {
        let dy_dw = [[[0.0; LANES]; LANES]; SIZE];
        let dy_db = [[0.0; LANES]; SIZE];

        let dy_dw_y = [0.0; LANES];
        let dy_db_y = 0.0;

        Self {
            dy_dw,
            dy_db,
            dy_dw_y,
            dy_db_y,
        }
    }

    #[inline]
    pub fn multiply_constant(&self, val: f32) -> Self {
        let mut res = Self::new();

        let val_ = f32x16::splat(val);
        
        // dy_dw
        for (x, y) in zip(self.dy_dw, &mut res.dy_dw) {
            for (x, y) in zip(x, y) 
            {
                let y_ = f32x16::from(x) * val_;
                *y = y_.into(); 
            }
        }

        // dy_db
        for (x, y) in zip(self.dy_db, &mut res.dy_db) 
        {
            let y_ = f32x16::from(x) * val_;
            *y = y_.into();
        }

        // dy_dw_y
        let x = self.dy_dw_y;
        let y = &mut res.dy_dw_y;
        {
            let y_ = f32x16::from(x) * val_;
            *y = y_.into();
        }

        // dy_db_y
        res.dy_db_y = self.dy_db_y * val;

        res

    }

    #[inline]
    pub fn add(&self, other: &Self) -> Self {
        let mut res = Self::new();

        // dy_dw
        for (a, b, y) in izip!(self.dy_dw, other.dy_dw, &mut res.dy_dw) {
            for (a, b, y) in izip!(a, b, y) 
            {
                let y_ = f32x16::from(a) + f32x16::from(b);
                *y = y_.into(); 
            }
        }

        // dy_db
        for (a, b, y) in izip!(self.dy_db, other.dy_db, &mut res.dy_db) {
            let y_ = f32x16::from(a) + f32x16::from(b);
            *y = y_.into(); 
        }

        // dy_dw_y
        let a = self.dy_dw_y;
        let b = other.dy_dw_y;
        let y = &mut res.dy_dw_y;
        {
            let y_ = f32x16::from(a) + f32x16::from(b);
            *y = y_.into(); 
        }

        // dy_db_y
        res.dy_db_y = self.dy_db_y + other.dy_db_y;

        res

    }

    #[inline]
    pub fn sub(&self, other: &Self) -> Self {
        let mut res = Self::new();

        // dy_dw
        for (a, b, y) in izip!(self.dy_dw, other.dy_dw, &mut res.dy_dw) {
            for (a, b, y) in izip!(a, b, y) 
            {
                let y_ = f32x16::from(a) - f32x16::from(b);
                *y = y_.into(); 
            }
        }

        // dy_db
        for (a, b, y) in izip!(self.dy_db, other.dy_db, &mut res.dy_db) {
            let y_ = f32x16::from(a) - f32x16::from(b);
            *y = y_.into(); 
        }

        // dy_dw_y
        let a = self.dy_dw_y;
        let b = other.dy_dw_y;
        let y = &mut res.dy_dw_y;
        {
            let y_ = f32x16::from(a) - f32x16::from(b);
            *y = y_.into(); 
        }

        // dy_db_y
        res.dy_db_y = self.dy_db_y - other.dy_db_y;

        res

    }

}