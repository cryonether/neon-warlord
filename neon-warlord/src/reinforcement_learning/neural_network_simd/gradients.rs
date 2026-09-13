//! Gradients of NeuralNetworkSimd

use std::iter::zip;

use itertools::izip;


use super::SVec;
use super::SMat;

pub struct GradientsSimd<const SIZE: usize, const N: usize, const L: usize> {
    pub dy_dw: [SMat<N, L>; SIZE],
    pub dy_db: [SVec<N, L>; SIZE],

    pub dy_dw_y: SMat<N, L>,
    pub dy_db_y: SVec<N, L>,
}

impl<const SIZE: usize, const NR_NEURONS: usize, const NR_LANES: usize> GradientsSimd<SIZE, NR_NEURONS, NR_LANES> {
    pub fn new() -> Self {
        let dy_dw = [SMat::new([[0.0; NR_NEURONS]; NR_NEURONS]); SIZE];
        let dy_db = [SVec::new([0.0; NR_NEURONS]); SIZE];

        let dy_dw_y = SMat::new([[0.0; NR_NEURONS]; NR_NEURONS]);
        let dy_db_y = SVec::new([0.0; NR_NEURONS]);

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

        // dy_dw
        for (x, y) in zip(self.dy_dw, &mut res.dy_dw) {
            for (x, y) in zip(x.as_array(), y.as_mut_array()) {
                for (x, y) in zip(x, y) {
                    *y = x * val;
                }
            }
        }

        // dy_db
        for (x, y) in zip(&self.dy_db, &mut res.dy_db) {
            for (x, y) in zip(x.as_array(), y.as_mut_array()) {
                *y = x * val;
            }
        }

        // dy_dw_y
        for (x, y) in zip(self.dy_dw_y.as_array(), res.dy_dw_y.as_mut_array()) {
                for (x, y) in zip(x, y) {
                    *y = x * val;
                }
        }

        // dy_db_y
        for (x, y) in zip(self.dy_db_y.as_array(), res.dy_db_y.as_mut_array()) {
            *y = x * val;
        }

        res
    }

    #[inline]
    pub fn add(&self, other: &Self) -> Self {
        let mut res = Self::new();

        // dy_dw
        for (a, b, y) in izip!(&self.dy_dw, &other.dy_dw, &mut res.dy_dw) {
            *y = a + b;
        }

        // dy_db
        for (a, b, y) in izip!(&self.dy_db, &other.dy_db, &mut res.dy_db) {
            *y = a + b;
        }

        // dy_dw_y
        res.dy_dw_y = &self.dy_dw_y + &other.dy_dw_y;

        // dy_db_y
        res.dy_db_y = &self.dy_db_y + &other.dy_db_y;

        res
    }

    #[inline]
    pub fn sub(&self, other: &Self) -> Self {
        let mut res = Self::new();

        // dy_dw
        for (a, b, y) in izip!(&self.dy_dw, &other.dy_dw, &mut res.dy_dw) {
            *y = a - b;
        }

        // dy_db
        for (a, b, y) in izip!(&self.dy_db, &other.dy_db, &mut res.dy_db) {
            *y = a - b;
        }

        // dy_dw_y
        res.dy_dw_y = &self.dy_dw_y - &other.dy_dw_y;

        // dy_db_y
        res.dy_db_y = &self.dy_db_y - &other.dy_db_y;

        res
    }
}

impl<const SIZE: usize, const NR_NEURONS: usize, const NR_LANES: usize> std::ops::Add for GradientsSimd<SIZE, NR_NEURONS, NR_LANES> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        GradientsSimd::add(&self, &rhs)
    }
}

impl<const SIZE: usize, const NR_NEURONS: usize, const NR_LANES: usize> std::ops::Sub for GradientsSimd<SIZE, NR_NEURONS, NR_LANES> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        GradientsSimd::sub(&self, &rhs)
    }
}

impl<const SIZE: usize, const NR_NEURONS: usize, const NR_LANES: usize> std::ops::Mul<f32> for &GradientsSimd<SIZE, NR_NEURONS, NR_LANES> {
    type Output = GradientsSimd<SIZE, NR_NEURONS, NR_LANES>;

    #[inline]
    fn mul(self, rhs: f32) -> Self::Output {
        GradientsSimd::multiply_constant(self, rhs)
    }
}

impl<const SIZE: usize, const NR_NEURONS: usize, const NR_LANES: usize> std::ops::AddAssign for GradientsSimd<SIZE, NR_NEURONS, NR_LANES> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = GradientsSimd::add(self, &rhs)
    }
}

impl<const SIZE: usize, const NR_NEURONS: usize, const NR_LANES: usize> std::ops::SubAssign for GradientsSimd<SIZE, NR_NEURONS, NR_LANES> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = GradientsSimd::sub(self, &rhs)
    }
}
