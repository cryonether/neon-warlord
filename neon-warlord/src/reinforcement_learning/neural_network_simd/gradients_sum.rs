//! Gradients of NeuralNetworkSimd

use std::iter::zip;

use itertools::izip;
use wide::f32x16;


use crate::reinforcement_learning::neural_network_simd::GradientsRef;

use super::SVec;
use super::SMat;

pub struct GradientsSum<const SIZE: usize, const N: usize, const L: usize> {
    pub dl_dw: [SMat<N, L>; SIZE],
    pub dl_db: [SVec<N, L>; SIZE],

    pub dl_dw_y: SMat<N, L>,
    pub dl_db_y: SVec<N, L>,
}

impl<const SIZE: usize, const N: usize, const L: usize> GradientsSum<SIZE, N, L> {
    pub fn new() -> Self {
        let dl_dw = [SMat::new([[0.0; N]; N]); SIZE];
        let dl_db = [SVec::new([0.0; N]); SIZE];

        let dl_dw_y = SMat::new([[0.0; N]; N]);
        let dl_db_y = SVec::new([0.0; N]);

        Self {
            dl_dw,
            dl_db,
            dl_dw_y,
            dl_db_y,
        }
    }

    // 220 ups -> 630 ups
    #[inline(never)]
    pub fn add_loss_gradients(&mut self, gradients_dy: &GradientsRef<SIZE, N, L>, d_loss_dy: f32)
    {
        let d_loss_dy = f32x16::splat(d_loss_dy);

        // dl_dw
        for (dl_dw, dy_dw) in zip(&mut self.dl_dw, gradients_dy.dy_dw) {
            for (dl_dw, dy_dw) in zip(&mut dl_dw.m, &dy_dw.m) {
                for (dl_dw, dy_dw) in zip(dl_dw, dy_dw) {
                    *dl_dw += dy_dw * d_loss_dy;
                }
            }
        }

        // dl_db
        for (dl_db, dy_db) in zip(&mut self.dl_db, gradients_dy.dy_db) {
            for (dl_db, dy_db) in zip(&mut dl_db.a, &dy_db.a) {
                *dl_db += dy_db * d_loss_dy;
            }
        }

        // dl_dw_y
        for (dl_dw_y, dy_dw_y) in zip(&mut self.dl_dw_y.m, &gradients_dy.dy_dw_y.m) {
            for (dl_dw_y, dy_dw_y) in zip( dl_dw_y, dy_dw_y) {
                *dl_dw_y += dy_dw_y * d_loss_dy;
            }
        }

        // dl_db_y
        for (dl_db_y, dy_db_y) in zip(&mut self.dl_db_y.a, &gradients_dy.dy_db_y.a) {
            *dl_db_y += dy_db_y * d_loss_dy;
        }
    }

    // // 630 ups -> 655 ups
    // #[inline(never)]
    // pub fn add_loss_gradients(
    //     &mut self,
    //     gradients_dy: &GradientsRef<SIZE, N, L>,
    //     d_loss_dy: f32,
    // ) {
    //     let scale = f32x16::splat(d_loss_dy);

    //     for i in 0..SIZE {
    //         let dst = &mut self.dl_dw[i].m;
    //         let src = &gradients_dy.dy_dw[i].m;

    //         for j in 0..N {
    //             for k in 0..L {
    //                 dst[j][k] += src[j][k] * scale;
    //             }
    //         }

    //         for j in 0..L {
    //             self.dl_db[i].a[j] += gradients_dy.dy_db[i].a[j] * scale;
    //         }
    //     }

    //     for j in 0..N {
    //         for k in 0..L {
    //             self.dl_dw_y.m[j][k] +=
    //                 gradients_dy.dy_dw_y.m[j][k] * scale;
    //         }
    //     }

    //     for j in 0..L {
    //         self.dl_db_y.a[j] += gradients_dy.dy_db_y.a[j] * scale;
    //     }
    // }


    #[inline]
    pub fn multiply_constant(&self, val: f32) -> Self {
        let mut res = Self::new();

        // dy_dw
        for (x, y) in zip(self.dl_dw, &mut res.dl_dw) {
            for (x, y) in zip(x.as_array(), y.as_mut_array()) {
                for (x, y) in zip(x, y) {
                    *y = x * val;
                }
            }
        }

        // dy_db
        for (x, y) in zip(&self.dl_db, &mut res.dl_db) {
            for (x, y) in zip(x.as_array(), y.as_mut_array()) {
                *y = x * val;
            }
        }

        // dy_dw_y
        for (x, y) in zip(self.dl_dw_y.as_array(), res.dl_dw_y.as_mut_array()) {
                for (x, y) in zip(x, y) {
                    *y = x * val;
                }
        }

        // dy_db_y
        for (x, y) in zip(self.dl_db_y.as_array(), res.dl_db_y.as_mut_array()) {
            *y = x * val;
        }

        res
    }

    #[inline]
    pub fn add(&self, other: &Self) -> Self {
        let mut res = Self::new();

        // dy_dw
        for (a, b, y) in izip!(&self.dl_dw, &other.dl_dw, &mut res.dl_dw) {
            *y = a + b;
        }

        // dy_db
        for (a, b, y) in izip!(&self.dl_db, &other.dl_db, &mut res.dl_db) {
            *y = a + b;
        }

        // dy_dw_y
        res.dl_dw_y = &self.dl_dw_y + &other.dl_dw_y;

        // dy_db_y
        res.dl_db_y = &self.dl_db_y + &other.dl_db_y;

        res
    }

    #[inline]
    pub fn sub(&self, other: &Self) -> Self {
        let mut res = Self::new();

        // dy_dw
        for (a, b, y) in izip!(&self.dl_dw, &other.dl_dw, &mut res.dl_dw) {
            *y = a - b;
        }

        // dy_db
        for (a, b, y) in izip!(&self.dl_db, &other.dl_db, &mut res.dl_db) {
            *y = a - b;
        }

        // dy_dw_y
        res.dl_dw_y = &self.dl_dw_y - &other.dl_dw_y;

        // dy_db_y
        res.dl_db_y = &self.dl_db_y - &other.dl_db_y;

        res
    }
}

impl<const SIZE: usize, const NR_NEURONS: usize, const NR_LANES: usize> std::ops::Add for GradientsSum<SIZE, NR_NEURONS, NR_LANES> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        GradientsSum::add(&self, &rhs)
    }
}

impl<const SIZE: usize, const NR_NEURONS: usize, const NR_LANES: usize> std::ops::Sub for GradientsSum<SIZE, NR_NEURONS, NR_LANES> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        GradientsSum::sub(&self, &rhs)
    }
}

impl<const SIZE: usize, const NR_NEURONS: usize, const NR_LANES: usize> std::ops::Mul<f32> for &GradientsSum<SIZE, NR_NEURONS, NR_LANES> {
    type Output = GradientsSum<SIZE, NR_NEURONS, NR_LANES>;

    #[inline]
    fn mul(self, rhs: f32) -> Self::Output {
        GradientsSum::multiply_constant(self, rhs)
    }
}

impl<const SIZE: usize, const NR_NEURONS: usize, const NR_LANES: usize> std::ops::AddAssign for GradientsSum<SIZE, NR_NEURONS, NR_LANES> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = GradientsSum::add(self, &rhs)
    }
}

impl<const SIZE: usize, const NR_NEURONS: usize, const NR_LANES: usize> std::ops::SubAssign for GradientsSum<SIZE, NR_NEURONS, NR_LANES> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = GradientsSum::sub(self, &rhs)
    }
}
