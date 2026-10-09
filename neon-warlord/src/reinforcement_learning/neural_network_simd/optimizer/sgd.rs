//! Stochastic Gradient Descend
//!

use itertools::izip;
use wide::f32x16;

use crate::reinforcement_learning::neural_network_simd::simd_math::{
    AlignedVec, simd_mat::SMat16, simd_vec::SVec16,
};

/// Stochastic Gradient Descend
#[derive(Clone)]
pub struct Sgd<const INPUTS: usize, const OUTPUTS: usize> {
    /// Step size
    alpha: f32,
}

impl<const INPUTS: usize, const OUTPUTS: usize> Sgd<INPUTS, OUTPUTS> {
    pub fn new() -> Self {

        let alpha = 0.001;

        Self {
            alpha,
        }
    }

    fn do_step<const N: usize>( 
        theta: &mut AlignedVec<N>, 
        d_theta: &mut AlignedVec<N>, 
        alpha: f32,
    ) 
    {
        let zero_ = f32x16::splat(0.0);
        let alpha_ = f32x16::splat(alpha);

        let zero = 0.0;
    
        for (theta_, d_theta_) in izip!(
            theta.simd_iter_mut(), 
            d_theta.simd_iter_mut(), 
        ) {
            *theta_ = *theta_ - alpha_ * *d_theta_;

            *d_theta_ = zero_;
        }

        for (theta, d_theta) in izip!(
            theta.remainder_mut(), 
            d_theta.remainder_mut(), 
        ) {
            *theta = *theta - alpha * *d_theta;

            *d_theta = zero;
        }
    
    }

    pub fn step(&mut self, 
        w: &mut SMat16<OUTPUTS, INPUTS>,
        b: &mut SVec16<OUTPUTS>,
        d_w: &mut SMat16<OUTPUTS, INPUTS>,
        d_b: &mut SVec16<OUTPUTS>,
    )
    {
        for (w, d_w) in izip!(w, d_w) {
           Self::do_step(w, d_w, self.alpha);
        }

        Self::do_step(b, d_b, self.alpha);
    }
}
