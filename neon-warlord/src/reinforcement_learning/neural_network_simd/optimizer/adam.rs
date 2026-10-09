//! Adam: A Method for Stochastic Optimization
//! Diederik P. Kingma, Jimmy Ba, 2015
//!

use itertools::izip;
use wide::f32x16;

use crate::reinforcement_learning::neural_network_simd::simd_math::{
    AlignedVec, simd_mat::SMat16, simd_vec::SVec16,
};

/// Adaptive Moment Estimation (Adam)
///
/// An algorithm for first-order gradient-based optimization of
/// stochastic objective functions, based on adaptive estimates
/// of lower-order moments.
///
/// Combines the advantages of two recently popular methods:
///     AdaGrad (Duchi et al., 2011): works well with sparse gradients
///     RMSProp (Tieleman & Hinton, 2012): works well in on-line and non-stationary settings
///
#[derive(Clone)]
pub struct Adam<const INPUTS: usize, const OUTPUTS: usize> {
    /// Step size
    alpha: f32,

    /// Exponential decay rates for the moment estimates
    beta_1: f32,
    beta_2: f32,

    /// Prevents division by zero
    epsilon: f32,

    /// First moment vector
    m_w: SMat16<OUTPUTS, INPUTS>,
    m_b: SVec16<OUTPUTS>,

    /// Second moment vector
    v_w: SMat16<OUTPUTS, INPUTS>,
    v_b: SVec16<OUTPUTS>,

    beta_1_pow_t: f32,
    beta_2_pow_t: f32,
}

impl<const INPUTS: usize, const OUTPUTS: usize> Adam<INPUTS, OUTPUTS> {
    pub fn new() -> Self {
        let alpha = 0.001;
        let beta_1 = 0.9;
        let beta_2 = 0.999;
        let epsilon = 1e-8;

        let m_w = SMat16::zero();
        let m_b = SVec16::zero();
        let v_w = SMat16::zero();
        let v_b = SVec16::zero();

        let beta_1_pow_t = beta_1;
        let beta_2_pow_t = beta_2;

        Self {
            alpha,
            beta_1,
            beta_2,
            epsilon,
            m_w,
            m_b,
            v_w,
            v_b,
            beta_1_pow_t,
            beta_2_pow_t,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn do_step<const N: usize>(
        theta: &mut AlignedVec<N>,
        d_theta: &mut AlignedVec<N>,
        m: &mut AlignedVec<N>,
        v: &mut AlignedVec<N>,
        alpha: f32,
        beta_1: f32,
        beta_2: f32,
        epsilon: f32,
        beta_1_pow_t: f32,
        beta_2_pow_t: f32,
    ) {
        let one_ = f32x16::splat(1.0);
        let alpha_ = f32x16::splat(alpha);
        let beta_1_ = f32x16::splat(beta_1);
        let beta_2_ = f32x16::splat(beta_2);
        let epsilon_ = f32x16::splat(epsilon);
        let beta_1_pow_t_ = f32x16::splat(beta_1_pow_t);
        let beta_2_pow_t_ = f32x16::splat(beta_2_pow_t);

        let one = 1.0;

        for (theta_, d_theta_, m, v) in izip!(
            theta.simd_iter_mut(),
            d_theta.simd_iter_mut(),
            m.simd_iter_mut(),
            v.simd_iter_mut()
        ) {
            // mt ← β1 · mt−1 + (1 − β1) · gt (Update biased first moment estimate)
            *m = beta_1 * *m + (one_ - beta_1_) * *d_theta_;

            // vt ← β2 · vt−1 + (1 − β2) · g2t (Update biased second raw moment estimate)
            *v = beta_2 * *v + (one_ - beta_2_) * *d_theta_ * *d_theta_;

            // mt ← mt/(1 − βt1) (Compute bias-corrected first moment estimate)
            let m_ = *m / (one_ - beta_1_pow_t_);

            // vt ← vt/(1 − βt2) (Compute bias-corrected second raw moment estimate)
            let v_ = *v / (one_ - beta_2_pow_t_);

            // θt ← θt−1 − α ·̂ mt/(√̂ vt + epsilon) (Update parameters)
            *theta_ -= alpha_ * m_ / (v_.sqrt() + epsilon_);
        }

        for (theta, d_theta, m, v) in izip!(
            theta.remainder_mut(),
            d_theta.remainder_mut(),
            m.remainder_mut(),
            v.remainder_mut()
        ) {
            // mt ← β1 · mt−1 + (1 − β1) · gt (Update biased first moment estimate)
            *m = beta_1 * *m + (one - beta_1) * *d_theta;

            // vt ← β2 · vt−1 + (1 − β2) · g2t (Update biased second raw moment estimate)
            *v = beta_2 * *v + (one - beta_2) * *d_theta * *d_theta;

            // mt ← mt/(1 − βt1) (Compute bias-corrected first moment estimate)
            let m_ = *m / (one - beta_1_pow_t);

            // vt ← vt/(1 − βt2) (Compute bias-corrected second raw moment estimate)
            let v_ = *v / (one - beta_2_pow_t);

            // θt ← θt−1 − α ·̂ mt/(√̂ vt + epsilon) (Update parameters)
            *theta -= alpha * m_ / (v_.sqrt() + epsilon);
        }
    }

    pub fn step(
        &mut self,
        w: &mut SMat16<OUTPUTS, INPUTS>,
        b: &mut SVec16<OUTPUTS>,
        d_w: &mut SMat16<OUTPUTS, INPUTS>,
        d_b: &mut SVec16<OUTPUTS>,
    ) {
        for (w, d_w, m, v) in izip!(w, d_w, &mut self.m_w, &mut self.v_w) {
            Self::do_step(
                w,
                d_w,
                m,
                v,
                self.alpha,
                self.beta_1,
                self.beta_2,
                self.epsilon,
                self.beta_1_pow_t,
                self.beta_2_pow_t,
            );
        }

        Self::do_step(
            b,
            d_b,
            &mut self.m_b,
            &mut self.v_b,
            self.alpha,
            self.beta_1,
            self.beta_2,
            self.epsilon,
            self.beta_1_pow_t,
            self.beta_2_pow_t,
        );

        self.beta_1_pow_t *= self.beta_1;
        self.beta_2_pow_t *= self.beta_2;
    }
}
