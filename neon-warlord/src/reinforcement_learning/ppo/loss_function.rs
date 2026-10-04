//! Functions to calculate the Loss of PPO
//!

use std::iter::zip;

use itertools::izip;

// const OUTPUTS: usize = 2;

///
/// Calculates the log probability density of a multi-dimensional continuous action
/// under an isotropic (spherical) Gaussian distribution.
///
/// ### Mathematical Formula:
///
/// ```text
///               1                          D
/// log p(x) = - ---   *  Σ (x_i - μ_i)²  - --- * ln(2πσ²)  
///              2σ²                         2
/// ```
///
/// Where:
/// - x_i = Elements of the `action` vector
/// - μ_i = Elements of the `mean` vector
/// - σ (sigma) = `std_dev` (scalar shared across all dimensions)
/// - σ² (variance) = `std_dev * std_dev`
/// - D = `num_dimensions` (length of the action vector)
///
pub struct GaussianLogProbability<const OUTPUTS: usize> {
    action: [f32; OUTPUTS],
    mean_action: [f32; OUTPUTS],
    std_dev: f32,
}

impl<const OUTPUTS: usize> GaussianLogProbability<OUTPUTS> {
    pub fn new() -> Self {
        Self {
            action: [0.0; OUTPUTS],
            mean_action: [0.0; OUTPUTS],
            std_dev: 0.0,
        }
    }

    pub fn calc(
        &mut self,
        action: &[f32; OUTPUTS],
        mean_action: &[f32; OUTPUTS],
        std_dev: f32,
    ) -> f32 {
        const PI: f32 = std::f32::consts::PI;

        let variance = std_dev * std_dev;

        let mut sum_squared_diff = 0.0;
        for (action, mean_action) in zip(action, mean_action) {
            let diff = action - mean_action;
            sum_squared_diff += diff * diff;
        }

        let res = -1.0 / (2.0 * variance) * sum_squared_diff
            - action.len() as f32 / 2.0 * f32::ln(2.0 * PI * variance);

        self.action = *action;
        self.mean_action = *mean_action;
        self.std_dev = std_dev;

        res
    }

    ///
    ///  d log p(x)     x - μ
    /// ------------ = ---------
    ///  d μ              σ²
    ///
    pub fn derivative(&self) -> [f32; OUTPUTS] {
        let variance = self.std_dev * self.std_dev;

        let mut res = [0.0; OUTPUTS];

        for (action, mean_action, res) in izip!(self.action, self.mean_action, &mut res) {
            *res = (action - mean_action) / variance;
        }

        res
    }
}

pub struct GaussianLogProbabilityTanH<const OUTPUTS: usize> {
    z: [f32; OUTPUTS],
    mean_action: [f32; OUTPUTS],
    std_dev: f32,
}

impl<const OUTPUTS: usize> GaussianLogProbabilityTanH<OUTPUTS> {
    pub fn new() -> Self {
        Self {
            z: [0.0; OUTPUTS],
            mean_action: [0.0; OUTPUTS],
            std_dev: 0.0,
        }
    }

    /// Calculates:
    ///
    /// log π(a|s)
    /// =
    /// log N(z | μ, σ)
    /// -
    /// Σ log(1 - tanh(z)^2)
    ///
    /// where:
    ///
    /// a = tanh(z)
    ///
    /// `z` is the unsquashed Gaussian sample.
    pub fn calc(
        &mut self,
        z: &[f32; OUTPUTS],
        mean_action: &[f32; OUTPUTS],
        std_dev: f32,
    ) -> f32 {
        const PI: f32 = std::f32::consts::PI;
        const EPSILON: f32 = 1e-6;

        let variance = std_dev * std_dev;

        let mut log_probability = 0.0;

        for (z, mean) in zip(z, mean_action) {
            // Gaussian log probability.
            let diff = z - mean;

            log_probability +=
                -0.5 * diff * diff / variance
                -0.5 * f32::ln(2.0 * PI * variance);

            // tanh change-of-variables correction.
            let action = z.tanh();

            log_probability -=
                (1.0 - action * action + EPSILON).ln();
        }

        self.z = *z;
        self.mean_action = *mean_action;
        self.std_dev = std_dev;

        log_probability
    }

    /// Derivative of the squashed Gaussian log probability
    /// with respect to the mean μ.
    ///
    /// For:
    ///
    /// log N(z | μ, σ)
    ///
    /// we have:
    ///
    /// ∂log p / ∂μ = (z - μ) / σ²
    ///
    /// The tanh correction does not directly depend on μ when
    /// differentiating with respect to μ while z is treated
    /// as the sampled value.
    pub fn derivative(&self) -> [f32; OUTPUTS] {
        let variance = self.std_dev * self.std_dev;

        let mut res = [0.0; OUTPUTS];

        for (z, mean, res) in izip!(self.z, self.mean_action, &mut res) {
            *res = (z - mean) / variance;
        }

        res
    }
}


pub struct PpoActorRatio {
    ratio: f32,
}

impl PpoActorRatio {
    pub fn new() -> Self {
        Self { ratio: 0.0 }
    }

    /// ```text
    ///          π_θ(at∣st)
    /// r_t(θ) = ------------
    ///         π_θ_old(at∣st)
    /// ```
    ///
    /// r=e^( log⁡(π_θ) − log(⁡π_old) )
    ///
    /// Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
    /// NOTE: we just subtract the logs, which is the same as
    /// dividing the values and then canceling the log with e^log.
    /// For why we use log probabilities instead of actual probabilities,
    /// here's a great explanation:
    /// https://cs.stackexchange.com/questions/70518/why-do-we-use-the-log-in-gradient-based-reinforcement-algorithms
    /// TL;DR makes gradient ascent easier behind the scenes.
    pub fn calc(&mut self, curr_log_probability: f32, log_probability: f32) -> f32 {
        self.ratio = f32::exp(curr_log_probability - log_probability);
        self.ratio
    }

    ///
    ///   ∂r
    /// ------- = e^( log⁡(π_θ) − log(⁡π_old) )
    /// ∂log⁡(π)
    ///
    ///   ∂r
    /// ------- = r
    /// ∂log⁡(π)
    ///
    ///
    pub fn derivative(&self) -> f32 {
        self.ratio
    }
}

pub struct PpoSurrogateLossClipped {
    advantage: f32,
    use_unclipped: bool,
}

impl PpoSurrogateLossClipped {
    pub fn new() -> Self {
        Self {
            advantage: 0.0,
            use_unclipped: false,
        }
    }

    ///
    /// L_tPPO = - min⁡( r_t * A_t, clip⁡( r_t, 1−ϵ, 1+ϵ) * A_t )
    ///
    pub fn calc(&mut self, ratio: f32, advantage: f32, clip: f32) -> f32 {
        let surrogate_loss_1 = ratio * advantage;
        let surrogate_loss_2 = f32::clamp(ratio, 1.0 - clip, 1.0 + clip) * advantage;

        self.advantage = advantage;

        if surrogate_loss_1 <= surrogate_loss_2 {
            self.use_unclipped = true;
            -surrogate_loss_1
        } else {
            self.use_unclipped = false;
            -surrogate_loss_2
        }
    }

    ///
    /// Unclipped branch:
    ///  ∂Lt
    /// ----- = −A
    ///   ∂r
    ///
    /// Clipped branch:
    ///  ∂L
    /// ------ = 0
    ///  ∂r
    ///
    pub fn derivative(&self) -> f32 {
        if self.use_unclipped {
            -self.advantage
        } else {
            0.0
        }
    }
}

pub struct MeanSquareError {
    diff: f32,
}

impl MeanSquareError {
    pub fn new() -> Self {
        Self { diff: 0.0 }
    }

    //      1    N-1
    // L = --- * ∑ (y_pred_i − y_i)²
    //      N    i=0
    pub fn calc(&mut self, y_pred: [f32; 1], y: f32) -> f32 {
        let diff = y_pred[0] - y;
        let res = diff * diff;

        self.diff = diff;

        res
    }

    // ∂L           2
    // --------- = --- * (y_pred_i − y_i)
    // ∂L_pred_i    N
    pub fn derivative(&self) -> [f32; 1] {
        [2.0 * (self.diff)]
    }
}
