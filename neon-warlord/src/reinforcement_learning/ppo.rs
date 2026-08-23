//! Partial implementation of a Proximal Policy Optimization (PPO) algorithm
//!
//! Details:
//!
//! PPO (Proximal Policy Optimization) is a reinforcement learning algorithm that 
//! improves a policy by rewarding good actions while limiting how much the policy 
//! can change in each update, making learning more stable and reliable.
//! 

use wide::f32x16;

const LANES: usize = 16;

pub struct PPO {

    inputs: Vec<f32>,

    /// Weights are stored transposed: 
    /// 
    /// weights[input * 16 + output] 
    /// 
    /// So weights for input 0 are: 
    /// [w00, w01, w02, ..., w0,15] 
    ///
    /// weights for input 1 are: 
    /// [w10, w11, w12, ..., w1,15]
    weights: Vec<f32>,

    /// One 16-float vector per layer.
    nodes: Vec<f32>,

    /// Same layout as `weights`.
    ///
    /// gradients[input * 16 + output]
    gradients: Vec<f32>,
}

impl PPO {
    pub fn forward_propagate(&mut self) {
        assert!(self.nodes.len().is_multiple_of(LANES));
        assert!(self.weights.len().is_multiple_of(LANES*LANES));
        assert!(self.inputs.len() == LANES);

        let mut input_ = f32x16::from(&self.inputs[..]);

        for(weights, nodes) in itertools::izip!(
            self.weights.chunks_exact(LANES*LANES),
            self.nodes.chunks_exact_mut(LANES),
        ) {
            let mut output_ = f32x16::splat(0.0);

            for weights in weights.chunks_exact(LANES) {
                let weights_ = f32x16::from(weights);

                output_ += input_ * weights_; 
            }

            output_ = Self::relu(output_);

            nodes.copy_from_slice(output_.as_array());
            input_ = output_;
        }
    }   


    ///
    ///     weights[input * 16 + output]
    /// 
    ///     y[o] = Σ_i x[i] * W[i,o]
    /// 
    ///     dX[i] = Σ_o dY[o] * W[i,o]
    /// 
    /// 
    pub fn backward_propagate(&mut self, expected_output: &[f32]) {
        assert!(self.nodes.len().is_multiple_of(LANES));
        assert!(self.weights.len().is_multiple_of(LANES*LANES));
        assert!(self.inputs.len() == LANES);
        assert!(expected_output.len() == LANES);
        assert_eq!(self.gradients.len(), self.weights.len());

        let layer_count = self.nodes.len() / LANES;

        assert_eq!(
            self.weights.len(),
            layer_count * LANES * LANES
        );

        // //
        // // Start with dL/dOutput.
        // //
        // // Loss:
        // //
        // //   L = 1/2 * Σ(y - expected)^2
        // //
        // // Therefore:
        // //
        // //   dL/dy = y - expected
        // //
        // let output = f32x16::from(
        //     &self.nodes[(layer_count - 1) * LANES..]
        // );

        // let expected = f32x16::from(expected_output);

        // let mut gradient = output - expected;

        // //
        // // Walk layers backwards.
        // //
        // for layer in (0..layer_count).rev() {
        //     let weight_offset = layer * LANES * LANES;

        //     let weights = &self.weights[
        //         weight_offset..weight_offset + LANES * LANES
        //     ];

        //     let gradients = &mut self.gradients[
        //         weight_offset..weight_offset + LANES * LANES
        //     ];

        //     //
        //     // Input to this layer.
        //     //
        //     let input = if layer == 0 {
        //         f32x16::from(&self.inputs[..])
        //     } else {
        //         f32x16::from(
        //             &self.nodes[
        //                 (layer - 1) * LANES
        //                     ..layer * LANES
        //             ]
        //         )
        //     };

        //     //
        //     // dL/dW[i,o] =
        //     //
        //     //     input[i] * dL/dY[o]
        //     //
        //     //
        //     // Since each weight row contains all 16 outputs,
        //     // this is one SIMD multiply per input.
        //     //
        //     for i in 0..LANES {
        //         let row_start = i * LANES;
        //         let row_end = row_start + LANES;

        //         let weight_gradient =
        //             input * gradient;

        //         gradients[row_start..row_end]
        //             .copy_from_slice(weight_gradient.as_array());
        //     }

        //     //
        //     // Calculate dL/dInput.
        //     //
        //     // dX[i] = Σ_o dY[o] * W[i,o]
        //     //
        //     // Each row is exactly one dot product.
        //     //
        //     let mut previous_gradient =
        //         f32x16::splat(0.0);

        //     for i in 0..LANES {
        //         let row_start = i * LANES;
        //         let row_end = row_start + LANES;

        //         let weights =
        //             f32x16::from(&weights[row_start..row_end]);

        //         let value =
        //             (weights * gradient).reduce_add();

        //         previous_gradient[i] = value;
        //     }

        //     //
        //     // Backpropagate through ReLU.
        //     //
        //     if layer > 0 {
        //         let previous_nodes =
        //             f32x16::from(
        //                 &self.nodes[
        //                     (layer - 1) * LANES
        //                         ..layer * LANES
        //                 ]
        //             );

        //         gradient =
        //             previous_gradient
        //                 * Self::relu_gradient(previous_nodes);
        //     }
        // }
    }

    #[inline]
    fn relu(x: f32x16) -> f32x16 {
        x.max(f32x16::splat(0.0))
    }

    #[inline]
    fn relu_gradient(x: f32x16) -> f32x16 {
        // 1 where x > 0, otherwise 0.
        x.simd_gt(f32x16::splat(0.0))
            .blend(
                f32x16::splat(1.0),
                f32x16::splat(0.0),
            )
    }
}

