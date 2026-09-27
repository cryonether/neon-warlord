
//! A layer of a neural network

use crate::reinforcement_learning::neural_network_simd::simd_math::SVec16;

const LANES: usize = 16;

/// A simd layer
/// Inputs and outputs are multiple of 16
pub struct LayerSimd<
    const INPUTS_16: usize,
    const OUTPUTS_16: usize,
    const ACTIVATION: bool,
    const RESIDUAL: bool
>
{
    x: SVec16<INPUTS_16>
}
