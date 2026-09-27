
//! A layer of a neural network

use crate::reinforcement_learning::neural_network_simd::simd_math::{simd_mat::SMat16, simd_vec::SVec16};


const LANES: usize = 16;

/// A simd layer
/// Inputs and outputs are multiple of 16
pub struct LayerSimd<
    const INPUTS: usize,
    const OUTPUTS: usize,
    const ACTIVATION: bool,
    const RESIDUAL: bool
>
{
    x: SVec16<INPUTS>,
    w: SMat16<INPUTS, OUTPUTS>,
    y: SVec16<OUTPUTS>,
}
