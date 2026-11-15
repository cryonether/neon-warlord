//! Tests safetensors_exporter

use crate::reinforcement_learning::neural_network_simd::{NeuralNetworkSimd, activation_function::{ActivationFunction, activation_none::ActivationNone}, safetensors_exporter::{export_ppo_bytes, import_ppo_bytes}};

#[test]
fn test_import_export_ppo() {
    let actor_0: NeuralNetworkSimd<2, 2, 2, 2, false, ActivationNone> = NeuralNetworkSimd::new_rand(42);
    let critic_0: NeuralNetworkSimd<2, 2, 2, 2, false, ActivationNone> = NeuralNetworkSimd::new_rand(42);

    let bytes = export_ppo_bytes(&actor_0, &critic_0).unwrap();

    let mut actor_1: NeuralNetworkSimd<2, 2, 2, 2, false, ActivationNone> = NeuralNetworkSimd::new_zero_one();
    let mut critic_1: NeuralNetworkSimd<2, 2, 2, 2, false, ActivationNone> = NeuralNetworkSimd::new_zero_one();

    let _ = import_ppo_bytes(&mut actor_1, &mut critic_1, bytes).unwrap();

    assert_network_equal(&actor_0, &actor_1);
    assert_network_equal(&critic_0, &critic_1);
}

fn assert_network_equal<
    const INPUTS: usize,
    const OUTPUTS: usize,
    const NEURONS: usize,
    const NR_LAYERS: usize,
    const RESIDUAL: bool,
    A: ActivationFunction<OUTPUTS>,
>(
    a: &NeuralNetworkSimd<
        INPUTS, OUTPUTS, NEURONS, NR_LAYERS, RESIDUAL, A
    >,
    b: &NeuralNetworkSimd<
        INPUTS, OUTPUTS, NEURONS, NR_LAYERS, RESIDUAL, A
    >,
) {
    // Input layer
    assert_eq!(a.input.w, b.input.w);
    assert_eq!(a.input.b, b.input.b);

    // Hidden layers
    assert_eq!(a.layers.len(), b.layers.len());

    for (i, (la, lb)) in a.layers.iter().zip(b.layers.iter()).enumerate() {
        assert_eq!(la.w, lb.w, "Hidden layer {i} weights differ");
        assert_eq!(la.b, lb.b, "Hidden layer {i} biases differ");
    }

    // Output layer
    assert_eq!(a.output.w, b.output.w);
    assert_eq!(a.output.b, b.output.b);
}