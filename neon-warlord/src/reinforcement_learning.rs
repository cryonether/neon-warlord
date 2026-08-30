//! Algorithms for reinforcement learning

pub mod dqn;
pub mod neat;
#[allow(dead_code)]
pub mod neural_network;
#[allow(dead_code)]
pub mod neural_network_dfdx;
#[allow(dead_code)]
pub mod neural_network_simd;
pub mod ppo;


pub fn _assert_f32_eq(a: f32, b: f32, epsilon: f32) {
    assert!(
        (a - b).abs() < epsilon,
        "expected {a} ≈ {b}, difference = {}",
        (a - b).abs()
    );
}

pub fn _assert_f32_eq_(a: f32, b: f32) {
    _assert_f32_eq(a, b, 0.1);
}