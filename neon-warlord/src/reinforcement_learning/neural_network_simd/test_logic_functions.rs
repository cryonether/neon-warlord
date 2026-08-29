//! Tries to approximate logic function

use crate::reinforcement_learning::{neural_network::epoch::NeuralNetworkEpoch, neural_network_simd::epoch::EpochSimd};

use super::*;

#[test]
fn model_and() {
    predict_logic(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        [
            [0.0],
            [0.0],
            [0.0],
            [1.0],
        ],
    )
}

#[test]
fn model_or() {
    predict_logic(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        [
            [0.0],
            [1.0],
            [1.0],
            [1.0],
        ],
    )
}

#[test]
fn model_not() {
    predict_logic(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        [
            [1.0],
            [1.0],
            [0.0],
            [0.0],
        ],
    )
}

#[test]
fn model_x0r() {
    predict_logic(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        [
            [0.0],
            [1.0],
            [1.0],
            [0.0],
        ],
    )
}

#[test]
fn model_nand() {
    predict_logic(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        [
            [1.0],
            [0.0],
            [0.0],
            [0.0],
        ],
    )
}

fn predict_logic(x_data: [[f32; 2]; 4], y_data: [[f32; 1]; 4])
{
    let mut model: EpochSimd<3> = EpochSimd::new();

    // println!("model: {}", model.model);

    let mut y_pred = [0.0, 0.0, 0.0, 0.0];

    for epoch in 0..500 {
        y_pred = model.learn(x_data, y_data);

        if epoch % 10 == 0 {
            println!("epoch: {}, target: {:?}, prediction: {:?}, loss: {}", 
                epoch,
                y_data,
                y_pred,
                model.loss,
            );
        }
    }

    // println!("model: {}", model.model);

    assert_f32_eq(y_pred[0], y_data[0][0], 0.1);
    assert_f32_eq(y_pred[1], y_data[1][0], 0.1);
    assert_f32_eq(y_pred[2], y_data[2][0], 0.1);
    assert_f32_eq(y_pred[3], y_data[3][0], 0.1);
}


fn assert_f32_eq(a: f32, b: f32, epsilon: f32) {
    assert!(
        (a - b).abs() < epsilon,
        "expected {a} ≈ {b}, difference = {}",
        (a - b).abs()
    );
}