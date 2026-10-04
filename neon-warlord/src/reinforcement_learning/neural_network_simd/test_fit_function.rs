//! Tries to learn a continuous function using a neural network

use crate::{
    print_color::{color::PrintColor, print_color}, reinforcement_learning::neural_network_simd::epoch::EpochSimd,
};

fn reward(x: f32) -> f32 {
    f32::clamp(1.0 - x.powi(2), 0.0, 1.0)
}

#[test]
fn test_fit_function() {
    let mut epoch: EpochSimd<1, 1, 8, 1, false> = EpochSimd::new();

    const BATCH_SIZE: usize = 100;

    for _j in 0..1000 {
        let mut batch_x: [[f32; 1]; BATCH_SIZE] = [[0.0]; BATCH_SIZE];
        let mut batch_y: [[f32; 1]; BATCH_SIZE] = [[0.0]; BATCH_SIZE];
        for i in 0..BATCH_SIZE {
            let x = (((i as f32) / BATCH_SIZE as f32) * 2.0) - 1.0;
            let y = reward(x);
            batch_x[i] = [x];
            batch_y[i] = [y];
        }

        let y_est = epoch.learn(batch_x, batch_y);

        for i in (0..BATCH_SIZE).step_by(5) {
            print_color(y_est[i], 0.0, 1.0, PrintColor::PurplePinkYellow);
        }

        println!("loss: {}", epoch.loss);
    }
}
