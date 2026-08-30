//! Tries to approximate logic function

use std::iter::zip;

use crate::reinforcement_learning::neural_network_simd::epoch::EpochSimd;


fn create_input_table() -> [[f32; 6]; 9] {
    let res: [[f32; 6]; 9] = [
        [1.0, 0.0, 0.0,   1.0, 0.0, 0.0,],
        [0.0, 1.0, 0.0,   1.0, 0.0, 0.0,],
        [0.0, 0.0, 1.0,   1.0, 0.0, 0.0,],

        [1.0, 0.0, 0.0,   0.0, 1.0, 0.0,],
        [0.0, 1.0, 0.0,   0.0, 1.0, 0.0,],
        [0.0, 0.0, 1.0,   0.0, 1.0, 0.0,],

        [1.0, 0.0, 0.0,   0.0, 0.0, 1.0,],
        [0.0, 1.0, 0.0,   0.0, 0.0, 1.0,],
        [0.0, 0.0, 1.0,   0.0, 0.0, 1.0,],
    ];

    res
}

fn create_output_table(
    maze: [[i32; 3]; 3],
) -> (
    [[f32; 1]; 9],
    [[f32; 1]; 9],
    [[f32; 1]; 9],
    [[f32; 1]; 9],
) {
    let mut up = [[-1.0; 1]; 9];
    let mut right = [[-1.0; 1]; 9];
    let mut down = [[-1.0; 1]; 9];
    let mut left = [[-1.0; 1]; 9];

    for row in 0..3 {
        for col in 0..3 {
            let i = row * 3 + col;

            // Wall
            if maze[row][col] != 0 {
                continue;
            }

            // Up
            if row > 0 && maze[row - 1][col] == 0 {
                up[i] = [1.0];
            }

            // Right
            if col < 2 && maze[row][col + 1] == 0 {
                right[i] = [1.0];
            }

            // Down
            if row < 2 && maze[row + 1][col] == 0 {
                down[i] = [1.0];
            }

            // Left
            if col > 0 && maze[row][col - 1] == 0 {
                left[i] = [1.0];
            }
        }
    }

    (up, right, down, left)
}


#[rustfmt::skip]
#[test]
fn test_3x3() {
    let maze: [[i32; 3]; 3] = [
            [0, 0, 0],
            [1, 1, 0],
            [0, 0, 0],
        ];

    let x_data = create_input_table();
    let y_data = create_output_table(maze);

    predict_logic(
        x_data,
        y_data.0,
        y_data.1,
        y_data.2,
        y_data.3,
    )
}

fn predict_logic<const INPUT_SIZE: usize, const BATCH_SIZE: usize>(
    x_data: [[f32; INPUT_SIZE]; BATCH_SIZE], 
    y_data_0: [[f32; 1]; BATCH_SIZE],
    y_data_1: [[f32; 1]; BATCH_SIZE],
    y_data_2: [[f32; 1]; BATCH_SIZE],
    y_data_3: [[f32; 1]; BATCH_SIZE],
) {
    let mut model: EpochSimd<3> = EpochSimd::new();

    // println!("model: {}", model.model);

    let mut y_pred_0 = [0.0; BATCH_SIZE];
    let mut y_pred_1 = [0.0; BATCH_SIZE];
    let mut y_pred_2 = [0.0; BATCH_SIZE];
    let mut y_pred_3 = [0.0; BATCH_SIZE];

    for epoch in 0..10000 {
        y_pred_0 = model.learn_output::<INPUT_SIZE, BATCH_SIZE>(x_data, y_data_0, 0);
        y_pred_1 = model.learn_output::<INPUT_SIZE, BATCH_SIZE>(x_data, y_data_1, 1);
        y_pred_2 = model.learn_output::<INPUT_SIZE, BATCH_SIZE>(x_data, y_data_2, 2);
        y_pred_3 = model.learn_output::<INPUT_SIZE, BATCH_SIZE>(x_data, y_data_3, 3);

        if epoch % 10 == 0 {
            println!("epoch: {}, target: {:?}, prediction: {:?}, loss: {}",
                epoch,
                y_data_3,
                y_pred_3,
                model.loss,
            );
        }

        if model.loss < 0.0001 {
            break;
        }
    }

    // println!("model: {}", model.model);

    for (y_pred, y_data) in zip(y_pred_0, y_data_0) {
        assert_f32_eq(y_pred, y_data[0], 0.1);
    }

    for (y_pred, y_data) in zip(y_pred_1, y_data_1) {
        assert_f32_eq(y_pred, y_data[0], 0.1);
    }

    for (y_pred, y_data) in zip(y_pred_2, y_data_2) {
        assert_f32_eq(y_pred, y_data[0], 0.1);
    }

    for (y_pred, y_data) in zip(y_pred_3, y_data_3) {
        assert_f32_eq(y_pred, y_data[0], 0.1);
    }
}

fn assert_f32_eq(a: f32, b: f32, epsilon: f32) {
    assert!(
        (a - b).abs() < epsilon,
        "expected {a} ≈ {b}, difference = {}",
        (a - b).abs()
    );
}
