//! Tests for NeuralNetworkSimd

use crate::reinforcement_learning::neural_network::NeuralNetwork;

use super::*;

use dfdx::prelude::*;

type Model = (
    (Linear<16, 16>, ReLU),
    (Linear<16, 16>, ReLU),
    (Linear<16, 16>, ReLU),
    Linear<16, 1>,
);

type ModelType = (
    (modules::Linear<16, 16, f32, Cpu>, ReLU),
    (modules::Linear<16, 16, f32, Cpu>, ReLU),
    (modules::Linear<16, 16, f32, Cpu>, ReLU),
    modules::Linear<16, 1, f32, Cpu>,
);

#[test]
fn compare() {
    let mut nn_0 = NeuralNetwork::new();
    nn_0.x = [1.0, 2.0].into();

    nn_0.w_0 = [[1.0, 1.0], [1.0, 1.0]].into();
    nn_0.w_1 = [[1.0, 1.0], [1.0, 1.0]].into();
    nn_0.w_2 = [[1.0, 1.0], [1.0, 1.0]].into();
    nn_0.w_3 = [[1.0, 1.0], [1.0, 1.0]].into();

    nn_0.b_0 = [1.0, 1.0].into();
    nn_0.b_1 = [1.0, 1.0].into();
    nn_0.b_2 = [1.0, 1.0].into();
    nn_0.b_3 = [1.0, 1.0].into();

    nn_0.forward();
    nn_0.backward(0);

    let mut nn_1: NeuralNetworkSimd<3> = NeuralNetworkSimd::new();

    nn_1.x = [
        1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];

    nn_1.w[0][0] = [
        1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    nn_1.w[0][1] = [
        1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];

    nn_1.w[1][0] = [
        1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    nn_1.w[1][1] = [
        1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];

    nn_1.w[2][0] = [
        1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    nn_1.w[2][1] = [
        1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];

    nn_1.w_y[0] = [
        1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    nn_1.w_y[1] = [
        1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];

    nn_1.b[0] = [
        1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    nn_1.b[1] = [
        1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    nn_1.b[2] = [
        1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];

    nn_1.b_y = [
        1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];

    nn_1.forward();
    nn_1.backward(0);

    println!("nn_0: {:}", nn_0);
    println!("nn_1: {:}", nn_1);

    assert_eq!(nn_0.y[0], nn_1.y[0]);

    assert_eq!(nn_0.dy_db0[0], nn_1.dy_db[0][0]);
    assert_eq!(nn_0.dy_db0[1], nn_1.dy_db[0][1]);

    assert_eq!(nn_0.dy_db1[0], nn_1.dy_db[1][0]);
    assert_eq!(nn_0.dy_db1[1], nn_1.dy_db[1][1]);

    assert_eq!(nn_0.dy_db2[0], nn_1.dy_db[2][0]);
    assert_eq!(nn_0.dy_db2[1], nn_1.dy_db[2][1]);

    assert_eq!(nn_0.dy_db3[0], nn_1.dy_db_y[0]);
    assert_eq!(nn_0.dy_db3[1], nn_1.dy_db_y[1]);

    assert_eq!(nn_0.dy_dw0[0], nn_1.dy_dw[0][0][0]);
    assert_eq!(nn_0.dy_dw0[1], nn_1.dy_dw[0][0][1]);
    assert_eq!(nn_0.dy_dw0[2], nn_1.dy_dw[0][1][0]);
    assert_eq!(nn_0.dy_dw0[3], nn_1.dy_dw[0][1][1]);

    assert_eq!(nn_0.dy_dw1[0], nn_1.dy_dw[1][0][0]);
    assert_eq!(nn_0.dy_dw1[1], nn_1.dy_dw[1][0][1]);
    assert_eq!(nn_0.dy_dw1[2], nn_1.dy_dw[1][1][0]);
    assert_eq!(nn_0.dy_dw1[3], nn_1.dy_dw[1][1][1]);

    assert_eq!(nn_0.dy_dw2[0], nn_1.dy_dw[2][0][0]);
    assert_eq!(nn_0.dy_dw2[1], nn_1.dy_dw[2][0][1]);
    assert_eq!(nn_0.dy_dw2[2], nn_1.dy_dw[2][1][0]);
    assert_eq!(nn_0.dy_dw2[3], nn_1.dy_dw[2][1][1]);

    assert_eq!(nn_0.dy_dw3[0], nn_1.dy_dw_y[0][0]);
    assert_eq!(nn_0.dy_dw3[1], nn_1.dy_dw_y[0][1]);
    assert_eq!(nn_0.dy_dw3[2], nn_1.dy_dw_y[1][0]);
    assert_eq!(nn_0.dy_dw3[3], nn_1.dy_dw_y[1][1]);
}

#[test]
fn compare_dfdx() {
    let mut nn_1: NeuralNetworkSimd<3> = NeuralNetworkSimd::new_zero_one();

    nn_1.x = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ];

    nn_1.forward();
    nn_1.backward(0);

    let dev = Cpu::default();
    let mut model = dev.build_module::<Model, f32>();

    model.0.0.weight = dev.ones() * 0.1;
    model.0.0.bias = dev.ones() * 0.1;

    model.1.0.weight = dev.ones() * 0.1;
    model.1.0.bias = dev.ones() * 0.1;

    model.2.0.weight = dev.ones() * 0.1;
    model.2.0.bias = dev.ones() * 0.1;

    model.3.weight = dev.ones() * 0.1;
    model.3.bias = dev.ones() * 0.1;

    let mut _grads: Gradients<f32, Cpu> = model.alloc_grads();

    let x: Tensor<Rank2<1, 16>, f32, Cpu> = dev.tensor([[
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ]]);
    let y = model.forward_mut(x.traced(_grads));
    // println!("y: {:?}", y.as_vec());
    let loss = y.sum();

    _grads = loss.backward();

    // println!("nn_1: {:}", nn_1);

    // println!("nn_2:");
    // print_model(&model);
    // print_grads(&model, &grads);
}

fn print_model(model: &ModelType) {
    for (i, w) in model.0.0.weight.array().iter().enumerate() {
        println!("w_0_{:02}: {:?}", i, w);
    }

    for (i, w) in model.1.0.weight.array().iter().enumerate() {
        println!("w_1_{:02}: {:?}", i, w);
    }

    for (i, w) in model.2.0.weight.array().iter().enumerate() {
        println!("w_2_{:02}: {:?}", i, w);
    }

    println!("w_3: {:?}", model.3.weight.array());
    println!();

    println!("b_0: {:?}", model.0.0.bias.array());
    println!("b_1: {:?}", model.1.0.bias.array());
    println!("b_2: {:?}", model.2.0.bias.array());
    println!("b_3: {:?}", model.3.bias.array());
    println!();
}

fn print_grads(model: &ModelType, grads: &Gradients<f32, Cpu>) {
    let dw_0 = grads.get(&model.0.0.weight);
    let db_0 = grads.get(&model.0.0.bias);

    let dw_1 = grads.get(&model.1.0.weight);
    let db_1 = grads.get(&model.1.0.bias);

    let dw_2 = grads.get(&model.2.0.weight);
    let db_2 = grads.get(&model.2.0.bias);

    let dw_3 = grads.get(&model.3.weight);
    let db_3 = grads.get(&model.3.bias);

    println!("###");
    println!("dw_0: {:?}", dw_0.as_vec());
    println!("");
    println!("dw_1: {:?}", dw_1.as_vec());
    println!("");
    println!("dw_2: {:?}", dw_2.as_vec());
    println!("");
    println!("dw_3: {:?}", dw_3.as_vec());
    println!("###");

    println!("db_0: {:?}", db_0.as_vec());
    println!("db_1: {:?}", db_1.as_vec());
    println!("db_2: {:?}", db_2.as_vec());
    println!("db_3: {:?}", db_3.as_vec());
}
