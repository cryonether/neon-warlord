//! Simple example for supervised learning

use super::*;
use dfdx::prelude::*;

type Model = (
    (Linear<2, 2>, ReLU),
    (Linear<2, 2>, ReLU),
    (Linear<2, 2>, ReLU),
    (Linear<2, 1>, ReLU),
);

#[test]
fn dfdx_and() {

    let x_data = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ];

    let y_data = [
        [0.0],
        [0.0],
        [0.0],
        [1.0],
    ];

    let dev = Cpu::default();

    let x = dev.tensor_from_vec(
        x_data.iter().flatten().copied().collect(),
        (4, Const::<2>),
    );

    let y = dev.tensor_from_vec(
        y_data.iter().flatten().copied().collect(),
        (4, Const::<1>),
    );

    // model

    let mut model = dev.build_module::<Model, f32>();

    let mut grads = model.alloc_grads();

    // prediction

    let y_pred = model.forward_mut(x.clone().traced(grads));

    println!("target:     {:?}", y.as_vec());
    println!("prediction: {:?}", y_pred.as_vec());

    // mean square error

    let diff = y_pred - y.clone();
    let loss = diff.square().mean::<Rank0, _>();

    println!("loss: {:?}", loss.array());

    // backward
    let grads = loss.backward();

    // gradients
    println!("gradients: {:?}", grads);

    // optimizer
    // we just use plain gradient descent
    // w_new = w_old - eta * dw
    const LEARNING_RATE: f32 = 0.1;

    let dw_0 = grads.get(&model.0.0.weight);
    let db_0 = grads.get(&model.0.0.bias);

    let dw_1 = grads.get(&model.1.0.weight);
    let db_1 = grads.get(&model.1.0.bias);

    let dw_2 = grads.get(&model.2.0.weight);
    let db_2 = grads.get(&model.2.0.bias);

    let dw_3 = grads.get(&model.3.0.weight);
    let db_3 = grads.get(&model.3.0.bias);

    model.0.0.weight = model.0.0.weight.clone() - dw_0 * LEARNING_RATE;
    model.0.0.bias   = model.0.0.bias.clone()   - db_0 * LEARNING_RATE;

    model.1.0.weight = model.1.0.weight.clone() - dw_1 * LEARNING_RATE;
    model.1.0.bias   = model.1.0.bias.clone()   - db_1 * LEARNING_RATE;

    model.2.0.weight = model.2.0.weight.clone() - dw_2 * LEARNING_RATE;
    model.2.0.bias   = model.2.0.bias.clone()   - db_2 * LEARNING_RATE;

    model.3.0.weight = model.3.0.weight.clone() - dw_3 * LEARNING_RATE;
    model.3.0.bias   = model.3.0.bias.clone()   - db_3 * LEARNING_RATE;


    // second forward pass
    let y_pred = model.forward(x.clone());

    let loss = (y_pred.clone() - y.clone())
        .square()
        .mean::<Rank0, _>();

    println!("prediction after: {:?}", y_pred.as_vec());
    println!("loss after: {:?}", loss.array());



    for epoch in 0..500 {
        // -------------------------
        // Forward
        // -------------------------

        let grads = model.alloc_grads();

        let y_pred = model.forward_mut(x.clone().traced(grads));

        // -------------------------
        // Loss
        // -------------------------

        if epoch % 10 == 0 {
            println!("target:     {:?}", y.as_vec());
            println!("prediction: {:?}", y_pred.as_vec());
        }

        let diff = y_pred - y.clone();

        let loss = diff
            .square()
            .mean::<Rank0, _>();

        // -------------------------
        // Progress
        // -------------------------

        if epoch % 10 == 0 {
            println!(
                "epoch {:5}  loss {:.6}",
                epoch,
                loss.array()
            );
        }

        // -------------------------
        // Backward
        // -------------------------

        let grads = loss.backward();

        // -------------------------
        // Gradient descent
        // -------------------------

        let dw_0 = grads.get(&model.0.0.weight);
        let db_0 = grads.get(&model.0.0.bias);

        let dw_1 = grads.get(&model.1.0.weight);
        let db_1 = grads.get(&model.1.0.bias);

        let dw_2 = grads.get(&model.2.0.weight);
        let db_2 = grads.get(&model.2.0.bias);

        let dw_3 = grads.get(&model.3.0.weight);
        let db_3 = grads.get(&model.3.0.bias);

        model.0.0.weight = model.0.0.weight.clone()
            - dw_0 * LEARNING_RATE;

        model.0.0.bias = model.0.0.bias.clone()
            - db_0 * LEARNING_RATE;

        model.1.0.weight = model.1.0.weight.clone()
            - dw_1 * LEARNING_RATE;

        model.1.0.bias = model.1.0.bias.clone()
            - db_1 * LEARNING_RATE;

        model.2.0.weight = model.2.0.weight.clone()
            - dw_2 * LEARNING_RATE;

        model.2.0.bias = model.2.0.bias.clone()
            - db_2 * LEARNING_RATE;

        model.3.0.weight = model.3.0.weight.clone()
            - dw_3 * LEARNING_RATE;

        model.3.0.bias = model.3.0.bias.clone()
            - db_3 * LEARNING_RATE;

    }


}