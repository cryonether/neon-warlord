//! Tests for NeuralNetwork

use super::*;

#[test]
fn evaluate_zeros() {
    let mut nn = NeuralNetwork::new();
    // nn.x = Vec2::zeros();

    nn.w_0 = Mat2::zeros();
    nn.w_1 = Mat2::zeros();
    nn.w_2 = Mat2::zeros();
    nn.w_3 = RowVec2::zeros();

    nn.b_0 = Vec2::zeros();
    nn.b_1 = Vec2::zeros();
    nn.b_2 = Vec2::zeros();
    nn.b_3 = 0.0;

    nn.forward();
    nn.backward();

    println!("nn: {:}", nn);
}

#[test]
fn evaluate_minus_one() {
    let mut nn = NeuralNetwork::new();
    // nn.x = Vec2::zeros();

    nn.w_0 = Mat2::new(-1.0, -1.0, -1.0, -1.0);
    nn.w_1 = Mat2::new(-1.0, -1.0, -1.0, -1.0);
    nn.w_2 = Mat2::new(-1.0, -1.0, -1.0, -1.0);
    nn.w_3 = RowVec2::new(-1.0, -1.0);

    nn.b_0 = Vec2::new(-1.0, -1.0);
    nn.b_1 = Vec2::new(-1.0, -1.0);
    nn.b_2 = Vec2::new(-1.0, -1.0);
    nn.b_3 = -1.0;

    nn.forward();
    nn.backward();

    println!("nn: {:}", nn);
}

#[test]
fn evaluate() {
    let mut nn = NeuralNetwork::new();
    nn.forward();
    nn.backward();

    println!("nn: {:}", nn);
}

#[test]
fn evaluate_diff() {
    let mut nn_0 = NeuralNetwork::new();
    nn_0.forward();
    nn_0.backward();

    println!("nn_0: {:}", nn_0);
    println!("");

    let mut nn_1 = nn_0.clone();
    nn_1.w_2[0] = 0.5;
    nn_1.forward();
    nn_1.backward();

    println!("nn_1: {:}", nn_1);
}

fn assert_weight(weight: usize, index_0: usize, index_1: usize) {
    let mut nn_0 = NeuralNetwork::new();
    nn_0.forward();
    nn_0.backward();

    let mut nn_1 = nn_0.clone();
    match weight {
        0 => {
            nn_1.w_0[(index_0, index_1)] = 0.5;
        }
        1 => {
            nn_1.w_1[(index_0, index_1)] = 0.5;
        }
        2 => {
            nn_1.w_2[(index_0, index_1)] = 0.5;
        }
        3 => {
            nn_1.w_3[(index_0, index_1)] = 0.5;
        }
        _ => {panic!("index out of bounds")}
    }

    nn_1.w_2[0] = 0.5;
    nn_1.forward();
    nn_1.backward();

    let dw = match weight {
        0 => {
            nn_1.dw_0[(index_0, index_1)]
        }
        1 => {
            nn_1.dw_1[(index_0, index_1)]
        }
        2 => {
            nn_1.dw_2[(index_0, index_1)]
        }
        3 => {
            nn_1.dw_3[(index_0, index_1)]
        }
        _ => {panic!("index out of bounds")}
    };

    let dw_ = (nn_0.y - nn_1.y) / 0.5;

    if dw != dw_ {
        println!("nn_0: {:}", nn_0);
        println!("");
        println!("nn_1: {:}", nn_1);
        assert_eq!(dw, dw_);
    }
}

#[test]
fn assert_weight_all() {
    assert_weight(3, 0, 0);
    assert_weight(3, 0, 1);
    assert_weight(3, 1, 0);
    assert_weight(3, 1, 1);
}

#[test]
fn compare_with_dfdx() {

    let mut nn = NeuralNetwork::new();
    nn.x[1] = 2.0;
    nn.forward();
    nn.backward();

    println!("nn: {:}", nn);
    println!("");

    use dfdx::prelude::*;

    type Model = (
        (Linear<2, 2>, ReLU),
        (Linear<2, 2>, ReLU),
        (Linear<2, 2>, ReLU),
        (Linear<2, 1>, ReLU),
    );

    let dev = Cpu::default();
    let mut model = dev.build_module::<Model, f32>();

    model.0.0.weight = dev.ones();
    model.0.0.bias   = dev.ones();

    model.1.0.weight = dev.ones();
    model.1.0.bias   = dev.ones();

    model.2.0.weight = dev.ones();
    model.2.0.bias   = dev.ones();

    model.3.0.weight = dev.ones();
    model.3.0.bias   = dev.ones();

    let x: Tensor<Rank2<1, 2>, f32, Cpu> =
    dev.tensor([[1.0, 2.0]]);

    // let y = model.forward(x.clone());

    // // Print parameters
    println!("dfdx {{");
    println!();

    // println!("x: {:?}", x.array());

    // println!();

    // println!("w_0: {:?}", model.0.0.weight.array());
    // println!("w_1: {:?}", model.1.0.weight.array());
    // println!("w_2: {:?}", model.2.0.weight.array());
    // println!("w_3: {:?}", model.3.0.weight.array());

    // println!();

    // println!("b_0: {:?}", model.0.0.bias.array());
    // println!("b_1: {:?}", model.1.0.bias.array());
    // println!("b_2: {:?}", model.2.0.bias.array());
    // println!("b_3: {:?}", model.3.0.bias.array());

    // println!();

    // println!("y: {:?}", y.array());

    // println!();
    // println!("}}");


    let z_0 = model.0.0.forward(x.clone());
    let a_0 = model.0.1.forward(z_0.clone());

    let z_1 = model.1.0.forward(a_0.clone());
    let a_1 = model.1.1.forward(z_1.clone());

    let z_2 = model.2.0.forward(a_1.clone());
    let a_2 = model.2.1.forward(z_2.clone());

    let z_3 = model.3.0.forward(a_2.clone());
    let y = model.3.1.forward(z_3.clone());

    println!("x:   {:?}", x.array());

    println!();

    println!("w_0: {:?}", model.0.0.weight.array());
    println!("w_1: {:?}", model.1.0.weight.array());
    println!("w_2: {:?}", model.2.0.weight.array());
    println!("w_3: {:?}", model.3.0.weight.array());

    println!();

    println!("b_0: {:?}", model.0.0.bias.array());
    println!("b_1: {:?}", model.1.0.bias.array());
    println!("b_2: {:?}", model.2.0.bias.array());
    println!("b_3: {:?}", model.3.0.bias.array());

    println!();

    println!("z_0: {:?}", z_0.array());
    println!("z_1: {:?}", z_1.array());
    println!("z_2: {:?}", z_2.array());
    println!("z_3: {:?}", z_3.array());

    println!();

    println!("a_0: {:?}", a_0.array());
    println!("a_1: {:?}", a_1.array());
    println!("a_2: {:?}", a_2.array());

    println!();

    println!("y: {:?}", y.array());

    println!();
    println!("}}");


    // Scalar loss.
    //
    // For a simple comparison with your implementation,
    // sum the output so dy/dy = 1.
    // let loss = y.sum::<(), _>();

    let mut grads = model.alloc_grads();

    let y = model.forward_mut(x.traced(grads));
    let loss = y.sum();

    grads = loss.backward();


    // Backward
    // grads = loss.backward();

    // Print gradients
    println!("dw_0: {:?}", grads.get(&model.0.0.weight).array());
    println!("dw_1: {:?}", grads.get(&model.1.0.weight).array());
    println!("dw_2: {:?}", grads.get(&model.2.0.weight).array());
    println!("dw_3: {:?}", grads.get(&model.3.0.weight).array());

    println!("");

    // And biases, if useful:
    println!("db_0: {:?}", grads.get(&model.0.0.bias).array());
    println!("db_1: {:?}", grads.get(&model.1.0.bias).array());
    println!("db_2: {:?}", grads.get(&model.2.0.bias).array());
    println!("db_3: {:?}", grads.get(&model.3.0.bias).array());
}
