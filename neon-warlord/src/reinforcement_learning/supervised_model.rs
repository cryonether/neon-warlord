//! Supervised learning of an universal function approximator

use crate::reinforcement_learning::neural_network::NeuralNetwork;

pub struct SupervisedModel{
    model: NeuralNetwork,

    pub loss: f32,
}

impl SupervisedModel {
    pub fn new() -> Self {

        let model = NeuralNetwork::new();


        println!("w_0: {:?}", model.w_0);
        println!("w_1: {:?}", model.w_1);
        println!("w_2: {:?}", model.w_2);
        println!("w_3: {:?}", model.w_2);
        println!();

        println!("b_0: {:?}", model.b_0);
        println!("b_1: {:?}", model.b_1);
        println!("b_2: {:?}", model.b_2);
        println!("b_3: {:?}", model.b_2);
        println!();

        Self { 
            model,
            loss: 0.0, 
        }
    }

    pub fn learn(&mut self, input: [[f32; 2]; 4], output: [[f32; 1]; 4]) -> [f32; 4]
    {
        // prediction
        self.model.x = input[0].into();
        self.model.forward();

        // mean square error

        // gradient
        self.model.backward();

        // optimize

        // let x = self.dev.tensor([input]);
        // let y = self.dev.tensor([output]);
        // let grads = self.model.alloc_grads();
        // let y_pred = self.model.forward_mut(x.traced(grads));

        // let res = y_pred.as_vec();

        // // mean square error
        // let diff = y_pred - y;
        // let loss = diff.square().mean::<Rank0, _>();
        // self.loss = loss.array();

        // // gradients
        // let grads: Gradients<f32, Cpu> = loss.backward();

        // // optimize
        // self.optimize(&grads);

        [0.0, 0.0, 0.0, 0.0]
    }
}