//! Supervised learning of an universal function approximator using dfdx

#[cfg(test)]
mod tests;

use dfdx::prelude::*;

type Model = (
    (Linear<2, 2>, ReLU),
    (Linear<2, 2>, ReLU),
    (Linear<2, 2>, ReLU),
    (Linear<2, 1>),
);

pub struct SupervisedModelDfDx {
    dev: Cpu,

    model: (
        (modules::Linear<2, 2, f32, Cpu>, ReLU), 
        (modules::Linear<2, 2, f32, Cpu>, ReLU), 
        (modules::Linear<2, 2, f32, Cpu>, ReLU), 
         modules::Linear<2, 1, f32, Cpu>
    ),

    pub loss: f32,
}

impl SupervisedModelDfDx {
    pub fn new() -> Self {
        // let dev = Cpu::default();
        let dev = Cpu::seed_from_u64(fastrand::u64(..));
        let model = dev.build_module::<Model, f32>();
        // let grads = model.alloc_grads();

        // model.0.0.weight = dev.ones();
        // model.0.0.bias   = dev.ones();

        // model.1.0.weight = dev.ones();
        // model.1.0.bias   = dev.ones();

        // model.2.0.weight = dev.ones();
        // model.2.0.bias   = dev.ones();

        // model.3.0.weight = dev.ones();
        // model.3.0.bias   = dev.ones();

        // model.0.0.bias = dev.zeros();
        // model.1.0.bias = dev.zeros();
        // model.2.0.bias = dev.zeros();
        // model.3.bias   = dev.zeros();

        println!("w_0: {:?}", model.0.0.weight.array());
        println!("w_1: {:?}", model.1.0.weight.array());
        println!("w_2: {:?}", model.2.0.weight.array());
        println!("w_3: {:?}", model.3.weight.array());
        println!();

        println!("b_0: {:?}", model.0.0.bias.array());
        println!("b_1: {:?}", model.1.0.bias.array());
        println!("b_2: {:?}", model.2.0.bias.array());
        println!("b_3: {:?}", model.3.bias.array());
        println!();

        Self { 
            dev,
            model,
            loss: 1.0,
         }
    }

    pub fn learn(&mut self, input: [[f32; 2]; 4], output: [[f32; 1]; 4]) -> [f32; 4]
    {
        // prediction
        let x = self.dev.tensor([input]);
        let y = self.dev.tensor([output]);
        let grads = self.model.alloc_grads();
        let y_pred = self.model.forward_mut(x.traced(grads));

        let res = y_pred.as_vec();

        // mean square error
        let diff = y_pred - y;
        let loss = diff.square().mean::<Rank0, _>();
        self.loss = loss.array();

        // gradients
        let grads: Gradients<f32, Cpu> = loss.backward();

        // optimize
        self.optimize(&grads);

        [res[0], res[1], res[2], res[3]]
    }

    fn optimize(&mut self, grads: &Gradients<f32, Cpu>)
    {
        let model = &mut self.model;

        // optimizer
        // plain gradient descent
        // w_new = w_old - eta * dw
        const LEARNING_RATE: f32 = 0.1;

        let dw_0 = grads.get(&model.0.0.weight);
        let db_0 = grads.get(&model.0.0.bias);

        let dw_1 = grads.get(&model.1.0.weight);
        let db_1 = grads.get(&model.1.0.bias);

        let dw_2 = grads.get(&model.2.0.weight);
        let db_2 = grads.get(&model.2.0.bias);

        let dw_3 = grads.get(&model.3.weight);
        let db_3 = grads.get(&model.3.bias);

        model.0.0.weight = model.0.0.weight.clone() - dw_0 * LEARNING_RATE;
        model.0.0.bias   = model.0.0.bias.clone()   - db_0 * LEARNING_RATE;

        model.1.0.weight = model.1.0.weight.clone() - dw_1 * LEARNING_RATE;
        model.1.0.bias   = model.1.0.bias.clone()   - db_1 * LEARNING_RATE;

        model.2.0.weight = model.2.0.weight.clone() - dw_2 * LEARNING_RATE;
        model.2.0.bias   = model.2.0.bias.clone()   - db_2 * LEARNING_RATE;

        model.3.weight = model.3.weight.clone() - dw_3 * LEARNING_RATE;
        model.3.bias   = model.3.bias.clone()   - db_3 * LEARNING_RATE;
    }
}