//! Deep Q Network (DQN)

#[cfg(test)]
mod test_maze;

use dfdx::{
    nn::{
        builders::Linear as LinearBuilder,
        modules::{Linear, ReLU},
        DeviceBuildExt,
        Module,
        ZeroGrads,
    },
    optim::{Sgd, SgdConfig},
    prelude::*,
};

/// Builder type.
///
/// dfdx uses the builder types when calling `build_module()`.
type ModelBuilder<
    const INPUTS: usize,
    const HIDDEN: usize,
    const OUTPUTS: usize,
> = (
    LinearBuilder<INPUTS, HIDDEN>,
    ReLU,
    LinearBuilder<HIDDEN, OUTPUTS>,
);

/// Actual model type stored by Dqn.
///
/// `build_module()` converts the builder type into these
/// device/dtype-specific modules.
type Model<
    const INPUTS: usize,
    const HIDDEN: usize,
    const OUTPUTS: usize,
> = (
    Linear<INPUTS, HIDDEN, f32, Cpu>,
    ReLU,
    Linear<HIDDEN, OUTPUTS, f32, Cpu>,
);

pub struct DqnDfdx<
    const INPUTS: usize,
    const HIDDEN: usize,
    const OUTPUTS: usize,
> {
    dev: Cpu,

    model: Model<INPUTS, HIDDEN, OUTPUTS>,

    optimizer: Sgd<
        Model<INPUTS, HIDDEN, OUTPUTS>,
        f32,
        Cpu,
    >,

    steps: Vec<Transition<INPUTS>>,

    epsilon: f32,
}

impl<
    const INPUTS: usize,
    const HIDDEN: usize,
    const OUTPUTS: usize,
> DqnDfdx<INPUTS, HIDDEN, OUTPUTS>
{
    pub fn new() -> Self {
        assert!(OUTPUTS > 0);

        let dev = Cpu::default();

        let model: Model<INPUTS, HIDDEN, OUTPUTS> =
            dev.build_module::<
                ModelBuilder<INPUTS, HIDDEN, OUTPUTS>,
                f32,
            >();

        let optimizer = Sgd::new(
            &model,
            SgdConfig {
                lr: 0.001,
                ..Default::default()
            },
        );

        Self {
            dev,
            model,
            optimizer,
            steps: Vec::new(),
            epsilon: 0.8,
        }
    }

    pub fn choose_action_u8(
        &mut self,
        inputs: &[u8; INPUTS],
    ) -> (usize, [f32; OUTPUTS]) {
        let inputs_f32 = inputs.map(|x| x as f32);

        self.choose_action(&inputs_f32)
    }

    pub fn choose_action(
        &mut self,
        inputs: &[f32; INPUTS],
    ) -> (usize, [f32; OUTPUTS]) {
        let input = self.dev.tensor(*inputs);

        let q_values = self.model.forward(input);

        let q_values: [f32; OUTPUTS] = q_values.array();

        let mut action = Self::pick_action(q_values);

        // Epsilon-greedy exploration.
        if fastrand::f32() < self.epsilon {
            action = fastrand::usize(0..OUTPUTS);
        }

        (action, q_values)
    }

    fn pick_action(q_values: [f32; OUTPUTS]) -> usize {
        let mut q_value_max = f32::NEG_INFINITY;
        let mut max_index = 0;

        for (i, &q_value) in q_values.iter().enumerate() {
            if q_value > q_value_max {
                q_value_max = q_value;
                max_index = i;
            }
        }

        max_index
    }

    pub fn set_reward_u8(
        &mut self,
        inputs: [u8; INPUTS],
        action: usize,
        reward: f32,
        next_inputs: [u8; INPUTS],
        finished: bool,
    ) {
        let inputs_f32 = inputs.map(|x| x as f32);
        let next_inputs_f32 = next_inputs.map(|x| x as f32);

        self.set_reward(
            inputs_f32,
            action,
            reward,
            next_inputs_f32,
            finished,
        );
    }

    pub fn set_reward(
        &mut self,
        inputs: [f32; INPUTS],
        action: usize,
        reward: f32,
        next_inputs: [f32; INPUTS],
        finished: bool,
    ) {
        assert!(
            action < OUTPUTS,
            "action {} is outside [0, {})",
            action,
            OUTPUTS
        );

        self.steps.push(Transition {
            inputs,
            action,
            reward,
            inputs_next: next_inputs,
            finished,
        });
    }

    pub fn learn(&mut self) -> f32 {
        const GAMMA: f32 = 0.99;

        if self.steps.is_empty() {
            return 0.0;
        }

        let n = self.steps.len() as f32;

        let mut loss_sum = 0.0;

        //
        // Allocate one gradient structure for the whole batch.
        //
        // We intentionally reuse this without zeroing between
        // transitions, so gradients accumulate.
        //
        let mut grads = self.model.alloc_grads();

        //
        // Iterate in reverse, matching the original implementation.
        //
        for step in self.steps.iter().rev() {
            let input = self.dev.tensor(step.inputs);
            let next_input = self.dev.tensor(step.inputs_next);

            //
            // ---------------------------------------------------------
            // Q(s)
            // ---------------------------------------------------------
            //
            // traced() starts the autograd tape and associates the
            // computation with `grads`.
            //
            let q_values = self
                .model
                .forward_mut(input.traced(grads));

            //
            // Save Q(s) before consuming the tensor below.
            //
            let q_values_array: [f32; OUTPUTS] =
                q_values.array();

            //
            // ---------------------------------------------------------
            // Q(s')
            // ---------------------------------------------------------
            //
            // This forward pass is deliberately NOT traced.
            //
            let q_values_next =
                self.model.forward(next_input);

            let q_values_next_array: [f32; OUTPUTS] =
                q_values_next.array();

            //
            // max_a Q(s', a)
            //
            let mut q_value_max_next =
                f32::NEG_INFINITY;

            for q_value in q_values_next_array {
                if q_value > q_value_max_next {
                    q_value_max_next = q_value;
                }
            }

            //
            // ---------------------------------------------------------
            // Bellman target
            // ---------------------------------------------------------
            //
            // terminal:
            //
            //     y = r
            //
            // non-terminal:
            //
            //     y = r + gamma * max Q(s', a)
            //
            let target = if step.finished {
                step.reward
            } else {
                step.reward + GAMMA * q_value_max_next
            };

            //
            // ---------------------------------------------------------
            // Selected action error
            // ---------------------------------------------------------
            //
            // Only Q(s, action) is supposed to contribute to the loss,
            // just like the original NeuralNetworkSimd implementation.
            //
            let prediction =
                q_values_array[step.action];

            let diff = prediction - target;

            loss_sum += diff * diff;

            //
            // Construct:
            //
            //     target = Q(s)
            //
            // except:
            //
            //     target[action] = Bellman target
            //
            // Therefore every non-selected action has zero error.
            //
            let mut target_values = q_values_array;

            target_values[step.action] = target;

            let target_tensor =
                self.dev.tensor(target_values);

            //
            // IMPORTANT:
            //
            // We use `sum()` rather than `mean()`.
            //
            // The original implementation computes:
            //
            //     loss = 1/N * (Q_action - target)^2
            //
            // where N is the number of transitions.
            //
            // `mean()` here would additionally divide by OUTPUTS.
            //
            let loss =
                (q_values - target_tensor)
                    .square()
                    .sum()
                    * (1.0 / n);

            //
            // Backward pass.
            //
            // The returned gradient object contains the accumulated
            // gradients for this computation.
            //
            grads = loss.backward();
        }

        //
        // Mean squared error reported to the caller.
        //
        let loss = loss_sum / n;

        //
        // Apply accumulated gradients.
        //
        self.optimizer
            .update(&mut self.model, &grads)
            .expect("DQN optimizer update failed");

        //
        // Clear transitions.
        //
        self.steps.clear();

        //
        // Epsilon decay.
        //
        self.epsilon =
            f32::max(self.epsilon * 0.99, 0.1);

        loss
    }

    pub fn epsilon(&self) -> f32 {
        self.epsilon
    }

    pub fn set_epsilon(&mut self, epsilon: f32) {
        self.epsilon = epsilon.clamp(0.0, 1.0);
    }
}

pub struct Transition<const INPUTS: usize> {
    pub inputs: [f32; INPUTS],
    pub action: usize,
    pub reward: f32,
    pub inputs_next: [f32; INPUTS],
    pub finished: bool,
}
