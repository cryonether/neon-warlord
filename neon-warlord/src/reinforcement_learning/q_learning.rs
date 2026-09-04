//! Learns the path of best quality

use std::collections::HashMap;

use crate::agents::agent_drawer;

#[cfg(test)]
mod test_maze;

pub struct QLearning<const INPUTS: usize, const OUTPUTS: usize> {
    state: HashMap<StateKey<INPUTS>, StateValue<OUTPUTS>>,
    
    steps: Vec<Transition<INPUTS>>,

    epsilon: f32,
}

impl<const INPUTS: usize, const OUTPUTS: usize> QLearning<INPUTS, OUTPUTS> {
    
    pub fn new() -> Self {
        let state: HashMap<StateKey<INPUTS>, StateValue<OUTPUTS>> = HashMap::new();
        let steps = Vec::new();
        let epsilon = 0.8;
        Self { state, steps, epsilon }
    }

    pub fn choose_action(&mut self, inputs: &[u8; INPUTS]) -> (usize, [f32; OUTPUTS])
    {
        let q_values: [f32; OUTPUTS] = match self.state.get(&StateKey{inputs: *inputs,})
        {
            Some(state_value) => state_value.q_values,
            None => [0.0; OUTPUTS],
        };

        let mut action = Self::pick_action(q_values);

        if fastrand::f32() < self.epsilon {
            action = fastrand::usize(0..OUTPUTS);
        }

        (action, q_values)
    }

    fn pick_action_probability(mut q_values: [f32; OUTPUTS]) -> usize {

        let min = q_values
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min);

        // Shift values so the smallest value becomes 0.
        for value in &mut q_values {
            *value -= min;
        }

        let sum: f32 = q_values.iter().sum();

        // All Q-values are equal (or invalid), so choose uniformly.
        if sum <= 0.0 || !sum.is_finite() {
            return fastrand::usize(..OUTPUTS);
        }

        let random = fastrand::f32() * sum;
        let mut cumulative = 0.0;

        for (i, &value) in q_values.iter().enumerate() {
            cumulative += value;

            if random < cumulative {
                return i;
            }
        }

        // Floating-point rounding fallback.
        OUTPUTS - 1
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

    pub fn set_reward(
        &mut self, 
        inputs: [u8; INPUTS], 
        action: usize, 
        reward: f32,
        next_inputs: [u8; INPUTS], 
    ) 
    {
        self.steps.push(Transition{
            inputs,
            action,
            reward,
            inputs_next: next_inputs,
        });
    }

    pub fn learn(&mut self) 
    {
        const GAMMA: f32 = 0.9;

        for step in self.steps.iter().rev() {
            let inputs = step.inputs;
            let action = step.action;
            let reward = step.reward;
            let inputs_next = step.inputs_next;

            let state_key = StateKey{inputs: inputs};
            let state_key_next = StateKey{inputs: inputs_next};

            let mut q_values: [f32; OUTPUTS] = match self.state.get(&state_key) {
                Some(state_value) => state_value.q_values,
                None => [0.0; OUTPUTS],
            };

            let q_values_next: [f32; OUTPUTS] = match self.state.get(&state_key_next) {
                Some(state_value) => state_value.q_values,
                None => [0.0; OUTPUTS],
            };

            let mut q_value_max_next = f32::NEG_INFINITY;
            for q_value in q_values_next {
                if q_value > q_value_max_next {
                    q_value_max_next = q_value
                }
            }

            let reward_2 = reward + GAMMA * (q_value_max_next - reward);

            // modify
            q_values[action] = reward_2;
            self.state.insert(state_key, StateValue{
                q_values,
            });
        }

        self.epsilon = f32::max(self.epsilon * 0.95, 0.1); 
    }
}

pub struct Transition<const INPUTS: usize> {
    pub inputs: [u8; INPUTS],
    pub action: usize,
    pub reward: f32,
    pub inputs_next: [u8; INPUTS],
}

#[derive(Hash, Eq, PartialEq)]
pub struct StateKey<const INPUTS: usize> {
    pub inputs: [u8; INPUTS],
}

pub struct StateValue<const OUTPUTS: usize> {
    pub q_values: [f32; OUTPUTS],
}

