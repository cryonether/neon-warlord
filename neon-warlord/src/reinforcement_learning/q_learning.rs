//! Learns the path of best quality

use std::collections::HashMap;

use crate::agents::agent_drawer;

#[cfg(test)]
mod test_maze;

pub struct QLearning<const INPUTS: usize, const OUTPUTS: usize> {
    state: HashMap<StateKey<INPUTS>, StateValue<OUTPUTS>>,
    
    steps: Vec<Transition<INPUTS>>,
}

impl<const INPUTS: usize, const OUTPUTS: usize> QLearning<INPUTS, OUTPUTS> {
    
    pub fn new() -> Self {
        let state: HashMap<StateKey<INPUTS>, StateValue<OUTPUTS>> = HashMap::new();
        let steps = Vec::new();
        Self { state, steps }
    }

    pub fn choose_action(&mut self, inputs: &[u8; INPUTS]) -> (usize, [f32; OUTPUTS])
    {
        let q_values: [f32; OUTPUTS] = match self.state.get(&StateKey{inputs: *inputs,})
        {
            Some(state_value) => state_value.q_values,
            None => [0.0; OUTPUTS],
        };

        let mut action = 0;
        for i in 1..OUTPUTS {
            if q_values[i] > q_values[action] {
                action = i;
            }
        }

        if fastrand::f32() < 0.1 {
            action = fastrand::usize(0..OUTPUTS);
        }

        (action, q_values)
    }

    pub fn set_reward(&mut self, inputs: &[u8; INPUTS], action: usize, reward: f32) 
    {
        self.steps.push(Transition{
            inputs: *inputs,
            action,
            reward,
        });
    }

    pub fn learn(&mut self) 
    {
        for step in &self.steps {
            let state_key = StateKey{inputs: step.inputs};

            let mut q_values: [f32; OUTPUTS] = match self.state.get(&state_key) {
                Some(state_value) => state_value.q_values,
                None => [0.0; OUTPUTS],
            };

            q_values[step.action] = step.reward;
            self.state.insert(state_key, StateValue{
                q_values,
            });
        }
    }
}

pub struct Transition<const INPUTS: usize> {
    pub inputs: [u8; INPUTS],
    pub action: usize,
    pub reward: f32,
}

#[derive(Hash, Eq, PartialEq)]
pub struct StateKey<const INPUTS: usize> {
    pub inputs: [u8; INPUTS],
}

pub struct StateValue<const OUTPUTS: usize> {
    pub q_values: [f32; OUTPUTS],
}

