//! Runs the simulation and generates transitions

use std::{thread, time::Duration};

use crate::{
    gym_simulation::{Vec3, gym::Gym, worker_thread_2::WorkerThread2Run}, reinforcement_learning::{
        neural_network_simd::activation_function::{activation_none::ActivationNone, activation_tan_h::ActivationTanH}, ppo::Ppo,
    },
};

/// Runs the simulation and generates transitions
pub struct GymWorker<
    const INPUTS: usize,
    const OUTPUTS: usize,
    const NEURONS: usize,
    const LAYERS: usize,
    const RESIDUAL: bool,
    ENV: Gym<INPUTS, OUTPUTS>,
> where
    ENV: std::clone::Clone,
{
    pub ppo: Ppo<INPUTS, OUTPUTS, NEURONS, LAYERS, RESIDUAL, ActivationTanH>,
    pub envs: Vec<ENV>,

    // pub nodes: Vec<forward_renderer::particle_shader::Instance>,
    // pub edges: Vec<forward_renderer::particle_shader_two_point::Instance>,

    pub nr_steps: usize,

    pub reward: f32,
    pub done: bool,

    ticks: usize,
}

impl<
    const INPUTS: usize,
    const OUTPUTS: usize,
    const NEURONS: usize,
    const LAYERS: usize,
    const RESIDUAL: bool,
    ENV: Gym<INPUTS, OUTPUTS>,
> GymWorker<INPUTS, OUTPUTS, NEURONS, LAYERS, RESIDUAL, ENV>
where
    ENV: std::clone::Clone,
{
    pub fn new(
        ppo: Ppo<INPUTS, OUTPUTS, NEURONS, LAYERS, RESIDUAL, ActivationTanH>,
        env: ENV,
        n: usize,
        m: usize,
        pos: Vec3,
        nr_steps: usize,
    ) -> Self {
        let radius = 0.1;

        let mut envs: Vec<ENV> = Vec::with_capacity(n * m);
        for j in 0..m {
            for i in 0..n {
                let pos = pos + Vec3::new(i as f32 * 10.0, j as f32 * 4.0, 0.0);

                let mut env = env.clone();
                env.set_drawer_position(&pos);
                env.set_drawer_radius(radius);

                envs.push(env);
            }
        }

        // let nodes: Vec<forward_renderer::particle_shader::Instance> = Vec::new();
        // let edges: Vec<forward_renderer::particle_shader_two_point::Instance> = Vec::new();

        Self {
            ppo,
            envs,
            // nodes,
            // edges,
            ticks: 0,
            nr_steps,
            reward: 0.0,
            done: false,
        }
    }
}

impl<
    const INPUTS: usize,
    const OUTPUTS: usize,
    const NEURONS: usize,
    const LAYERS: usize,
    const RESIDUAL: bool,
    ENV: Gym<INPUTS, OUTPUTS>,
> WorkerThread2Run for GymWorker<INPUTS, OUTPUTS, NEURONS, LAYERS, RESIDUAL, ENV>
where
    ENV: std::clone::Clone,
{
    fn run(&mut self) {
        let dt = 1.0 / 60.0;
        let max_steps = 4000;

        // self.nodes.clear();
        // self.edges.clear();
        self.reward = 0.0;



        for env in &mut self.envs {
            {
                // current state and action
                let state = env.get_state();
                let (action, _mean_action, log_probability) = self.ppo.get_action(&state);

                // update
                env.update(&action, dt);
                env.update_verlet_physics(dt);

                // reward
                let (reward, terminated) = env.get_reward();
                self.reward += reward;

                // save reward
                let truncated = self.ticks.is_multiple_of(self.nr_steps) || self.ticks >= max_steps;
                self.ppo.save_reward(
                    state,
                    action,
                    log_probability,
                    reward,
                    terminated,
                    truncated,
                );
            }
        }

        if self.ticks.is_multiple_of(self.nr_steps) {
            self.done = true;
        }


        if self.ticks >= max_steps {
            self.ticks = 0;
            for env in &mut self.envs {
                env.reset();
            }
        } else {
            self.ticks += 1;
        }

    }
    
    fn draw(&self, 
        nodes: &mut Vec<forward_renderer::particle_shader::Instance>,
        edges: &mut Vec<forward_renderer::particle_shader_two_point::Instance>,
    ) {
        nodes.clear();
        edges.clear();

        for env in &self.envs {
            env.draw(nodes, edges);
        }
    }
    
    fn done(&self) -> bool {
        self.done
    }
    
    fn init(&mut self) {
        self.done = false;
        self.reward = 0.0;
    }
}
