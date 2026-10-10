//! Generates transitions

use std::iter::zip;

use crate::{
    gym_simulation::{
        Vec3, gym::Gym, verlet_physics_drawer::VerletPhysicsDrawer,
        worker_thread_2::WorkerThread2Run,
    },
    reinforcement_learning::{
        neural_network_simd::activation_function::activation_tan_h::ActivationTanH, ppo::Ppo,
    },
};

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
    pub drawers: Vec<VerletPhysicsDrawer>,

    pub nodes: Vec<forward_renderer::particle_shader::Instance>,
    pub edges: Vec<forward_renderer::particle_shader_two_point::Instance>,

    pub nr_steps: usize,

    pub reward: f32,

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
        let mut drawers = Vec::with_capacity(n * m);
        for j in 0..m {
            for i in 0..n {
                let pos = pos + Vec3::new(i as f32 * 10.0, j as f32 * 4.0, 0.0);

                let env = env.clone();
                let drawer = VerletPhysicsDrawer::new(env.get_verlet_physics(), radius, pos);

                envs.push(env);
                drawers.push(drawer);
            }
        }

        let nodes: Vec<forward_renderer::particle_shader::Instance> = Vec::new();
        let edges: Vec<forward_renderer::particle_shader_two_point::Instance> = Vec::new();

        Self {
            ppo,
            envs,
            drawers,
            nodes,
            edges,
            ticks: 0,
            nr_steps,
            reward: 0.0,
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
        let max_steps = 1000;

        self.nodes.clear();
        self.edges.clear();
        self.reward = 0.0;

        for (env, drawer) in zip(&mut self.envs, &mut self.drawers) {
            for i in 0..self.nr_steps {
                // current state and action
                let state = env.get_state();
                let (action, _mean_action, log_probability) = self.ppo.get_action(&state);

                // update
                env.update(&action, dt);
                env.update_verlet_physics(dt);

                // reward
                let (reward, done) = env.get_reward();
                self.reward += reward;

                // save reward
                let done = done || i >= self.nr_steps - 1 || self.ticks + i >= max_steps;
                self.ppo
                    .save_reward(state, action, log_probability, reward, done);

                // count ticks and reset
                if self.ticks + i >= max_steps {
                    self.ticks = 0;
                    env.reset();
                } else {
                    self.ticks += 1;
                }
            }

            // draw
            drawer.update(env.get_verlet_physics(), &mut self.nodes, &mut self.edges);
        }
    }
}
