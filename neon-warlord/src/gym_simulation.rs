//! Simulates an inverted pendulum

mod pendulum;
mod gym;
pub mod gym_line;

use std::collections::VecDeque;

use forward_renderer::{height_map::HeightMapInterface, to_rgb};
use wgpu_renderer::performance_monitor::{Fps, watch::Watch};

use crate::{
    gym_simulation::gym::Gym, pendulum_cart_simulation::{graph_lines::{GraphLines, GraphLinesDrawer}, pendulum_cart::{PendulumAction, PendulumCart, PendulumState}, verlet_physics_drawer::VerletPhysicsDrawer}, physics_simulation_v3_drawer::DrawerObjects, print_color::print_color, reinforcement_learning::{dqn2::Dqn2, ppo::Ppo}, triple_buffer, worker_thread,
};

pub const WATCH_POINTS_SIZE: usize = 10;
type Vec3 = cgmath::Vector3<f32>;

// const INPUTS: usize = 4;
// const OUTPUTS: usize = 2;
// const NR_LAYERS: usize = 2;
// const RESIDUAL: bool = false;

pub struct GymSimulation<
    const INPUTS: usize,
    const OUTPUTS: usize,
    const NEURONS: usize,
    const LAYERS: usize,
    const RESIDUAL: bool,
    ENV: Gym<INPUTS, OUTPUTS>,
>
{
    // Physics
    ticks: u64,

    ppo: Ppo<INPUTS, OUTPUTS, NEURONS, LAYERS, RESIDUAL>,

    graph_actor_loss: GraphLines<1>,
    graph_critic_loss: GraphLines<1>,
    graph_actions: GraphLines<OUTPUTS>,
    graph_mean_actions: GraphLines<OUTPUTS>,
    graph_reward: GraphLines<1>,
    graph_inputs: GraphLines<INPUTS>,

    drawer_graph_actor_loss: GraphLinesDrawer<1>,
    drawer_graph_critic_loss: GraphLinesDrawer<1>,
    drawer_graph_actions: GraphLinesDrawer<OUTPUTS>,
    drawer_graph_mean_actions: GraphLinesDrawer<OUTPUTS>,
    drawer_graph_reward: GraphLinesDrawer<1>,
    drawer_graph_inputs: GraphLinesDrawer<INPUTS>,

    env: ENV,
    verlet_physics_drawer: VerletPhysicsDrawer,
    steps: u64,
    episode: u64,

    // Debug
    ups: Fps,
    last_render_time: instant::Instant,
    watch_ups: Watch<WATCH_POINTS_SIZE>,
}

// unsafe impl<
//     const INPUTS: usize,
//     const OUTPUTS: usize,
//     const NEURONS: usize,
//     const LAYERS: usize,
//     const RESIDUAL: bool,
//     ENV: Gym<INPUTS, OUTPUTS>,
// > Send for GymSimulation<
//     INPUTS,
//     OUTPUTS,
//     NEURONS,
//     LAYERS,
//     RESIDUAL,
//     ENV,
// >{}

impl<
    const INPUTS: usize,
    const OUTPUTS: usize,
    const NEURONS: usize,
    const LAYERS: usize,
    const RESIDUAL: bool,
    ENV: Gym<INPUTS, OUTPUTS>,
> GymSimulation<
    INPUTS,
    OUTPUTS,
    NEURONS,
    LAYERS,
    RESIDUAL,
    ENV,
> {
    pub fn new(env: ENV) -> Self {
        // agent 0
        let pos = Vec3::new(0.0, 0.0, 2.0);

        let pos_graph_actor_loss = pos + Vec3::new(-4.2, 1.0, 2.2);
        let pos_graph_critic_loss = pos + Vec3::new(-4.2, 1.0, 0.0);
        let pos_graph_actions = pos + Vec3::new(-2.0, 1.0, 2.2);
        let pos_graph_mean_actions = pos + Vec3::new(-2.0, 1.0, 0.0);
        let pos_graph_quality = pos + Vec3::new(2.2, 1.0, 0.0);

        let pos_graph_inputs = pos + Vec3::new(2.2, 1.0, 2.2);

        let pos_env = pos + Vec3::new(2.0, -0.5, 1.0);


        let scale = 0.1;

        let ppo = Ppo::new(0);

        // Debug
        let ups = Fps::new();
        let watch_ups = Watch::new();

        // Graph
        let graph_x: VecDeque<f32> = (0..100).map(|i| i as f32 * 0.1).collect();
        // let graph_y: VecDeque<f32> = (0..100).map(|i| (i as f32 * 0.1).sin()).collect();
        let graph_y: VecDeque<f32> = (0..100).map(|_i| 0.0).collect();
        let graph_actor_loss = GraphLines {
            x: graph_x.clone(),
            y: [graph_y.clone()],
        };

        let graph_critic_loss = GraphLines {
            x: graph_x.clone(),
            y: [graph_y.clone()],
        };

        let y: [VecDeque<f32>; OUTPUTS] =
            std::array::from_fn(|_| graph_y.clone());
        let graph_actions = GraphLines {
            x: graph_x.clone(),
            y: y,
        };

        let y: [VecDeque<f32>; OUTPUTS] =
            std::array::from_fn(|_| graph_y.clone());
        let graph_mean_actions = GraphLines {
            x: graph_x.clone(),
            y: y,
        };

        let graph_reward = GraphLines {
            x: graph_x.clone(),
            y: [graph_y.clone()],
        };

        let y: [VecDeque<f32>; INPUTS] =
            std::array::from_fn(|_| graph_y.clone());
        let graph_inputs = GraphLines {
            x: graph_x.clone(),
            y: y,
        };



        let drawer_graph_actor_loss =
            GraphLinesDrawer::new(scale, pos_graph_actor_loss).colors([to_rgb("#00ff62").into()]);
        let drawer_graph_critic_loss = GraphLinesDrawer::new(scale, pos_graph_critic_loss)
            .colors([to_rgb("#ffd000").into()]);

        let colors: [Vec3; OUTPUTS] =
            std::array::from_fn(|_| to_rgb("#ff00e6").into());
        let drawer_graph_actions = GraphLinesDrawer::new(scale, pos_graph_actions).colors(colors);

        let colors: [Vec3; OUTPUTS] =
            std::array::from_fn(|_| to_rgb("#ff0932").into());
        let drawer_graph_mean_actions = GraphLinesDrawer::new(scale, pos_graph_mean_actions).colors(colors);


        let drawer_graph_reward = GraphLinesDrawer::new(scale, pos_graph_quality)
            .colors([to_rgb("#00d9ae").into()]);

        let colors: [Vec3; INPUTS] =
            std::array::from_fn(|_| to_rgb("#7700d9").into());
        let drawer_graph_inputs = GraphLinesDrawer::new(scale, pos_graph_inputs)
            .colors(colors);


        // Pendulum
        let verlet_physics_drawer =
            VerletPhysicsDrawer::new(&env.get_verlet_physics(), scale, pos_env);

        // Dqn

        Self {
            ticks: 0,
            steps: 0,
            episode: 0,

            ups,
            last_render_time: instant::Instant::now(),
            watch_ups,
            ppo,
            graph_actor_loss,
            graph_critic_loss,
            graph_actions,
            graph_mean_actions,
            graph_reward,
            graph_inputs,
            drawer_graph_actor_loss,
            drawer_graph_critic_loss,
            drawer_graph_actions,
            drawer_graph_mean_actions,
            drawer_graph_reward,
            drawer_graph_inputs,
            env,
            verlet_physics_drawer,

        }
    }

    pub fn update_physics(&mut self) {
        self.watch_ups.update();

        let dt = 1.0 / 60.0;
        self.ticks += 1;

        self.watch_ups.start("Solver");
        let state = self.env.get_state();


        let (action, mean_action, log_probability) = self.ppo.get_action(&state);

        self.env.update(&action, dt);
        let new_state = self.env.get_state();
        let reward = self.env.get_reward();

        self.ppo.save_reward(state, action, log_probability, reward, false);

        for (i, val) in new_state.iter().enumerate() {
            self.graph_inputs.y_push_pop(i, *val);
        }

        for (i, val) in action.iter().enumerate() {
            self.graph_actions.y_push_pop(i, *val);
        }

        for (i, val) in mean_action.iter().enumerate() {
            self.graph_mean_actions.y_push_pop(i, *val);
        }

        self.graph_reward.y_push_pop(0, reward);

        self.env.update_verlet_physics(dt);
        self.watch_ups.stop();

        if self.ticks.is_multiple_of(1000) {
            let (actor_loss, critic_loss) = self.ppo.learn();
            

            fn create_input<const INPUTS: usize>(i: usize, size: usize) -> [f32; INPUTS] {
                let x = i as f32 / (size - 1) as f32 * 2.0 - 1.0;
                let mut input: [f32; INPUTS] = [0.0; INPUTS];
                input[0] = x;
                input
            }

            print!("{}, ", self.ticks / 1000);
            print!("actor: [ ");
            let size = 10;
            for i in 0 .. size {
                let input = create_input(i, size); 
                let y_pred = self.ppo.actor.forward(&input);

                print_color(y_pred[0], 0.0, 1.0, crate::print_color::PrintColor::GreenCyanBlue);
            }

            print!("], critic: [ ");
            let size = 10;
            for i in 0 .. size {
                let input = create_input(i, size); 
                let y_pred = self.ppo.critic.forward(&input);

                print_color(y_pred[0], 0.0, 1.0, crate::print_color::PrintColor::BluePurpleRed);
            }
            print!("], ");

            println!("actor_loss: {}, critic_loss: {}", actor_loss, critic_loss);
            self.env.reset();
        }

        // ups
        let now = instant::Instant::now();
        let dt = now - self.last_render_time;
        self.last_render_time = now;
        self.ups.update(dt);
    }

    pub fn update_drawer(&mut self, objects: &mut DrawerObjects) {
        let nodes = &mut objects.genome_nodes;
        let edges = &mut objects.genome_edges;

        self.watch_ups.start("Draw Model");

        self.drawer_graph_actor_loss.update(&self.graph_actor_loss, edges);
        self.drawer_graph_critic_loss
            .update(&self.graph_critic_loss, edges);
        self.drawer_graph_actions.update(&self.graph_actions, edges);
        self.drawer_graph_mean_actions.update(&self.graph_mean_actions, edges);
        self.drawer_graph_reward.update(&self.graph_reward, edges);
        self.drawer_graph_inputs
            .update(&self.graph_inputs, edges);

        self.verlet_physics_drawer
            .update(&self.env.get_verlet_physics(), nodes, edges);

        self.watch_ups.stop();

        objects.ups = self.ups.get();
        // self.watch_ups.update();
        objects.watch_ups = self.watch_ups.get_viewer_data();
    }
}

pub trait GymSimulationInterface {
    fn update_physics(&mut self);
    fn update_drawer(&mut self, objects: &mut DrawerObjects);
}

impl<
    const INPUTS: usize,
    const OUTPUTS: usize,
    const NEURONS: usize,
    const LAYERS: usize,
    const RESIDUAL: bool,
    ENV: Gym<INPUTS, OUTPUTS>,
> GymSimulationInterface for GymSimulation<
    INPUTS,
    OUTPUTS,
    NEURONS,
    LAYERS,
    RESIDUAL,
    ENV,
> {
    fn update_physics(&mut self) {
        self.update_physics();
    }

    fn update_drawer(&mut self, objects: &mut DrawerObjects) {
        self.update_drawer(objects);
    }
}

pub struct PendulumSimulationThread
{
    pub sim: Box<dyn GymSimulationInterface>,
    pub producer: triple_buffer::Producer<DrawerObjects>,
}

impl worker_thread::Update for PendulumSimulationThread
{
    fn update_physics(&mut self) {
        
        self.sim.update_physics();
    }

    fn update_drawer(&mut self) {

        let data = self.producer.buffer();
        data.clear();

        self.sim.update_drawer(data);

        self.producer.publish();
    }
}

unsafe impl<> Send for PendulumSimulationThread<>{}