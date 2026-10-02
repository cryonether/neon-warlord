//! Simulates an inverted pendulum

mod pendulum;

use std::collections::VecDeque;

use forward_renderer::{height_map::HeightMapInterface, to_rgb};
use wgpu_renderer::performance_monitor::{Fps, watch::Watch};

use crate::{
    pendulum_cart_simulation::{graph_lines::{GraphLines, GraphLinesDrawer}, pendulum_cart::{PendulumAction, PendulumCart, PendulumState}, verlet_physics_drawer::VerletPhysicsDrawer}, pendulum_simulation::pendulum::Pendulum, physics_simulation_v3_drawer::DrawerObjects, reinforcement_learning::{dqn2::Dqn2, ppo2::Ppo2}, triple_buffer, worker_thread,
};

pub const WATCH_POINTS_SIZE: usize = 10;
type Vec3 = cgmath::Vector3<f32>;

// const INPUTS: usize = 4;
// const OUTPUTS: usize = 2;
// const NR_LAYERS: usize = 2;
// const RESIDUAL: bool = false;

pub struct PendulumSimulation {
    // Physics
    ticks: u64,

    ppo: Ppo2<4, 1, 64, 3, false>,

    graph_actor_loss: GraphLines<1>,
    graph_critic_loss: GraphLines<1>,
    graph_acceleration: GraphLines<1>,
    graph_quality: GraphLines<1>,
    graph_sin_cos: GraphLines<2>,
    graph_sin_cos_vel: GraphLines<2>,

    drawer_graph_actor_loss: GraphLinesDrawer<1>,
    drawer_graph_critic_loss: GraphLinesDrawer<1>,
    drawer_graph_acceleration: GraphLinesDrawer<1>,
    drawer_graph_quality: GraphLinesDrawer<1>,
    drawer_graph_sin_cos: GraphLinesDrawer<2>,
    drawer_graph_sin_cos_vel: GraphLinesDrawer<2>,

    pendulum: Pendulum,
    initial_pendulum: Pendulum,
    verlet_physics_drawer: VerletPhysicsDrawer,
    steps: u64,
    episode: u64,

    // Debug
    ups: Fps,
    last_render_time: instant::Instant,
    watch_ups: Watch<WATCH_POINTS_SIZE>,
}

unsafe impl Send for PendulumSimulation {}

impl PendulumSimulation {
    pub fn new() -> Self {
        // agent 0
        let pos = Vec3::new(0.0, 0.0, 2.0);

        let pos_graph_actor_loss = pos + Vec3::new(-4.2, 1.0, 1.0);
        let pos_graph_critic_loss = pos + Vec3::new(-2.0, 1.0, 0.0);
        let pos_graph_acceleration = pos + Vec3::new(-2.0, 1.0, 2.2);
        let pos_graph_quality = pos + Vec3::new(2.2, 1.0, 0.0);

        let pos_graph_sin_cos = pos + Vec3::new(2.2, 1.0, 2.2);
        let pos_graph_sin_cos_vel = pos + Vec3::new(4.4, 1.0, 2.2);

        let pos_pendulum = pos + Vec3::new(2.0, -0.5, 1.0);


        let scale = 0.1;

        let ppo = Ppo2::new();

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

        let graph_acceleration = GraphLines {
            x: graph_x.clone(),
            y: [
                graph_y.clone(),
            ],
        };

        let graph_quality = GraphLines {
            x: graph_x.clone(),
            y: [graph_y.clone()],
        };
        let graph_sin_cos = GraphLines {
            x: graph_x.clone(),
            y: [graph_y.clone(), graph_y.clone()],
        };
        let graph_sin_cos_vel = GraphLines {
            x: graph_x.clone(),
            y: [graph_y.clone(), graph_y.clone()],
        };

        let drawer_graph_actor_loss =
            GraphLinesDrawer::new(scale, pos_graph_actor_loss).colors([to_rgb("#12d900").into()]);
        let drawer_graph_critic_loss = GraphLinesDrawer::new(scale, pos_graph_critic_loss)
            .colors([to_rgb("#b1d900").into()]);
        let drawer_graph_acceleration = GraphLinesDrawer::new(scale, pos_graph_acceleration).colors([
            to_rgb("#950187").into(),
        ]);
        let drawer_graph_quality = GraphLinesDrawer::new(scale, pos_graph_quality)
            .colors([to_rgb("#d9ae00").into()])
            .y_lim(std::f32::consts::PI);
        let drawer_graph_sin_cos = GraphLinesDrawer::new(scale, pos_graph_sin_cos)
            .colors([to_rgb("#7700d9").into(), to_rgb("#c884ff").into()])
            .y_lim(std::f32::consts::PI);
        let drawer_graph_sin_cos_vel =
            GraphLinesDrawer::new(scale, pos_graph_sin_cos_vel).colors([to_rgb("#0070d9").into(), to_rgb("#5fb2ff").into()]);

        // Pendulum
        let pendulum = Pendulum::new();
        let initial_pendulum = pendulum.clone();
        let verlet_physics_drawer =
            VerletPhysicsDrawer::new(&pendulum.verlet_physics, scale, pos_pendulum);

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
            graph_acceleration,
            graph_quality,
            graph_sin_cos,
            graph_sin_cos_vel,
            drawer_graph_actor_loss,
            drawer_graph_critic_loss,
            drawer_graph_acceleration,
            drawer_graph_quality,
            drawer_graph_sin_cos,
            drawer_graph_sin_cos_vel,
            pendulum,
            initial_pendulum,
            verlet_physics_drawer,
 
        }
    }

    pub fn update_physics(&mut self, _height_map: &impl HeightMapInterface) {
        let dt = 1.0 / 60.0;
        self.ticks += 1;

        self.watch_ups.start("Solver");
        let pendulum_state = self.pendulum.state();
        let observation = [
            pendulum_state.sin_theta, 
            pendulum_state.cos_theta, 
            pendulum_state.angular_velocity, 
            pendulum_state.cos_alpha_v
        ];

        let (action, log_probability) = self.ppo.get_action(&observation);
        let acc = action[0] * 1.0;

        let pendulum_state_new = self.pendulum.update(acc, dt);
        let reward = self.calculate_reward(&pendulum_state_new);

        self.ppo.save_reward(observation, action, log_probability, reward, false);

        self.graph_sin_cos.y_push_pop(0, pendulum_state_new.sin_theta);
        self.graph_sin_cos.y_push_pop(1, pendulum_state_new.cos_theta);
        self.graph_sin_cos_vel.y_push_pop(0, pendulum_state_new.angular_velocity * 10.0);
        self.graph_sin_cos_vel.y_push_pop(1, pendulum_state_new.cos_alpha_v * 10.0);
        self.graph_acceleration.y_push_pop(0, acc);

        self.pendulum.update_verlet_physics(dt);
        self.watch_ups.stop();

        if self.ticks.is_multiple_of(1000) {
            let (actor_loss, critic_loss) = self.ppo.learn();
            println!("actor_loss: {}, critic_loss: {}", actor_loss, critic_loss);
            self.pendulum.rand_pos();
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
        // self.model_drawer.update(&self.dqn.target_net, nodes, edges);

        self.drawer_graph_actor_loss.update(&self.graph_actor_loss, edges);
        self.drawer_graph_critic_loss
            .update(&self.graph_critic_loss, edges);
        self.drawer_graph_acceleration.update(&self.graph_acceleration, edges);
        self.drawer_graph_quality.update(&self.graph_quality, edges);
        self.drawer_graph_sin_cos
            .update(&self.graph_sin_cos, edges);
        self.drawer_graph_sin_cos_vel.update(&self.graph_sin_cos_vel, edges);

        self.verlet_physics_drawer
            .update(&self.pendulum.verlet_physics, nodes, edges);

        self.watch_ups.stop();

        objects.ups = self.ups.get();
        self.watch_ups.update();
        objects.watch_ups = self.watch_ups.get_viewer_data();
    }
    
    fn calculate_reward(&self, pendulum_state: &pendulum::PendulumState) -> f32 {
        pendulum_state.sin_theta + 1.0
    }
}

pub struct PendulumSimulationThread<T>
where
    T: HeightMapInterface,
{
    pub sim: PendulumSimulation,
    pub producer: triple_buffer::Producer<DrawerObjects>,
    pub height_map: T,
}

impl<T> worker_thread::Update for PendulumSimulationThread<T>
where
    T: HeightMapInterface,
{
    fn update(&mut self) {
        let data = self.producer.buffer();
        data.clear();

        self.sim.update_physics(&self.height_map);
        self.sim.update_drawer(data);

        self.producer.publish();
    }
}
