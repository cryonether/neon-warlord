//! Simulates an inverted pendulum

mod graph_lines;
mod neural_network_drawer;
mod pendulum;
mod verlet_physics_drawer;
mod test_pendulum_simulation_dfdx;
mod test_pendulum_simulation_dfdx_2;
mod test_pendulum_simulation_dfdx_3;
mod test_pendulum_simulation_dfdx_4;
mod test_pendulum_simulation_dfdx_5;

use std::collections::VecDeque;

use forward_renderer::{height_map::HeightMapInterface, to_rgb};
use wgpu_renderer::performance_monitor::{Fps, watch::Watch};

use crate::{
    pendulum_simulation::{
        graph_lines::{GraphLines, GraphLinesDrawer}, pendulum::{Pendulum, PendulumAction, PendulumState}, verlet_physics_drawer::VerletPhysicsDrawer,
    }, physics_simulation_v3_drawer::DrawerObjects, reinforcement_learning::{dqn::{self}, dqn_dfdx2::DqnDfdx2}, triple_buffer, worker_thread,
};

pub const WATCH_POINTS_SIZE: usize = 10;
type Vec3 = cgmath::Vector3<f32>;

const INPUTS: usize = 4;
const OUTPUTS: usize = 2;
const NR_LAYERS: usize = 2;
const RESIDUAL: bool = false;

pub struct PendulumSimulation {
    // Physics
    ticks: u64,

    // model: Box<NeuralNetworkSimd<INPUTS, OUTPUTS, NR_LAYERS, RESIDUAL>>,
    // model_drawer: NeuralNetworkDrawer<INPUTS, OUTPUTS, NR_LAYERS, RESIDUAL, 128, 8>,

    dqn: DqnDfdx2,

    graph_loss: GraphLines<1>,
    graph_chosen_action: GraphLines<1>,
    graph_actions: GraphLines<OUTPUTS>,
    graph_angle: GraphLines<1>,
    graph_angle_vel: GraphLines<1>,
    graph_cart: GraphLines<1>,
    graph_cart_vel: GraphLines<1>,

    graph_drawer_loss: GraphLinesDrawer<1>,
    graph_drawer_chosen_action: GraphLinesDrawer<1>,
    graph_drawer_actions: GraphLinesDrawer<OUTPUTS>,
    graph_drawer_angle: GraphLinesDrawer<1>,
    graph_drawer_angle_vel: GraphLinesDrawer<1>,
    graph_drawer_cart: GraphLinesDrawer<1>,
    graph_drawer_cart_vel: GraphLinesDrawer<1>,

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
        let pos_model = pos;

        let pos_graph_loss = pos + Vec3::new(-4.2, 1.0, 1.0);
        let pos_graph_chosen_action = pos + Vec3::new(-2.0, 1.0, 0.0);
        let pos_graph_actions = pos + Vec3::new(-2.0, 1.0, 2.2);
        let pos_pendulum = pos + Vec3::new(2.0, -0.5, 1.0);

        let pos_graph_angle = pos + Vec3::new(2.2, 1.0, 0.0);
        let pos_graph_angle_vel = pos + Vec3::new(2.2, 1.0, 2.2);
        let pos_graph_cart = pos + Vec3::new(4.4, 1.0, 0.0);
        let pos_graph_cart_vel = pos + Vec3::new(4.4, 1.0, 2.2);

        let scale = 0.1;

        // let model = Box::new(NeuralNetworkSimd::new());
        // let dqn = Dqn::new();
        let dqn = DqnDfdx2::new();
        // let model_drawer: NeuralNetworkDrawer<4, 2, 1, false, 128, 8> = NeuralNetworkDrawer::new(&dqn.target_net, scale, pos_model);

        // Debug
        let ups = Fps::new();
        let watch_ups = Watch::new();

        // Graph
        let graph_x: VecDeque<f32> = (0..100).map(|i| i as f32 * 0.1).collect();
        // let graph_y: VecDeque<f32> = (0..100).map(|i| (i as f32 * 0.1).sin()).collect();
        let graph_y: VecDeque<f32> = (0..100).map(|_i| 0.0).collect();
        let graph_loss = GraphLines {
            x: graph_x.clone(),
            y: [graph_y.clone()],
        };

        let graph_chosen_action = GraphLines {
            x: graph_x.clone(),
            y: [graph_y.clone()],
        };

        let graph_actions = GraphLines {
            x: graph_x.clone(),
            y: [
                graph_y.clone(), 
                graph_y.clone(), 
                // graph_y.clone(), 
                // graph_y.clone(), 
                // graph_y.clone(), 
                // graph_y.clone(), 
                // graph_y.clone()
            ],
        };

        let graph_angle = GraphLines {
            x: graph_x.clone(),
            y: [graph_y.clone()],
        };
        let graph_angle_vel = GraphLines {
            x: graph_x.clone(),
            y: [graph_y.clone()],
        };
        let graph_cart = GraphLines {
            x: graph_x.clone(),
            y: [graph_y.clone()],
        };
        let graph_cart_vel = GraphLines {
            x: graph_x.clone(),
            y: [graph_y.clone()],
        };

        let graph_drawer_loss =
            GraphLinesDrawer::new(scale, pos_graph_loss).colors([to_rgb("#12d900").into()]);
        let graph_drawer_chosen_action =
            GraphLinesDrawer::new(scale, pos_graph_chosen_action).colors([to_rgb("#b1d900").into()]);
        let graph_drawer_actions =
            GraphLinesDrawer::new(scale, pos_graph_actions).colors([
                // to_rgb("#ff005d").into(), 
                // to_rgb("#ff00e6").into(), 
                to_rgb("#950187").into(), 
                // to_rgb("#100010").into(), 
                to_rgb("#1f0090").into(),
                // to_rgb("#3700ff").into(),
                // to_rgb("#0400ff").into(),
            ]);
        let graph_drawer_angle = GraphLinesDrawer::new(scale, pos_graph_angle)
            .colors([to_rgb("#d9ae00").into()])
            .y_lim(std::f32::consts::PI);
        let graph_drawer_angle_vel = GraphLinesDrawer::new(scale, pos_graph_angle_vel)
            .colors([to_rgb("#7700d9").into()])
            .y_lim(std::f32::consts::PI);
        let graph_drawer_cart =
            GraphLinesDrawer::new(scale, pos_graph_cart).colors([to_rgb("#0070d9").into()]);
        let graph_drawer_cart_vel =
            GraphLinesDrawer::new(scale, pos_graph_cart_vel).colors([to_rgb("#0041d9").into()]);

        // Pendulum
        let pendulum = Pendulum::new();
        let verlet_physics_drawer =
            VerletPhysicsDrawer::new(&pendulum.verlet_physics, scale, pos_pendulum);

        // Dqn
        

        Self {
            ticks: 0,
            steps: 0,
            episode:0,

            // model,
            // model_drawer,
            graph_loss,
            graph_angle,
            graph_angle_vel,
            graph_cart,
            graph_cart_vel,
            pendulum: pendulum.clone(),
            initial_pendulum: pendulum.clone(),
            verlet_physics_drawer,

            ups,
            last_render_time: instant::Instant::now(),
            watch_ups,
            graph_drawer_loss,
            graph_drawer_angle,
            graph_drawer_angle_vel,
            graph_drawer_cart,
            graph_drawer_cart_vel,
            dqn,
            graph_actions,
            graph_drawer_actions,
            graph_chosen_action,
            graph_drawer_chosen_action,
        }
    }

    pub fn update_physics(&mut self, _height_map: &impl HeightMapInterface) {
        let dt = 1.0 / 60.0;
        self.ticks += 1;

        self.watch_ups.start("Solver");
        let pendulum_state = self.pendulum.state();
        let pendulum_action = self.get_pendulum_action(&pendulum_state);
        let pendulum_state_new = self.pendulum.update(pendulum_action, dt);
        self.set_pendulum_reward(&pendulum_state, pendulum_action, &pendulum_state_new);

       

        self.graph_angle.y_push_pop(0, pendulum_state_new.alpha);
        self.graph_angle_vel
            .y_push_pop(0, pendulum_state_new.angular_velocity);
        self.graph_cart.y_push_pop(0, pendulum_state_new.cart_pos);
        self.graph_cart_vel.y_push_pop(0, pendulum_state_new.cart_velocity);

        self.pendulum.update_verlet_physics(dt);
        self.watch_ups.stop();

        // ups
        let now = instant::Instant::now();
        let dt = now - self.last_render_time;
        self.last_render_time = now;
        self.ups.update(dt);
    }

    fn get_pendulum_action(&mut self, pendulum_state: &PendulumState) -> PendulumAction {
        let alpha = pendulum_state.alpha;
        let angular_velocity = pendulum_state.angular_velocity;
        let cart_pos = pendulum_state.cart_pos;
        let cart_velocity = pendulum_state.cart_velocity;

        let inputs = [
            alpha, 
            angular_velocity, 
            cart_pos, 
            cart_velocity, 
        ];

        let action = self.dqn.choose_action(&inputs);

        // update graph
        self.graph_actions.y_push_pop(0, action.1[0]);
        self.graph_actions.y_push_pop(1, action.1[1]);
        // self.graph_actions.y_push_pop(2, action.1[2]);
        // self.graph_actions.y_push_pop(3, action.1[3]);
        // self.graph_actions.y_push_pop(4, action.1[4]);
        // self.graph_actions.y_push_pop(5, action.1[5]);
        // self.graph_actions.y_push_pop(6, action.1[6]);
        self.graph_chosen_action.y_push_pop(0, (action.0 as f32 - 1.0) * 0.2);

        let pendulum_action: PendulumAction = (action.0 as u8).into();

        pendulum_action
    }

    fn set_pendulum_reward(&mut self, 
        pendulum_state: &PendulumState, 
        pendulum_action: PendulumAction, 
        pendulum_state_next: &PendulumState,
    ) {
        self.steps += 1;

        // let alpha = pendulum_state.alpha;
        // let angular_velocity = pendulum_state.angular_velocity;
        // let cart_pos = pendulum_state.cart_pos;
        // let cart_velocity = pendulum_state.cart_velocity;
        let action: u8 = pendulum_action.into();
        let action = action as usize;
        // println!("action: {}", action);

        let state: [f32; 4] = [
            pendulum_state.alpha, 
            pendulum_state.angular_velocity, 
            pendulum_state.cart_pos, 
            pendulum_state.cart_velocity, 
        ];

        let next_state: [f32; 4] = [
            pendulum_state_next.alpha, 
            pendulum_state_next.angular_velocity, 
            pendulum_state_next.cart_pos, 
            pendulum_state_next.cart_velocity, 
        ];

        // Winkel fortlaufend auf den Bereich [-PI, PI] normalisieren.
        // Dadurch ist 0.0 perfekt oben, PI und -PI sind unten.
        let out_of_bounds = pendulum_state.cart_pos.abs() > 0.9;
        let done = out_of_bounds || self.steps >= 1000;
        if done {
            self.steps = 0;
        }

        let reward = if !out_of_bounds {
            // cos(theta) ist +1.0 wenn perfekt oben, und -1.0 wenn unten.
            // Durch (+1.0) / 2.0 normieren wir den Wert perfekt auf den Bereich [0.0 bis 1.0].
            let target_reward = (pendulum_state.alpha.cos() + 1.0) / 2.0;
            
            // Wir nehmen den Wert hoch 2, damit "fast oben" extrem viel mehr belohnt wird
            // als das bloße Herabhängen im Keller.
            let mut r = target_reward.powi(2);

            // Kleine Strafen für zu wildes Bewegen, damit er oben stabilisiert
            r -= 0.01 * pendulum_state.angular_velocity.powi(2);
            r -= 0.01 * pendulum_state.cart_velocity.powi(2);

            r.max(0.0) // Verhindert, dass der Reward negativ wird!
        } else {
            0.0 // Keine harte negative Strafe, einfach Null bei Out-of-Bounds
        };


        let loss = self.dqn.set_reward_learn(state, action, reward, next_state, done);
        self.graph_loss.y_push_pop(0, loss * 2.0);

        if done {
            self.pendulum = self.initial_pendulum.clone(); 

            self.dqn.epsilon_decay();

            self.episode += 1;
            if self.episode >= 10 {
                self.episode = 0;
                self.dqn.update_target_net();
                
            }          
        }
    }

    pub fn update_drawer(&mut self, objects: &mut DrawerObjects) {
        let nodes = &mut objects.genome_nodes;
        let edges = &mut objects.genome_edges;

        self.watch_ups.start("Draw Model");
        // self.model_drawer.update(&self.dqn.target_net, nodes, edges);

        self.graph_drawer_loss.update(&self.graph_loss, edges);
        self.graph_drawer_chosen_action.update(&self.graph_chosen_action, edges);
        self.graph_drawer_actions.update(&self.graph_actions, edges);
        self.graph_drawer_angle.update(&self.graph_angle, edges);
        self.graph_drawer_angle_vel
            .update(&self.graph_angle_vel, edges);
        self.graph_drawer_cart.update(&self.graph_cart, edges);
        self.graph_drawer_cart_vel
            .update(&self.graph_cart_vel, edges);

        self.verlet_physics_drawer
            .update(&self.pendulum.verlet_physics, nodes, edges);

        self.watch_ups.stop();

        objects.ups = self.ups.get();
        self.watch_ups.update();
        objects.watch_ups = self.watch_ups.get_viewer_data();
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
