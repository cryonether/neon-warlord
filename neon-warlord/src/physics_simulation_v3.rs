//! Next iteration of the verlet physics simulation

use forward_renderer::height_map::HeightMapInterface;
use wgpu_renderer::performance_monitor::{Fps, watch::Watch};

use crate::{
    advanced_composition::{
        definition::{
            ParsedDefinition, get_pendulum_definition, get_pendulum_definition_fitness_function,
        },
        swarm::Swarm,
    }, physics_simulation_v3_drawer::DrawerObjects, triple_buffer, verlet_physics::solver::Solver, worker_thread,
};

type Vec3 = cgmath::Vector3<f32>;

pub const WATCH_POINTS_SIZE: usize = 10;


pub struct PhysicsSimulationV3 {
    producer: triple_buffer::Producer<DrawerObjects>,

    // Physics
    swarm: Swarm,
    solver: Solver,
    ticks: u64,

    // Debug
    ups: Fps,
    last_render_time: instant::Instant,
    watch_ups: Watch<WATCH_POINTS_SIZE>,
}

impl PhysicsSimulationV3 {
    pub fn new(
        producer: triple_buffer::Producer<DrawerObjects>
    ) -> Self {
        // agent 0
        let pos = Vec3::new(0.0, 0.0, 2.0);
        let scale = 0.1;
        let definition = get_pendulum_definition();
        let fitness_function = get_pendulum_definition_fitness_function();
        let parsed_definition = ParsedDefinition::parse(&definition, pos, scale);

        let swarm_size = 1000;
        // let swarm_size = 40000;
        let swarm = Swarm::new(&parsed_definition, swarm_size)
            .set_fitness_functions(&[fitness_function]);

        // solver
        let solver = Solver::new();

        // Debug
        let ups = Fps::new();
        let watch_ups = Watch::new();

        Self {
            producer,
        
            swarm,
            solver,
            ticks: 0,

            ups,
            last_render_time: instant::Instant::now(),
            watch_ups,
        }
    }

    // Update

    pub fn update_physics(&mut self, height_map: &impl HeightMapInterface) {
        let dt = 1.0 / 60.0;
        self.ticks += 1;

        self.watch_ups.start(0, "swarm.update_physics");
            self.swarm.update_physics(dt);
        self.watch_ups.stop(0);

        self.watch_ups.start(1, "Solver");
            self.solver.update_advanced_composites(
                &mut self.swarm.advanced_composition,
                height_map,
                dt,
            );
        self.watch_ups.stop(1);

        // ups
        let now = instant::Instant::now();
        let dt = now - self.last_render_time;
        self.last_render_time = now;
        self.ups.update(dt);
        }
        
        pub fn update_drawer(&mut self) {
            // self.watch_ups.start(2, "update_drawer");
            let data = self.producer.buffer();
            data.clear();
            
        self.watch_ups.start(2, "swarm.update_drawer");
        self.swarm.update_drawer(data);
        self.watch_ups.stop(2);
            
        data.ups = self.ups.get(); 
        self.watch_ups.update();
        data.watch_ups = self.watch_ups.get_viewer_data();
        
        self.producer.publish();
    }
}

pub struct PhysicSimThread<T> {
    pub sim: PhysicsSimulationV3,
    pub height_map: T,
}

impl<T> worker_thread::Update for PhysicSimThread<T>
where T: HeightMapInterface
{
    fn update(&mut self) {
        self.sim.update_physics(&self.height_map);
        self.sim.update_drawer();
    }
}