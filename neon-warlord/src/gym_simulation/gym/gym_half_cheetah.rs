//! A pendulum, try to balance it


use cgmath::Zero;

use crate::{
    gym_simulation::{Vec3, gym::Gym}, physics::verlet_velocity::{VerletVelocityPhysics, verlet_velocity_drawer::VerletVelocityDrawer},
};

const OBSERVATIONS: usize = 11;
const ACTIONS: usize = 4;

const MASS: f32 = 0.1;

#[derive(Clone)]
pub struct GymHalfCheetah {
    pub verlet_physics: VerletVelocityPhysics,
    pub drawer: VerletVelocityDrawer,

    particles_body:  [usize; 4],
    _particles_leg_0: [usize; 2],
    _particles_leg_1: [usize; 2],

    state: State,

    vel_x_average: f32,
}

impl GymHalfCheetah {
    pub fn new() -> Self {
        let mut verlet_physics = VerletVelocityPhysics::new();
        let drawer = VerletVelocityDrawer::new(&verlet_physics, 0.1, Vec3::zero());
        
        let radius = 0.1;

        let stiffness_distance = 800.0;
        let damping_distance = 0.999;

        let stiffness_angle = 800.0;
        let damping_angle = 0.999;


        let stiffness_motor = 800.0;
        let damping_motor = 0.999;
        let max_torque = 100.0;
        
        // body
        let particles_body = [
            verlet_physics.particles.push(Vec3::new(-1.0, 0.0, 1.5), radius, MASS * 4.0),
            verlet_physics.particles.push(Vec3::new(0.0, 0.0, 1.5), radius, MASS * 4.0),
            verlet_physics.particles.push(Vec3::new(1.0, 0.0, 1.5), radius, MASS * 4.0),
            verlet_physics.particles.push(Vec3::new(1.3, 0.0, 1.8), radius, MASS * 4.0),
        ];

        verlet_physics.push_constraint_distance_infer(particles_body[0], particles_body[1], stiffness_distance, damping_distance);
        verlet_physics.push_constraint_distance_infer(particles_body[1], particles_body[2], stiffness_distance, damping_distance);
        verlet_physics.push_constraint_distance_infer(particles_body[2], particles_body[3], stiffness_distance, damping_distance);

        verlet_physics.push_constraint_angle_infer(particles_body[0], particles_body[1], particles_body[2], stiffness_angle, damping_angle);
        verlet_physics.push_constraint_angle_infer(particles_body[1], particles_body[2], particles_body[3], stiffness_angle, damping_angle);

        // leg 0
        let _particles_leg_0 = [
            verlet_physics.particles.push(Vec3::new(-1.0, 0.0, 1.0), radius, MASS),
            verlet_physics.particles.push(Vec3::new(-1.0, 0.0, 0.5), radius, MASS),
        ];

        verlet_physics.push_constraint_distance_infer(particles_body[0], _particles_leg_0[0], stiffness_distance, damping_distance);
        verlet_physics.push_constraint_distance_infer(_particles_leg_0[0], _particles_leg_0[1], stiffness_distance, damping_distance);

        verlet_physics.push_constraint_angle_motor(particles_body[1], particles_body[0], _particles_leg_0[0], 
        f32::to_radians(20.0),
        f32::to_radians(160.0),
        max_torque,
        stiffness_motor,
        damping_motor, 
        );
        verlet_physics.push_constraint_angle_motor(particles_body[0], _particles_leg_0[0], _particles_leg_0[1], 
        f32::to_radians(90.0),
        f32::to_radians(270.0),
        max_torque,
        stiffness_motor,
        damping_motor, 
        );


        // leg 1
        let _particles_leg_1 = [
            verlet_physics.particles.push(Vec3::new(1.0, 0.0, 1.0), radius, MASS),
            verlet_physics.particles.push(Vec3::new(1.0, 0.0, 0.5), radius, MASS),
        ];
        
        verlet_physics.push_constraint_distance_infer(particles_body[2], _particles_leg_1[0], stiffness_distance, damping_distance);
        verlet_physics.push_constraint_distance_infer(_particles_leg_1[0], _particles_leg_1[1], stiffness_distance, damping_distance);


        verlet_physics.push_constraint_angle_motor(particles_body[1], particles_body[2], _particles_leg_1[0], 
        f32::to_radians(20.0),
        f32::to_radians(160.0),
        max_torque,
        stiffness_motor,
        damping_motor, 
        );
        verlet_physics.push_constraint_angle_motor(particles_body[2], _particles_leg_1[0], _particles_leg_1[1], 
        f32::to_radians(90.0),
        f32::to_radians(270.0),
        max_torque,
        stiffness_motor,
        damping_motor, 
        );

        let state = State {
            _pos_x: 1.3,
            pos_z: 2.3,
            vel_x: 0.0,
            vel_z: 0.0,
            leg_angle_0: 0.0,
            leg_angle_1: 0.0,
            leg_angle_2: 0.0,
            leg_angle_3: 0.0,
            leg_angular_velocity_0: 0.0,
            leg_angular_velocity_1: 0.0,
            leg_angular_velocity_2: 0.0,
            leg_angular_velocity_3: 0.0,
        };

        Self {
            verlet_physics,
            drawer,
            particles_body,
            _particles_leg_0,
            _particles_leg_1,
            state,
            vel_x_average: 0.0,
        }
    }

    // Calculates the sin_x and cos_x of the pole and the velocities
    fn update(&mut self, force: [f32; 4], _dt: f32) -> State {

        self.verlet_physics.set_external_torque(0, force[0]);
        self.verlet_physics.set_external_torque(1, force[1]);
        self.verlet_physics.set_external_torque(2, force[2]);
        self.verlet_physics.set_external_torque(3, force[3]);

        self.verlet_physics.step();

        let angle_0 = self.verlet_physics.get_motor_angle(0);
        let angle_1 = self.verlet_physics.get_motor_angle(1);
        let angle_2 = self.verlet_physics.get_motor_angle(2);
        let angle_3 = self.verlet_physics.get_motor_angle(3);

        let angular_velocity_0 = self.verlet_physics.get_motor_angular_velocity(0);
        let angular_velocity_1 = self.verlet_physics.get_motor_angular_velocity(1);
        let angular_velocity_2 = self.verlet_physics.get_motor_angular_velocity(2);
        let angular_velocity_3 = self.verlet_physics.get_motor_angular_velocity(3);

        let front_tip = self.particles_body[3];
        let pos = self.verlet_physics.particles.get_pos(front_tip);
        let vel = self.verlet_physics.particles.get_vel(front_tip);

        State {
            _pos_x: pos.x,
            pos_z: pos.z,
            vel_x: vel.x,
            vel_z: vel.z,

            leg_angle_0: angle_0,
            leg_angle_1: angle_1,
            leg_angle_2: angle_2,
            leg_angle_3: angle_3,

            leg_angular_velocity_0: angular_velocity_0,
            leg_angular_velocity_1: angular_velocity_1,
            leg_angular_velocity_2: angular_velocity_2,
            leg_angular_velocity_3: angular_velocity_3,
        }
    }

    fn apply_static_constraint(&mut self) {

    }
}

impl Gym<OBSERVATIONS, ACTIONS> for GymHalfCheetah {
    fn set_drawer_position(&mut self, pos: &Vec3) {
        self.drawer.position = *pos;
    }

    fn set_drawer_radius(&mut self, radius: f32) {
        self.drawer.radius = radius;
    }

    fn draw(
        &self,
        producer_nodes: &mut Vec<forward_renderer::particle_shader::Instance>,
        producer_edges: &mut Vec<forward_renderer::particle_shader_two_point::Instance>,
    ) {
        self.drawer.update(&self.verlet_physics, producer_nodes, producer_edges);
    }


    fn update_verlet_physics(&mut self, _dt: f32) {

    }

    fn get_state(&self) -> [f32; OBSERVATIONS] {
        let state = &self.state;
        [
            state.pos_z,
            state.vel_x,
            state.vel_z,
            state.leg_angle_0,
            state.leg_angle_1,
            state.leg_angle_2,
            state.leg_angle_3,
            state.leg_angular_velocity_0,
            state.leg_angular_velocity_1,
            state.leg_angular_velocity_2,
            state.leg_angular_velocity_3,
        ]
    }

    fn update(&mut self, actions: &[f32; ACTIONS], dt: f32) {
        self.apply_static_constraint();

        let force = [
            actions[0] * 20.0,
            actions[1] * 20.0,
            actions[2] * 20.0,
            actions[3] * 20.0,
        ];

        // let force = actions[0] * 0.7;
        self.state = self.update(force, dt);

        self.vel_x_average = (self.vel_x_average + self.state.vel_x) * 0.92; 
    }

    fn get_reward(&self) -> (f32, bool) {

        let vel_x = self.state.vel_x * 0.1;
        let reward = vel_x.clamp(0.0, 1.0);
        (reward, false)
    }

    fn reset(&mut self) {
        let alpha = (fastrand::f32() * 2.0 - 1.0) * std::f32::consts::PI;
        // let alpha = -std::f32::consts::PI * 0.5  + (fastrand::f32() * 2.0 - 1.0) * 0.3;

        let _pol_sin_alpha = alpha.sin();
        let _pol_cos_alpha = alpha.cos();

        let drawer = self.drawer.clone();
        
        *self = Self::new();

        self.drawer = drawer;
    }
    

    

}


#[derive(Clone)]
pub struct State {
    _pos_x: f32,
    pos_z: f32,
    vel_x: f32,
    vel_z: f32,
    leg_angle_0: f32,
    leg_angle_1: f32,
    leg_angle_2: f32,
    leg_angle_3: f32,
    leg_angular_velocity_0: f32,
    leg_angular_velocity_1: f32,
    leg_angular_velocity_2: f32,
    leg_angular_velocity_3: f32,
}
