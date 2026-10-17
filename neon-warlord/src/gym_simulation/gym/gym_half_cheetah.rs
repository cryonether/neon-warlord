//! A pendulum, try to balance it

use std::iter::zip;

use cgmath::{InnerSpace, Zero};
use wgpu::rwh;

use crate::{
    gym_simulation::{Vec3, gym::Gym}, physics::{verlet_physics_simd::VerletPhysicsSimd, verlet_velocity::{VerletVelocity, verlet_velocity_drawer::VerletVelocityDrawer}},
};

const OBSERVATIONS: usize = 11;
const ACTIONS: usize = 4;

const MASS: f32 = 0.1;

#[derive(Clone)]
pub struct GymHalfCheetah {
    pub verlet_physics: VerletVelocity,
    pub drawer: VerletVelocityDrawer,

    particles_body:  [usize; 4],
    particles_leg_0: [usize; 2],
    particles_leg_1: [usize; 2],

    state: State,

    vel_x_average: f32,
}

impl GymHalfCheetah {
    pub fn new() -> Self {
        let mut verlet_physics = VerletVelocity::new();
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
            verlet_physics.push_particle(Vec3::new(-1.0, 0.0, 1.5), radius, MASS * 4.0),
            verlet_physics.push_particle(Vec3::new(0.0, 0.0, 1.5), radius, MASS * 4.0),
            verlet_physics.push_particle(Vec3::new(1.0, 0.0, 1.5), radius, MASS * 4.0),
            verlet_physics.push_particle(Vec3::new(1.3, 0.0, 1.8), radius, MASS * 4.0),
        ];

        verlet_physics.push_constraint_distance_infer(particles_body[0], particles_body[1], stiffness_distance, damping_distance);
        verlet_physics.push_constraint_distance_infer(particles_body[1], particles_body[2], stiffness_distance, damping_distance);
        verlet_physics.push_constraint_distance_infer(particles_body[2], particles_body[3], stiffness_distance, damping_distance);

        verlet_physics.push_constraint_angle_infer(particles_body[0], particles_body[1], particles_body[2], stiffness_angle, damping_angle);
        verlet_physics.push_constraint_angle_infer(particles_body[1], particles_body[2], particles_body[3], stiffness_angle, damping_angle);

        // leg 0
        let particles_leg_0 = [
            verlet_physics.push_particle(Vec3::new(-1.0, 0.0, 1.0), radius, MASS),
            verlet_physics.push_particle(Vec3::new(-1.0, 0.0, 0.5), radius, MASS),
        ];

        verlet_physics.push_constraint_distance_infer(particles_body[0], particles_leg_0[0], stiffness_distance, damping_distance);
        verlet_physics.push_constraint_distance_infer(particles_leg_0[0], particles_leg_0[1], stiffness_distance, damping_distance);

        verlet_physics.push_constraint_angle_motor(particles_body[1], particles_body[0], particles_leg_0[0], 
        f32::to_radians(20.0),
        f32::to_radians(160.0),
        max_torque,
        stiffness_motor,
        damping_motor, 
        );
        verlet_physics.push_constraint_angle_motor(particles_body[0], particles_leg_0[0], particles_leg_0[1], 
        f32::to_radians(90.0),
        f32::to_radians(270.0),
        max_torque,
        stiffness_motor,
        damping_motor, 
        );


        // leg 1
        let particles_leg_1 = [
            verlet_physics.push_particle(Vec3::new(1.0, 0.0, 1.0), radius, MASS),
            verlet_physics.push_particle(Vec3::new(1.0, 0.0, 0.5), radius, MASS),
        ];
        
        verlet_physics.push_constraint_distance_infer(particles_body[2], particles_leg_1[0], stiffness_distance, damping_distance);
        verlet_physics.push_constraint_distance_infer(particles_leg_1[0], particles_leg_1[1], stiffness_distance, damping_distance);


        verlet_physics.push_constraint_angle_motor(particles_body[1], particles_body[2], particles_leg_1[0], 
        f32::to_radians(20.0),
        f32::to_radians(160.0),
        max_torque,
        stiffness_motor,
        damping_motor, 
        );
        verlet_physics.push_constraint_angle_motor(particles_body[2], particles_leg_1[0], particles_leg_1[1], 
        f32::to_radians(90.0),
        f32::to_radians(270.0),
        max_torque,
        stiffness_motor,
        damping_motor, 
        );

        let state = State {
            pos_x: 1.3,
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
            particles_leg_0,
            particles_leg_1,
            state,
            vel_x_average: 0.0,
        }
    }

    // fn apply_angle_force(&mut self, index_base: usize, index_pole: usize, force: f32, state_leg: StateLeg, dt: f32) -> StateLeg {
    //     let index_pole: usize = index_pole;
    //     let index_base: usize = index_base;
    //     let pos_pole: cgmath::Vector3<f32> = self.verlet_physics.particles.position(index_pole);
    //     let pos_base: cgmath::Vector3<f32> = self.verlet_physics.particles.position(index_base);

    //     // move to the center
    //     let pos_pole = pos_pole - pos_base;
    //     // let _pos_static = pos_static - pos_static;

    //     // normalize
    //     let pos_pole = pos_pole.normalize();

    //     // tangent
    //     let tangent = cgmath::Vector3::new(-pos_pole.z, 0.0, pos_pole.x);

    //     // apply force
    //     let acc = tangent * force / MASS;
    //     self.verlet_physics.particles.accelerate(index_pole, acc);

    //     // get sin and cos
    //     let pol_sin_alpha = pos_pole.z;
    //     let pol_cos_alpha = pos_pole.x;

    //     // get velocities
    //     let pol_sin_alpha_v = (pol_sin_alpha - state_leg.sin_alpha) / dt;
    //     let pol_cos_alpha_v = (pol_cos_alpha - state_leg.cos_alpha) / dt;

    //     //
    //     // d sin(α)/dt = cos(α) α'
    //     // d cos(α)/dt = -sin(α) α'
    //     //
    //     // cos(α) · (cos(α) α') - sin(α) · (-sin(α) α') = α'
    //     //
    //     let angular_velocity = pol_cos_alpha * pol_sin_alpha_v - pol_sin_alpha * pol_cos_alpha_v;


    //     StateLeg {
    //         sin_alpha: pol_sin_alpha,
    //         cos_alpha: pol_cos_alpha,
    //         angular_velocity,
    //     }

    // }

    // Calculates the sin_x and cos_x of the pole and the velocities
    fn update(&mut self, force: [f32; 4], dt: f32) -> State {

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
        let pos = self.verlet_physics.get_pos(front_tip);
        let vel = self.verlet_physics.get_vel(front_tip);

        State {
            pos_x: pos.x,
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
        // let friction = 1.00; // 0.0 = no friction, 1.0 = full stop

        // let particles = &mut self.verlet_physics.particles;
        // let len = particles.len();

        // for i in 0..len {
        //     let radius = particles.radius[i];

        //     if particles.z[i] < radius {
        //         // Resolve collision with the floor
        //         particles.z[i] = radius;
        //         particles.prev_z[i] = radius;

        //         // Apply friction to horizontal velocity
        //         let vx = particles.x[i] - particles.prev_x[i];
        //         // particles.prev_x[i] = particles.x[i] - vx * (1.0 - friction);
        //         particles.prev_x[i] = particles.x[i];
        //     }
        // }
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


    fn update_verlet_physics(&mut self, dt: f32) {
        // self.verlet_physics.update(dt * 0.4);
        // self.verlet_physics.update(dt * 1.0);
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

        // let mut vel_x_sum = 0.0;
        // for elem in &self.verlet_physics.vel_x {
        //     vel_x_sum += elem;
        // }

        // let vel_x_average = vel_x_sum / self.verlet_physics.vel_x.len() as f32;



        let vel_x = self.state.vel_x * 0.1;


        // let pos_x = self.state.pos_x * 0.1;

        let reward = vel_x.clamp(0.0, 1.0);

        // let reward = self.vel_x_average * 0.1;

        // let reward = vel_x * (2.0 - vel_x);

        // let reward = vel_x_average * 0.1;

        // let reward = reward.max(0.0);

        (reward, false)
    }

    fn reset(&mut self) {
        let alpha = (fastrand::f32() * 2.0 - 1.0) * std::f32::consts::PI;
        // let alpha = -std::f32::consts::PI * 0.5  + (fastrand::f32() * 2.0 - 1.0) * 0.3;

        let pol_sin_alpha = alpha.sin();
        let pol_cos_alpha = alpha.cos();

        // self.verlet_physics.particles.reset_position(
        //     self.particle_pole[0],
        //     Vec3::new(pol_cos_alpha, 0.0, pol_sin_alpha),
        // );

        // self.state = State {
        //     pol_sin_alpha,
        //     pol_cos_alpha,
        //     angular_velocity: 0.0,
        // }

        let drawer = self.drawer.clone();
        
        *self = Self::new();

        self.drawer = drawer;
    }
    

    

}


#[derive(Clone)]
pub struct StateLeg {
    sin_alpha: f32,
    cos_alpha: f32,
    angular_velocity: f32,

//     leg_0_sin_alpha, leg_0_cos_alpha, log_0_angular_velocity
// leg_1_sin_alpha, leg_1_cos_alpha, log_1_angular_velocity
// leg_2_sin_alpha, leg_2_cos_alpha, log_2_angular_velocity
// leg_3_sin_alpha, leg_3_cos_alpha, log_3_angular_velocity
}


#[derive(Clone)]
pub struct State {
    pos_x: f32,
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
