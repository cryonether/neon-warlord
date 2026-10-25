//! A pendulum, try to balance it

use std::{f32::consts::PI, iter::zip};

use cgmath::{InnerSpace, Zero};
use wgpu::rwh;

use crate::{
    gym_simulation::{Vec3, gym::Gym}, physics::{verlet_physics_simd::VerletPhysicsSimd, verlet_velocity::{VerletVelocityPhysics, verlet_velocity_drawer::VerletVelocityDrawer}, xpbd::{Xpbd, xpbd_angular_constraint::XpbdAngularConstraint, xpbd_drawer::XpbdDrawer, xpbd_hinge_joint::{RotationAxis, XpbdHingeJoint}, xpbd_positional_constraint::XpbdPositionalConstraint, xpbd_rigid_body::XpbdRigidBody}},
};

const OBSERVATIONS: usize = 11;
const ACTIONS: usize = 4;

const MASS: f32 = 0.1;

#[derive(Clone)]
pub struct GymHalfCheetah2 {
    pub physics: Xpbd,
    pub drawer: XpbdDrawer,

    index_body: usize,
    index_log_0_0: usize,
    index_log_1_0: usize,
    index_log_0_1: usize,
    index_log_1_1: usize,

    index_hinge_joint_0_0: usize,
    index_hinge_joint_1_0: usize,
    index_hinge_joint_0_1: usize,
    index_hinge_joint_1_1: usize,
}

impl GymHalfCheetah2 {
    pub fn new() -> Self {
        let mut physics = Xpbd::new();
        let drawer = XpbdDrawer::new(&physics, 0.1, Vec3::zero());
        
        let body = XpbdRigidBody::new(
            Vec3::new(-1.0, 0.0, 1.5), 
            &[
                Vec3::zero(), 
                Vec3::new(1.0, 0.0, 0.0), 
                Vec3::new(2.0, 0.0, 0.0),
                Vec3::new(2.3, 0.0, 0.3)
            ],
            MASS
            // 0.0
        );

        let log_0_0 = XpbdRigidBody::new(
            Vec3::new(-1.0, 0.0, 1.5), 
            &[Vec3::zero(), Vec3::new(0.0, 0.0, -0.5)],
            MASS
        );

        let log_1_0 = XpbdRigidBody::new(
            Vec3::new(1.0, 0.0, 1.5), 
            &[Vec3::zero(), Vec3::new(0.0, 0.0, -0.5)],
            MASS
        );

        let log_0_1 = XpbdRigidBody::new(
            Vec3::new(-1.0, 0.0, 1.0), 
            &[Vec3::zero(), Vec3::new(0.0, 0.0, -0.5)],
            MASS
        );

        let log_1_1 = XpbdRigidBody::new(
            Vec3::new(1.0, 0.0, 1.0), 
            &[Vec3::zero(), Vec3::new(0.0, 0.0, -0.5)],
            MASS
        );

        let index_body = physics.add_rigid_body(&body);
        let index_log_0_0 = physics.add_rigid_body(&log_0_0);
        let index_log_1_0 = physics.add_rigid_body(&log_1_0);
        let index_log_0_1 = physics.add_rigid_body(&log_0_1);
        let index_log_1_1 = physics.add_rigid_body(&log_1_1);

        let hinge_joint_0_0 = XpbdHingeJoint::new(
            body.particles[0], 
            log_0_0.particles[0], 
            -0.5 * PI, 
            0.5 * PI, 
            RotationAxis::Y,
            RotationAxis::Y,
            0.0002, 
            0.0002
        );
        let index_hinge_joint_0_0 = physics.add_hinge_joint(
            index_body, 
            index_log_0_0, 
            hinge_joint_0_0
        );


        let hinge_joint_1_0 = XpbdHingeJoint::new(
            body.particles[2], 
            log_1_0.particles[0], 
            -0.5 * PI, 
            0.5 * PI, 
            RotationAxis::Y,
            RotationAxis::Y,
            0.0002, 
            0.0002
        );
        let index_hinge_joint_1_0 = physics.add_hinge_joint(
            index_body, 
            index_log_1_0, 
            hinge_joint_1_0
        );


        let hinge_joint_0_1 = XpbdHingeJoint::new(
            log_0_0.particles[1], 
            log_0_1.particles[0], 
            -0.5 * PI, 
            0.5 * PI, 
            RotationAxis::Y,
            RotationAxis::Y,
            0.0002, 
            0.0002
        );
         let index_hinge_joint_0_1 = physics.add_hinge_joint(
            index_log_0_0, 
            index_log_0_1, 
            hinge_joint_0_1
        );


        let hinge_joint_1_1 = XpbdHingeJoint::new(
            log_1_0.particles[1], 
            log_1_1.particles[0], 
            -0.5 * PI, 
            0.5 * PI, 
            RotationAxis::Y,
            RotationAxis::Y,
            0.0002, 
            0.0002
        );
         let index_hinge_joint_1_1 = physics.add_hinge_joint(
            index_log_1_0, 
            index_log_1_1, 
            hinge_joint_1_1
        );

        Self {
            physics,
            drawer,
            index_body,
            index_log_0_0,
            index_log_1_0,
            index_log_0_1,
            index_log_1_1,
            index_hinge_joint_0_0,
            index_hinge_joint_1_0,
            index_hinge_joint_0_1,
            index_hinge_joint_1_1,

        }
    }
}

impl Gym<OBSERVATIONS, ACTIONS> for GymHalfCheetah2 {
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
        self.drawer.update(&self.physics, producer_nodes, producer_edges);
    }


    fn update_verlet_physics(&mut self, dt: f32) {
        // self.physics.step(dt * 0.04);
        self.physics.step(dt * 1.0);
        // self.verlet_physics.update(dt * 0.4);
    }

    fn get_state(&self) -> [f32; OBSERVATIONS] {
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    }

    fn update(&mut self, actions: &[f32; ACTIONS], dt: f32) {
        // self.apply_static_constraint();

        let force = [
            actions[0] * 20.0,
            actions[1] * 20.0,
            actions[2] * 20.0,
            actions[3] * 20.0,
        ];

        // // let force = actions[0] * 0.7;
        // self.state = self.update(force, dt);

        // self.vel_x_average = (self.vel_x_average + self.state.vel_x) * 0.92; 
    }

    fn get_reward(&self) -> (f32, bool) {

        // let mut vel_x_sum = 0.0;
        // for elem in &self.verlet_physics.vel_x {
        //     vel_x_sum += elem;
        // }

        // let vel_x_average = vel_x_sum / self.verlet_physics.vel_x.len() as f32;



        // let vel_x = self.state.vel_x * 0.1;


        // let pos_x = self.state.pos_x * 0.1;

        // let reward = vel_x.clamp(0.0, 1.0);

        // let reward = self.vel_x_average * 0.1;

        // let reward = vel_x * (2.0 - vel_x);

        // let reward = vel_x_average * 0.1;

        // let reward = reward.max(0.0);

        (0.0, false)
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
