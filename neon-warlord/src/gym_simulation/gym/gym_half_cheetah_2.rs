//! A pendulum, try to balance it

use std::f32::consts::PI;

use cgmath::Zero;

use crate::{
    gym_simulation::{Vec3, gym::Gym},
    physics::xpbd::{
        Xpbd,
        xpbd_drawer::XpbdDrawer,
        xpbd_hinge_joint::{RotationAxis, XpbdHingeJoint},
        xpbd_rigid_body::XpbdRigidBody,
    },
};

const OBSERVATIONS: usize = 12;
const ACTIONS: usize = 4;

const MASS: f32 = 0.1;

#[derive(Clone)]
pub struct GymHalfCheetah2 {
    pub physics: Xpbd,
    pub drawer: XpbdDrawer,

    index_body: usize,
    _index_log_0_0: usize,
    _index_log_1_0: usize,
    _index_log_0_1: usize,
    _index_log_1_1: usize,

    index_hinge_joint_0_0: usize,
    index_hinge_joint_1_0: usize,
    index_hinge_joint_0_1: usize,
    index_hinge_joint_1_1: usize,

    state: State,
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
                Vec3::new(2.3, 0.0, 0.3),
            ],
            MASS, // 0.0
        );

        let log_0_0 = XpbdRigidBody::new(
            Vec3::new(-1.0, 0.0, 1.5),
            &[Vec3::zero(), Vec3::new(0.0, 0.0, -0.5)],
            MASS,
        );

        let log_1_0 = XpbdRigidBody::new(
            Vec3::new(1.0, 0.0, 1.5),
            &[Vec3::zero(), Vec3::new(0.0, 0.0, -0.5)],
            MASS,
        );

        let log_0_1 = XpbdRigidBody::new(
            Vec3::new(-1.0, 0.0, 1.0),
            &[Vec3::zero(), Vec3::new(0.0, 0.0, -0.5)],
            MASS,
        );

        let log_1_1 = XpbdRigidBody::new(
            Vec3::new(1.0, 0.0, 1.0),
            &[Vec3::zero(), Vec3::new(0.0, 0.0, -0.5)],
            MASS,
        );

        let index_body = physics.add_rigid_body(&body);
        let _index_log_0_0 = physics.add_rigid_body(&log_0_0);
        let _index_log_1_0 = physics.add_rigid_body(&log_1_0);
        let _index_log_0_1 = physics.add_rigid_body(&log_0_1);
        let _index_log_1_1 = physics.add_rigid_body(&log_1_1);

        let positional_compliance = 0.000;
        let angular_compliance = 0.000;

        let hinge_joint_0_0 = XpbdHingeJoint::new(
            body.particles[0],
            log_0_0.particles[0],
            -0.5 * PI,
            0.5 * PI,
            RotationAxis::Y,
            RotationAxis::Y,
            positional_compliance,
            angular_compliance,
        );
        let index_hinge_joint_0_0 =
            physics.add_hinge_joint(index_body, _index_log_0_0, hinge_joint_0_0);

        let hinge_joint_1_0 = XpbdHingeJoint::new(
            body.particles[2],
            log_1_0.particles[0],
            -0.5 * PI,
            0.5 * PI,
            RotationAxis::Y,
            RotationAxis::Y,
            positional_compliance,
            angular_compliance,
        );
        let index_hinge_joint_1_0 =
            physics.add_hinge_joint(index_body, _index_log_1_0, hinge_joint_1_0);

        let hinge_joint_0_1 = XpbdHingeJoint::new(
            log_0_0.particles[1],
            log_0_1.particles[0],
            -0.8 * PI,
            0.0 * PI,
            RotationAxis::Y,
            RotationAxis::Y,
            positional_compliance,
            angular_compliance,
        );
        let index_hinge_joint_0_1 =
            physics.add_hinge_joint(_index_log_0_0, _index_log_0_1, hinge_joint_0_1);

        let hinge_joint_1_1 = XpbdHingeJoint::new(
            log_1_0.particles[1],
            log_1_1.particles[0],
            -0.8 * PI,
            0.0 * PI,
            RotationAxis::Y,
            RotationAxis::Y,
            positional_compliance,
            angular_compliance,
        );
        let index_hinge_joint_1_1 =
            physics.add_hinge_joint(_index_log_1_0, _index_log_1_1, hinge_joint_1_1);

        let state = State {
            _pos_x: 0.0,
            pos_z: 0.0,
            vel_x: 0.0,
            vel_z: 0.0,
            rot_y: 0.0,
            leg_angle_0: 0.0,
            leg_angle_1: 0.0,
            leg_angle_2: 0.0,
            leg_angle_3: 0.0,
            leg_angular_velocity_0: 0.0,
            leg_angular_velocity_1: 0.0,
            leg_angular_velocity_2: 0.0,
            leg_angular_velocity_3: 0.0,
            actions: [0.0; 4]
        };

        Self {
            physics,
            drawer,
            index_body,
            _index_log_0_0,
            _index_log_1_0,
            _index_log_0_1,
            _index_log_1_1,
            index_hinge_joint_0_0,
            index_hinge_joint_1_0,
            index_hinge_joint_0_1,
            index_hinge_joint_1_1,
            state,
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
        self.drawer
            .update(&self.physics, producer_nodes, producer_edges);
    }

    fn update_verlet_physics(&mut self, dt: f32) {
        // self.physics.step(dt * 0.04);
        self.physics.step(dt * 1.0);
        // self.verlet_physics.update(dt * 0.4);

        let body = &self.physics.rigid_bodies[self.index_body];

        let hinge_0_0 = &self.physics.hinge_joints[self.index_hinge_joint_0_0].constraint;
        let hinge_1_0 = &self.physics.hinge_joints[self.index_hinge_joint_1_0].constraint;
        let hinge_0_1 = &self.physics.hinge_joints[self.index_hinge_joint_0_1].constraint;
        let hinge_1_1 = &self.physics.hinge_joints[self.index_hinge_joint_1_1].constraint;

        let state = State {
            _pos_x: body.p.pos.x,
            pos_z: body.p.pos.z,
            vel_x: body.p.vel.x,
            vel_z: body.p.vel.z,
            rot_y: body.r.get_rotation_as_euler().y,
            leg_angle_0: hinge_0_0.get_angle(),
            leg_angle_1: hinge_1_0.get_angle(),
            leg_angle_2: hinge_0_1.get_angle(),
            leg_angle_3: hinge_1_1.get_angle(),
            leg_angular_velocity_0: hinge_0_0.get_angular_velocity(),
            leg_angular_velocity_1: hinge_1_0.get_angular_velocity(),
            leg_angular_velocity_2: hinge_0_1.get_angular_velocity(),
            leg_angular_velocity_3: hinge_1_1.get_angular_velocity(),
            actions: self.state.actions
        };

        self.state = state;
    }

    fn get_state(&self) -> [f32; OBSERVATIONS] {
        let state = &self.state;
        [
            state.pos_z,
            state.vel_x,
            state.vel_z,
            state.rot_y,
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

    fn update(&mut self, actions: &[f32; ACTIONS], _dt: f32) {
        // self.apply_static_constraint();

        let torque = [
            actions[0] * 10.0,
            actions[1] * 10.0,
            actions[2] * 10.0,
            actions[3] * 10.0,
        ];

        

        let hinge_0_0 = &mut self.physics.hinge_joints[self.index_hinge_joint_0_0].constraint;
        hinge_0_0.set_external_torque(torque[0]);

        let hinge_1_0 = &mut self.physics.hinge_joints[self.index_hinge_joint_1_0].constraint;
        hinge_1_0.set_external_torque(torque[1]);

        let hinge_0_1 = &mut self.physics.hinge_joints[self.index_hinge_joint_0_1].constraint;
        hinge_0_1.set_external_torque(torque[2]);

        let hinge_1_1 = &mut self.physics.hinge_joints[self.index_hinge_joint_1_1].constraint;
        hinge_1_1.set_external_torque(torque[3]);

        self.state.actions = *actions;
    }

    fn get_reward(&self) -> (f32, bool) {

        // Reward target velocity
        let target_velocity = 6.0;
        let mut reward = 1.0 - ((self.state.vel_x - target_velocity)/target_velocity).powi(2);

        // Penalize rotation
        let rot_cost = self.state.rot_y.powi(2);
        reward -= rot_cost * 0.1;
        // println!("self.state.rot_y: {}", self.state.rot_y);

        // Penalize vel z
        // let vel_z_cost = self.state.vel_z.powi(2);
        // reward -= vel_z_cost * 0.1;

        // Penalize huge actions
        let mut action_cost = 0.0;
        for action in self.state.actions {
            action_cost += action.powi(2);
        }
        reward -= action_cost * 0.1;
        
        // No negative rewards
        let reward = reward.max(0.0);

        (reward, false)
    }

    fn reset(&mut self) {
        let alpha = (fastrand::f32() * 2.0 - 1.0) * std::f32::consts::PI;

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
    rot_y: f32,
    leg_angle_0: f32,
    leg_angle_1: f32,
    leg_angle_2: f32,
    leg_angle_3: f32,
    leg_angular_velocity_0: f32,
    leg_angular_velocity_1: f32,
    leg_angular_velocity_2: f32,
    leg_angular_velocity_3: f32,

    actions: [f32; 4],
}
