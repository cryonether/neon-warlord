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

const OBSERVATIONS: usize = 25;
const ACTIONS: usize = 8;

const MASS: f32 = 0.1;

#[derive(Clone)]
pub struct GymDog {
    pub physics: Xpbd,
    pub drawer: XpbdDrawer,

    index_body: usize,
    index_leg_0_0: usize,
    index_leg_1_0: usize,
    index_leg_2_0: usize,
    index_leg_3_0: usize,

    index_leg_0_1: usize,
    index_leg_1_1: usize,
    index_leg_2_1: usize,
    index_leg_3_1: usize,

    index_hinge_joint_0_0: usize,
    index_hinge_joint_1_0: usize,
    index_hinge_joint_2_0: usize,
    index_hinge_joint_3_0: usize,

    index_hinge_joint_0_1: usize,
    index_hinge_joint_1_1: usize,
    index_hinge_joint_2_1: usize,
    index_hinge_joint_3_1: usize,

    state: State,
}

impl GymDog {
    pub fn new() -> Self {
        let mut physics = Xpbd::new();
        let radius = 0.1;
        let drawer = XpbdDrawer::new(&physics, radius, Vec3::zero());

        let width2 = 0.3;
        let length2 = 0.65;

        let body = XpbdRigidBody::new(
            Vec3::new(-length2, 0.0, 1.5),
            &[
                Vec3::new(0.0, width2, 0.0),
                Vec3::new(length2, width2, 0.0),
                Vec3::new(length2 * 2.0, width2, 0.0),

                Vec3::new(0.0, -width2, 0.0),
                Vec3::new(length2, -width2, 0.0),
                Vec3::new(length2 * 2.0, -width2, 0.0),

                Vec3::new(length2 * 2.0, 0.0, 0.0),
                Vec3::new(length2 * 2.0 + 0.3, 0.0, 0.3),
            ],
            MASS, // 0.0
            radius,
        );

        // Upper legs

        let leg_0_0 = XpbdRigidBody::new(
            Vec3::new(-length2, width2, 1.5),
            &[Vec3::zero(), Vec3::new(0.0, 0.0, -0.5)],
            MASS,
            radius,
        );

        let leg_1_0 = XpbdRigidBody::new(
            Vec3::new(length2, width2, 1.5),
            &[Vec3::zero(), Vec3::new(0.0, 0.0, -0.5)],
            MASS,
            radius,
        );

        let leg_2_0 = XpbdRigidBody::new(
            Vec3::new(-length2, -width2, 1.5),
            &[Vec3::zero(), Vec3::new(0.0, 0.0, -0.5)],
            MASS,
            radius,
        );


        let leg_3_0 = XpbdRigidBody::new(
            Vec3::new(length2, -width2, 1.5),
            &[Vec3::zero(), Vec3::new(0.0, 0.0, -0.5)],
            MASS,
            radius,
        );

        // Lower legs

        let leg_0_1 = XpbdRigidBody::new(
            Vec3::new(-length2, width2, 1.0),
            &[Vec3::zero(), Vec3::new(0.0, 0.0, -0.5)],
            MASS,
            radius,
        );

        let leg_1_1 = XpbdRigidBody::new(
            Vec3::new(length2, width2, 1.0),
            &[Vec3::zero(), Vec3::new(0.0, 0.0, -0.5)],
            MASS,
            radius,
        );

        let leg_2_1 = XpbdRigidBody::new(
            Vec3::new(-length2, -width2, 1.0),
            &[Vec3::zero(), Vec3::new(0.0, 0.0, -0.5)],
            MASS,
            radius,
        );

        let leg_3_1 = XpbdRigidBody::new(
            Vec3::new(length2, -width2, 1.0),
            &[Vec3::zero(), Vec3::new(0.0, 0.0, -0.5)],
            MASS,
            radius,
        );


        let index_body = physics.add_rigid_body(&body);
        
        let index_leg_0_0 = physics.add_rigid_body(&leg_0_0);
        let index_leg_1_0 = physics.add_rigid_body(&leg_1_0);
        let index_leg_2_0 = physics.add_rigid_body(&leg_2_0);
        let index_leg_3_0 = physics.add_rigid_body(&leg_3_0);

        let index_leg_0_1 = physics.add_rigid_body(&leg_0_1);
        let index_leg_1_1 = physics.add_rigid_body(&leg_1_1);
        let index_leg_2_1 = physics.add_rigid_body(&leg_2_1);
        let index_leg_3_1 = physics.add_rigid_body(&leg_3_1);

        let positional_compliance = 0.000;
        let angular_compliance = 0.000;

        // Upper legs

        let hinge_joint_0_0 = XpbdHingeJoint::new(
            body.particles[0],
            leg_0_0.particles[0],
            -0.8 * PI,
            0.8 * PI,
            RotationAxis::Y,
            RotationAxis::Y,
            positional_compliance,
            angular_compliance,
        );
        let index_hinge_joint_0_0 =
            physics.add_hinge_joint(index_body, index_leg_0_0, hinge_joint_0_0);

        let hinge_joint_1_0 = XpbdHingeJoint::new(
            body.particles[2],
            leg_1_0.particles[0],
            -0.8 * PI,
            0.8 * PI,
            RotationAxis::Y,
            RotationAxis::Y,
            positional_compliance,
            angular_compliance,
        );
        let index_hinge_joint_1_0 =
            physics.add_hinge_joint(index_body, index_leg_1_0, hinge_joint_1_0);


        let hinge_joint_2_0 = XpbdHingeJoint::new(
            body.particles[3],
            leg_2_0.particles[0],
            -0.8 * PI,
            0.8 * PI,
            RotationAxis::Y,
            RotationAxis::Y,
            positional_compliance,
            angular_compliance,
        );
        let index_hinge_joint_2_0 =
            physics.add_hinge_joint(index_body, index_leg_2_0, hinge_joint_2_0);

        let hinge_joint_3_0 = XpbdHingeJoint::new(
            body.particles[5],
            leg_3_0.particles[0],
            -0.8 * PI,
            0.8 * PI,
            RotationAxis::Y,
            RotationAxis::Y,
            positional_compliance,
            angular_compliance,
        );
        let index_hinge_joint_3_0 =
            physics.add_hinge_joint(index_body, index_leg_3_0, hinge_joint_3_0);


        // Lower legs

        let hinge_joint_0_1 = XpbdHingeJoint::new(
            leg_0_0.particles[1],
            leg_0_1.particles[0],
            -0.8 * PI,
            0.0 * PI,
            RotationAxis::Y,
            RotationAxis::Y,
            positional_compliance,
            angular_compliance,
        );
        let index_hinge_joint_0_1 =
            physics.add_hinge_joint(index_leg_0_0, index_leg_0_1, hinge_joint_0_1);

        let hinge_joint_1_1 = XpbdHingeJoint::new(
            leg_1_0.particles[1],
            leg_1_1.particles[0],
            -0.8 * PI,
            0.0 * PI,
            RotationAxis::Y,
            RotationAxis::Y,
            positional_compliance,
            angular_compliance,
        );
        let index_hinge_joint_1_1 =
            physics.add_hinge_joint(index_leg_1_0, index_leg_1_1, hinge_joint_1_1);


        let hinge_joint_2_1 = XpbdHingeJoint::new(
            leg_2_0.particles[1],
            leg_2_1.particles[0],
            -0.8 * PI,
            0.0 * PI,
            RotationAxis::Y,
            RotationAxis::Y,
            positional_compliance,
            angular_compliance,
        );
        let index_hinge_joint_2_1 =
            physics.add_hinge_joint(index_leg_2_0, index_leg_2_1, hinge_joint_2_1);

        let hinge_joint_3_1 = XpbdHingeJoint::new(
            leg_3_0.particles[1],
            leg_3_1.particles[0],
            -0.8 * PI,
            0.0 * PI,
            RotationAxis::Y,
            RotationAxis::Y,
            positional_compliance,
            angular_compliance,
        );
        let index_hinge_joint_3_1 =
            physics.add_hinge_joint(index_leg_3_0, index_leg_3_1, hinge_joint_3_1);


        let state = State {
            pos_z_target: 0.0,
            vel_x_target: 0.0,
            _pos_x: 0.0,
            pos_z: 0.0,
            vel_x: 0.0,
            vel_y: 0.0,
            vel_z: 0.0,
            rot_x: 0.0,
            rot_y: 0.0,
            rot_z: 0.0,
            leg_angle_0_0: 0.0,
            leg_angle_1_0: 0.0,
            leg_angle_2_0: 0.0,
            leg_angle_3_0: 0.0,
            leg_angle_0_1: 0.0,
            leg_angle_1_1: 0.0,
            leg_angle_2_1: 0.0,
            leg_angle_3_1: 0.0,
            leg_angular_velocity_0_0: 0.0,
            leg_angular_velocity_1_0: 0.0,
            leg_angular_velocity_2_0: 0.0,
            leg_angular_velocity_3_0: 0.0,
            leg_angular_velocity_0_1: 0.0,
            leg_angular_velocity_1_1: 0.0,
            leg_angular_velocity_2_1: 0.0,
            leg_angular_velocity_3_1: 0.0,

            actions: [0.0; 8],
        };

        let mut obj = Self {
            physics,
            drawer,
            index_body,
            index_leg_0_0,
            index_leg_1_0,
            index_leg_2_0,
            index_leg_3_0,
            index_leg_0_1,
            index_leg_1_1,
            index_leg_2_1,
            index_leg_3_1,
            index_hinge_joint_0_0,
            index_hinge_joint_1_0,
            index_hinge_joint_2_0,
            index_hinge_joint_3_0,
            index_hinge_joint_0_1,
            index_hinge_joint_1_1,
            index_hinge_joint_2_1,
            index_hinge_joint_3_1,
            state,
        };

        obj.set_target();

        obj
    }

    fn set_target(&mut self) {
        self.state.pos_z_target = 0.95;
        self.state.vel_x_target = 2.0 + fastrand::f32() * 8.0
    }
}

impl Gym<OBSERVATIONS, ACTIONS> for GymDog {
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
        // self.physics.step(dt * 0.0001);
        self.physics.step(dt * 1.0);
        // self.physics.step(dt * 0.4);

        let body = &self.physics.rigid_bodies[self.index_body];
        let pos = body.p.pos;
        let vel = body.p.vel;
        let rot = body.r.get_rotation_as_euler();

        let hinge_0_0 = &self.physics.hinge_joints[self.index_hinge_joint_0_0].constraint;
        let hinge_1_0 = &self.physics.hinge_joints[self.index_hinge_joint_1_0].constraint;
        let hinge_2_0 = &self.physics.hinge_joints[self.index_hinge_joint_2_0].constraint;
        let hinge_3_0 = &self.physics.hinge_joints[self.index_hinge_joint_3_0].constraint;

        let hinge_0_1 = &self.physics.hinge_joints[self.index_hinge_joint_0_1].constraint;
        let hinge_1_1 = &self.physics.hinge_joints[self.index_hinge_joint_1_1].constraint;
        let hinge_2_1 = &self.physics.hinge_joints[self.index_hinge_joint_2_1].constraint;
        let hinge_3_1 = &self.physics.hinge_joints[self.index_hinge_joint_3_1].constraint;

        let state = State {
            pos_z_target: self.state.pos_z_target,
            vel_x_target: self.state.vel_x_target,
            _pos_x: pos.x,
            pos_z: pos.z,
            vel_x: vel.x,
            vel_y: vel.y,
            vel_z: vel.z,
            rot_x: rot.x,
            rot_y: rot.y,
            rot_z: rot.z,
            leg_angle_0_0: hinge_0_0.get_angle(),
            leg_angle_1_0: hinge_1_0.get_angle(),
            leg_angle_2_0: hinge_2_0.get_angle(),
            leg_angle_3_0: hinge_3_0.get_angle(),
            leg_angle_0_1: hinge_0_1.get_angle(),
            leg_angle_1_1: hinge_1_1.get_angle(),
            leg_angle_2_1: hinge_2_1.get_angle(),
            leg_angle_3_1: hinge_3_1.get_angle(),
            leg_angular_velocity_0_0: hinge_0_0.get_angular_velocity(),
            leg_angular_velocity_1_0: hinge_1_0.get_angular_velocity(),
            leg_angular_velocity_2_0: hinge_2_0.get_angular_velocity(),
            leg_angular_velocity_3_0: hinge_3_0.get_angular_velocity(),
            leg_angular_velocity_0_1: hinge_0_1.get_angular_velocity(),
            leg_angular_velocity_1_1: hinge_1_1.get_angular_velocity(),
            leg_angular_velocity_2_1: hinge_2_1.get_angular_velocity(),
            leg_angular_velocity_3_1: hinge_3_1.get_angular_velocity(),
            actions: self.state.actions,
        };

        self.state = state;

        for elem in self.get_state() {
            if !elem.is_finite() {
                self.reset();
                return;
            }
        }
    }

    fn get_state(&self) -> [f32; OBSERVATIONS] {
        let state = &self.state;
        [
            state.pos_z_target,
            state.vel_x_target,
            state.pos_z,
            state.vel_x,
            state.vel_y,
            state.vel_z,
            state.rot_x,
            state.rot_y,
            state.rot_z,
            state.leg_angle_0_0,
            state.leg_angle_1_0,
            state.leg_angle_2_0,
            state.leg_angle_3_0,
            state.leg_angle_0_1,
            state.leg_angle_1_1,
            state.leg_angle_2_1,
            state.leg_angle_3_1,
            state.leg_angular_velocity_0_0,
            state.leg_angular_velocity_1_0,
            state.leg_angular_velocity_2_0,
            state.leg_angular_velocity_3_0,
            state.leg_angular_velocity_0_1,
            state.leg_angular_velocity_1_1,
            state.leg_angular_velocity_2_1,
            state.leg_angular_velocity_3_1,
        ]
    }

    fn update(&mut self, actions: &[f32; ACTIONS], _dt: f32) {
        // self.apply_static_constraint();

        let torque = [
            actions[0] * 10.0,
            actions[1] * 10.0,
            actions[2] * 10.0,
            actions[3] * 10.0,

            
            actions[4] * 10.0,
            actions[5] * 10.0,
            actions[6] * 10.0,
            actions[7] * 10.0,
        ];

        self.physics.hinge_joints[self.index_hinge_joint_0_0].constraint.set_external_torque(torque[0]);
        self.physics.hinge_joints[self.index_hinge_joint_1_0].constraint.set_external_torque(torque[1]);
        self.physics.hinge_joints[self.index_hinge_joint_2_0].constraint.set_external_torque(torque[2]);
        self.physics.hinge_joints[self.index_hinge_joint_3_0].constraint.set_external_torque(torque[3]);

        self.physics.hinge_joints[self.index_hinge_joint_0_1].constraint.set_external_torque(torque[4]);
        self.physics.hinge_joints[self.index_hinge_joint_1_1].constraint.set_external_torque(torque[5]);
        self.physics.hinge_joints[self.index_hinge_joint_2_1].constraint.set_external_torque(torque[6]);
        self.physics.hinge_joints[self.index_hinge_joint_3_1].constraint.set_external_torque(torque[7]);

        self.state.actions = *actions;
    }

    fn get_reward(&self) -> (f32, bool) {
        // Reward target velocity
        // let target_velocity = 6.0;
        let target_velocity = self.state.vel_x_target;
        let velocity_cost = 1.0 - ((self.state.vel_x - target_velocity) / target_velocity).powi(2);
        let mut reward = 1.0 * velocity_cost;

        // Reward target height
        let target_height = self.state.pos_z_target;
        let height_cost = 1.0 - ((self.state.pos_z - target_height) / target_height).powi(2);
        reward += 1.0 * height_cost;

        // Penalize rotation
        let rot_cost = self.state.rot_y.powi(2) + self.state.rot_x.powi(2);
        reward -= rot_cost * 1.0;
        // println!("self.state.rot_y: {}", self.state.rot_y);

        // Penalize vel z
        // let vel_z_cost = self.state.vel_z.powi(2);
        // reward -= vel_z_cost * 0.1;

        // Penalize huge actions
        let mut action_cost = 0.0;
        for action in self.state.actions {
            action_cost += action.powi(2);
        }
        reward -= action_cost * 0.01;

        if self.state.rot_y.powi(2) > 2.0  || self.state.rot_x.powi(2) > 2.0 {
            return (0.0, true)
        }

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
    pos_z_target: f32,
    vel_x_target: f32,

    _pos_x: f32,
    pos_z: f32,

    vel_x: f32,
    vel_y: f32,
    vel_z: f32,
    
    rot_x: f32,
    rot_y: f32,
    rot_z: f32,

    // Upper leg
    leg_angle_0_0: f32,
    leg_angle_1_0: f32,
    leg_angle_2_0: f32,
    leg_angle_3_0: f32,

    // Lower leg
    leg_angle_0_1: f32,
    leg_angle_1_1: f32,
    leg_angle_2_1: f32,
    leg_angle_3_1: f32,

    // Upper log
    leg_angular_velocity_0_0: f32,
    leg_angular_velocity_1_0: f32,
    leg_angular_velocity_2_0: f32,
    leg_angular_velocity_3_0: f32,

     // Lower leg
    leg_angular_velocity_0_1: f32,
    leg_angular_velocity_1_1: f32,
    leg_angular_velocity_2_1: f32,
    leg_angular_velocity_3_1: f32,


    actions: [f32; 8],
}
