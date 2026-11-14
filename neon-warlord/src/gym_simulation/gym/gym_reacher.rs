//! Implementation of a reacher
//! https://gymnasium.farama.org/environments/mujoco/reacher/

use std::f32::consts::PI;

use cgmath::{InnerSpace, Zero};

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
const ACTIONS: usize = 3;

const MASS: f32 = 0.1;

#[derive(Clone)]
pub struct GymReacher {
    pub physics: Xpbd,
    pub drawer: XpbdDrawer,

    _index_stator: usize,
    _index_rotor: usize,
    _index_first_arm: usize,
    index_second_arm: usize,
    index_goal: usize,

    index_hinge_rotor: usize,
    index_hinge_first_arm: usize,
    index_hinge_second_arm: usize,

    state: State,
}

impl GymReacher {
    pub fn new() -> Self {
        let mut physics = Xpbd::new();
        let stator_pos = 3.0;

        let radius = 0.1;
        let drawer = XpbdDrawer::new(&physics, radius, Vec3::zero());

        let stator = XpbdRigidBody::new(
            Vec3::new(0.0, 0.0, stator_pos),
            &[Vec3::zero(), Vec3::new(0.0, 0.0, -1.0)],
            0.0,
            radius,
        );

        let rotor = XpbdRigidBody::new(
            Vec3::new(0.0, 0.0, 2.0),
            &[Vec3::zero(), Vec3::new(0.4, 0.0, 0.0)],
            MASS,
            radius,
        );

        let first_arm = XpbdRigidBody::new(
            Vec3::new(0.4, 0.0, 2.0),
            &[Vec3::zero(), Vec3::new(0.8, 0.0, 0.0)],
            MASS,
            radius,
        );

        let second_arm = XpbdRigidBody::new(
            Vec3::new(0.5 + 0.8, 0.0, 2.0),
            &[Vec3::zero(), Vec3::new(0.8, 0.0, 0.0)],
            MASS,
            radius,
        );

        let goal = XpbdRigidBody::new(
            Vec3::new(2.0, 0.0, stator_pos - 0.2),
            &[Vec3::zero()],
            0.0,
            radius,
        );

        let index_stator = physics.add_rigid_body(&stator);
        let index_rotor = physics.add_rigid_body(&rotor);
        let _index_first_arm = physics.add_rigid_body(&first_arm);
        let index_second_arm = physics.add_rigid_body(&second_arm);
        let index_goal = physics.add_rigid_body(&goal);

        let positional_compliance = 0.00;
        let angular_compliance = 0.0001;

        let hinge_rotor = XpbdHingeJoint::new(
            stator.particles[1],
            rotor.particles[0],
            -1.5 * PI,
            1.5 * PI,
            RotationAxis::Z,
            RotationAxis::Z,
            positional_compliance,
            angular_compliance,
        );
        let index_hinge_rotor = physics.add_hinge_joint(index_stator, index_rotor, hinge_rotor);

        let hinge_first_arm = XpbdHingeJoint::new(
            rotor.particles[1],
            first_arm.particles[0],
            -1.9 * PI,
            1.9 * PI,
            RotationAxis::Y,
            RotationAxis::Y,
            positional_compliance,
            angular_compliance,
        );
        let index_hinge_first_arm =
            physics.add_hinge_joint(index_rotor, _index_first_arm, hinge_first_arm);

        let hinge_second_arm = XpbdHingeJoint::new(
            first_arm.particles[1],
            second_arm.particles[0],
            -1.9 * PI,
            1.9 * PI,
            RotationAxis::Y,
            RotationAxis::Y,
            positional_compliance,
            angular_compliance,
        );
        let index_hinge_second_arm =
            physics.add_hinge_joint(_index_first_arm, index_second_arm, hinge_second_arm);

        let state = State {
            goal_pos_x: 0.0,
            goal_pos_y: 0.0,
            goal_pos_z: 0.0,
            fingertip_pos_x: 0.0,
            fingertip_pos_y: 0.0,
            fingertip_pos_z: 0.0,
            rotor_angle: 0.0,
            first_arm_angle: 0.0,
            second_arm_angle: 0.0,
            rotor_angular_velocity: 0.0,
            first_arm_angular_velocity: 0.0,
            second_arm_angular_velocity: 0.0,

            actions: [0.0; 3],
        };

        let mut obj = Self {
            physics,
            drawer,
            _index_stator: index_stator,
            _index_rotor: index_rotor,
            _index_first_arm,
            index_second_arm,
            index_goal,
            index_hinge_rotor,
            index_hinge_first_arm,
            index_hinge_second_arm,
            state,
        };

        obj.reset();

        obj
    }

    fn hard_reset(&mut self) {
        let drawer = self.drawer.clone();
        *self = Self::new();
        self.drawer = drawer;
    }
}

impl Gym<OBSERVATIONS, ACTIONS> for GymReacher {
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
        self.physics.step(dt * 1.0);
        // self.physics.step(dt * 1.0);

        let index_goal = self.index_goal;
        let index_second_arm = self.index_second_arm;

        let index_hinge_rotor = self.index_hinge_rotor;
        let index_hinge_first_arm = self.index_hinge_first_arm;
        let index_hinge_second_arm = self.index_hinge_second_arm;

        let goal = &self.physics.rigid_bodies[index_goal];
        let second_arm = &self.physics.rigid_bodies[index_second_arm];

        let pos_goal = goal.p.pos;
        let pos_fingertip = second_arm.particles[1];

        let hinge_rotor = &self.physics.hinge_joints[index_hinge_rotor].constraint;
        let hinge_first_arm = &self.physics.hinge_joints[index_hinge_first_arm].constraint;
        let hinge_second_arm = &self.physics.hinge_joints[index_hinge_second_arm].constraint;

        // let state = State {
        self.state.goal_pos_x = pos_goal.x;
        self.state.goal_pos_y = pos_goal.y;
        self.state.goal_pos_z = pos_goal.z;
        self.state.fingertip_pos_x = pos_fingertip.x;
        self.state.fingertip_pos_y = pos_fingertip.y;
        self.state.fingertip_pos_z = pos_fingertip.z;
        self.state.rotor_angle = hinge_rotor.get_angle();
        self.state.first_arm_angle = hinge_first_arm.get_angle();
        self.state.second_arm_angle = hinge_second_arm.get_angle();
        self.state.rotor_angular_velocity = hinge_rotor.get_angular_velocity();
        self.state.first_arm_angular_velocity = hinge_first_arm.get_angular_velocity();
        self.state.second_arm_angular_velocity = hinge_second_arm.get_angular_velocity();
        // };

        for elem in self.get_state() {
            if !elem.is_finite() {
                self.hard_reset();
                return;
            }
        }

        // self.state = state;
    }

    fn get_state(&self) -> [f32; OBSERVATIONS] {
        let state = &self.state;
        [
            state.goal_pos_x,
            state.goal_pos_y,
            state.goal_pos_z,
            state.fingertip_pos_x,
            state.fingertip_pos_y,
            state.fingertip_pos_z,
            state.rotor_angle,
            state.first_arm_angle,
            state.second_arm_angle,
            state.rotor_angular_velocity,
            state.first_arm_angular_velocity,
            state.second_arm_angular_velocity,
        ]
    }

    fn update(&mut self, actions: &[f32; ACTIONS], _dt: f32) {
        let torque = [actions[0] * 10.0, actions[1] * 10.0, actions[2] * 10.0];

        let hinge_0 = &mut self.physics.hinge_joints[self.index_hinge_rotor].constraint;
        hinge_0.set_external_torque(torque[0]);

        let hinge_1 = &mut self.physics.hinge_joints[self.index_hinge_first_arm].constraint;
        hinge_1.set_external_torque(torque[1]);

        let hinge_2 = &mut self.physics.hinge_joints[self.index_hinge_second_arm].constraint;
        hinge_2.set_external_torque(torque[2]);

        self.state.actions = *actions;
    }

    fn get_reward(&self) -> (f32, bool) {
        let index_goal = self.index_goal;
        let index_second_arm = self.index_second_arm;

        let goal = &self.physics.rigid_bodies[index_goal];
        let second_arm = &self.physics.rigid_bodies[index_second_arm];

        let pos_goal = goal.p.pos;
        let pos_fingertip = second_arm.particles[1];
        let pos_fingertip = second_arm.local_to_world(&pos_fingertip);

        let distance2 = (pos_fingertip - pos_goal).magnitude2();

        let mut reward = 1.0 - distance2 * 0.2;

        // Penalize angular velocity
        let angular_velocity_cost = self.state.rotor_angular_velocity.powi(2)
            + self.state.first_arm_angular_velocity.powi(2)
            + self.state.second_arm_angular_velocity.powi(2);
        reward -= angular_velocity_cost * 0.01;

        // Penalize huge actions
        let mut action_cost = 0.0;
        for action in self.state.actions {
            action_cost += action.powi(2);
        }
        reward -= action_cost * 0.1;

        // No negative rewards
        let reward = reward.max(0.0);

        if distance2 < 0.1 {
            return (10.0, true);
        }

        (reward, false)
    }

    fn reset(&mut self) {
        let v = get_point_sphere(&Vec3::new(0.0, 0.0, 2.0), 1.8);
        let pos_goal = v;

        let index_goal = self.index_goal;
        let particle_goal = &mut self.physics.rigid_bodies[index_goal];

        particle_goal.p.pos = pos_goal;
        particle_goal.p.prev_pos = pos_goal;

        // let drawer = self.drawer.clone();
        // *self = Self::new();
        // self.drawer = drawer;
    }
    
    fn get_current_position(&self) -> Vec3 {
        Vec3::zero()
    }
}

#[derive(Clone)]
pub struct State {
    goal_pos_x: f32,
    goal_pos_y: f32,
    goal_pos_z: f32,

    fingertip_pos_x: f32,
    fingertip_pos_y: f32,
    fingertip_pos_z: f32,

    rotor_angle: f32,
    first_arm_angle: f32,
    second_arm_angle: f32,

    rotor_angular_velocity: f32,
    first_arm_angular_velocity: f32,
    second_arm_angular_velocity: f32,

    actions: [f32; 3],
}

fn get_point_sphere(p: &Vec3, r: f32) -> Vec3 {
    loop {
        let x = (fastrand::f32() * 2.0 - 1.0) * r;
        let y = (fastrand::f32() * 2.0 - 1.0) * r;
        let z = (fastrand::f32() * 2.0 - 1.0) * r;

        let v = Vec3::new(x, y, z);
        if v.magnitude2() <= r.powi(2) {
            return p + v;
        }
    }
}
