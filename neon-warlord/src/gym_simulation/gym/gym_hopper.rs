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
pub struct GymHopper {
    pub physics: Xpbd,
    pub drawer: XpbdDrawer,

    index_leg_2: usize,
    index_leg_1: usize,
    _index_leg_2: usize,
    index_foot: usize,

    index_hinge_leg_0: usize,
    index_hinge_leg_1: usize,
    index_hinge_leg_2: usize,

    state: State,
}

impl GymHopper {
    pub fn new() -> Self {
        let mut physics = Xpbd::new();

        let radius = 0.1;
        let drawer = XpbdDrawer::new(&physics, radius, Vec3::zero());

        let leg_0 = XpbdRigidBody::new(
            Vec3::new(0.0, 0.0, 3.4),
            &[Vec3::zero(), Vec3::new(0.0, 0.0, -1.0)],
            MASS,
            radius,
        );

        let leg_1 = XpbdRigidBody::new(
            Vec3::new(0.0, 0.0, 2.4),
            &[Vec3::zero(), Vec3::new(0.0, 0.0, -1.0)],
            MASS,
            radius,
        );

        let leg_2 = XpbdRigidBody::new(
            Vec3::new(0.0, 0.0, 1.4),
            &[Vec3::zero(), Vec3::new(0.0, 0.0, -1.0)],
            MASS,
            radius,
        );

        let a = 0.4;
        let foot = XpbdRigidBody::new(
            Vec3::new(0.0, 0.0, 0.4),
            &[
                Vec3::zero(), 
                Vec3::new( a, -a, -0.2),
                Vec3::new( a, a, -0.2),
                Vec3::new(-a, a, -0.2),
                Vec3::new(-a, -a, -0.2),
            ],
            MASS,
            radius,
        );


        let index_leg_0 = physics.add_rigid_body(&leg_0);
        let index_leg_1 = physics.add_rigid_body(&leg_1);
        let _index_leg_2 = physics.add_rigid_body(&leg_2);
        let index_foot = physics.add_rigid_body(&foot);

        let positional_compliance = 0.00;
        let angular_compliance = 0.0001;

        let hinge_leg_0 = XpbdHingeJoint::new(
            leg_0.particles[1],
            leg_1.particles[0],
            -0.5 * PI,
            0.5 * PI,
            RotationAxis::Y,
            RotationAxis::Y,
            positional_compliance,
            angular_compliance,
        );
        let index_hinge_leg_0 = physics.add_hinge_joint(index_leg_0, index_leg_1, hinge_leg_0);

        let hinge_leg_1 = XpbdHingeJoint::new(
            leg_1.particles[1],
            leg_2.particles[0],
            -0.5 * PI,
            0.5 * PI,
            RotationAxis::Y,
            RotationAxis::Y,
            positional_compliance,
            angular_compliance,
        );
        let index_hinge_leg_1 =
            physics.add_hinge_joint(index_leg_1, _index_leg_2, hinge_leg_1);

        let hinge_leg_2 = XpbdHingeJoint::new(
            leg_2.particles[1],
            foot.particles[0],
            -0.5 * PI,
            0.5 * PI,
            RotationAxis::Y,
            RotationAxis::Y,
            positional_compliance,
            angular_compliance,
        );
        let index_hinge_leg_2 =
            physics.add_hinge_joint(_index_leg_2, index_foot, hinge_leg_2);

        let state = State {
            pos_z: 0.0,
            angle_x: 0.0,
            angle_y: 0.0,
            vel_x: 0.0,
            vel_y: 0.0,
            vel_z: 0.0,
            angle_joint_0: 0.0,
            angle_joint_1: 0.0,
            angle_joint_2: 0.0,
            angular_velocity_joint_0: 0.0,
            angular_velocity_joint_1: 0.0,
            angular_velocity_joint_2: 0.0,
            
            actions: [0.0; 3],
        };


        let obj = Self {
            physics,
            drawer,
            index_leg_2: index_leg_0,
            index_leg_1,
            _index_leg_2,
            index_foot,
            index_hinge_leg_0,
            index_hinge_leg_1,
            index_hinge_leg_2,
            state,
        };

        // obj.reset();

        obj
    }

    fn hard_reset(&mut self) {
        let drawer = self.drawer.clone();
        *self = Self::new();
        self.drawer = drawer;
    }
}

impl Gym<OBSERVATIONS, ACTIONS> for GymHopper {
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
        // self.physics.step(dt * 0.01);

        let index_leg_0 = self.index_leg_2;
        let index_leg_1 = self.index_leg_1;
        let index_leg_2 = self.index_leg_2;
        let index_foot = self.index_foot;

        let index_hinge_leg_0 = self.index_hinge_leg_0;
        let index_hinge_leg_1 = self.index_hinge_leg_1;
        let index_hinge_leg_2 = self.index_hinge_leg_2;

        let leg_0 = &self.physics.rigid_bodies[index_leg_0];
        let leg_1 = &self.physics.rigid_bodies[index_leg_1];
        let leg_2 = &self.physics.rigid_bodies[index_leg_2];
        let _foot =  &self.physics.rigid_bodies[index_foot];

        let hinge_leg_0 = &self.physics.hinge_joints[index_hinge_leg_0].constraint;
        let hinge_leg_1 = &self.physics.hinge_joints[index_hinge_leg_1].constraint;
        let hinge_leg_2 = &self.physics.hinge_joints[index_hinge_leg_2].constraint;


        let pos_leg_0 = leg_0.p.pos;
        let rot_leg_0 = leg_0.r.get_rotation_as_euler();

        let vel_leg_0 = leg_0.p.vel;
        let _vel_leg_1 = leg_1.p.vel;
        let _vel_leg_2 = leg_2.p.vel;


        self.state.pos_z = pos_leg_0.z;
        self.state.angle_x = rot_leg_0.x;
        self.state.angle_y = rot_leg_0.y;
        self.state.vel_x = vel_leg_0.x;
        self.state.vel_y = vel_leg_0.y;
        self.state.vel_z = vel_leg_0.z;
        self.state.angle_joint_0 = hinge_leg_0.get_angle();
        self.state.angle_joint_1 = hinge_leg_1.get_angle();
        self.state.angle_joint_2 = hinge_leg_2.get_angle();
        self.state.angular_velocity_joint_0 = hinge_leg_0.get_angular_velocity();
        self.state.angular_velocity_joint_1 = hinge_leg_1.get_angular_velocity();
        self.state.angular_velocity_joint_2 = hinge_leg_2.get_angular_velocity();

        for elem in self.get_state() {
            if !elem.is_finite() {
                self.hard_reset();
                return;
            }
        }
    }

    fn get_state(&self) -> [f32; OBSERVATIONS] {
        let state = &self.state;
        [
            state.pos_z,
            state.angle_x,
            state.angle_y,
            state.vel_x,
            state.vel_y,
            state.vel_z,
            state.angle_joint_0,
            state.angle_joint_1,
            state.angle_joint_2,
            state.angular_velocity_joint_0,
            state.angular_velocity_joint_1,
            state.angular_velocity_joint_2,
        ]
    }

    fn update(&mut self, actions: &[f32; ACTIONS], _dt: f32) {
        let torque = [
            actions[0] * 40.0, 
            actions[1] * 40.0, 
            actions[2] * 40.0
        ];

        let hinge_0 = &mut self.physics.hinge_joints[self.index_hinge_leg_0].constraint;
        hinge_0.set_external_torque(torque[0]);

        let hinge_1 = &mut self.physics.hinge_joints[self.index_hinge_leg_1].constraint;
        hinge_1.set_external_torque(torque[1]);

        let hinge_2 = &mut self.physics.hinge_joints[self.index_hinge_leg_2].constraint;
        hinge_2.set_external_torque(torque[2]);

        self.state.actions = *actions;
    }

    fn get_reward(&self) -> (f32, bool) {
        let pos_z = self.state.pos_z;
        let _vel_x = self.state.vel_x;

        let angle_x = self.state.angle_x;
        let angle_y = self.state.angle_y;

        let angle_hinge_0 = self.state.angle_joint_0;
        let angle_hinge_1 = self.state.angle_joint_1;
        let angle_hinge_2 = self.state.angle_joint_2;

        // println!("angle_hinge_0: {}, angle_hinge_1: {}, angle_hinge_2: {}", 
        //     angle_hinge_0, 
        //     angle_hinge_1,
        //     angle_hinge_2,
        // );

        let target_velocity = 3.0;
        let vel =  1.0 - ((self.state.vel_x - target_velocity) / target_velocity).powi(2);

        let mut reward = 1.0 + vel * 0.5;

        // Penalize angle
        let angle_cost = angle_x.powi(2) + angle_y.powi(2);
        reward -= angle_cost * 1.0;


        // Penalize hinge angle
        let angle_hinge_cost = angle_hinge_0.powi(2) + angle_hinge_1.powi(2) + angle_hinge_2.powi(2);
        reward -= angle_hinge_cost * 0.01;


        // // Penalize angular velocity
        // let angular_velocity_cost = self.state.leg_1_angular_velocity.powi(2)
        //     + self.state.leg_2_angular_velocity.powi(2)
        //     + self.state.foot_angular_velocity.powi(2);
        // reward -= angular_velocity_cost * 0.01;

        // // Penalize huge actions
        // let mut action_cost = 0.0;
        // for action in self.state.actions {
        //     action_cost += action.powi(2);
        // }
        // reward -= action_cost * 0.1;

        // // No negative rewards
        let reward = reward.max(0.0);

        // if distance2 < 0.1 {
        //     return (10.0, true);
        // }

        if angle_x.powi(2) > 0.4 || angle_y.powi(2) > 0.4 || pos_z < 1.0{
            return (0.0, true);
        }

        (reward, false)
    }

    fn reset(&mut self) {
        let v = get_point_sphere(&Vec3::new(0.0, 0.0, 2.0), 1.8);
        let _pos_goal = v;

        // let index_goal = self.index_goal;
        // let particle_goal = &mut self.physics.rigid_bodies[index_goal];

        // particle_goal.p.pos = pos_goal;
        // particle_goal.p.prev_pos = pos_goal;

        let drawer = self.drawer.clone();
        *self = Self::new();
        self.drawer = drawer;
    }
}

#[derive(Clone)]
pub struct State {
    // pos_x: f32,
    // pos_y: f32,
    pos_z: f32,

    angle_x: f32,
    angle_y: f32,
    // angle_z: f32,

    vel_x: f32,
    vel_y: f32,
    vel_z: f32,

    angle_joint_0: f32,
    angle_joint_1: f32,
    angle_joint_2: f32,

    angular_velocity_joint_0: f32,
    angular_velocity_joint_1: f32,
    angular_velocity_joint_2: f32,

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
