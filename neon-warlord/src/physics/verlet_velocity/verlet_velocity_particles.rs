//! Particles for verlet velocity integration

use cgmath::Zero;

use super::Vec3;

#[derive(Clone)]
pub struct VerletVelocityParticles {
    pos_x: Vec<f32>,
    pos_y: Vec<f32>,
    pos_z: Vec<f32>,

    vel_half_x: Vec<f32>,
    vel_half_y: Vec<f32>,
    vel_half_z: Vec<f32>,

    vel_x: Vec<f32>,
    vel_y: Vec<f32>,
    vel_z: Vec<f32>,

    acc_x: Vec<f32>,
    acc_y: Vec<f32>,
    acc_z: Vec<f32>,

    pub mass: Vec<f32>,
    pub inv_mass: Vec<f32>,
    pub radius: Vec<f32>,

    // particle is integrated in rigid body integrator
    pub is_rigid_body: Vec<bool>,
}

impl VerletVelocityParticles {
    pub fn new() -> Self {
        let pos_x = Vec::new();
        let pos_y = Vec::new();
        let pos_z = Vec::new();
        let vel_half_x = Vec::new();
        let vel_half_y = Vec::new();
        let vel_half_z = Vec::new();
        let vel_x = Vec::new();
        let vel_y = Vec::new();
        let vel_z = Vec::new();
        let acc_x = Vec::new();
        let acc_y = Vec::new();
        let acc_z = Vec::new();
        let mass = Vec::new();
        let inv_mass = Vec::new();
        let radius = Vec::new();
        let is_rigid_body = Vec::new();

        Self {
            pos_x,
            pos_y,
            pos_z,
            vel_half_x,
            vel_half_y,
            vel_half_z,
            vel_x,
            vel_y,
            vel_z,
            acc_x,
            acc_y,
            acc_z,
            mass,
            inv_mass,
            radius,
            is_rigid_body,
        }
    }

    pub fn integrate_position(&mut self, dt: f32) {
        for i in 0..self.len() {
            if self.inv_mass[i] == 0.0 || self.is_rigid_body[i] {
                // static particle
                continue;
            }

            let mut pos = self.get_pos(i);
            let vel = self.get_vel(i);
            let acc = self.get_acc(i);

            let vel_half = vel + 0.5 * acc * dt;
            pos += vel_half * dt;

            self.set_pos(i, pos);
            self.set_vel_half(i, vel_half);
        }
    }

    pub fn zero_acc(&mut self) {
        for i in 0..self.len() {
            self.set_acc(i, Vec3::zero());
        }
    }

    pub fn integrate_velocity(&mut self, dt: f32) {
        for i in 0..self.len() {
            if self.inv_mass[i] == 0.0 || self.is_rigid_body[i] {
                // static particle
                continue;
            }

            let vel_half = self.get_vel_half(i);
            let acc = self.get_acc(i);

            let vel = vel_half + 0.5 * acc * dt;

            self.set_vel(i, vel);
        }
    }

    pub fn push(&mut self, pos: Vec3, radius: f32, mass: f32) -> usize {
        let index = self.len();

        self.pos_x.push(pos.x);
        self.pos_y.push(pos.y);
        self.pos_z.push(pos.z);

        self.vel_half_x.push(0.0);
        self.vel_half_y.push(0.0);
        self.vel_half_z.push(0.0);

        self.vel_x.push(0.0);
        self.vel_y.push(0.0);
        self.vel_z.push(0.0);

        self.acc_x.push(0.0);
        self.acc_y.push(0.0);
        self.acc_z.push(0.0);

        self.mass.push(mass);
        self.inv_mass.push(if mass > 0.0 {
            1.0 / mass
        } else {
            // static particle
            0.0
        });

        self.radius.push(radius);
        self.is_rigid_body.push(false);

        index
    }

    pub fn get_pos(&self, i: usize) -> Vec3 {
        Vec3::new(self.pos_x[i], self.pos_y[i], self.pos_z[i])
    }

    pub fn get_vel_half(&self, i: usize) -> Vec3 {
        Vec3::new(self.vel_half_x[i], self.vel_half_y[i], self.vel_half_z[i])
    }

    pub fn get_vel(&self, i: usize) -> Vec3 {
        Vec3::new(self.vel_x[i], self.vel_y[i], self.vel_z[i])
    }

    pub fn get_acc(&self, i: usize) -> Vec3 {
        Vec3::new(self.acc_x[i], self.acc_y[i], self.acc_z[i])
    }

    pub fn set_pos(&mut self, i: usize, val: Vec3) {
        self.pos_x[i] = val.x;
        self.pos_y[i] = val.y;
        self.pos_z[i] = val.z;
    }

    pub fn set_vel_half(&mut self, i: usize, val: Vec3) {
        self.vel_half_x[i] = val.x;
        self.vel_half_y[i] = val.y;
        self.vel_half_z[i] = val.z;
    }

    pub fn set_vel(&mut self, i: usize, val: Vec3) {
        self.vel_x[i] = val.x;
        self.vel_y[i] = val.y;
        self.vel_z[i] = val.z;
    }

    pub fn set_acc(&mut self, i: usize, val: Vec3) {
        self.acc_x[i] = val.x;
        self.acc_y[i] = val.y;
        self.acc_z[i] = val.z;
    }

    fn assert_bounds_invariants(&self) {
        let len = self.pos_x.len();

        assert!(self.pos_x.len() == len);
        assert!(self.pos_y.len() == len);
        assert!(self.pos_z.len() == len);

        assert!(self.vel_half_x.len() == len);
        assert!(self.vel_half_y.len() == len);
        assert!(self.vel_half_z.len() == len);

        assert!(self.vel_x.len() == len);
        assert!(self.vel_y.len() == len);
        assert!(self.vel_z.len() == len);

        assert!(self.acc_x.len() == len);
        assert!(self.acc_y.len() == len);
        assert!(self.acc_z.len() == len);

        assert!(self.inv_mass.len() == len);
        assert!(self.radius.len() == len);
        assert!(self.is_rigid_body.len() == len);
    }

    pub fn len(&self) -> usize {
        self.assert_bounds_invariants();

        self.pos_x.len()
    }
}
