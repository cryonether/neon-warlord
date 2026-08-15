//! Physics simulation using verlet

pub mod fixed;
pub mod fixed_link;
pub mod link;
pub mod solver;

#[allow(dead_code)]
pub mod sticky_link;
#[allow(dead_code)]
pub mod verlet_composition;

use cgmath::Zero;

type Vec3 = cgmath::Vector3<f32>;

#[derive(Clone)]
pub struct VerletObject {
    position_current: Vec3,
    position_old: Vec3,
    acceleration: Vec3,
    radius: f32,

    pub is_static: bool,
}

impl VerletObject {
    pub fn new(position_current: Vec3, radius: f32) -> Self {
        let position_old = position_current;
        let acceleration = Vec3::zero();

        Self {
            position_current,
            position_old,
            acceleration,
            radius,
            is_static: false,
        }
    }

    pub fn reset_position(&mut self, position: Vec3) {
        self.position_current = position;
        self.position_old = position;
    }

    pub fn update_position(&mut self, dt: f32) {
        if self.is_static {
            return;
        }

        let velocity = self.position_current - self.position_old;
        // Save current position
        self.position_old = self.position_current;
        // Perform Verlet integration
        self.position_current = self.position_current + velocity + self.acceleration * dt * dt;
        // Reset acceleration
        self.acceleration = Vec3::zero();
    }

    pub fn accelerate(&mut self, acc: Vec3) {
        self.acceleration += acc;
    }

    pub fn position(&self) -> Vec3 {
        self.position_current
    }

    pub fn set_position(&mut self, pos: Vec3) {
        self.position_current = pos;
    }

    pub fn radius(&self) -> f32 {
        self.radius
    }

    pub fn damp(&mut self, val: f32) {
        let velocity = self.position_current - self.position_old;
        let velocity = velocity * val;
        self.position_old = self.position_current - velocity;
    }
}

