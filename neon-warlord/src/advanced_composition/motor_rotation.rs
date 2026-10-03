use cgmath::{InnerSpace, Vector3};

use crate::verlet_physics_simd::verlet_particles::VerletParticles;

/// An actor free to rotate along the axis described by two nodes.
#[derive(Clone)]
pub struct MotorRotation {
    pub node_id: usize,
    pub node_a_id: usize,
    pub node_b_id: usize,

    // Angular acceleration in radians/s².
    acceleration: f32,

    cos_alpha: f32,
    sin_alpha: f32,
}

impl MotorRotation {
    pub fn new(node_id: usize, node_a_id: usize, node_b_id: usize) -> Self {
        Self {
            node_id,
            node_a_id,
            node_b_id,
            acceleration: 0.0,
            cos_alpha: 0.0,
            sin_alpha: 0.0,
        }
    }

    pub fn update(&mut self, particles: &mut VerletParticles) {
        let pos = particles.position(self.node_id);
        let a = particles.position(self.node_a_id);
        let b = particles.position(self.node_b_id);

        // Unit rotation axis.
        let axis = b - a;
        let axis_len_sq = axis.dot(axis);

        if axis_len_sq <= f32::EPSILON {
            return;
        }

        let axis = axis / axis_len_sq.sqrt();

        // Position relative to the axis.
        let r = pos - a;
        let perpendicular = r - axis * r.dot(axis);
        let radius_sq = perpendicular.dot(perpendicular);

        if radius_sq <= f32::EPSILON {
            return;
        }

        let radius = radius_sq.sqrt();
        let radial = perpendicular / radius;

        // Construct a stable orthonormal basis around the axis.
        let reference = if axis.x.abs() < 0.9 {
            Vector3::unit_x()
        } else {
            Vector3::unit_y()
        };

        let basis_x = axis.cross(reference).normalize();
        let basis_y = axis.cross(basis_x);

        self.cos_alpha = radial.dot(basis_x);
        self.sin_alpha = radial.dot(basis_y);

        // a = α × r
        particles.accelerate(self.node_id, axis.cross(perpendicular) * self.acceleration);
    }

    pub fn accelerate(&mut self, acceleration: f32) {
        self.acceleration = acceleration;
    }

    pub fn get_sin_alpha(&self) -> f32 {
        -self.cos_alpha
    }

    pub fn get_cos_alpha(&self) -> f32 {
        -self.sin_alpha
    }
}
