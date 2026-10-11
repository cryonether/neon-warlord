//! Verlet physics with SIMD optimizations

pub mod distance_constraints;
pub mod verlet_particles;
pub mod angle_constraints;

use cgmath::{InnerSpace, MetricSpace};
use dfdx::tensor_ops::cos;

use crate::verlet_physics_simd::{
    angle_constraints::AngleConstraints, distance_constraints::DistanceConstraints, verlet_particles::VerletParticles,
};

type Vec3 = cgmath::Vector3<f32>;

/// Verlet physics with SIMD optimizations
#[derive(Clone)]
pub struct VerletPhysicsSimd {
    pub particles: VerletParticles,
    pub distance_constraints: DistanceConstraints,
    pub angle_constraints: AngleConstraints,
}

impl VerletPhysicsSimd {
    pub fn new() -> Self {
        let particles = VerletParticles::new();
        let distance_constraints = DistanceConstraints::new();
        let angle_constraints = AngleConstraints::new();

        Self {
            particles,
            distance_constraints,
            angle_constraints,
        }
    }

    pub fn update(&mut self, dt: f32) {
        self.apply_gravity();
        self.distance_constraints.solve(&mut self.particles);
        self.angle_constraints.solve(&mut self.particles);
        self.particles.integrate(dt);
    }

    fn apply_gravity(&mut self) {
        // const GRAVITY: Vec3 = Vec3::new(0.0, 0.0, -1.0);
        const GRAVITY: Vec3 = Vec3::new(0.0, 0.0, -9.81);
        self.particles.apply_gravity(&GRAVITY);
    }

    pub fn push_particle(&mut self, pos: Vec3, radius: f32, mass: f32) -> usize {
        self.particles.push(pos, radius, mass)
    }

    pub fn push_constraint_distance(
        &mut self,
        a: usize,
        b: usize,
        rest: f32,
        stiffness: f32,
    ) -> usize {
        self.distance_constraints
            .push_constraint_distance(a, b, rest, stiffness)
    }

    pub fn push_constraint_distance_infer(
        &mut self,
        a: usize,
        b: usize,
        stiffness: f32,
    ) -> usize {
        let a_x = self.particles.x[a];
        let a_y = self.particles.y[a];
        let a_z = self.particles.z[a];

        let b_x = self.particles.x[b];
        let b_y = self.particles.y[b];
        let b_z = self.particles.z[b];

        let len = Vec3::new(a_x, a_y, a_z).distance(Vec3::new(b_x, b_y, b_z));

        self.distance_constraints
            .push_constraint_distance(a, b, len, stiffness)
    }


    pub fn push_constraint_position(
        &mut self,
        a: usize,
        b: usize,
        rest: Vec3,
        stiffness: f32,
    ) -> usize {
        self.distance_constraints
            .push_constraint_position(a, b, rest, stiffness)
    }

    pub fn push_constraint_none(&mut self, a: usize, b: usize) -> usize {
        self.distance_constraints.push_constraint_none(a, b)
    }

    pub fn push_constraint_angle(
        &mut self,
        a: usize,
        b: usize,
        c: usize,
        cos_alpha: f32,
        stiffness: f32,
    ) -> usize {
        self.angle_constraints
            .push(a, b, c, cos_alpha, stiffness)
    }


    pub fn push_constraint_angle_infer(
        &mut self,
        a: usize,
        b: usize,
        c: usize,
        stiffness: f32,
    ) -> usize {
        let ab_x = self.particles.x[a] - self.particles.x[b];
        let ab_y = self.particles.y[a] - self.particles.y[b];
        let ab_z = self.particles.z[a] - self.particles.z[b];

        let cb_x = self.particles.x[c] - self.particles.x[b];
        let cb_y = self.particles.y[c] - self.particles.y[b];
        let cb_z = self.particles.z[c] - self.particles.z[b];

        let dot = ab_x * cb_x + ab_y * cb_y + ab_z * cb_z;

        let len_ab_sq = ab_x * ab_x + ab_y * ab_y + ab_z * ab_z;
        let len_cb_sq = cb_x * cb_x + cb_y * cb_y + cb_z * cb_z;

        let cos_alpha = if len_ab_sq > 0.0 && len_cb_sq > 0.0 {
            dot / (len_ab_sq * len_cb_sq).sqrt()
        } else {
            1.0 // Fallback for degenerate geometry. (either segment has zero length)
        };

        self.angle_constraints
            .push(a, b, c, cos_alpha.clamp(-1.0, 1.0), stiffness)
    }


    pub fn get_particle_position(&self, index: usize) -> Vec3 {
        let pos_x = self.particles.x[index];
        let pos_y = self.particles.y[index];
        let pos_z = self.particles.z[index];
        Vec3::new(pos_x, pos_y, pos_z)
    }
}
