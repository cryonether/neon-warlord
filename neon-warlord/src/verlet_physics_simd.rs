//! Verlet physics with SIMD optimizations

pub mod distance_constraints;
pub mod verlet_particles;

use crate::verlet_physics_simd::{
    distance_constraints::DistanceConstraints, verlet_particles::VerletParticles,
};

type Vec3 = cgmath::Vector3<f32>;

/// Verlet physics with SIMD optimizations
#[derive(Clone)]
pub struct VerletPhysicsSimd {
    pub particles: VerletParticles,
    pub distance_constraints: DistanceConstraints,
}

impl VerletPhysicsSimd {
    pub fn new() -> Self {
        let particles = VerletParticles::new();
        let distance_constraints = DistanceConstraints::new();

        Self {
            particles,
            distance_constraints,
        }
    }

    pub fn update(&mut self, dt: f32) {
        self.apply_gravity();
        self.apply_distance_constraints();
        self.integrate(dt);
    }

    fn apply_gravity(&mut self) {
        const GRAVITY: Vec3 = Vec3::new(0.0, 0.0, -1.0);
        self.particles.apply_gravity(&GRAVITY);
    }

    fn apply_distance_constraints(&mut self) {
        self.distance_constraints.solve(&mut self.particles);
    }

    fn integrate(&mut self, dt: f32) {
        self.particles.integrate(dt);
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

    pub fn get_particle_position(&self, index: usize) -> Vec3 {
        let pos_x = self.particles.x[index];
        let pos_y = self.particles.y[index];
        let pos_z = self.particles.z[index];
        Vec3::new(pos_x, pos_y, pos_z)
    }
}
