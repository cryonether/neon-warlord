//! A sensor which tracks the position relative to another node

use cgmath::Zero;

use crate::verlet_physics_simd::verlet_particles::VerletParticles;

type Vec3 = cgmath::Vector3<f32>;

/// A sensor which tracks the position relative to another node
#[derive(Clone)]
pub struct SensorRelativePosition {
    pub node_id: usize,
    pub node_a_id: usize,

    position_previous: Vec3,
    position: Vec3,
    velocity: Vec3,
}

impl SensorRelativePosition {
    pub fn new(node_id: usize, node_a_id: usize) -> Self {
        Self {
            node_id,
            node_a_id,
            position_previous: Vec3::zero(),
            position: Vec3::zero(),
            velocity: Vec3::zero(),
        }
    }

    pub fn update_simd(&mut self, verlet_particles: &VerletParticles, _dt: f32) {
        // apply constraint
        let pos = verlet_particles.position(self.node_id);
        let pos_a = verlet_particles.position(self.node_a_id);

        let vec_a_s = pos - pos_a;
        self.position_previous = self.position;
        self.position = vec_a_s;

        self.velocity = self.position - self.position_previous;
    }

    pub fn get_position_vec(&self) -> &Vec3 {
        &self.position
    }

    pub fn get_velocity_vec(&self) -> &Vec3 {
        &self.velocity
    }
}
