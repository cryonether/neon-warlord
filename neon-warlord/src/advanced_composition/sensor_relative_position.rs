//! A sensor which tracks the position relative to another node

use cgmath::Zero;

use crate::{verlet_physics::VerletObject, verlet_physics_simd::verlet_particles::VerletParticles};

type Vec3 = cgmath::Vector3<f32>;

/// A sensor which tracks the position relative to another node
#[derive(Clone)]
pub struct SensorRelativePosition {
    pub node_id: usize,
    pub node_a_id: usize,

    val: Vec3,
}

impl SensorRelativePosition {
    pub fn new(node_id: usize, node_a_id: usize) -> Self {
        Self {
            node_id,
            node_a_id,
            val: Vec3::zero(),
        }
    }

    pub fn update(&mut self, verlet_objects: &[VerletObject]) {
        // apply constraint
        let pos = verlet_objects[self.node_id].position();
        let pos_a = verlet_objects[self.node_a_id].position();

        let vec_a_s = pos - pos_a;
        self.val = vec_a_s;
    }


    pub fn update_simd(&mut self, verlet_particles: &VerletParticles) {
        // apply constraint
        let pos = verlet_particles.position(self.node_id);
        let pos_a = verlet_particles.position(self.node_a_id);

        let vec_a_s = pos - pos_a;
        self.val = vec_a_s;
    }

    pub fn get_val(&self) -> &Vec3 {
        &self.val
    }
}
