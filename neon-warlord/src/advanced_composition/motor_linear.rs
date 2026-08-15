//! An actor free to move between two nodes

use cgmath::InnerSpace;

use crate::verlet_physics_simd::verlet_particles::VerletParticles;

/// An actor free to move between two nodes
#[derive(Clone)]
pub struct MotorLinear {
    pub node_id: usize,
    pub node_a_id: usize,
    pub node_b_id: usize,

    acceleration: f32,
}

impl MotorLinear {
    pub fn new(node_id: usize, node_a_id: usize, node_b_id: usize) -> Self {
        Self {
            node_id,
            node_a_id,
            node_b_id,
            acceleration: 0.0,
        }
    }

    pub fn update_simd(&mut self, verlet_particles: &mut VerletParticles) {
        // apply constraint
        let radius = verlet_particles.radius[self.node_id];
        let pos = verlet_particles.position(self.node_id);
        let pos_a = verlet_particles.position(self.node_a_id);
        let pos_b = verlet_particles.position(self.node_b_id);

        let vec_a_b = pos_b - pos_a;
        let vec_a_b_norm = vec_a_b.normalize();
        let vec_m_b = pos_b - pos;

        let vec_a_b = (pos_b - pos_a).normalize();

        let new_pos;
        if (pos_b - pos).magnitude() < 2.0 * radius {
            // left side
            new_pos = pos_b - vec_a_b_norm * 2.0 * radius;
            verlet_particles.set_position(self.node_id, new_pos);
        } else if (pos_a - pos).magnitude() < 2.0 * radius {
            // right side
            new_pos = pos_a + vec_a_b_norm * 2.0 * radius;
            verlet_particles.set_position(self.node_id, new_pos);
        } else {
            // in the middle
            new_pos = pos_b - (vec_a_b.dot(vec_m_b) * vec_a_b) / (vec_a_b.dot(vec_a_b));
        }

        verlet_particles.set_position(self.node_id, new_pos);

        // apply acceleration
        let acceleration = vec_a_b * self.acceleration;
        verlet_particles.accelerate(self.node_id, acceleration);
    }

    pub fn accelerate(&mut self, val: f32) {
        self.acceleration = (val - 0.5) * 200.0;
    }
}
