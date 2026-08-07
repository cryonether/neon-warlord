//! An actor free to move between two nodes

use cgmath::InnerSpace;

use crate::verlet_physics::VerletObject;

/// An actor free to move between two nodes
pub struct MotorLinear {
    pub node_id: usize,
    pub node_a_id: usize,
    pub node_b_id: usize,
}

impl MotorLinear {
    pub fn update(&mut self, verlet_objects: &mut [VerletObject]) {
        // apply constraint
        let radius = verlet_objects[self.node_id].radius();
        let pos = verlet_objects[self.node_id].position();
        let pos_a = verlet_objects[self.node_a_id].position();
        let pos_b = verlet_objects[self.node_b_id].position();

        let vec_a_b = pos_b - pos_a;
        let vec_a_b_norm = vec_a_b.normalize();
        let vec_m_b = pos_b - pos;

        let vec_a_b = (pos_b - pos_a).normalize();

        let new_pos;
        if (pos_b - pos).magnitude() < 2.0 * radius {
            // left side
            new_pos = pos_b - vec_a_b_norm * 2.0 * radius;
            verlet_objects[self.node_id].set_position(new_pos);
        } else if (pos_a - pos).magnitude() < 2.0 * radius {
            // right side
            new_pos = pos_a + vec_a_b_norm * 2.0 * radius;
            verlet_objects[self.node_id].set_position(new_pos);
        } else {
            // in the middle
            new_pos = pos_b - (vec_a_b.dot(vec_m_b) * vec_a_b) / (vec_a_b.dot(vec_a_b));
        }

        verlet_objects[self.node_id].set_position(new_pos);
    }

    pub fn accelerate(&mut self, val: f32, verlet_objects: &mut [VerletObject]) {
        let pos_a = verlet_objects[self.node_a_id].position();
        let pos_b = verlet_objects[self.node_b_id].position();

        let vec_a_b = (pos_b - pos_a).normalize();

        let acceleration = vec_a_b * val;

        verlet_objects[self.node_id].accelerate(acceleration);
    }
}
