//! A link between two objects with fixed angle

use crate::verlet_physics::{Vec3, VerletObject};

pub struct FixedLink {
    pub node_id_1: usize,
    pub node_id_2: usize,
    target_vector: Vec3,
    stiffness: f32,
    damping: f32,
    force_split: f32,
}

impl FixedLink {
    pub fn new(node_id_2: usize, node_id_1: usize, target_vector: Vec3) -> Self {
        Self {
            node_id_1,
            node_id_2,
            target_vector,
            stiffness: 1.0,
            damping: 1.0,
            force_split: 0.6,
        }
    }

    pub fn stiffness(self, val: f32) -> Self {
        Self {
            node_id_1: self.node_id_1,
            node_id_2: self.node_id_2,
            target_vector: self.target_vector,
            stiffness: val,
            damping: self.damping,
            force_split: self.force_split,
        }
    }

    pub fn damping(self, val: f32) -> Self {
        Self {
            node_id_1: self.node_id_1,
            node_id_2: self.node_id_2,
            target_vector: self.target_vector,
            stiffness: self.stiffness,
            damping: val,
            force_split: self.force_split,
        }
    }

    pub fn force_split(self, val: f32) -> Self {
        Self {
            node_id_1: self.node_id_1,
            node_id_2: self.node_id_2,
            target_vector: self.target_vector,
            stiffness: self.stiffness,
            damping: self.damping,
            force_split: val,
        }
    }

    pub fn apply(&self, verlet_objects: &mut [VerletObject]) {
        if self.node_id_1 >= verlet_objects.len()
            || self.node_id_2 >= verlet_objects.len()
            || self.node_id_1 == self.node_id_2
        {
            return;
        }

        let (object_1, object_2) = if self.node_id_1 < self.node_id_2 {
            let (left, right) = verlet_objects.split_at_mut(self.node_id_2);
            (&mut left[self.node_id_1], &mut right[0])
        } else {
            let (left, right) = verlet_objects.split_at_mut(self.node_id_1);
            (&mut right[0], &mut left[self.node_id_2])
        };

        let axis = self.target_vector - (object_1.position() - object_2.position());

        let stiffness = 2500.0 * self.stiffness;
        let damping = self.damping;

        object_1.accelerate(self.force_split * axis * stiffness);
        object_2.accelerate((self.force_split - 1.0) * axis * stiffness);

        object_1.damp(damping);
        object_2.damp(damping);
    }
}
