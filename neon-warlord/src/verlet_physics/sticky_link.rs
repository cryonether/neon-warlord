//! A StickyLink between two objects

use cgmath::InnerSpace;

use crate::verlet_physics::VerletObject;

pub struct StickyLink {
    node_id_1: usize,
    node_id_2: usize,
    target_distance: f32,
}

impl StickyLink {
    pub fn new(node_id_1: usize, node_id_2: usize, target_distance: f32) -> Self {
        Self {
            node_id_1,
            node_id_2,
            target_distance,
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

        let axis = object_1.position() - object_2.position();
        let dist = axis.magnitude();

        if dist == 0.0 {
            return;
        }

        let n = axis / dist;
        let delta = self.target_distance - dist;

        object_1.set_position(object_1.position() + 0.5 * delta * n * 1.0);
        object_2.set_position(object_2.position() - 0.5 * delta * n * 1.0);
    }
}
