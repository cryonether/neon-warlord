//! A link between two objects

use crate::physics::verlet_physics::{Vec3, VerletObject};

pub struct Fixed {
    node_id_1: usize,
    position: Vec3,
}

impl Fixed {
    pub fn new(node_id_1: usize, position: Vec3) -> Self {
        Self {
            node_id_1,
            position,
        }
    }

    pub fn apply(&self, verlet_objects: &mut [VerletObject]) {
        if self.node_id_1 >= verlet_objects.len() {
            return;
        }

        let object = &mut verlet_objects[self.node_id_1];

        object.set_position(self.position);
    }
}
