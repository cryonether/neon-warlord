//! A link between two objects

use crate::verlet_physics::VerletObject;

#[derive(Clone)]
pub struct LooseLink {
    pub node_id_1: usize,
    pub node_id_2: usize,
}

impl LooseLink {
    pub fn new(node_id_1: usize, node_id_2: usize) -> Self {
        Self {
            node_id_1,
            node_id_2,
        }
    }

    pub fn apply(&self, _verlet_objects: &mut [VerletObject]) {
        // no constraint to apply
    }
}
