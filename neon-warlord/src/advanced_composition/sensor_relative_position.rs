//! A sensor which tracks the position relative to another node

use std::vec;

use cgmath::{InnerSpace, Zero};

use crate::{advanced_composition::Vec3, verlet_physics::VerletObject};


/// A sensor which tracks the position relative to another node
pub struct SensorRelativePosition {
    pub node_id: usize,
    pub node_a_id: usize,

    val: Vec3,
}

impl SensorRelativePosition {
    pub fn new(node_id: usize, node_a_id: usize) -> Self {
        Self { node_id, node_a_id, val:Vec3::zero() }
    }
    
    pub fn update(&mut self, verlet_objects: &[VerletObject]) {
        
        // apply constraint
        let pos = verlet_objects[self.node_id].position();
        let pos_a = verlet_objects[self.node_a_id].position();

        let vec_a_s = pos - pos_a;
        self.val = vec_a_s;
    }

    pub fn get_val(&self) -> &Vec3 {
        &self.val
    }
}
