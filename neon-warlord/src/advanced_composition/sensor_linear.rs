//! An actor free to move between two nodes

use cgmath::InnerSpace;

use crate::verlet_physics_simd::verlet_particles::VerletParticles;
type Vec3 = cgmath::Vector3<f32>;

/// An actor free to move between two nodes
#[derive(Clone)]
pub struct SensorLinear {
    pub node_id: usize,
    pub node_a_id: usize,
    pub node_b_id: usize,

    // value between -1 to 1
    position_previous: f32,
    position: f32,
    velocity: f32,
}

impl SensorLinear {
    pub fn new(node_id: usize, node_a_id: usize, node_b_id: usize) -> Self {
        Self {
            node_id,
            node_a_id,
            node_b_id,
            position_previous: 0.0,
            position: 0.0,
            velocity: 0.0,
        }
    }

    pub fn update_simd(&mut self, verlet_particles: &VerletParticles) {
        let pos: cgmath::Vector3<f32> = verlet_particles.position(self.node_id);
        let pos_a: cgmath::Vector3<f32> = verlet_particles.position(self.node_a_id);
        let pos_b: cgmath::Vector3<f32> = verlet_particles.position(self.node_b_id);

        self.position_previous = self.position;
        self.position = Self::calculate(&pos, &pos_a, &pos_b);
        self.velocity = self.position - self.position_previous;

    }

    fn calculate(pos: &Vec3, pos_a: &Vec3, pos_b: &Vec3) -> f32 {
        let ab = pos_b - pos_a;
        let t = (pos - pos_a).dot(ab) / ab.dot(ab);

        t * 2.0 - 1.0
    }

    pub fn position(&self) -> f32 {
        self.position
    }

    pub fn velocity(&self) -> f32 {
        self.velocity
    }
}

#[test]
fn test_0() {
    let pos = Vec3::new(0.0, 0.0, 0.0);
    let pos_a = Vec3::new(-10.0, 0.0, 0.0);
    let pos_b = Vec3::new(10.0, 0.0, 0.0);

    let res = SensorLinear::calculate(&pos, &pos_a, &pos_b);

    assert_eq!(res, 0.0);
}

#[test]
fn test_1() {
    let pos = Vec3::new(-10.0, 0.0, 0.0);
    let pos_a = Vec3::new(-10.0, 0.0, 0.0);
    let pos_b = Vec3::new(10.0, 0.0, 0.0);

    let res = SensorLinear::calculate(&pos, &pos_a, &pos_b);

    assert_eq!(res, -1.0);
}

#[test]
fn test_2() {
    let pos = Vec3::new(10.0, 0.0, 0.0);
    let pos_a = Vec3::new(-10.0, 0.0, 0.0);
    let pos_b = Vec3::new(10.0, 0.0, 0.0);

    let res = SensorLinear::calculate(&pos, &pos_a, &pos_b);

    assert_eq!(res, 1.0);
}
