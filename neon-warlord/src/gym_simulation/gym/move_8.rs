//! Moves in an 8

use cgmath::{InnerSpace, Zero};
use forward_renderer::{particle_shader, to_rgb};

use crate::gym_simulation::Vec3;

/// Moves in an 8
#[derive(Clone)]
pub struct Move8 {
    points: Vec<Vec3>,

    index: usize,
}

impl Move8 {
    pub fn new() -> Self {
        let middle = Vec3::zero();
        let nr_points = 100; // Nr of points in the 8 path
        let radius = 20.0; // Radius of each circle on each side
        let distance = 120.0; // Distance between the middle of both circles

        let mut points: Vec<Vec3> = Vec::new();

        // Generates a path on an 8
        for i in 0..nr_points {
            let t = (i as f32 / nr_points as f32) * 2.0 * std::f32::consts::PI;

            let x = (distance / 2.0) * t.sin();
            let y = radius * (2.0 * t).sin();

            points.push(middle + Vec3::new(x, y, 0.0));
        }

        Self { points, index: 0 }
    }

    pub fn get_next_position(&mut self, pos: &Vec3) -> Vec3 {
        let min_distance = 4.0;

        let next_pos = self.points[self.index];
        let distance2 = (next_pos - pos).magnitude2();
        if distance2 > min_distance * min_distance {
            next_pos
        } else {
            self.index = (self.index + 1) % self.points.len();

            self.points[self.index]
        }
    }

    pub fn get_dirction(&mut self, pos: &Vec3) -> Vec3 {
        let next_pos = self.get_next_position(pos);
        next_pos - pos
    }

    pub fn set_random_position(&mut self) {
        self.index = fastrand::usize(0..self.points.len());
    }

    pub fn get_position(&self) -> Vec3 {
        self.points[self.index]
    }
}


// #######################################################################
// Move8Drawer

/// Draws Move8
#[derive(Clone)]
pub struct Move8Drawer {
    nodes_color_0: Vec3,

    radius: f32,
    pub position: Vec3,
}

impl Move8Drawer {
    pub fn new(radius: f32, position: Vec3) -> Self {
        let nodes_color_0: Vec3 = to_rgb("#51ffd9").into();

        Self {
            nodes_color_0,
            radius,
            position,
        }
    }

    pub fn draw(&self, move_8: &Move8, producer_nodes: &mut Vec<particle_shader::Instance>) {
        for point in &move_8.points {
            let pos = *point + Vec3::new(0.0, 0.0, self.radius) + self.position;
            producer_nodes.push(particle_shader::Instance {
                position: pos.into(),
                color: self.nodes_color_0.into(),
                time: 1.0,
                size: self.radius * 2.0,
            });
        }
    }
}
