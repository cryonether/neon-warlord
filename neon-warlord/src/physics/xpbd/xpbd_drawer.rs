//! Draws AdvancedCompositions

use forward_renderer::{particle_shader, particle_shader_two_point, to_rgb};

use super::*;

type Vec3 = cgmath::Vector3<f32>;

/// Draws AdvancedCompositions
#[derive(Clone)]
pub struct XpbdDrawer {
    nodes_color_0: Vec3,
    nodes_color_1: Vec3,
    links_color: Vec3,
    pos_color: Vec3,

    _nr_nodes: usize,
    _nr_edges: usize,

    pub radius: f32,
    pub position: Vec3,
}

impl XpbdDrawer {
    pub fn new(physics: &Xpbd, radius: f32, position: Vec3) -> Self {
        let _nr_nodes = physics.rigid_bodies.len();
        let _nr_edges = physics.hinge_joints.len();

        let nodes_color_0: Vec3 = to_rgb("#ce51ff").into();
        let nodes_color_1: Vec3 = to_rgb("#371643").into();
        let links_color: Vec3 = to_rgb("#131922").into();
        let pos_color: Vec3 = to_rgb("#00814b").into();

        Self {
            nodes_color_0,
            nodes_color_1,
            links_color,
            pos_color,
            _nr_nodes,
            _nr_edges,
            radius,
            position,
        }
    }

    pub fn update(
        &self,
        physics: &Xpbd,
        producer_nodes: &mut Vec<particle_shader::Instance>,
        producer_edges: &mut Vec<particle_shader_two_point::Instance>,
    ) {
        // Particles
        for rigid_body in &physics.rigid_bodies {
            let position = rigid_body.p.pos + self.position;
            let radius = self.radius;

            producer_nodes.push(particle_shader::Instance {
                position: position.into(),
                color: self.pos_color.into(),
                time: 1.0,
                size: radius,
            });

            if rigid_body.particles.is_empty() {
                continue;
            }

            let particle_0 = rigid_body.particles[0];
            let position_0 = rigid_body.local_to_world(&particle_0) + self.position;
            producer_nodes.push(particle_shader::Instance {
                position: position_0.into(),
                color: self.nodes_color_0.into(),
                time: 1.0,
                size: radius * 2.0,
            });

            let mut prev_position = position_0;

            for i in 1..rigid_body.particles.len() {
                let particle = &rigid_body.particles[i];

                let position = rigid_body.local_to_world(particle) + self.position;

                producer_nodes.push(particle_shader::Instance {
                    position: position.into(),
                    color: self.nodes_color_0.into(),
                    time: 1.0,
                    size: radius * 2.0,
                });

                producer_edges.push(particle_shader_two_point::Instance {
                    position_0: prev_position.into(),
                    position_1: position.into(),
                    color: self.nodes_color_1.into(),
                    time: 1.0,
                    size: self.radius * 0.1,
                });

                prev_position = position;
            }
        }

        // Position constraints
        for constraint in &physics.hinge_joints {
            let index_a = constraint.index_a;
            let index_b = constraint.index_b;

            let rigid_body_a = &physics.rigid_bodies[index_a];
            let rigid_body_b = &physics.rigid_bodies[index_b];

            let pos_a = rigid_body_a.local_to_world(&constraint.constraint.attachment_point_0)
                + self.position;
            let pos_b = rigid_body_b.local_to_world(&constraint.constraint.attachment_point_1)
                + self.position;

            producer_edges.push(particle_shader_two_point::Instance {
                position_0: pos_a.into(),
                position_1: pos_b.into(),
                color: self.links_color.into(),
                time: 1.0,
                size: self.radius * 0.1,
            });
        }
    }
}
