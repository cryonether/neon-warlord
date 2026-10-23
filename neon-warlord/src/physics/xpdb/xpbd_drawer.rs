//! Draws AdvancedCompositions

use forward_renderer::{particle_shader, particle_shader_two_point, to_rgb};

use super::*;

type Vec3 = cgmath::Vector3<f32>;

/// Draws AdvancedCompositions
#[derive(Clone)]
pub struct XpbdDrawer {
    nodes_color_0: Vec3,
    links_color_0: Vec3,

    _nr_nodes: usize,
    _nr_edges: usize,

    pub radius: f32,
    pub position: Vec3,
}

impl XpbdDrawer {
    pub fn new(physics: &Xpbd, radius: f32, position: Vec3) -> Self {
        let _nr_nodes = physics.particles.len();
        let _nr_edges = physics.position_constraints.len();

        let nodes_color_0: Vec3 = to_rgb("#ce51ff").into();
        let _nodes_color_1: Vec3 = to_rgb("#a72ebc").into();
        let links_color_0: Vec3 = to_rgb("#131922").into();
        let _links_color_1: Vec3 = to_rgb("#2d2e27").into();

        Self {
            nodes_color_0,
            links_color_0,
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
        for rigid_body  in &physics.rigid_bodies {
            let position = rigid_body.p.pos + self.position;
            let radius = self.radius;

            producer_nodes.push(particle_shader::Instance {
                position: position.into(),
                color: self.nodes_color_0.into(),
                time: 1.0,
                size: radius * 2.0,
            });
        }

        // Position constraints
        for constraint in &physics.position_constraints {
            let index_a = constraint.index_a;
            let index_b = constraint.index_b;

            let pos_0 = physics.rigid_bodies[index_a].p.pos + self.position;
            let pos_1 = physics.rigid_bodies[index_b].p.pos + self.position;

            producer_edges.push(particle_shader_two_point::Instance {
                position_0: pos_0.into(),
                position_1: pos_1.into(),
                color: self.links_color_0.into(),
                time: 1.0,
                size: self.radius * 0.1,
            });
        }
    }
}
