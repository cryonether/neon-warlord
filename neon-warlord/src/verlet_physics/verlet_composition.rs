//! Creates an agent using verlet physics

use cgmath::MetricSpace;

use crate::verlet_physics::{
    VerletObject, fixed::Fixed, fixed_link::FixedLink, link::Link, sticky_link::StickyLink,
};
type Vec3 = cgmath::Vector3<f32>;

pub struct VerletComposition {
    pub verlet_objects: Vec<VerletObject>,
    pub links: Vec<Link>,
    pub fixed_links: Vec<FixedLink>,
    pub fixed: Vec<Fixed>,
    pub sticky_links: Vec<StickyLink>,
}

impl VerletComposition {
    pub fn create(nodes: &[Node], radius: f32) -> VerletComposition {
        let pos = Vec3::new(0.0, 0.0, 0.0);

        let mut verlet_objects = Vec::new();
        let mut links = Vec::new();
        let mut fixed_links = Vec::new();
        let fixed = Vec::new();
        let mut sticky_links = Vec::new();

        if nodes.is_empty() {
            return Self {
                verlet_objects,
                links,
                fixed_links,
                fixed,
                sticky_links,
            };
        }

        // Create all nodes
        for node in nodes {
            let pos = pos + node.pos;

            // Create a physics node
            verlet_objects.push(VerletObject::new(pos, radius));
        }

        // create all links
        for node in nodes {
            let id_0 = node.id;
            let id_1 = node.link_target;
            let pos_0 = verlet_objects[id_0].position();
            let pos_1 = verlet_objects[id_1].position();

            match node.link_kind {
                LinkKind::Fixed => {
                    fixed_links.push(
                        FixedLink::new(id_0, id_1, pos_1 - pos_0)
                            .damping(0.9)
                            .force_split(0.45),
                    );
                }
                LinkKind::Linked => links.push(Link::new(id_0, id_1, pos_0.distance(pos_1))),
                LinkKind::Sticky => {
                    sticky_links.push(StickyLink::new(id_0, id_1, pos_0.distance(pos_1)))
                }
                LinkKind::Origin => {
                    // fixed.push(Fixed::new(id_0, pos_0));
                }
            }
        }

        Self {
            verlet_objects,
            links,
            fixed_links,
            fixed,
            sticky_links,
        }
    }
}

pub enum LinkKind {
    /// Node is fixed in location to the other node
    Fixed,
    /// Node is fixed in distance to the other node
    Linked,
    /// Node is fixed in distance to the other node and is sticky to the ground
    Sticky,
    /// Treated as origin of the structure (has no parent links)
    Origin,
}

pub struct Node {
    pub id: usize,
    pub link_kind: LinkKind,
    pub link_target: usize,
    pub pos: Vec3,
}
