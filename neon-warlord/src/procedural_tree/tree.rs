//! A tree with nodes and leaves

use std::ops::Add;

use crate::procedural_tree::node::Node;
use cgmath::{InnerSpace, Vector3};
use cgmath::{Quaternion, Rotation};
use fastrand::Rng;

type Vec3 = cgmath::Vector3<f32>;

pub struct Tree {
    nodes: Vec<Node>,

    root: usize,
    depth: usize,

    nr_nodes: usize,
    nr_leaves: usize,
}

impl Tree {
    pub fn new(depth: usize, seed: u64) -> Self {
        let mut nodes = Vec::new();
        let root = 0;

        let mut rng = Rng::with_seed(seed);

        let pos = Vec3::new(-2.0, 15.0, 0.1);
        let root_node = Node::new(pos);
        nodes.push(root_node);

        let tree_details = Self::add_children(root, pos, &mut nodes, depth - 1, depth, &mut rng);

        Self {
            nodes,
            root,
            depth,

            nr_nodes: tree_details.nr_nodes,
            nr_leaves: tree_details.nr_leaves,
        }
    }

    fn add_children(
        node_index: usize,
        absolute_position: Vec3,
        nodes: &mut Vec<Node>,
        depth: usize,
        max_depth: usize,
        rng: &mut Rng,
    ) -> TreeDetails {
        let mut res = if depth == 0 {
            TreeDetails {
                nr_nodes: 0,
                nr_leaves: 1,
            }
        } else {
            TreeDetails {
                nr_nodes: 1,
                nr_leaves: 0,
            }
        };

        if depth == 0 {
            return res;
        }

        let nr_children: usize = if depth == 1 {
            rng.usize(3..=5)
            // rng.random_range(1..=1)
        } else {
            rng.usize(1..=(max_depth - depth).clamp(1, 3))
            // rng.random_range(1..=1)
        };

        let index = nodes.len();
        let node = &mut nodes[node_index];
        let node_position = node.position;
        node.children_base_index = index;
        node.nr_children = nr_children;

        for _i in 0..nr_children {
            let x = rng.f32() * 2.0 - 1.0;
            let y = rng.f32() * 2.0 - 1.0;
            let z = rng.f32() * 0.8 + 0.2;

            let height = (depth - 1) as f32 * 0.3;
            // let z = height;

            let pos = Vec3::new(x, y, z);

            let from = Vector3::new(0.0, 0.0, 1.0);
            let to = node_position.normalize();
            let rotation = Quaternion::between_vectors(from, to);
            let rotated = rotation.rotate_vector(pos);

            let mut pos2 = rotated;
            pos2.z += height * height;
            // pos2.z += height;

            pos2 /= 1.8;

            // let pos = (Vec3::new(x, y, z)).normalize();

            nodes.push(Node::new(Vec3::new(pos2.x, pos2.y, pos2.z)));
        }

        for i in index..index + nr_children {
            let pos = nodes[i].position;
            let res_child =
                Self::add_children(i, absolute_position + pos, nodes, depth - 1, max_depth, rng);
            res = res + res_child
        }

        res
    }

    pub fn traverse_tree(&self, tree_interface: &mut dyn TreeInterface) {
        let node = &self.nodes[self.root];
        tree_interface.create_node(
            self.root,
            None,
            &node.position,
            &node.position,
            self.depth,
            node.nr_children == 0,
        );
        Self::traverse(self, self.root, node.position, self.depth, tree_interface);
    }

    fn traverse(
        &self,
        node_index: usize,
        absolute_position: Vec3,
        depth: usize,
        tree_interface: &mut dyn TreeInterface,
    ) {
        let node = &self.nodes[node_index];
        let _pos_0 = &node.position;
        let index = node.children_base_index;
        let nr_children = node.nr_children;

        #[allow(clippy::needless_range_loop)]
        for i in index..index + nr_children {
            let node = &self.nodes[i];
            let pos_1 = node.position;
            tree_interface.create_node(
                i,
                Some(node_index),
                &absolute_position,
                &pos_1,
                depth - 1,
                node.nr_children == 0,
            );
        }

        for i in index..index + nr_children {
            let node = &self.nodes[i];
            let pos_1 = node.position;

            Self::traverse(
                self,
                i,
                absolute_position + pos_1,
                depth - 1,
                tree_interface,
            );
        }
    }

    pub fn size(&self) -> usize {
        self.nodes.len()
    }

    pub fn nr_leaves(&self) -> usize {
        self.nr_leaves
    }

    pub fn nr_nodes(&self) -> usize {
        self.nr_nodes
    }

    pub fn nr_links(&self) -> usize {
        self.nr_nodes - 1 + self.nr_leaves
    }
}

pub trait TreeInterface {
    fn create_node(
        &mut self,
        node_index: usize,
        parent_index: Option<usize>,
        absolute_position: &Vec3,
        relative_position: &Vec3,
        depth: usize,
        is_leave: bool,
    );
}

struct TreeDetails {
    nr_nodes: usize,
    nr_leaves: usize,
}

impl Add for TreeDetails {
    type Output = TreeDetails;

    fn add(self, other: TreeDetails) -> TreeDetails {
        TreeDetails {
            nr_nodes: self.nr_nodes + other.nr_nodes,
            nr_leaves: self.nr_leaves + other.nr_leaves,
        }
    }
}
