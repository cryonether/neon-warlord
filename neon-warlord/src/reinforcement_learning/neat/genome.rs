//! A neural network
//!

use cgmath::Zero;

use crate::reinforcement_learning::neat::{self, node::NodeKind};
type Vec3 = cgmath::Vector3<f32>;

/// A neural network
#[derive(Clone)]
pub struct Genome {
    pub nr_sensors: usize,
    pub nr_outputs: usize,
    pub nodes: Vec<neat::Node>,
    pub edges: Vec<neat::Edge>,

    innovation: usize,

    pub fitness: f32,

    // Hack to save the position in the world because the structure of the application is way too complicated
    pub world_position: Vec3,
}

impl Genome {
    pub fn new(nr_sensors: usize, nr_outputs: usize) -> Self {
        let mut nodes: Vec<neat::Node> = Vec::with_capacity(nr_sensors + nr_outputs);
        let edges: Vec<neat::Edge> = Vec::new();

        // Sensors
        for i in 0..nr_sensors {
            nodes.push(neat::Node {
                id: i,
                kind: neat::node::NodeKind::Sensor,
                value: 0.0,
                layer: 0,
                bias: 0.0,
            });
        }

        // Outputs
        for i in 0..nr_outputs {
            nodes.push(neat::Node {
                id: nr_sensors + i,
                kind: neat::node::NodeKind::Output,
                value: 0.0,
                layer: 1,
                bias: 0.0,
            });
        }

        Self {
            nr_sensors,
            nr_outputs,
            nodes,
            edges,
            innovation: 0,
            fitness: 0.0,

            world_position: Vec3::zero(),
        }
    }

    pub fn add_edge(&mut self, id_from: usize, id_to: usize, weight: f32) -> bool {
        // Get node indices
        let node_from = self
            .nodes
            .iter()
            .enumerate()
            .find(|(_, node)| node.id == id_from);

        let node_to = self
            .nodes
            .iter()
            .enumerate()
            .find(|(_, node)| node.id == id_to);

        let (index_from, node_from) = match node_from {
            Some(val) => val,
            None => return false,
        };

        let (index_to, node_to) = match node_to {
            Some(val) => val,
            None => return false,
        };

        // Check layers
        let layer_from = node_from.layer;
        let layer_to = node_to.layer;
        if layer_from >= layer_to {
            return false;
        }

        // Create edge
        let edge = neat::Edge {
            _id_from: id_from,
            _id_to: id_to,
            weight,
            enabled: true,
            _innovation: self.innovation,
            index_from,
            index_to,
        };

        self.innovation += 1;

        // Get element position
        let pos = self.edges.partition_point(|e| e.index_to < edge.index_to);

        // Check if element is duplicated
        for i in pos..self.edges.len() {
            let edge = &self.edges[i];
            if edge.index_to == index_to && edge.index_from == index_from {
                // duplicated
                return false;
            }

            if edge.index_to != index_to {
                break;
            }
        }

        // Insert new element
        self.edges.insert(pos, edge);

        true
    }

    pub fn add_layer(&mut self, layer: usize) -> bool {
        if layer == 0 {
            return false;
        }

        // Add new layer
        for node in &mut self.nodes {
            if node.layer >= layer {
                node.layer += 1;
            }
        }

        true
    }

    /// Creates a new node
    /// Returns the id of the new node
    pub fn add_node(&mut self, layer: usize) -> usize {
        // Create node
        let id = self.nodes.len();

        let node = neat::Node {
            id,
            kind: neat::node::NodeKind::Hidden,
            layer,
            value: 0.0,
            bias: 0.0,
        };

        // Insert node
        let pos = self.nodes.partition_point(|n| n.layer <= node.layer);

        // Update edge indices
        for edge in &mut self.edges {
            if edge.index_from >= pos {
                edge.index_from += 1;
            }
            if edge.index_to >= pos {
                edge.index_to += 1;
            }
        }

        self.nodes.insert(pos, node);

        id
    }

    pub fn sensors(&mut self) -> &mut [neat::Node] {
        let size = self.nr_sensors;

        &mut self.nodes[0..size]
    }

    pub fn outputs(&self) -> &[neat::Node] {
        let size = self.nr_outputs;
        let len = self.nodes.len();

        &self.nodes[len - size..len]
    }

    /// Get number of layers
    pub fn _layers(&self) -> usize {
        self.nodes.last().unwrap().layer
    }

    pub fn evaluate(&mut self) {
        let mut index_edge = 0;
        for i in 0..self.nodes.len() {
            if self.nodes[i].kind == NodeKind::Sensor {
                continue;
            }

            // Sum up all edges
            let mut sum = 0.0;
            while index_edge < self.edges.len() && self.edges[index_edge].index_to == i {
                let edge = &self.edges[index_edge];
                let node_from = &self.nodes[edge.index_from];

                sum += node_from.value * edge.weight;

                index_edge += 1;
            }

            let value = sum + self.nodes[i].bias;
            // let value = sum;
            self.nodes[i].value = Self::_activation_sigmoid(value)
        }
    }

    fn _activation_identity(value: f32) -> f32 {
        value
    }

    fn _activation_binary_step(value: f32) -> f32 {
        (value >= 0.0) as u8 as f32
    }

    fn _activation_re_lu(value: f32) -> f32 {
        // best balance
        value.max(0.0)
    }

    fn _activation_leaky_re_lu(value: f32) -> f32 {
        if value >= 0.0 { value } else { 0.01 * value }
    }

    fn _activation_absolute_value(value: f32) -> f32 {
        value.abs()
    }

    fn _activation_tan_h(value: f32) -> f32 {
        value.tanh()
    }

    fn _activation_sigmoid(value: f32) -> f32 {
        // used by the original neat algorithm
        1.0 / (1.0 + (-value).exp())
    }

    fn _activation_sine(value: f32) -> f32 {
        value.sin()
    }

    fn _activation_gaussian(value: f32) -> f32 {
        (-(value * value)).exp()
    }
}
