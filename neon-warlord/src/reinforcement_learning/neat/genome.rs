//! A neural network
//! 

use crate::{procedural_tree::node, reinforcement_learning::neat};

/// A neural network
pub struct Genome {
    pub nr_sensors: usize,
    pub nr_outputs: usize,
    pub nodes: Vec<neat::Node>,
    pub edges: Vec<neat::Edge>,

    innovation: usize,
}

impl Genome {
    pub fn new(
        nr_sensors: usize,
        nr_outputs: usize,
    ) -> Self {
       let mut nodes: Vec<neat::Node> =  Vec::with_capacity(nr_sensors + nr_outputs);
       let edges: Vec<neat::Edge> = Vec::new();

       // Sensors  
       for i in 0..nr_sensors {
        nodes.push(neat::Node{
            id: i,
            kind: neat::node::NodeKind::Sensor,
            value: 0.0,
            layer: 0,
        });
       }

       // Outputs
       for i in 0..nr_outputs {
        nodes.push(neat::Node{
            id: nr_sensors+i,
            kind: neat::node::NodeKind::Output,
            value: 0.0,
            layer: 1,
        });
       }

        Self {
            nr_sensors,
            nr_outputs,
            nodes,
            edges,
            innovation:0,
        }
    }

    pub fn add_connection(&mut self, id_from: usize, id_to: usize) -> bool {
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

        let (index_from, node_from) = match node_from  {
            Some(val) => val,
            None => return false,
        };

        let (index_to, node_to) = match node_to  {
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
            id_from,
            id_to,
            weight: 0.0,
            enabled: true,
            innovation: self.innovation,
            index_from,
            index_to,
        };

        self.innovation += 1;

        // Insert element
        let pos = self.edges.partition_point(|e| e.index_to <= edge.index_to);
        self.edges.insert(pos, edge);

        return true;
    }

    pub fn add_layer(&mut self, layer: usize) -> bool {
        if layer > self.nodes.last().unwrap().layer {
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

    pub fn add_node(&mut self, layer: usize) -> usize {
        // Create node
        let id = self.nodes.len();

        let node = neat::Node{
            id,
            kind: neat::node::NodeKind::Hidden,
            layer,
            value: 0.0,
        };

        // Insert node
        let pos = self.nodes.partition_point(|n| n.layer <= node.layer);
        self.nodes.insert(pos, node);

        // Update edge indices
        for edge in &mut self.edges {
            if edge.index_from > pos {
                edge.index_from += 1;
            }
            if edge.index_to > pos {
                edge.index_to += 1;
            }
        }

        id
    }

    pub fn sensors(&mut self) -> &mut [neat::Node] {
        let size = self.nr_sensors;

        &mut self.nodes[0..size]
    }

    pub fn outputs(&self) -> &[neat::Node] {
        let size = self.nr_outputs;
        let len = self.nodes.len();

        & self.nodes[len-size..len]
    }

    pub fn evaluate(&mut self) {
        let mut index_edge = 0;
        for i in 0..self.nodes.len() {

            // Sum up all edges
            let mut sum = 0.0;
            while index_edge < self.edges.len() && self.edges[index_edge].index_to == i {
                let edge = &self.edges[index_edge];
                let node_from = &self.nodes[edge.index_from];

                sum += node_from.value * edge.weight;

                index_edge += 1;
            } 

            self.nodes[i].value = sum;
        }
    }
}
