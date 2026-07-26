use std::collections::VecDeque;

use gltf::texture::WrappingMode::ClampToEdge;

///! A neural network
use crate::reinforcement_learning::neat;

/// A Directed Acyclic Graph
#[derive(Debug)]
pub struct Dag {
    pub nodes: Vec<f32>,
    pub edges: Vec<(usize, usize)>,
}

impl Dag {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
        }
    }

    pub fn add_node(&mut self, value: f32) -> usize {
        let id = self.nodes.len();
        self.nodes.push(value);
        id
    }

    pub fn add_edge(&mut self, from: usize, to: usize) -> Result<(), &'static str> {
        if from >= self.nodes.len() || to >= self.nodes.len() {
            return Err("node index out of bounds");
        }

        if from == to {
            return Err("self edge creates cycle");
        }

        // Already ordered, easy case
        if from < to {
            if self.creates_cycle(from, to) {
                return Err("edge creates cycle");
            }

            self.edges.push((from, to));
            self.edges.sort_by_key(|&(a, b)| (a, b));
            return Ok(());
        }

        // Need to move `from` before `to`
        if self.creates_cycle(from, to) {
            return Err("edge creates cycle");
        }

        self.edges.push((from, to));
        self.retopologize(from, to);

        Ok(())
    }

    fn creates_cycle(&self, from: usize, to: usize) -> bool {
        // Adding from -> to creates a cycle if to already reaches from
        let mut stack = vec![to];
        let mut visited = vec![false; self.nodes.len()];

        while let Some(n) = stack.pop() {
            if n == from {
                return true;
            }

            if visited[n] {
                continue;
            }

            visited[n] = true;

            for &(a, b) in &self.edges {
                if a == n {
                    stack.push(b);
                }
            }
        }

        false
    }

    fn retopologize(&mut self, from: usize, to: usize) {
        // Move `from` and its outgoing dependencies before `to`
        //
        // Simple stable version: recompute full topological ordering.

        let mut indegree = vec![0usize; self.nodes.len()];

        for &(a, b) in &self.edges {
            indegree[b] += 1;
        }

        let mut queue = Vec::new();

        for i in 0..self.nodes.len() {
            if indegree[i] == 0 {
                queue.push(i);
            }
        }

        let mut order = Vec::with_capacity(self.nodes.len());

        while let Some(n) = queue.pop() {
            order.push(n);

            for &(a, b) in &self.edges {
                if a == n {
                    indegree[b] -= 1;
                    if indegree[b] == 0 {
                        queue.push(b);
                    }
                }
            }
        }

        debug_assert_eq!(order.len(), self.nodes.len());

        // Remap node indices
        let mut map = vec![0; self.nodes.len()];
        for (new, old) in order.iter().enumerate() {
            map[*old] = new;
        }

        let mut new_nodes = Vec::with_capacity(self.nodes.len());
        for old in order {
            new_nodes.push(self.nodes[old]);
        }

        self.nodes = new_nodes;

        for (a, b) in &mut self.edges {
            *a = map[*a];
            *b = map[*b];
        }

        self.edges.sort_by_key(|&(a, b)| (a, b));
    }
}

pub struct Genome {
    pub sensors: Vec<neat::Node>,
    pub outputs: Vec<neat::Node>,
    pub layers_nodes: Vec<Vec<neat::Node>>,
    pub layers_connections: Vec<Vec<neat::Edge>>,
}

impl Genome {
    pub fn new(
        nr_sensors: usize,
        nr_outputs: usize,
    ) -> Self {
       let mut sensors: Vec<neat::Node> = Vec::with_capacity(nr_sensors);
       let mut outputs: Vec<neat::Node> = Vec::with_capacity(nr_outputs);
       let layers_nodes: Vec<Vec<neat::Node>> = Vec::new();
       let layers_connections: Vec<Vec<neat::Edge>> = Vec::new();

       for i in 0..nr_sensors {
        sensors.push(neat::Node{
            id: i,
            kind: neat::node::NodeKind::Sensor,
            value: 0.0,
        });
       }


       for i in 0..nr_outputs {
        outputs.push(neat::Node{
            id: i,
            kind: neat::node::NodeKind::Output,
            value: 0.0,
        });
       }

        Self {
            sensors,
            outputs,
            layers_nodes,
            layers_connections,
        }
    }

    pub fn evaluate(&mut self) {}
}
