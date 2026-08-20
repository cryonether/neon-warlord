//! Definitions for a advanced composition

use cgmath::{InnerSpace, MetricSpace, Zero};

use crate::advanced_composition::neural_network::FitnessFunction;

type Vec3 = cgmath::Vector3<f32>;

#[derive(Clone)]
pub struct Node {
    pub id: usize,
    pub kind: NodeKind,
    pub edge: EdgeKind,
}

impl Node {
    fn f(mut self, n: usize) -> Self {
        self.edge = EdgeKind::Fixed(n);
        self
    }

    fn d(mut self, n: usize) -> Self {
        self.edge = EdgeKind::FixedDistance(n);
        self
    }

    fn l(mut self, n: usize) -> Self {
        self.edge = EdgeKind::Loose(n);
        self
    }
}

#[derive(Clone, PartialEq)]
pub enum NodeKind {
    // None
    None,
    // Regular Node
    Regular,
    // Fixed
    Static,
    // Linear Motor
    MotorLinear(usize, usize),
    // Sensor tracking the relative position to another node
    SensorRelativePosition(usize),
    // Neural Network
    NeuralNetwork,
}

#[derive(Clone)]
pub enum EdgeKind {
    // None
    None,
    // Loose
    Loose(usize),
    // Fixed
    Fixed(usize),
    // Fixed distance
    FixedDistance(usize),
}

const Z: Node = Node {
    id: 0,
    kind: NodeKind::None,
    edge: EdgeKind::None,
};

fn a(n: usize) -> Node {
    Node {
        id: n,
        kind: NodeKind::Regular,
        edge: EdgeKind::None,
    }
}

fn ml(n: usize, a: usize, b: usize) -> Node {
    Node {
        id: n,
        kind: NodeKind::MotorLinear(a, b),
        edge: EdgeKind::None,
    }
}

fn z(n: usize) -> Node {
    Node {
        id: n,
        kind: NodeKind::Static,
        edge: EdgeKind::None,
    }
}

fn srp(n: usize, a: usize) -> Node {
    Node {
        id: n,
        kind: NodeKind::SensorRelativePosition(a),
        edge: EdgeKind::None,
    }
}

fn n(n: usize) -> Node {
    Node {
        id: n,
        kind: NodeKind::NeuralNetwork,
        edge: EdgeKind::None,
    }
}

pub struct LocatedNode {
    pub node: Node,
    pub pos: Vec3,
}

pub struct ParsedDefinition {
    pub nodes: Vec<LocatedNode>,
    pub scale: f32,
}

impl ParsedDefinition {
    pub fn parse<const NR_SLICES: usize, const R: usize, const C: usize>(
        layers: &[[[Node; C]; R]; NR_SLICES],
        pos: Vec3,
        scale: f32,
    ) -> ParsedDefinition {
        let mut res: Vec<LocatedNode> = Vec::new();

        // add elements
        #[allow(clippy::needless_range_loop)]
        for nr_slice in 0..NR_SLICES {
            for r in 0..R {
                for c in 0..C {
                    let node = layers[nr_slice][r][c].clone();

                    let pos = Vec3::new(nr_slice as f32, r as f32, c as f32);

                    let agent_node = LocatedNode { node, pos };

                    res.push(agent_node);
                }
            }
        }

        res.sort_by_key(|elem| elem.node.id);

        let origin = res[0].pos;
        for elem in &mut res {
            let local_pos = Vec3::new(
                elem.pos.z - origin.z,
                elem.pos.y - origin.y,
                -elem.pos.x - origin.x,
            );

            elem.pos = local_pos;
        }

        // move and scale elements
        for elem in &mut res {
            elem.pos = elem.pos * scale + pos;
        }

        ParsedDefinition { nodes: res, scale }
    }

    pub fn count_nr_neural_networks(&self) -> usize {
        let mut sum = 0;
        for elem in &self.nodes {
            if elem.node.kind == NodeKind::NeuralNetwork {
                sum += 1;
            }
        }

        sum
    }

    // Get number of neural network input lines
    pub fn count_nr_neural_network_inputs(&self) -> usize {
        let mut sum = 0;
        for elem in &self.nodes {
            sum += match elem.node.kind {
                NodeKind::None => 0,
                NodeKind::Regular => 0,
                NodeKind::Static => 0,
                NodeKind::MotorLinear(_, _) => 2,
                NodeKind::SensorRelativePosition(_) => 6,
                NodeKind::NeuralNetwork => 0,
            };
        }

        sum
    }

    // Get number of neural network output lines
    pub fn count_nr_neural_network_outputs(&self) -> usize {
        let mut sum = 0;
        for elem in &self.nodes {
            sum += match elem.node.kind {
                NodeKind::None => 0,
                NodeKind::Regular => 0,
                NodeKind::Static => 0,
                NodeKind::MotorLinear(_, _) => 1,
                NodeKind::SensorRelativePosition(_) => 0,
                NodeKind::NeuralNetwork => 0,
            };
        }

        sum
    }
}

#[allow(dead_code)]
#[rustfmt::skip]
pub fn get_agent_0_definition() -> [[[Node; 9]; 9]; 3] {

    let layer_0 = [
        [a(13).f(9) , Z          , Z          , Z          , Z          , Z          , Z          , Z          , a(14).f(10)],
        [Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          ],
        [Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          ],
        [Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          ],
        [Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          ],
        [Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          ],
        [Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          ],
        [Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          ],
        [a(16).f(12), Z          , Z          , Z          , Z          , Z          , Z          , Z          , a(15).f(11)],
    ];

    let layer_1 = [
        [Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          ],
        [Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          ],
        [Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          ],
        [Z          , Z          , Z          , a(8).f(0)  , a(1).f(0)  , a(2).f(0)  , Z          , Z          , Z          ],
        [Z          , Z          , Z          , a(7).f(0)  , a(0)       , a(3).f(0)  , Z          , Z          , Z          ],
        [Z          , Z          , Z          , a(6).f(0)  , a(5).f(0)  , a(4).f(0)  , Z          , Z          , Z          ],
        [Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          ],
        [Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          ],
        [Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          ],
    ];

    let layer_2 = [
        [Z          , Z          , Z          , Z          , Z          , Z          , Z         , Z          , Z           ],
        [Z          , Z          , Z          , Z          , Z          , Z          , Z         , Z          , Z           ],
        [Z          , Z          , a(9).f(8)  , Z          , Z          , Z          , a(10).f(2), Z          , Z           ],
        [Z          , Z          , Z          , Z          , Z          , Z          , Z         , Z          , Z           ],
        [Z          , Z          , Z          , Z          , Z          , Z          , a(17).f(3), Z          , Z           ],
        [Z          , Z          , Z          , Z          , Z          , Z          , Z         , Z          , Z           ],
        [Z          , Z          , a(12).f(6) , Z          , Z          , Z          , a(11).f(4), Z          , Z           ],
        [Z          , Z          , Z          , Z          , Z          , Z          , Z         , Z          , Z           ],
        [Z          , Z          , Z          , Z          , Z          , Z          , Z         , Z          , Z           ],
    ];

    [layer_0, layer_1, layer_2]
}

// Nodes:
// Z ... None
// a ... Regular Node
// s ... Static Node

// Links:
// .f ... Fixed position to another node

#[rustfmt::skip]
pub fn get_pendulum_definition() -> [[[Node; 11]; 9]; 4] {

    let layer_0 = [
        [Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          ],
        [Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          ],
        [Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          ],
        [Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          ],
        [Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          ],
        [Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          ],
        [Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          ],
        [Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          ],
        [Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          ],
    ];

    let layer_1 = [
        [Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          ],
        [Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          ],
        [Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          ],
        [Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          ],
        [z(1).l(0)  , Z          , Z          , Z          , Z          , ml(0, 1, 2), Z   , Z        , Z            , Z          , z(2).l(0)  ],
        [Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          ],
        [Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          ],
        [Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          ],
        [Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z          ],
    ];

    let layer_2 = [
        [Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z         , Z          , Z          , Z           ],
        [Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z         , Z          , Z          , Z           ],
        [Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z         , Z          , Z          , Z           ],
        [Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z         , Z          , Z          , Z           ],
        [n(4).f(1)  , Z          , Z          , Z          , Z          , Z          , Z          , Z         , Z          , Z          , Z],
        [Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z         , Z          , Z          , Z           ],
        [Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z         , Z          , Z          , Z           ],
        [Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z         , Z          , Z          , Z           ],
        [Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z         , Z          , Z          , Z           ],
    ];

    let layer_3 = [
        [Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z         , Z         , Z          , Z           ],
        [Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z         , Z         , Z          , Z           ],
        [Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z         , Z         , Z          , Z           ],
        [Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z         , Z         , Z          , Z           ],
        [Z          , Z          , Z          , Z          , Z          , srp(3, 0).d(0) , Z , Z         , Z        , Z           , Z           ],
        [Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z         , Z         , Z          , Z           ],
        [Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z         , Z         , Z          , Z           ],
        [Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z         , Z         , Z          , Z           ],
        [Z          , Z          , Z          , Z          , Z          , Z          , Z          , Z         , Z         , Z          , Z           ],
    ];

    [layer_0, layer_1, layer_2, layer_3]
}

pub fn get_pendulum_definition_fitness_function() -> Box<dyn FitnessFunction + 'static + Send> {
    struct FitnessFunctionAccumulateZ {
        pub sum: f32,
        pub last_position: Vec3,
    }
    impl FitnessFunction for FitnessFunctionAccumulateZ {
        fn calculate_fitness(&mut self, inputs: &[f32]) -> f32 {
            assert!(inputs.len() == 8);

            let linear_motor_position = inputs[0];
            let linear_motor_velocity = inputs[1];
            let pos = Vec3::new(inputs[2], inputs[3], inputs[4]);
            let volocity = Vec3::new(inputs[5], inputs[6], inputs[7]);

            self.sum += (1.0 + pos.z) * (1.0 + pos.z) - 0.1 * linear_motor_velocity.abs() - 0.4 * volocity.magnitude() * volocity.magnitude() - 0.4 * linear_motor_position.abs() * linear_motor_position.abs();

            self.last_position = pos;

            self.sum
        }

        fn clone_box(&self) -> Box<dyn FitnessFunction + Send> {
            Box::new(Self {
                sum: self.sum,
                last_position: self.last_position,
            })
        }
    }

    let fitness: Box<dyn FitnessFunction + Send> = Box::new(FitnessFunctionAccumulateZ {
        sum: 0.0,
        last_position: Vec3::zero(),
    });

    fitness
}
