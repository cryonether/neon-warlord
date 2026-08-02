//! Definitions for a advanced composition

type Vec3 = cgmath::Vector3<f32>;

#[derive(Clone)]
pub struct Node {
    pub id: usize,
    pub kind: NodeKind,
    pub edge: EdgeKind,
}

impl Node {
    fn f(mut self, n:usize) -> Self {
        self.edge = EdgeKind::Fixed(n);
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
    LinearMotor(usize, usize),
    // Neural Network
    NeuralNetwork,

}

#[derive(Clone)]
pub enum EdgeKind {
    // None
    None,
    // Fixed
    Fixed(usize)
}

const Z: Node = Node {
    id: 0,
    kind: NodeKind::None,
    edge: EdgeKind::None,
};

fn a(n:usize) -> Node {
    Node {
        id: n,
        kind: NodeKind::Regular,
        edge: EdgeKind::None,
    }    
}

fn l(n:usize, a:usize, b:usize) -> Node {
    Node {
        id: n,
        kind:  NodeKind::LinearMotor(a, b),
        edge: EdgeKind::None,
    }    
}

fn f(n:usize) -> Node {
    Node {
        id: n,
        kind: NodeKind::Static,
        edge: EdgeKind::None,
    }    
}

pub struct LocatedNode {
    pub node: Node,
    pub pos: Vec3,
}

fn parse_definition<const NR_SLICES: usize, const R: usize, const C: usize>(
    layers: &[[[Node; C]; R]; NR_SLICES],
    pos: Vec3,
    scale: f32,
) -> Vec<LocatedNode> {
    let mut res: Vec<LocatedNode> = Vec::new();

    // add elements
    for nr_slice in 0..NR_SLICES {
        for r in 0..R {
            for c in 0..C {
                let node = layers[nr_slice][r][c].clone();

                let pos = Vec3::new(nr_slice as f32, r as f32, c as f32);

                let agent_node = LocatedNode{
                    node,
                    pos,
                };

                res.push(agent_node);
            }
        }
    }

    res.sort_by_key(|elem| elem.node.id);

    // change coordinate system
    let origin = Vec3::new(
        res[0].pos.y,
        res[0].pos.x,
        res[0].pos.z,
    );

    for elem in &mut res {
        let local_pos = Vec3::new(
                elem.pos.y - origin.x,
                -(elem.pos.x - origin.y),
                elem.pos.z - origin.z,
            );

        elem.pos = local_pos;
    }

    // move and scale elements
    for elem in &mut res {
        elem.pos = elem.pos * scale + pos;
    }

    res
}


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

// #[rustfmt::skip]
// pub fn get_pendulum_definition() -> [[[&'static str; 9]; 9]; 3] {

//     // L(a,b)  ... Linear Motor (left_range, right_range)
//     // R(a,b,c)... Rotator(target, axis0, axis1)
//     // N       ... Neural Network
//     // F       ... Fixed
//     // A       ... Regular Node

//     let layer_0 = [
//         ["     ", "     ", "     ", "     ", "     ", "     ", "     ", "     ", "     "],
//         ["     ", "     ", "     ", "     ", "     ", "     ", "     ", "     ", "     "],
//         ["     ", "     ", "     ", "     ", "     ", "     ", "     ", "     ", "     "],
//         ["     ", "     ", "     ", "     ", "     ", "     ", "     ", "     ", "     "],
//         ["     ", "     ", "     ", "     ", "     ", "     ", "     ", "     ", "     "],
//         ["     ", "     ", "     ", "     ", "     ", "     ", "     ", "     ", "     "],
//         ["     ", "     ", "     ", "     ", "     ", "     ", "     ", "     ", "     "],
//         ["     ", "     ", "     ", "     ", "     ", "     ", "     ", "     ", "     "],
//         ["     ", "     ", "     ", "     ", "     ", "     ", "     ", "     ", "     "],
//     ];

//     // let layer_1 = [
//     //     ["     ", "     ", "     ", "     ", "4          ", "     ", "     ", "     ", "     "],
//     //     ["     ", "     ", "     ", "     ", "           ", "     ", "     ", "     ", "     "],
//     //     ["     ", "     ", "     ", "     ", "           ", "     ", "     ", "     ", "     "],
//     //     ["     ", "     ", "     ", "     ", "R3+4+0+2-F0", "     ", "     ", "     ", "     "],
//     //     ["F1   ", "     ", "     ", "     ", "L0+1+2     ", "     ", "     ", "     ", "F2   "],
//     //     ["N5-F1", "     ", "     ", "     ", "           ", "     ", "     ", "     ", "     "],
//     //     ["     ", "     ", "     ", "     ", "           ", "     ", "     ", "     ", "     "],
//     //     ["     ", "     ", "     ", "     ", "           ", "     ", "     ", "     ", "     "],
//     //     ["     ", "     ", "     ", "     ", "           ", "     ", "     ", "     ", "     "],
//     // ];


//     let layer_1 = [
//         ["     ", "     ", "     ", "     ", "           ", "     ", "     ", "     ", "     "],
//         ["     ", "     ", "     ", "     ", "           ", "     ", "     ", "     ", "     "],
//         ["     ", "     ", "     ", "     ", "           ", "     ", "     ", "     ", "     "],
//         ["     ", "     ", "     ", "     ", "           ", "     ", "     ", "     ", "     "],
//         ["F1   ", "     ", "     ", "     ", "L0(1,2)    ", "     ", "     ", "     ", "F2   "],
//         ["     ", "     ", "     ", "     ", "           ", "     ", "     ", "     ", "     "],
//         ["     ", "     ", "     ", "     ", "           ", "     ", "     ", "     ", "     "],
//         ["     ", "     ", "     ", "     ", "           ", "     ", "     ", "     ", "     "],
//         ["     ", "     ", "     ", "     ", "           ", "     ", "     ", "     ", "     "],
//     ];


//     let layer_2 = [
//         ["     ", "     ", "     ", "     ", "     ", "     ", "     ", "     ", "     "],
//         ["     ", "     ", "     ", "     ", "     ", "     ", "     ", "     ", "     "],
//         ["     ", "     ", "     ", "     ", "     ", "     ", "     ", "     ", "     "],
//         ["     ", "     ", "     ", "     ", "     ", "     ", "     ", "     ", "     "],
//         ["     ", "     ", "     ", "     ", "     ", "     ", "     ", "     ", "     "],
//         ["     ", "     ", "     ", "     ", "     ", "     ", "     ", "     ", "     "],
//         ["     ", "     ", "     ", "     ", "     ", "     ", "     ", "     ", "     "],
//         ["     ", "     ", "     ", "     ", "     ", "     ", "     ", "     ", "     "],
//         ["     ", "     ", "     ", "     ", "     ", "     ", "     ", "     ", "     "],
//     ];

//     [layer_0, layer_1, layer_2]
// }




#[rustfmt::skip]
pub fn get_pendulum_definition2() -> [[[Node; 9]; 9]; 1] {

    // L(a,b)  ... Linear Motor (left_range, right_range)
    // R(a,b,c)... Rotator(target, axis0, axis1)
    // N       ... Neural Network
    // F       ... Fixed
    // A       ... Regular Node

    let layer_0 = [
        [Z          ,        Z          , Z          , Z          , Z          ,                  Z          , Z          , Z          , Z  ],
        [Z          ,        Z          , Z          , Z          , Z          ,                  Z          , Z          , Z          , Z  ],
        [Z          ,        Z          , Z          , Z          , Z          ,                  Z          , Z          , Z          , Z  ],
        [a(3).f(1),Z          , Z          , Z          , Z          ,                  Z          , Z          , Z          , Z  ],
        [f(1),     Z          , Z          , Z          , l(0, 1, 2), Z          , Z          , Z          , f(2)],
        [Z          ,        Z          , Z          , Z          , Z          ,                  Z          , Z          , Z          , Z  ],
        [Z          ,        Z          , Z          , Z          , Z          ,                  Z          , Z          , Z          , Z  ],
        [Z          ,        Z          , Z          , Z          , Z          ,                  Z          , Z          , Z          , Z  ],
        [Z          ,        Z          , Z          , Z          , Z          ,                  Z          , Z          , Z          , Z  ],
    ];

    // let layer_1 = [
    //     ["     ", "     ", "     ", "     ", "4          ", "     ", "     ", "     ", "     "],
    //     ["     ", "     ", "     ", "     ", "           ", "     ", "     ", "     ", "     "],
    //     ["     ", "     ", "     ", "     ", "           ", "     ", "     ", "     ", "     "],
    //     ["     ", "     ", "     ", "     ", "R3+4+0+2-F0", "     ", "     ", "     ", "     "],
    //     ["F1   ", "     ", "     ", "     ", "L0+1+2     ", "     ", "     ", "     ", "F2   "],
    //     ["N5-F1", "     ", "     ", "     ", "           ", "     ", "     ", "     ", "     "],
    //     ["     ", "     ", "     ", "     ", "           ", "     ", "     ", "     ", "     "],
    //     ["     ", "     ", "     ", "     ", "           ", "     ", "     ", "     ", "     "],
    //     ["     ", "     ", "     ", "     ", "           ", "     ", "     ", "     ", "     "],
    // ];

    [layer_0]
}
