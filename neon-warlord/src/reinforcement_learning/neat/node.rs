//! Node of an element of the NEAT algorithm

#[derive(Clone)]
pub struct Node {
    // topologically fixed

    /// Id of the node
    pub id: usize,
    /// Kind of the node
    pub kind: NodeKind,
    // current layer of the node
    pub layer: usize,

    // variable

    /// Value of the node
    pub value: f32,
    pub bias: f32,
}

#[derive(Clone)]
pub enum NodeKind {
    Sensor,
    Output,
    Hidden,
}
