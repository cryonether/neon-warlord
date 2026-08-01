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

    // Bias for the activation function
    pub bias: f32,
    /// Value of the node
    pub value: f32,
}
#[derive(Clone, PartialEq)]
pub enum NodeKind {
    Sensor,
    Output,
    Hidden,
}
