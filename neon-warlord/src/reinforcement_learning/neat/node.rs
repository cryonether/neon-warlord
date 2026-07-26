//! Node of an element of the NEAT algorithm

#[derive(Clone, Copy)]
pub struct Node {
    /// Id of the node
    pub id: usize,
    /// Kind of the node
    pub kind: NodeKind,
    /// Value of the node
    pub value: f32,
}

#[derive(Clone, Copy)]
pub enum NodeKind {
    Sensor,
    Output,
    Hidden,
}
