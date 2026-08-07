//! Connection between two elements of the NEAT algorithm

#[derive(Clone)]
pub struct Edge {
    // topologically fixed
    /// Id of the incoming node
    pub _id_from: usize,
    /// Id of the outgoing node
    pub _id_to: usize,
    /// Weight of this connection
    pub weight: f32,
    /// If this connection is enabled or not
    pub enabled: bool,
    /// In which innovation cycle this connection was created
    pub _innovation: usize,

    // variable
    /// Index in the edges array
    pub index_from: usize,
    /// Index in the edges array
    pub index_to: usize,
}
