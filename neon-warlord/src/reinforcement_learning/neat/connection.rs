//! Connection between two elements of the NEAT algorithm

pub struct Edge {
    /// Id of the incoming node
    pub in_id: usize,
    /// Id of the outgoing node
    pub out_id: usize,
    /// Weight of this connection
    pub weight: f32,
    /// If this connection is enabled or not
    pub enabled: bool,
    /// In which innovation cycle this connection was created
    pub innovation: usize,
}

