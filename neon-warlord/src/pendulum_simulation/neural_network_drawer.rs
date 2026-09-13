//! Draws the neural Network

use cgmath::VectorSpace;
use forward_renderer::{particle_shader, particle_shader_two_point, to_rgb};

use crate::{
    pendulum_simulation::Vec3, reinforcement_learning::neural_network_simd::NeuralNetworkSimd,
};

pub struct NeuralNetworkDrawer<const INPUTS: usize, const OUTPUTS: usize, const NR_LAYERS: usize, const RESIDUAL: bool, const NR_NEURONS: usize, const NR_LANES: usize>
{
    size: f32,
    color_negative: Vec3,
    color_zero: Vec3,
    color_positive: Vec3,

    _nr_nodes: usize,

    position: Vec3,
}

const LANES: usize = 16;

impl<const INPUTS: usize, const OUTPUTS: usize, const NR_LAYERS: usize, const RESIDUAL: bool, const NR_NEURONS: usize, const NR_LANES: usize>
    NeuralNetworkDrawer<INPUTS, OUTPUTS, NR_LAYERS, RESIDUAL, NR_NEURONS, NR_LANES>
{
    pub fn new(
        _model: &NeuralNetworkSimd<INPUTS, OUTPUTS, NR_LAYERS, RESIDUAL, NR_NEURONS, NR_LANES>,
        radius: f32,
        position: Vec3,
    ) -> Self {
        let nr_nodes = NR_LAYERS * LANES;

        let color_negative = to_rgb("#0911ff");
        let color_zero = to_rgb("#282428");
        let color_positive = to_rgb("#ff0d0d");

        // let size = radius*2.0;
        let size = radius;

        Self {
            size,
            color_negative: color_negative.into(),
            color_zero: color_zero.into(),
            color_positive: color_positive.into(),
            _nr_nodes: nr_nodes,
            position,
        }
    }

    pub fn update(
        &mut self,
        model: &NeuralNetworkSimd<INPUTS, OUTPUTS, NR_LAYERS, RESIDUAL, NR_NEURONS, NR_LANES>,
        producer_nodes: &mut Vec<particle_shader::Instance>,
        _producer_edges: &mut Vec<particle_shader_two_point::Instance>,
    ) {
        self.update_nodes(model, producer_nodes);
    }

    fn update_nodes(
        &mut self,
        model: &NeuralNetworkSimd<INPUTS, OUTPUTS, NR_LAYERS, RESIDUAL, NR_NEURONS, NR_LANES>,
        producer_nodes: &mut Vec<particle_shader::Instance>,
    ) {
        let w_iter = model.w.iter().chain([&model.w_y]);

        for (k, layer) in w_iter.enumerate() {
            for (j, node) in layer.as_array().iter().enumerate() {
                for (i, &w) in node.iter().enumerate() {
                    let position = Vec3::new(k as f32, j as f32, i as f32);

                    let position = position * self.size + self.position;

                    let color =
                        gradient(w, self.color_negative, self.color_zero, self.color_positive);

                    let time = 1.0;

                    let size = self.size * 0.5;

                    let instance = particle_shader::Instance {
                        position: position.into(),
                        color: color.into(),
                        time,
                        size,
                    };

                    producer_nodes.push(instance);
                }
            }
        }
    }
}

fn gradient(t: f32, negative: Vec3, zero: Vec3, positive: Vec3) -> Vec3 {
    let t = t.clamp(-1.0, 1.0);

    if t < 0.0 {
        // Map [-1, 0] -> [0, 1]
        negative.lerp(zero, t + 1.0)
    } else {
        // Map [0, 1] -> [0, 1]
        zero.lerp(positive, t)
    }
}
