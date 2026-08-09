//! Creates vertex data to draw a circle
//!

use wgpu_renderer::vertex_color_shader::Color;
use wgpu_renderer::vertex_color_shader::Vertex;

type Vec3 = cgmath::Vector3<f32>;

pub struct Lines {
    pub vertices: Vec<Vertex>,
    pub colors: Vec<Color>,
    pub indices: Vec<u32>,

    size: usize,
}

impl Lines {
    pub fn new(n: usize) -> Self {
        Self::new_color_fade(n, [1.0, 0.0, 1.0], [0.0, 1.0, 1.0])
    }

    pub fn new_color_fade(n: usize, color0: [f32; 3], color1: [f32; 3]) -> Self {
        let mut vertices = Vec::<Vertex>::new();
        let mut colors = Vec::<Color>::new();
        let mut indices = Vec::<u32>::new();

        let color0 = cgmath::Vector3::from(color0);
        let color1 = cgmath::Vector3::from(color1);

        for i in 0..n * 2 {
            let x = 0.0;
            let _y = 0.0;
            let z = 0.0;

            let dist_y = 1.0 / (n * 2) as f32 * i as f32;
            let y = i as f32;

            let color = color0 * dist_y + color1 * (1.0 - dist_y);

            vertices.push(Vertex {
                position: [x, y, z],
            });

            colors.push(Color {
                color: color.into(),
            });

            indices.push(i as u32);
        }

        Self {
            vertices,
            colors,
            indices,
            size: n,
        }
    }

    pub fn len(&self) -> usize {
        self.size
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn set_line_position(&mut self, index: usize, pos_0: Vec3, pos_1: Vec3) {
        if index * 2 >= self.vertices.len() {
            return;
        }

        self.vertices[index * 2].position = pos_0.into();
        self.vertices[index * 2 + 1].position = pos_1.into();
    }
}
