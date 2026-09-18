//! Draws a multi-line graph

use std::collections::VecDeque;

use forward_renderer::{particle_shader_two_point, to_rgb};

use crate::pendulum_simulation::Vec3;

pub struct GraphLines<const SIZE: usize> {
    pub x: VecDeque<f32>,
    pub y: [VecDeque<f32>; SIZE],
}

impl<const SIZE: usize> GraphLines<SIZE> {
    /// Pushes a new X value and removes the oldest one.
    pub fn _x_push_pop(&mut self, val: f32) {
        self.x.pop_front();
        self.x.push_back(val);
    }

    // Pushes a value onto one of the Y lines and removes the oldest value.
    pub fn y_push_pop(&mut self, line: usize, val: f32) {
        if line >= SIZE {
            return;
        }

        self.y[line].pop_front();
        self.y[line].push_back(val);
    }

    /// Pushes a complete sample containing values for every line.
    pub fn _push(&mut self, x: f32, values: [f32; SIZE]) {
        self.x.pop_front();
        self.x.push_back(x);

        for (line, value) in values.into_iter().enumerate() {
            self.y[line].pop_front();
            self.y[line].push_back(value);
        }
    }
}

pub struct GraphLinesDrawer<const SIZE: usize> {
    size: f32,
    colors: [Vec3; SIZE],
    position: Vec3,

    x_lim_start: f32,
    x_lim_end: f32,
    y_lim_start: f32,
    y_lim_end: f32,

    // Grid settings
    grid_color: Vec3,
    grid_spacing: f32,
    grid_extent: f32,
}

impl<const SIZE: usize> GraphLinesDrawer<SIZE> {
    pub fn new(size: f32, position: Vec3) -> Self {
        let default_colors = [
            to_rgb("#c300d9"),
            to_rgb("#00d9c3"),
            to_rgb("#d9c300"),
            to_rgb("#ff4444"),
            to_rgb("#4488ff"),
            to_rgb("#44dd44"),
            to_rgb("#ff8800"),
            to_rgb("#ffffff"),
        ];

        let colors = std::array::from_fn(|i| {
            default_colors[i % default_colors.len()].into()
        });

        Self {
            size,
            colors,
            position,

            x_lim_start: 0.0,
            x_lim_end: 100.0,
            y_lim_start: -1.0,
            y_lim_end: 1.0,

            grid_color: to_rgb("#333333").into(),
            grid_spacing: 2.0,
            grid_extent: 10.0,
        }
    }

    /// Set the color of a particular line.
    pub fn _line_color(mut self, line: usize, color: [f32; 3]) -> Self {
        if line < SIZE {
            self.colors[line] = color.into();
        }

        self
    }

    /// Set all line colors.
    pub fn colors(mut self, colors: [Vec3; SIZE]) -> Self {
        self.colors = colors;
        self
    }

    pub fn _x_lim(mut self, start: f32, end: f32) -> Self {
        self.x_lim_start = start;
        self.x_lim_end = end;
        self
    }

    pub fn y_lim(mut self, y_lim: f32) -> Self {
        self.y_lim_start = -y_lim;
        self.y_lim_end = y_lim;
        self
    }

    pub fn _y_lim_range(mut self, start: f32, end: f32) -> Self {
        self.y_lim_start = start;
        self.y_lim_end = end;
        self
    }

    pub fn _grid_spacing(mut self, spacing: f32) -> Self {
        self.grid_spacing = spacing;
        self
    }

    pub fn _grid_extent(mut self, extent: f32) -> Self {
        self.grid_extent = extent;
        self
    }

    pub fn update(
        &mut self,
        graph: &GraphLines<SIZE>,
        edges: &mut Vec<particle_shader_two_point::Instance>,
    ) {
        self.draw_grid(edges);
        self.draw_graph(graph, edges);
    }

    fn draw_grid(
        &self,
        edges: &mut Vec<particle_shader_two_point::Instance>,
    ) {
        let size = self.size * 0.02;

        let extent = self.grid_extent;
        let spacing = self.grid_spacing;

        let mut value = -extent;

        while value <= extent {
            // Vertical grid line.
            let p0 =
                self.position
                + Vec3::new(value, 0.0, -extent) * self.size;

            let p1 =
                self.position
                + Vec3::new(value, 0.0, extent) * self.size;

            edges.push(particle_shader_two_point::Instance {
                position_0: p0.into(),
                position_1: p1.into(),
                color: self.grid_color.into(),
                time: 1.0,
                size,
            });

            // Horizontal grid line.
            let p0 =
                self.position
                + Vec3::new(-extent, 0.0, value) * self.size;

            let p1 =
                self.position
                + Vec3::new(extent, 0.0, value) * self.size;

            edges.push(particle_shader_two_point::Instance {
                position_0: p0.into(),
                position_1: p1.into(),
                color: self.grid_color.into(),
                time: 1.0,
                size,
            });

            value += spacing;
        }
    }

    fn draw_graph(
        &self,
        graph: &GraphLines<SIZE>,
        edges: &mut Vec<particle_shader_two_point::Instance>,
    ) {
        let size = self.size * 0.05;

        let x_range = self.x_lim_end - self.x_lim_start;
        let y_range = self.y_lim_end - self.y_lim_start;

        if x_range == 0.0 || y_range == 0.0 {
            return;
        }

        // Draw every line independently.
        for line in 0..SIZE {
            let y = &graph.y[line];

            let count = graph.x.len().min(y.len());

            if count < 2 {
                continue;
            }

            for i in 0..count - 1 {
                let x0 = graph.x[i];
                let x1 = graph.x[i + 1];

                let y0 = y[i];
                let y1 = y[i + 1];

                // Normalize X and Y to [0, 1].
                let x0 = (x0 - self.x_lim_start) / x_range;
                let x1 = (x1 - self.x_lim_start) / x_range;

                let y0 = (y0 - self.y_lim_start) / y_range;
                let y1 = (y1 - self.y_lim_start) / y_range;

                let x0 = x0 * self.grid_extent;
                let x1 = x1 * self.grid_extent;

                // Map [0, 1] -> [-grid_extent, grid_extent].
                let x0 = x0 * 2.0 * self.grid_extent - self.grid_extent;
                let x1 = x1 * 2.0 * self.grid_extent - self.grid_extent;

                let y0 = y0 * 2.0 * self.grid_extent - self.grid_extent;
                let y1 = y1 * 2.0 * self.grid_extent - self.grid_extent;

                let p0 =
                    self.position
                    + Vec3::new(x0, 0.0, y0) * self.size;

                let p1 =
                    self.position
                    + Vec3::new(x1, 0.0, y1) * self.size;

                edges.push(particle_shader_two_point::Instance {
                    position_0: p0.into(),
                    position_1: p1.into(),
                    color: self.colors[line].into(),
                    time: 1.0,
                    size,
                });
            }
        }
    }
}
