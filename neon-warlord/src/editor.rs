//! Edit physic based actors
//!
pub mod mouse_selector;

use cgmath::Zero;
use forward_renderer::to_rgb;
use wgpu_renderer::wgpu_renderer::camera::{Camera, Projection};

use crate::{
    editor::mouse_selector::MouseSelector,
    physics::xpbd::{Xpbd, xpbd_drawer::XpbdDrawer},
};

type Vec3 = cgmath::Vector3<f32>;
type Vec2 = cgmath::Vector2<f32>;

const N: usize = 10;

/// Edit physic based actors
pub struct Editor {
    physics: Xpbd,
    mouse_selector: MouseSelector,
    drawer: XpbdDrawer,
    // cursor_index: usize,
    radius: f32,

    // Local variables
    mouse_pos: Vec2,
    _grid: Box<[[[Field; N]; N]; N]>,

    cursor_pos: Vec3,
}

impl Editor {
    pub fn new(width: u32, height: u32) -> Self {
        // Physics
        let physics = Xpbd::new();

        // Physics Drawer
        let radius = 0.05;
        let position = Vec3::zero();
        let drawer = XpbdDrawer::new(&physics, radius, position);

        // Mouse selector
        let mouse_selector =
            MouseSelector::new(width, height);

        let _grid = Box::new([[[Field::None; N]; N]; N]);

        // Add objects
        // let cursor_index = Self::create_objects(&mut physics, radius);

        Self {
            physics,
            mouse_selector,
            drawer,

            // cursor_index,
            radius,

            mouse_pos: Vec2::zero(),
            _grid,
            cursor_pos: Vec3::zero(),
        }
    }

    pub fn resize(&mut self, with: u32, height: u32) {
        self.mouse_selector.resize(with, height);
    }

    pub fn mouse_event(&mut self, mouse_pos: Vec2) {
        self.mouse_pos = mouse_pos
    }

    pub fn _key_event(&mut self) {}

    pub fn update(&mut self, 
        camera: &Camera,
        projection: &Projection,
    ) {
        let dt = 1.0 / 60.0;

        let mouse_pos = self.mouse_pos;

        let cursor_pos = self.mouse_selector.calculate_position(mouse_pos, camera, projection);
        if let Some(cursor_pos) = cursor_pos {
            self.cursor_pos = cursor_pos + Vec3::new(0.0, 0.0, 0.0);
        }

        self.physics.step(dt);
    }

    pub fn draw(
        &self,
        nodes: &mut Vec<forward_renderer::particle_shader::Instance>,
        edges: &mut Vec<forward_renderer::particle_shader_two_point::Instance>,
    ) {
        self.drawer.update(&self.physics, nodes, edges);

        let pos_color: Vec3 = to_rgb("#32ddff").into();
        nodes.push(forward_renderer::particle_shader::Instance{
            position: self.cursor_pos.into(),
            color: pos_color.into(),
            time: 1.0,
            size: self.radius * 2.0,
        });
    }

    // fn create_objects(physics: &mut Xpbd, radius: f32) -> usize {
    //     const MASS: f32 = 0.1;
    //     let n = 10;
    //     let d = 1.0;

    //     // for y in 0..n {
    //     //     for x in 0..n {
    //     //         let elem = XpbdRigidBody::new(
    //     //             Vec3::new(
    //     //                 x as f32 * d - (n / 2) as f32,
    //     //                 y as f32 * d - (n / 2) as f32,
    //     //                 1.5
    //     //             ),
    //     //             &[Vec3::zero(), Vec3::new(-0.5, 0.0, 0.25)],
    //     //             MASS,
    //     //             radius,
    //     //         );

    //     //         physics.add_rigid_body(&elem);
    //     //     }
    //     // }

    //     // let cursor = XpbdRigidBody::new(Vec3::new(0.0, 0.0, radius), &[Vec3::zero()], 0.0, radius);

    //     // physics.add_rigid_body(&cursor)
    //     0
    // }
}

#[derive(Clone, Copy)]
enum Field {
    None,
    _Node,
}
