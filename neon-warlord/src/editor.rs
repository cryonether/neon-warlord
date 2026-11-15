//! Edit physic based actors
//! 
pub mod mouse_selector;

use cgmath::{One, Zero};

use crate::{
    editor::mouse_selector::MouseSelector,
    physics::xpbd::{Xpbd, xpbd_drawer::XpbdDrawer},
};

type Vec3 = cgmath::Vector3<f32>;
type Vec2 = cgmath::Vector2<f32>;
type Quat = cgmath::Quaternion<f32>;

/// Edit physic based actors
pub struct Editor {
    physics: Xpbd,
    mouse_selector: MouseSelector,

    drawer: XpbdDrawer,

    // Local variables
    mouse_pos: Vec2,
}

impl Editor {
    pub fn new(
        surface_width: u32,
        surface_height: u32,
        fovy_half_tan: f32,

    ) -> Self {

        // Physics
        let physics = Xpbd::new();

        // Physics Drawer
        let radius = 0.1;
        let position = Vec3::zero();
        let drawer = XpbdDrawer::new(&physics, radius, position);
        
        // Mouse selector
        let plane_position: Vec3 = Vec3::zero();
        let plane_rotation: Quat = Quat::one();
        let mouse_selector = MouseSelector::new(
            surface_width,
            surface_height,
            fovy_half_tan,
            plane_position,
            plane_rotation,
        );

        Self {
            physics,
            mouse_selector,
            drawer,
        }
    }

    pub fn resize(&mut self, with: u32, height: u32) {
        self.mouse_selector.resize(with, height);
    }

    pub fn mouse_event(&mut self, mouse_pos: Vec2) {
        self.mouse_pos = mouse_pos
    }

    pub fn key_event(&mut self) {

    }

    pub fn update(&mut self) {
        self.mouse_selector.calculate_position(self.mouse_pos);
    }

    pub fn draw(&self,
        nodes: &mut Vec<forward_renderer::particle_shader::Instance>,
        edges: &mut Vec<forward_renderer::particle_shader_two_point::Instance>,
    ) {
        self.drawer.update(&self.physics, nodes, edges);
    }
}
