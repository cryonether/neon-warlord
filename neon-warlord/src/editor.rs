//! Edit physic based actors
//! 
pub mod mouse_selector;

use cgmath::{One, Zero};

use crate::{
    physics::xpbd::{Xpbd, xpbd_drawer::XpbdDrawer, xpbd_rigid_body::XpbdRigidBody},
};

type Vec3 = cgmath::Vector3<f32>;
type Vec2 = cgmath::Vector2<f32>;
type Quat = cgmath::Quaternion<f32>;

/// Edit physic based actors
pub struct Editor {
    physics: Xpbd,
    // mouse_selector: MouseSelector,

    drawer: XpbdDrawer,

    cursor_index: usize,

    // Local variables
    mouse_pos: Vec2,
}

impl Editor {
    pub fn new(
        width: u32,
        height: u32,
        fovy_half_tan: f32,

    ) -> Self {

        // Physics
        let mut physics = Xpbd::new();

        // Physics Drawer
        let radius = 0.1;
        let position = Vec3::zero();
        let drawer = XpbdDrawer::new(&physics, radius, position);
        
        // Mouse selector
        let plane_position: Vec3 = Vec3::zero();
        let plane_rotation: Quat = Quat::one();
        // let mouse_selector = MouseSelector::new(
        //     width,
        //     height,
        //     fovy_half_tan,
        //     plane_position,
        //     plane_rotation,
        // );

        // Add objects
        let cursor_index = Self::create_objects(&mut physics, radius);

        Self {
            physics,
            // mouse_selector,
            drawer,
          
            cursor_index,

            mouse_pos: Vec2::zero(),
        }
    }

    pub fn resize(&mut self, with: u32, height: u32) {
        // self.mouse_selector.resize(with, height);
    }

    pub fn mouse_event(&mut self, mouse_pos: Vec2) {
        self.mouse_pos = mouse_pos
    }

    pub fn key_event(&mut self) {

    }

    pub fn update(&mut self) {
        let dt = 1.0 / 60.0;

        let mouse_pos = self.mouse_pos;
        
        // self.mouse_selector.calculate_position(self.mouse_pos);

        let cursor = &mut self.physics.rigid_bodies[self.cursor_index];
        cursor.p.pos = Vec3::new(mouse_pos.x * 0.01, mouse_pos.y * 0.01, 2.0);

        self.physics.step(dt * 0.1);
    }

    pub fn draw(&self,
        nodes: &mut Vec<forward_renderer::particle_shader::Instance>,
        edges: &mut Vec<forward_renderer::particle_shader_two_point::Instance>,
    ) {
        self.drawer.update(&self.physics, nodes, edges);
    }

    fn create_objects(physics: &mut Xpbd, radius: f32) -> usize {
        const MASS: f32 = 0.1;
        let n = 10;
        let d = 1.0;
        
        for y in 0..n {
            for x in 0..n {
                let elem = XpbdRigidBody::new(
                    Vec3::new(
                        x as f32 * d - (n / 2) as f32, 
                        y as f32 * d - (n / 2) as f32, 
                        1.5
                    ),
                    &[Vec3::zero(), Vec3::new(-0.5, 0.0, 0.25)],
                    MASS,
                    radius,
                );

                physics.add_rigid_body(&elem);
            }
        }

        let cursor = XpbdRigidBody::new(
            Vec3::new(
                0.0, 
                0.0, 
                radius
            ),
            &[Vec3::zero()],
            MASS,
            radius,
        );

        physics.add_rigid_body(&cursor)
    }
}
