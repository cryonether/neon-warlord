//! Calcutes the direction the mouse is pointing at analogous to the camera, and determines the point on a plane clicked on by the mouse 
//! 

use wgpu_renderer::{vertex_color_shader, wgpu_renderer::camera::Camera};

pub struct MouseSelector
{
    width: u32,
    height: u32,
    fovy_half_tan: f32,
    trans: vertex_color_shader::Instance,
    mouse_pos: glam::Vec3,
}

impl MouseSelector
{
    pub fn new(width: u32, height: u32, fovy_half_tan: f32, trans: vertex_color_shader::Instance) -> Self
    {
        Self { width, height, fovy_half_tan, trans, mouse_pos: glam::Vec3::ZERO, }
    }

    pub fn resize(&mut self, new_width: u32, new_height: u32) {
        self.width = new_width;
        self.height = new_height;
    }

    pub fn calc_mouse_position_on_screen(&mut self, x_pos: f32, y_pos: f32) {
        // Invert y
        self.mouse_pos.y = self.height as f32 - y_pos;

        // Centering
        self.mouse_pos.x = x_pos - self.width as f32 / 2.;
        self.mouse_pos.y -= self.height as f32 / 2.;

        // Norming and switching coordinates to global coordinate system
        self.mouse_pos.z = self.mouse_pos.x * self.fovy_half_tan * 2. / (self.height as f32);
        self.mouse_pos.x = 1.;
        self.mouse_pos.y *= self.fovy_half_tan * 2. / (self.height as f32);
    }

    /// returns the direction of the mouse cursor in the camera coordinate system
    pub fn _get_mouse_position_on_screen(&self) -> glam::Vec3 {
        self.mouse_pos
    }

    pub fn mouse_position_on_grid(&self, camera: &Camera) -> (f32, f32) {
        let pos = camera.position;
        let mut pos2 = glam::Vec3::new(pos.x, pos.y, pos.z);
        let cam_trans = glam::Mat3::from_rotation_y(-camera.yaw.0) * glam::Mat3::from_rotation_z(camera.pitch.0);

        let mut direction = cam_trans * self.mouse_pos;
        pos2 -= self.trans.position;
        pos2 = self.trans.rotation.mul_vec3(pos2); //Validate!!
        direction = self.trans.rotation.mul_vec3(direction); //Validate!!
        let lambda = -pos2.z / direction.z;
        
        (pos2.y + lambda * direction.y, pos2.x + lambda * direction.x)
    }
}