//! Calcutes the direction the mouse is pointing at analogous to the camera, and determines the point on a plane clicked on by the mouse 
//! 

use cgmath::Zero;
use wgpu_renderer::wgpu_renderer::camera::Camera;

use crate::editor::{Quat, Vec3};

pub struct MouseSelector
{
    width: u32,
    height: u32,
    fovy_half_tan: f32,
    plane_position: Vec3,
    plane_rotation: Quat,
}

impl MouseSelector
{
    pub fn new(width: u32, height: u32, fovy_half_tan: f32, plane_position: Vec3, plane_rotation: Quat) -> Self
    {
        Self { width, height, fovy_half_tan, plane_position, plane_rotation}
    }

    pub fn resize(&mut self, new_width: u32, new_height: u32) {
        self.width = new_width;
        self.height = new_height;
    }
   
    // Calculates the intersection of the ray cast by the mouse with global plane z=0
    pub(crate) fn calculate_position(&self, mouse_pos: cgmath::Vector2<f32>, camera: &Camera) -> Option<Vec3> {
        let mut direction:Vec3 = Vec3::zero();

        // Centering (0 is the bottom left corner of the screen)
        direction.x = mouse_pos.x - self.width as f32 / 2.;
        direction.y = mouse_pos.y - self.height as f32 / 2.;

        // Norming and switching coordinates to global rhs coordinate system (z is vertical)
        let display_scale = self.fovy_half_tan * 2. / (self.height as f32);
        direction.z = direction.y * display_scale;
        direction.y = - direction.x * display_scale;
        direction.x = 1.;

        let cam_pos = Vec3::new(camera.position.x, camera.position.y, camera.position.z);
        let cam_rotation = cgmath::Matrix3::from_angle_z(camera.yaw) * cgmath::Matrix3::from_angle_y(-camera.pitch);
        direction = cam_rotation * direction;

        // You can select points with distance 1000 * camera_y from your position
        let relative_range = 1000.;
        if direction.z.abs() < 1. / relative_range {
            return Option::None;
        }

        let lambda = -cam_pos.z / direction.z;

        // Looking away from plane
        if lambda < 0. {
            return Option::None;
        }

        Some(cam_pos + lambda * direction)
    }
}
