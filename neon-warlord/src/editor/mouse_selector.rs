//! Calcutes the direction the mouse is pointing at analogous to the camera, and determines the point on a plane clicked on by the mouse
//!

use std::f32::consts::FRAC_PI_2;

use cgmath::{InnerSpace, Rad, SquareMatrix, Zero};
use wgpu_renderer::wgpu_renderer::camera::{Camera, Projection};

use crate::editor::{Quat, Vec3};
type Vec4 = cgmath::Vector4<f32>;

pub struct MouseSelector {
    width: u32,
    height: u32,
    fovy_half_tan: f32,
    plane_position: Vec3,
    plane_rotation: Quat,

    pub n: Vec3,
    pub p: Vec3,
}

impl MouseSelector {
    pub fn new(
        width: u32,
        height: u32,
        fovy: f32,
        plane_position: Vec3,
        plane_rotation: Quat,
    ) -> Self {
        // Somehow this needs an factor 1.5 to work, we dont know why.
        // (fovy/2.).tan() is the offset of the vertical borders from the center, on the plane with distance 1.
        Self {
            width,
            height,
            fovy_half_tan: (fovy / 2.).tan() * 1.5,
            plane_position,
            plane_rotation,
            n: Vec3::zero(),
            p: Vec3::zero(),
        }
    }

    pub fn resize(&mut self, new_width: u32, new_height: u32) {
        self.width = new_width;
        self.height = new_height;
    }

    // Calculates the intersection of the ray cast by the mouse with global plane z=0
    pub(crate) fn calculate_position(
        &mut self,
        mouse_pos: cgmath::Vector2<f32>,
        camera: &Camera,
        projection: &Projection,
    ) -> Option<Vec3> {

        // Centering (0 is the bottom left corner of the screen)
        let ndc_x = (mouse_pos.x / self.width as f32) * 2.0 - 1.0;
        let ndc_y = (mouse_pos.y / self.height as f32) * 2.0 - 1.0;

        let near = Vec4::new(ndc_x, ndc_y, 0.0, 1.0);
        let far = Vec4::new(ndc_x, ndc_y, 1.0, 1.0);

        let inverse_view_projection = (projection.calc_matrix() * camera.calc_matrix()).invert();
        if inverse_view_projection.is_none() {
            return None;
        }

        let inverse_view_projection = inverse_view_projection.unwrap();
        let near_world = inverse_view_projection * near;
        let far_world = inverse_view_projection * far;

        let near_world = near_world / near_world.w;
        let far_world = far_world / far_world.w;

        self.p = near_world.truncate();
        // self.n = (far_world.truncate() -  near_world.truncate()).normalize();    // somehow points in the opposite direction
        self.n = (near_world.truncate() - far_world.truncate()).normalize();

        Self::intersect_plane_z0(self.p, self.n)
    }

    /// Intersects a point with direction with the z-plane
    fn intersect_plane_z0(
        point: Vec3,
        direction: Vec3,
    ) -> Option<Vec3> {
        // Select points with distance 1000 * camera_y from the current position
        let relative_range = 1000.0;

        if direction.z.abs() < 1.0 / relative_range {
            return None;
        }

        let lambda = -point.z / direction.z;

        if lambda < 0.0 {
            return None;
        }

        Some(point + direction * lambda)
    }
}

