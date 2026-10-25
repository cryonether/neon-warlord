//! Implementation of a hinge joint with xpbd physics

use std::f32::consts::PI;

use cgmath::{InnerSpace, Quaternion, Rad, Rotation, Rotation3, Zero};

use crate::physics::xpbd::{Quad, Vec3, xpbd_angular_constraint::XpbdAngularConstraint, xpbd_positional_constraint::XpbdPositionalConstraint, xpbd_rigid_body::XpbdRigidBody};

#[derive(Clone)]
pub struct XpbdHingeJoint {
    // Attachment point 0 defined relative to the center of mass of body 0
    pub attachment_point_0: Vec3,

    // Attachment point 1 defined relative to the center of mass of body 1
    pub attachment_point_1: Vec3,

    // Perpendicular unit axes defined in relative to the center of mass of body 0
    pub a0: Vec3,
    pub b0: Vec3,
    pub c0: Vec3,

    // Perpendicular unit axes defined in relative to the center of mass of body 1
    pub a1: Vec3,
    pub b1: Vec3,
    pub c1: Vec3,

    // Target angle alpha
    // pub target_angle: f32,


    // Limits the angle between the axes n1 and n2 of two bodies
    // to be in the interval [alpha, beta] using the common rotation axis n.
    pub angle_limit_0: f32,
    pub angle_limit_1: f32, 

    pub positional_constraint: XpbdPositionalConstraint,
    pub angular_constraint: XpbdAngularConstraint, 


    pub external_torque: f32,
}

impl XpbdHingeJoint {
    pub fn new(
        attachment_point_0: Vec3,
        attachment_point_1: Vec3,
        angle_limit_0: f32,
        angle_limit_1: f32,
        rotation_axis_0: RotationAxis,
        rotation_axis_1: RotationAxis,
        positional_compliance: f32,
        angular_compliance: f32,
    ) -> Self {

        let (a0, b0, c0) = rotation_axis_0.local_frame();
        let (a1, b1, c1) = rotation_axis_1.local_frame();

        let positional_constraint = XpbdPositionalConstraint::new(positional_compliance);
        let angular_constraint = XpbdAngularConstraint::new(angular_compliance);

        Self {
            attachment_point_0,
            attachment_point_1,
            a0,
            b0,
            c0,
            a1,
            b1,
            c1,
            angle_limit_0,
            angle_limit_1,
            positional_constraint,
            angular_constraint,
            // external_torque: -2.0,
            external_torque: 0.0,
        }
    }

    pub fn solve(&mut self, obj_0: &mut XpbdRigidBody, obj_1: &mut XpbdRigidBody, dt: f32) {


        let attachment_point_0 = obj_0.local_to_world(&self.attachment_point_0);
        let attachment_point_1 = obj_1.local_to_world(&self.attachment_point_1);
    
        self.positional_constraint.solve(
            obj_0, 
            obj_1,
            &attachment_point_0,
            &attachment_point_1,
            dt, 
        );

        let p0 = &obj_0.p;
        let r0 = &obj_0.r;
        let p1 = &obj_1.p;
        let r1 = &obj_1.r;


        let a0 = r0.rot.rotate_vector(self.a0);
        let a1 = r1.rot.rotate_vector(self.a1);

        let b0 = r0.rot.rotate_vector(self.b0);
        let b1 = r1.rot.rotate_vector(self.b1);

        let c0 = r0.rot.rotate_vector(self.c0);
        let c1 = r1.rot.rotate_vector(self.c1);
        

        // ##########################################################################
        // Apply external torque
        obj_0.r.external_torque += self.external_torque * a0;
        obj_1.r.external_torque -= self.external_torque * a1;



        // ##########################################################################
        // Test 0: Align both angles

        // let mut q_err = (r0.rot * r1.rot.conjugate()).normalize();

        // // Choose the shortest rotation.
        // if q_err.s < 0.0 {
        //     q_err = -q_err;
        // }

        // let v = Vec3::new(q_err.v.x, q_err.v.y, q_err.v.z);
        // let sin_half = v.magnitude();

        // let (axis, angle) = if sin_half < 1e-6 {
        //     (Vec3::unit_x(), 0.0)
        // } else {
        //     let axis = v / sin_half;
        //     let angle = 2.0 * sin_half.atan2(q_err.s);
        //     (axis, angle)
        // };

        // self.angular_constraint.solve(
        //     obj_0, 
        //     obj_1, 
        //     &axis,
        //     angle,
        //     dt
        // );

        // ##########################################################################
        // Test 1: Free rotation around the y axis

        // let cross = a0.cross(a1);
        // let sin_theta = cross.magnitude();
        // let cos_theta = a0.dot(a1).clamp(-1.0, 1.0);

        // let (axis, angle) = if sin_theta > 1e-6 {
        //     (
        //         cross / sin_theta,
        //         sin_theta.atan2(cos_theta),
        //     )
        // } else {
        //     (Vec3::unit_x(), 0.0)
        // };

        // self.angular_constraint.solve(
        //     obj_0, 
        //     obj_1, 
        //     &axis,
        //     angle,
        //     dt
        // );

        // ##########################################################################
        // Test 2: Drive a hinge joint towards a specified target angle alpha

        // let alpha = 0.5 * PI;

        // // Rotate body 0's reference vector around its hinge axis.
        // let rotation = Quad::from_axis_angle(a0.normalize(), Rad(alpha));
        // let b_target = rotation.rotate_vector(b0);

        // // Angular correction from the paper, Equation (21).
        // let angular_error = b_target.cross(b1);

        // let angle = angular_error.magnitude();
        // let axis = if angle > 1e-6 {
        //     angular_error / angle
        // } else {
        //     Vec3::unit_x()
        // };

        // self.angular_constraint.solve(
        //     obj_0, 
        //     obj_1, 
        //     &axis,
        //     angle,
        //     dt
        // );

        // ##########################################################################
        // Test 3: Limit the target angle between alpha and beta
        
        let alpha = self.angle_limit_0;
        let beta = self.angle_limit_1;

        let n = a0;
        let n0 = b0;
        let n1 = b1;

        let sin_phi = (n0.cross(n1)).dot(n);
        let cos_phi = n0.dot(n1);
        let phi = sin_phi.atan2(cos_phi);

        let phi_target = phi.clamp(alpha, beta);

        let rotation = Quad::from_axis_angle(n, Rad(phi_target));
        let n_target = rotation.rotate_vector(-n0);

        // Angular correction from the paper, Equation (21).
        let angular_error = n_target.cross(n1);

        let angle = angular_error.magnitude();
        let axis = if angle > 1e-6 {
            angular_error / angle
        } else {
            // Vec3::unit_y()
            n
        };
        
        // println!("alpha: {}, beta: {}, phi_target: {}, phi: {}",
        //     alpha,
        //     beta,
        //     phi_target,
        //     phi,
        // );

        self.angular_constraint.solve(
            obj_0, 
            obj_1, 
            &axis,
            angle,
            dt
        );

        

    }

    pub fn reset_lagrange_multiplier(&mut self) {
        self.positional_constraint.reset_lagrange_multiplier();
        self.angular_constraint.reset_lagrange_multiplier();
    }

    // /// Approximate angular error vector between two unit vectors.
    // /// Returns a0 × a1.
    // fn angular_error_approximate(a0: Vec3, a1: Vec3) -> Vec3 {
    //     a0.cross(a1)
    // }

    // /// Computes the actual unsigned angle between two vectors in radians.
    // fn angular_error(a0: Vec3, a1: Vec3) -> Vec3 {
    //     let cross = a0.cross(a1);
    //     let sin_theta = cross.magnitude();
    //     let cos_theta = a0.dot(a1);

    //     if sin_theta < 1e-6 {
    //         return Vec3::new(0.0, 0.0, 0.0);
    //     }

    //     let theta = sin_theta.atan2(cos_theta);
    //     cross * (theta / sin_theta)
    // }

}


pub enum RotationAxis {
    X,
    Y, 
    Z,
}

impl RotationAxis {
    fn local_frame(self) -> (Vec3, Vec3, Vec3) {
        match self {
            RotationAxis::X => (
                Vec3::unit_x(),
                Vec3::unit_y(),
                Vec3::unit_z(),
            ),
            RotationAxis::Y => (
                Vec3::unit_y(),
                Vec3::unit_z(),
                Vec3::unit_x(),
            ),
            RotationAxis::Z => (
                Vec3::unit_z(),
                Vec3::unit_x(),
                Vec3::unit_y(),
            ),
        }
    }
}