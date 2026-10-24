//! A angular constraint of two rigid bodies

use cgmath::{InnerSpace, Rotation};

use crate::physics::xpdb::{Quad, Vec3, xpbd_rigid_body::XpbdRigidBody};

pub struct XpbdAngularConstraint {

    // Rotation axis defined in local space
    pub rotation_axis: Vec3,

    // magnitude, which is the rotation angle
    pub rotation_angle: f32,

    // The compliance of the constraint
    // compliance = 0 simulates infinite stiffness
    pub compliance: f32,

    // Lagrange multiplier in angular impulse multiplied by time, equivalently torque times time squared.
    pub lagrange_multiplier: f32,

    // Torque applied
    torque: Vec3,
}

impl XpbdAngularConstraint {
    pub fn solve(&mut self, obj_0: &mut XpbdRigidBody, obj_1: &mut XpbdRigidBody, dt: f32) {
        // let p0 = &mut obj_0.p;
        let r0 = &mut obj_0.r;
        // let p1 = &mut obj_1.p;
        let r1 = &mut obj_1.r;

        // If the rotation axis is stored in body 0s local frame
        // let n = r0.rot.rotate_vector(self.rotation_axis).normalize();

        let (n, theta) = Self::calculate_rotation_error(&r0.rot, &r1.rot);

        //
        // Calculates the generalized inverse mass of a rigid body for an angular
        // constraint, accounting for rotational motion around the body's center of mass.
        //
        // ### Mathematical Formula:
        //
        // ```text
        // w₁ = nᵀ I₁⁻¹ n
        // ```
        //
        // Where:
        // - w₁ = Generalized inverse mass of body 1 for the angular constraint
        // - n = Normalized direction of the rotational correction Δq, representing
        //       the rotation axis
        // - I₁⁻¹ = Inverse inertia tensor of body 1
        // - ᵀ = Transpose
        //
        // The inverse inertia tensor determines the body's resistance to rotation
        // about the correction axis. Unlike positional constraints, angular constraints
        // do not involve a translational mass term or a lever arm, since they affect
        // only the body's orientation.
        //
        // Reference: Detailed Rigid Body Simulation with Extended Position Based
        // Dynamics (XPBD), Section 3.3.2, Equation (11).
        //
        let i_n0 = r0.apply_inverse_inertia_world(&n);
        let i_n1 = r1.apply_inverse_inertia_world(&n);
        
        let w0 = n.dot(i_n0);
        let w1 = n.dot(i_n1);


        //
        // Calculates the Lagrange multiplier correction for a compliant angular
        // constraint using Extended Position Based Dynamics (XPBD).
        //
        // ### Mathematical Formula:
        //
        // ```text
        //             -θ - α̃ λ
        // Δλ = ─────────────────────────
        //        w₁ + w₂ + α̃
        //
        //         α
        // α̃ = ───────
        //        h²
        // ```
        //
        // Where:
        // - Δλ = Change in the Lagrange multiplier
        // - θ = Angular constraint error (rotation angle in radians)
        // - λ = Accumulated Lagrange multiplier
        // - α̃ (alpha tilde) = Scaled angular compliance
        // - α = Angular compliance (inverse angular stiffness)
        // - h = Simulation substep duration
        // - w₁ = Generalized inverse mass of body 1 for the angular constraint
        // - w₂ = Generalized inverse mass of body 2 for the angular constraint
        //
        // The compliance is scaled by the square of the substep duration to make
        // the constraint behavior consistent across different time steps.
        // For a rigid angular constraint, set α = 0.
        //
        let alpha_tilde = self.compliance / (dt * dt);
        let delta_lagrange_multiplier = (-theta - alpha_tilde * self.lagrange_multiplier) / 
                                                        (w0 + w1 + alpha_tilde);
        self.lagrange_multiplier += delta_lagrange_multiplier;

        // Compute the angular impulse
        let p = delta_lagrange_multiplier * n;

        //
        // Applies the angular constraint correction to the orientation of body 0
        // using Extended Position Based Dynamics (XPBD).
        //
        // ### Mathematical Formula:
        //
        // ```text
        // q₀ ← q₀ + ½ [I₀⁻¹ p, 0] q₀
        //
        // p = Δλ n
        // ```
        //
        // Where:
        // - q₀ = Orientation quaternion of body 0
        // - Δλ = Change in the Lagrange multiplier
        // - n = World-space rotation axis
        // - p = Angular constraint correction vector
        // - I₀⁻¹ = Inverse inertia tensor of body 0
        // - [I₀⁻¹ p, 0] = Quaternion with I₀⁻¹ p as its vector part
        //   and zero as its scalar part
        let i_p0 = r0.apply_inverse_inertia_world(
            &p
        );
        let i_p1 = r1.apply_inverse_inertia_world(
            &p
        );

        r0.rot = r0.rot + 0.5 * Quad::new(0.0, i_p0.x, i_p0.y, i_p0.z) * r0.rot;
        r1.rot = r1.rot - 0.5 * Quad::new(0.0, i_p1.x, i_p1.y, i_p1.z) * r1.rot;

        self.torque = self.lagrange_multiplier * n / (dt * dt);
    }

    pub fn get_torque(&self) -> cgmath::Vector3<f32> {
        self.torque
    }

    pub fn reset_lagrange_multiplier(&mut self) {
        self.lagrange_multiplier = 0.0;
    }

    pub fn calculate_rotation_error_2(
        rotation_0: &Quad,
        rotation_1: &Quad,
    ) -> (Vec3, f32) {
        let mut q_err = (rotation_0 * rotation_1.conjugate()).normalize();

        // Choose the shortest rotation.
        if q_err.s < 0.0 {
            q_err = -q_err;
        }

        let v = Vec3::new(q_err.v.x, q_err.v.y, q_err.v.z);
        let sin_half = v.magnitude();

        if sin_half < 1e-6 {
            return (Vec3::unit_x(), 0.0);
        }

        let axis = v / sin_half;
        // let angle = 2.0 * sin_half.atan2(q_err.s);

        // Suitable for small angle errors (paper's approximation)
        let angle = 2.0 * sin_half;
                
        (axis, angle)
    }

    pub fn calculate_rotation_error(
        rotation_0: &Quad,
        rotation_1: &Quad,
    ) -> (Vec3, f32) {
        let dq = (*rotation_0 * rotation_1.conjugate()).normalize();

        let sign = if dq.s > 0.0 { -1.0 } else { 1.0 };

        let corr = Vec3::new(dq.v.x, dq.v.y, dq.v.z) * (2.0 * sign);
        let angle = corr.magnitude();

        if angle < 1e-6 {
            return (Vec3::unit_x(), 0.0);
        }

        (corr / angle, angle)
    }
}