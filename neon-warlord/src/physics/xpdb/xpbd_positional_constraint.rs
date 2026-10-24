//! A positional constraint of two rigid bodies

use cgmath::{InnerSpace, Rotation};

use crate::physics::xpdb::{Mat3, Quad, Vec3, xpbd_rigid_body::XpbdRigidBody};

#[derive(Clone)]
pub struct XpbdPositionalConstraint {

    // Point 0 defined relative to the center of mass
    pub relative_position_0: Vec3,

    // Point 1 defined relative to the center of mass
    pub relative_position_1: Vec3,

    // The compliance of the constraint
    // compliance = 0 simulates infinite stiffness
    pub compliance: f32,

    // Lagrange multiplier in m/N
    pub lagrange_multiplier: f32,
}

impl XpbdPositionalConstraint {
    pub fn new(relative_position_0: Vec3, relative_position_1: Vec3, compliance: f32) -> Self {
        
        Self {
            relative_position_0,
            relative_position_1,
            compliance,
            lagrange_multiplier: 0.0,
        }
    }


    pub fn solve(&mut self, obj_0: &mut XpbdRigidBody, obj_1: &mut XpbdRigidBody, dt: f32){
        let p0 = &mut obj_0.p;
        let r0 = &mut obj_0.r;
        let p1 = &mut obj_1.p;
        let r1 = &mut obj_1.r;

        // Calculate the world space anchor points
        let anchor_0 = p0.pos
            + r0.rot.rotate_vector(self.relative_position_0);

        let anchor_1 = p1.pos
            + r1.rot.rotate_vector(self.relative_position_1);

        let r0_world = anchor_0 - p0.pos;
        let r1_world = anchor_1 - p1.pos;


        let dx = anchor_0 - anchor_1;


        // To apply a positional correction Δx at positions r1 and r2, we first
        // split it into its direction n and its magnitude n.
        let magnitude= dx.magnitude();
        if magnitude < 1e-6 {
            // Both objects occupies the same space
            return;
        }

        let direction = dx / magnitude;

        //
        // Calculates the generalized inverse mass of a rigid body for a positional
        // constraint, accounting for both translational and rotational motion.
        //
        // ### Mathematical Formula:
        //
        // ```text
        // w₁ = 1/m₁ + (r₁ × n)ᵀ I₁⁻¹ (r₁ × n)
        // ```
        //
        // Where:
        // - w₁ = Generalized inverse mass of body 1
        // - m₁ = Mass of body 1
        // - r₁ = Lever arm from the center of mass to the constraint point
        // - n = Normalized direction of the positional correction Δx
        // - I₁⁻¹ = Inverse inertia tensor of body 1
        // - × = Cross product
        // - ᵀ = Transpose
        //
        // The first term accounts for translational motion, while the second
        // accounts for rotational motion around the center of mass.
        //
        let rn0 = r0_world.cross(direction);
        let rn1 = r1_world.cross(direction);

        let i_rn0 = apply_inverse_inertia_world(
            &r0.rot, 
            &r0.inv_rot, 
            &r0.inverse_inertia_tensor, 
            &rn0
        );
        let i_rn1 = apply_inverse_inertia_world(
            &r1.rot, 
            &r1.inv_rot, 
            &r1.inverse_inertia_tensor, 
            &rn1
        );
        
        let w0 = p0.inv_mass + rn0.dot(i_rn0);
        let w1 = p1.inv_mass + rn1.dot(i_rn1);

        //
        // Calculates the Lagrange multiplier correction for a compliant positional
        // constraint using Extended Position Based Dynamics (XPBD).
        //
        // ### Mathematical Formula:
        //
        // ```text
        //           -c - α̃ λ
        // Δλ = ─────────────────────────
        //          w₁ + w₂ + α̃
        //
        //          α
        // α̃ = ───────
        //       h²
        // ```
        //
        // Where:
        // - Δλ = Change in the Lagrange multiplier
        // - c = Constraint error (distance error for a positional constraint)
        // - λ = Accumulated Lagrange multiplier
        // - α̃ (alpha tilde) = Scaled compliance
        // - α = Compliance of the constraint (inverse stiffness)
        // - h = Simulation substep duration
        // - w₁ = Generalized inverse mass of body 1
        // - w₂ = Generalized inverse mass of body 2
        //
        // The compliance is scaled by the square of the substep duration to make
        // the constraint behavior consistent across different time steps.
        // For a rigid constraint, set α = 0.
        //
        let alpha_tilde = self.compliance / (dt * dt);
        let delta_lagrange_multiplier = (-magnitude - alpha_tilde * self.lagrange_multiplier) / 
                                                        (w0 + w1 + alpha_tilde);
        self.lagrange_multiplier += delta_lagrange_multiplier;

        // Compute the positional impulse
        let p = delta_lagrange_multiplier * direction;

        // Update the states of the bodies

        //
        // Calculates the positional correction for a rigid body by applying the
        // Lagrange multiplier correction along the constraint direction.
        //
        // ### Mathematical Formula:
        //
        // ```text
        // Δp₁ = w₁ Δλ n
        // Δp₂ = -w₂ Δλ n
        // ```
        //
        // Where:
        // - Δp₁ = Translational position correction of body 1
        // - Δp₂ = Translational position correction of body 2
        // - w₁ = Generalized inverse mass of body 1
        // - w₂ = Generalized inverse mass of body 2
        // - Δλ = Change in the Lagrange multiplier
        // - n = Normalized constraint direction
        //
        // The positional corrections move the two bodies in opposite directions,
        // weighted by their generalized inverse masses. Bodies with greater
        // inverse mass receive larger corrections.
        //
        p0.pos += p * p0.inv_mass;
        p1.pos -= p * p1.inv_mass;

        //
        // Applies the rotational positional correction to a rigid body's orientation
        // using the inverse inertia tensor and the constraint impulse.
        //
        // ### Mathematical Formula:
        //
        // ```text
        // Δθ₁ = I₁⁻¹ (r₁ × p)
        //
        // q₁' = q₁ + ½ (0, Δθ₁) ⊗ q₁
        //
        // ```
        //
        // Where:
        // - Δθ₁ = Angular positional correction
        // - I₁⁻¹ = Inverse world-space inertia tensor of body 1
        // - r₁ = World-space lever arm from the center of mass to the constraint point
        // - p = Positional constraint impulse, p = Δλ n
        // - q₁ = Current orientation quaternion
        // - q₁' = Corrected orientation quaternion
        // - ⊗ = Quaternion multiplication
        //
        // The corrected quaternion should be normalized after applying the update
        // to maintain a valid unit rotation.
        //
        // Reference: Detailed Rigid Body Simulation with Extended Position Based
        // Dynamics, Section 3.3.1, equation (8).
        // https://matthias-research.github.io/pages/publications/PBDBodies.pdf
        //
        let dq0 = apply_inverse_inertia_world(
            &r0.rot, 
            &r0.inv_rot, 
            &r0.inverse_inertia_tensor, 
            &r0_world.cross(p)
        );
        let dq1 = apply_inverse_inertia_world(
            &r1.rot, 
            &r1.inv_rot, 
            &r1.inverse_inertia_tensor, 
            &r1_world.cross(p)
        );
        
        r0.rot = r0.rot + 0.5 * Quad::new(0.0, dq0.x, dq0.y, dq0.z) * r0.rot;
        r1.rot = r1.rot - 0.5 * Quad::new(0.0, dq1.x, dq1.y, dq1.z) * r1.rot;
    }

    ///
    /// Estimates the force acting along a constraint from the accumulated
    /// Lagrange multiplier after the constraint-solving iterations.
    ///
    /// ### Mathematical Formula:
    ///
    /// ```text
    ///       λ
    /// f = ─────
    ///      h²
    /// ```
    ///
    /// Where:
    /// - f = Estimated constraint force (in newtons)
    /// - λ = Accumulated Lagrange multiplier after solving the constraint
    /// - h = Simulation timestep or substep duration used by the solver
    ///
    /// The force is recovered by dividing the accumulated Lagrange multiplier
    /// by the square of the timestep. This provides an estimate of the force
    /// required to satisfy the constraint.
    ///
    /// Note:
    /// - Use a consistent timestep convention when calculating the force.
    /// - The multiplier should be accumulated during constraint solving.
    ///
    pub fn get_force(&self, dt:f32) -> f32 {
        self.lagrange_multiplier / (dt * dt)
    }

    pub fn reset_lagrange_multiplier(&mut self) {
        self.lagrange_multiplier = 0.0;
    }
}

/// Applies a local space inverse inertia tensor to a world space vector
fn _apply_inverse_inertia_world(
    rotation: Quad,
    inverse_rotation: Quad,
    inverse_inertia_local: Vec3,
    v_world: Vec3,
) -> Vec3 {
    let v_local = inverse_rotation.rotate_vector(v_world);

    // For a diagonal local-space inverse inertia tensor.
    let result_local = Vec3::new(
        inverse_inertia_local.x * v_local.x,
        inverse_inertia_local.y * v_local.y,
        inverse_inertia_local.z * v_local.z,
    );

    rotation.rotate_vector(result_local)
}

fn apply_inverse_inertia_world(
    rotation: &Quad,
    inverse_rotation: &Quad,
    inverse_inertia_local: &Mat3,
    v_world: &Vec3,
) -> Vec3 {
    // Transform the vector into local space.
    let v_local = inverse_rotation.rotate_vector(*v_world);

    // Apply the full local-space inverse inertia tensor.
    let result_local = inverse_inertia_local * v_local;

    // Transform the result back into world space.
    rotation.rotate_vector(result_local)
}