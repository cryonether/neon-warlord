//! A ground collision

use cgmath::InnerSpace;

use crate::physics::xpbd::{Quat, Vec3, xpbd_rigid_body::XpbdRigidBody};

// Height of the ground plane
const GROUND_HEIGHT_Z: f32 = 0.0;

#[derive(Clone)]
pub struct XpbdGround {
    pub contact_points: Vec<(usize, ContactPoint)>,
}

impl XpbdGround {
    pub fn new() -> Self {
        Self {
            contact_points: Vec::new(),
        }
    }

    pub fn find_contact_points(&mut self, objs: &[XpbdRigidBody]) {
        self.contact_points.clear();

        // The rigid bodies consist of multiple spherical particles with a defined radius
        let particle_radius = 0.1;

        for (i, obj) in objs.iter().enumerate() {
            for particle in &obj.particles {
                // The position of the particle is relative to the body center of mass

                let particle_local = *particle;
                let pos_world = obj.local_to_world(&particle_local);

                if pos_world.z - particle_radius < GROUND_HEIGHT_Z {
                    // Collision with the ground detected

                    let contact = ContactPoint {
                        particle_local,
                        particle_radius,
                        lagrange_multiplier_n: 0.0,
                        lagrange_multiplier_t: 0.0,
                    };

                    self.contact_points.push((i, contact));
                }
            }
        }
    }

    pub fn solve(&mut self, bodies: &mut [XpbdRigidBody], dt: f32) {
        for (index, contact_point) in &mut self.contact_points {
            let obj = &mut bodies[*index];
            contact_point.solve(obj, dt);
        }
    }

    ///
    /// Handle dynamic friction and restitution
    /// 
    /// Details:
    ///     Here we iterate once through all the contacts and update the new velocities
    /// 
    pub fn solve_velocity_level(&mut self, bodies: &mut [XpbdRigidBody], dt: f32) {
        for (index, contact_point) in &mut self.contact_points {
            let obj = &mut bodies[*index];
            contact_point.solve_velocity_level(obj, dt);
        }
    }
}

#[derive(Clone)]
pub struct ContactPoint {
    pub particle_local: Vec3,
    pub particle_radius: f32,

    // Accumulated XPBD multipliers.
    pub lagrange_multiplier_n: f32,
    pub lagrange_multiplier_t: f32,
}

impl ContactPoint {
    pub fn solve(&mut self, obj: &mut XpbdRigidBody, dt: f32) {
        let p = &obj.p;
        let r = &obj.r;

        let pos_world = r.rot * self.particle_local + p.pos;
        let prev_pos_world = r.prev_rot * self.particle_local + p.prev_pos;
        let normal = Vec3::unit_z();
        let contact_point = pos_world - normal * self.particle_radius;
        let separation = contact_point.z - GROUND_HEIGHT_Z;
        let r_ = contact_point - p.pos;

        let p = &mut obj.p;
        let r = &mut obj.r;

        // #########################################################
        // Solve Collision
        {
            // Parameters
            let compliance = 0.00;
            let r0_world = r_;
            let distance = separation;
            let direction = normal;

            // Solve Collision

            // Calculates the generalized inverse mas
            let rn0 = r0_world.cross(direction);
            let i_rn0 = r.apply_inverse_inertia_world(&rn0);
            let w0 = p.inv_mass + rn0.dot(i_rn0);

            // Calculates the Lagrange multiplier correction
            let alpha_tilde = compliance / (dt * dt);
            let delta_lagrange_multiplier =
                (-distance - alpha_tilde * self.lagrange_multiplier_n) / (w0 + alpha_tilde);
            self.lagrange_multiplier_n += delta_lagrange_multiplier;

            // Compute the positional impulse
            let p_ = delta_lagrange_multiplier * direction;

            // println!("p: {:?}", p_);

            // Calculates the positional correction
            p.pos += p_ * p.inv_mass;

            // Applies the rotational positional correction
            let dq0 = r.apply_inverse_inertia_world(&r0_world.cross(p_));

            r.rot += 0.5 * Quat::new(0.0, dq0.x, dq0.y, dq0.z) * r.rot;
        }

        // #########################################################
        // Solve Friction
        {
            // Parameters
            let friction_coefficient = 1.0;
            let compliance = 0.0000;

            // Recompute contact position after normal correction.
            let pos_world = r.rot * self.particle_local + p.pos;
            let contact_point = pos_world - normal * self.particle_radius;
            let prev_contact_point = prev_pos_world - normal * self.particle_radius;

            // Compute the relative motion of the contact points and its tangential component
            let dp = contact_point - prev_contact_point;
            let dp_t = dp - normal * dp.dot(normal);

            // Solve Collision
            let dx = dp_t;
            let distance = dx.magnitude();
            if distance < 1e-06 {
                return;
            }

            let direction = dx / distance;
            let r_contact = contact_point - p.pos;

            // Calculates the generalized inverse mas
            let rn0 = r_contact.cross(direction);
            let i_rn0 = r.apply_inverse_inertia_world(&rn0);
            let w0 = p.inv_mass + rn0.dot(i_rn0);

            // Calculates the Lagrange multiplier correction
            let alpha_tilde = compliance / (dt * dt);
            let delta_lagrange_multiplier =
                (-distance - alpha_tilde * self.lagrange_multiplier_t) / (w0 + alpha_tilde);
            let new_lagrange_multiplier_t = self.lagrange_multiplier_t + delta_lagrange_multiplier;

            self.lagrange_multiplier_t += delta_lagrange_multiplier;
            if new_lagrange_multiplier_t.abs() <= friction_coefficient * self.lagrange_multiplier_n
            {
                // Compute the positional impulse
                let p_ = delta_lagrange_multiplier * direction;
                // println!("p: {:?}, db_t: {:?}", p_, dp_t);

                // Calculates the positional correction
                p.pos += p_ * p.inv_mass;

                // Applies the rotational positional correction
                let dq0 = r.apply_inverse_inertia_world(&r_contact.cross(p_));

                r.rot += 0.5 * Quat::new(0.0, dq0.x, dq0.y, dq0.z) * r.rot;
            }
        }
    }
    
    /// Solve dynamic friction
    /// 
    /// // Typical friction coefficients for different materials.
    ///
    /// Material          | Static Friction (μs) | Dynamic Friction (μd)
    /// ------------------|----------------------|-----------------------
    /// Ice on Ice        | 0.05                 | 0.03
    /// Ice on Steel      | 0.03                 | 0.02
    /// Steel on Steel    | 0.6                  | 0.4
    /// Wood on Wood      | 0.5                  | 0.3
    /// Rubber on Concrete| 1.0                  | 0.8
    /// Rubber on Asphalt | 1.0                  | 0.8
    /// Concrete on Concrete | 0.7               | 0.6
    /// Leather on Wood   | 0.5                  | 0.4
    /// Glass on Glass    | 0.9                  | 0.4
    /// Teflon on Steel   | 0.04                 | 0.04
    /// Aluminum on Steel | 0.6                  | 0.4
    /// Sand on Sand      | 0.6                  | 0.5
    ///
    /// Note:
    /// - μs: Static friction coefficient (positional friction).
    /// - μd: Dynamic friction coefficient (velocity-level friction).
    /// - Values are approximate and depend on surface finish, pressure,
    ///   lubrication, and environmental conditions.
    pub fn solve_velocity_level(&mut self, obj: &mut XpbdRigidBody, dt: f32) {
        
        // Parameters
        let friction_coefficient = 0.8;

        let p = &mut obj.p;
        let r = &mut obj.r;

        let pos_world = r.rot * self.particle_local + p.pos;
        let prev_pos_world = r.prev_rot * self.particle_local + p.prev_pos;
        let normal = Vec3::unit_z();
        let contact_point = pos_world - normal * self.particle_radius;
        let separation = contact_point.z - GROUND_HEIGHT_Z;
        let r_ = contact_point - p.pos;

        // Computes the relative velocity at the contact point:
        //
        // v = (v1 + ω1 × r1) - (v2 + ω2 × r2)
        //
        // Computes the relative normal velocity by projecting v
        // onto the contact normal:
        //
        // vn = n · v
        //
        // Computes the relative tangential velocity by removing
        // the normal component from the relative velocity:
        //
        // vt = v - n * vn

        let v1 = p.vel;
        let omega1 = r.omega;
        let r1 = r_;
        let n = normal;

        let v = v1 + omega1.cross(r1);
        let v_n = n.dot(v);
        let v_t = v - n *v_n;

        // Computes the explicit velocity correction due to dynamic Coulomb friction.
        //
        // Δv = - (v_t / |v_t|) * min(h * μ_d * |f_n|, |v_t|)
        //
        // Where:
        // - v_t: Relative tangential velocity at the contact point.
        // - μ_d: Dynamic friction coefficient.
        // - h: Time step.
        // - f_n: Normal contact force, computed as λ_n / h².
        //
        // The correction opposes the tangential velocity and is limited
        // so that its magnitude never exceeds |v_t|.
        //
        // This prevents the friction update from reversing the tangential
        // velocity. Combined with the Gauss-Seidel update, this explicit
        // friction step is unconditionally stable.

        let v_t_magnitude = v_t.magnitude();
        if v_t_magnitude < 1e-6 {
            return;
        }

        let u_d = friction_coefficient;
        let h = dt;
        let lambda_n = self.lagrange_multiplier_n;
        let f_n = lambda_n / (h*h);
        let f_n_magnitude = f_n.abs();

        // Tangential direction
        let tangent = v_t / v_t_magnitude;

        let delta_v = - tangent * f32::min(h * u_d * f_n_magnitude, v_t_magnitude);


        // Applies a velocity update Δv at the contact points of two bodies.
        //
        // The impulse p is computed by dividing the velocity correction
        // by the sum of the inverse masses:
        //
        // p = Δv / (w1 + w2)
        //
        // The linear velocities are updated according to the inverse masses:
        //
        // v1 ← v1 + p / m1
        // v2 ← v2 - p / m2
        //
        // The angular velocities are updated using the inverse inertia tensors
        // and the cross product between the contact offsets and the impulse:
        //
        // ω1 ← ω1 + I1⁻¹ (r1 × p)
        // ω2 ← ω2 - I2⁻¹ (r2 × p)
        //
        // Where:
        // - Δv: Desired relative velocity correction.
        // - p: Contact impulse.
        // - w1, w2: Inverse masses of the two bodies.
        // - m1, m2: Masses of the two bodies.
        // - r1, r2: Contact offsets relative to the body centers of mass.
        // - I1⁻¹, I2⁻¹: Inverse world-space inertia tensors.
        // - ω1, ω2: Angular velocities of the two bodies.

        // Calculate effective inverse mass
        let rn = r1.cross(tangent);
        let i_rn = r.apply_inverse_inertia_world(&rn);

        let w1 = p.inv_mass + rn.dot(i_rn);

        // Impulse
        let p_ = delta_v / (w1);

        // Update linear velocity
        p.vel += p_ * p.inv_mass;

        // Updater angular velocity
        let angular_impulse = r1.cross(p_);
        r.omega += r.apply_inverse_inertia_world(&angular_impulse);
    }
}
