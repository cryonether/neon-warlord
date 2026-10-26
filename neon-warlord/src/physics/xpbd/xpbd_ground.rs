//! A ground collision

use cgmath::InnerSpace;

use crate::physics::xpbd::{Quad, Vec3, xpbd_positional_constraint::XpbdPositionalConstraint, xpbd_rigid_body::XpbdRigidBody};

// Hight of the ground plane
const GROUND_HIGHT_Z: f32 = 0.0;

#[derive(Clone)]
pub struct XpbdGround {
    pub contact_points: Vec<(usize, ContactPoint)>
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

                if pos_world.z - particle_radius < GROUND_HIGHT_Z {
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
    pub fn solve(&mut self, obj: &mut XpbdRigidBody, dt: f32){
        let p = &obj.p;
        let r = &obj.r;

        let pos_world = r.rot * self.particle_local + p.pos;
        let prev_pos_world = r.prev_rot * self.particle_local + p.prev_pos;
        let normal = Vec3::unit_z();
        let contact_point = pos_world - normal * self.particle_radius;
        let separation = contact_point.z - GROUND_HIGHT_Z;
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
            let i_rn0 = r.apply_inverse_inertia_world(
                &rn0
            );
            let w0 = p.inv_mass + rn0.dot(i_rn0);

            // Calculates the Lagrange multiplier correction
            let alpha_tilde = compliance / (dt * dt);
            let delta_lagrange_multiplier = (-distance - alpha_tilde * self.lagrange_multiplier_n) / 
                                                            (w0 + alpha_tilde);
            self.lagrange_multiplier_n += delta_lagrange_multiplier;

            // Compute the positional impulse
            let p_ = delta_lagrange_multiplier * direction;

            // println!("p: {:?}", p_);

            // Calculates the positional correction
            p.pos += p_ * p.inv_mass;

            // Applies the rotational positional correction 
            let dq0 = r.apply_inverse_inertia_world(
                    &r0_world.cross(p_)
                );

            r.rot += 0.5 * Quad::new(0.0, dq0.x, dq0.y, dq0.z) * r.rot;
        }

        // #########################################################
        // Solve Friction
        {   
            // Parameters
            let friction_coefficient = 10.0;
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
            let i_rn0 = r.apply_inverse_inertia_world(
                &rn0
            );
            let w0 = p.inv_mass + rn0.dot(i_rn0);

            // Calculates the Lagrange multiplier correction
            let alpha_tilde = compliance / (dt * dt);
            let delta_lagrange_multiplier = (-distance - alpha_tilde * self.lagrange_multiplier_t) / 
                                                            (w0 + alpha_tilde);
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
                let dq0 = r.apply_inverse_inertia_world(
                        &r_contact.cross(p_)
                    );

                r.rot += 0.5 * Quad::new(0.0, dq0.x, dq0.y, dq0.z) * r.rot;
            }
        }
        

    }
}
