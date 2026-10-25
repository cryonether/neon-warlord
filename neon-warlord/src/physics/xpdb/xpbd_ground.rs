//! A ground collision

use cgmath::InnerSpace;

use crate::physics::xpdb::{Quad, Vec3, xpbd_positional_constraint::XpbdPositionalConstraint, xpbd_rigid_body::XpbdRigidBody};

// Hight of the ground plane
const GROUND_HIGHT_Z: f32 = 0.0;
const RADIUS: f32 = 0.5;

pub struct XpbdGround {
    pub contact_points: Vec<(usize, ContactPoint)>
}

impl XpbdGround {

    pub fn find_contact_points(&mut self, bodies: &[XpbdRigidBody]) {

        // The rigid bodies consist of multiple spherical particles with a defined radius
        let radius = 0.1;

        for (i, body) in bodies.iter().enumerate() {
            for particle in &body.particles {
                // The position of the particle is relative to the body center of mass

                let pos_local = particle + body.p.pos;
                let pos_world = body.local_to_world(&pos_local);

                if pos_world.z - radius < GROUND_HIGHT_Z {
                    // Collision with the ground detected

                    let object_index = i;
                    let normal = Vec3::unit_z();
                    let contact_point = pos_world - normal * radius;
                    let separation = contact_point.z - GROUND_HIGHT_Z;
                    let r = contact_point - body.p.pos;

                    let contact_point = ContactPoint {
                        particle_local: pos_local,
                        particle_radius: radius,
                        r,
                        normal,
                        separation,
                        lagrange_multiplier_n: 0.0,
                        lagrange_multiplier_t: 0.0,
                    };

                    self.contact_points.push((object_index, contact_point));
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





pub struct ContactPoint {
    pub particle_local: Vec3,
    pub particle_radius: f32,

    // Contact point relative to the body's center of mass,
    // expressed in world space.
    pub r: Vec3,

    // Contact normal pointing out of the ground.
    pub normal: Vec3,

    // Signed separation: positive when separated,
    // negative when penetrating.
    pub separation: f32,

    // Accumulated XPBD multipliers.
    pub lagrange_multiplier_n: f32,
    pub lagrange_multiplier_t: f32,
}

impl ContactPoint {
    pub fn solve(&mut self, obj: &mut XpbdRigidBody, dt: f32){
        let p = &obj.p;
        let r = &obj.r;

        let pos_world = obj.local_to_world(&self.particle_local);
        let contact_point = pos_world - self.normal * self.particle_radius;
        self.separation = contact_point.z - GROUND_HIGHT_Z;
        self.r = contact_point - p.pos;

        let p = &mut obj.p;
        let r = &mut obj.r;

        // Parameters
        let compliance = 0.0;
        let r0_world = self.r;
        let distance = self.separation;
        let direction = self.normal;

        // #########################################################
        // Solve

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
        self.lagrange_multiplier_n = self.lagrange_multiplier_n.max(0.0);

        // Compute the positional impulse
        let p_ = delta_lagrange_multiplier * direction;

        // Calculates the positional correction
        p.pos += p_ * p.inv_mass;

        // Applies the rotational positional correction 
        let dq0 = r.apply_inverse_inertia_world(
                &r0_world.cross(p_)
            );

        r.rot += 0.5 * Quad::new(0.0, dq0.x, dq0.y, dq0.z) * r.rot;
    }
}