//! Extended Position Based Dynamics
//! https://matthias-research.github.io/pages/publications/PBDBodies.pdf
//! 

pub mod xpbd_positional_constraint;

#[cfg(test)]
mod tests;

use cgmath::{InnerSpace, Rotation, Zero};

use crate::physics::{rigid_body::RigidBody, xpdb::xpbd_positional_constraint::XpbdPositionalConstraint};

type Vec3 = cgmath::Vector3<f32>;
type Quad = cgmath::Quaternion<f32>;
type Mat3 = cgmath::Matrix3<f32>;

const GRAVITY: Vec3 = Vec3::new(0.0, 0.0, -10.0);

pub struct Xpbd {
    particles: Vec<XpbdParticle>,
    rigid_bodies: Vec<XpbdRigidBody>,

    position_constraints: Vec<PositionalConstraint>,
    // velocity_constraints: Vec<f32>,
}

impl Xpbd {
    pub fn new() -> Self {
        let particles = Vec::new();
        let rigid_bodies = Vec::new();
        let position_constraints = Vec::new();
        // let velocity_constraints = Vec::new();

        Self {
            particles,
            rigid_bodies,
            position_constraints,
            // velocity_constraints,
        }
    }

    pub fn step(&mut self) {
        const DT: f32 = 1.0 / 60.0;
        const SUB_STEPS: usize = 4;
        let sub_dt = DT / SUB_STEPS as f32;

        for i in 0..SUB_STEPS {
            for particle in &mut self.particles {
                particle.p.integrate_position(sub_dt);
            }

            for rigid_body in &mut self.rigid_bodies {
                rigid_body.p.integrate_position(sub_dt);
                rigid_body.r.integrate_orientation(sub_dt);
            }

            for constraint in &mut self.position_constraints {
                // SolvePositions(x_1, . . . x_n, q_1, . . . q_n);

                let index_a = constraint.index_a;
                let index_b = constraint.index_b;
                let constraint = &mut constraint.constraint;

                assert_ne!(index_a, index_b, "Constraint indices must differ");
                let (obj_a, obj_b) = {
                    let (left, right) = self.rigid_bodies.split_at_mut(index_b);
                    (&mut left[index_a], &mut right[0])
                };

                constraint.solve(obj_a, obj_b, sub_dt);
            }

            for particle in &mut self.particles {
                particle.p.integrate_velocity(sub_dt);
            }

            for rigid_body in &mut self.rigid_bodies {
                rigid_body.p.integrate_velocity(sub_dt);
                rigid_body.r.integrate_angular_velocity(sub_dt);
            }

            // for constraint in &self.velocity_constraints {
            //     // SolveVelocities(v_1, . . . v_n, w_1, . . . w_n)
            // }
        }


    }
}

pub struct XpbdParticle {
    p: PositionalQuantities,
}

pub struct XpbdRigidBody {
    p: PositionalQuantities,
    r: RotationalQuantities,
}

pub struct PositionalQuantities {
    // Position
    pos: Vec3,

    // Previous position
    pos_prev: Vec3,

    // Velocity
    vel: Vec3,

    // mass
    mass: f32,

    // Inverse mass
    inv_mass: f32,

    // Inverse of stiffness
    compliance: f32,
}

impl PositionalQuantities {
    fn integrate_position(&mut self, dt: f32) {
        // Save previous position
        self.pos_prev = self.pos;

        // Apply external forces
        let force_external = GRAVITY * self.mass;
        self.vel = self.vel + dt * force_external * self.inv_mass;

        // Update position
        self.pos = self.pos + dt * self.vel;
    }

    fn integrate_velocity(&mut self, dt: f32) {
        self.vel = (self.pos - self.pos_prev) / dt;
    }
}

pub struct RotationalQuantities {
    // Rotation
    pub orientation: Quad,

    // Previous rotation
    pub orientation_prev: Quad,

    // Angular velocity
    pub angular_velocity: Vec3,

    // Inertia tensor
    pub inertia_tensor: Mat3,

    // Inverse inertia tensor
    pub inverse_inertia_tensor: Mat3
}

impl RotationalQuantities {
    fn integrate_orientation(&mut self, dt: f32) {
        // Save previous rotation
        self.orientation_prev = self.orientation;

        // Apply external torques
        let external_torque = Vec3::zero();
        self.angular_velocity = self.angular_velocity + 
            dt * self.inverse_inertia_tensor * 
            (external_torque - self.angular_velocity.cross(
                self.inertia_tensor * self.angular_velocity
            )
        );

        // Update orientation
        self.orientation = self.orientation + dt * 0.5 * Quad::new(
            0.0,
            self.angular_velocity.x, 
            self.angular_velocity.y, 
            self.angular_velocity.z, 
        ) * self.orientation;

        // Normalize
        self.orientation = self.orientation / self.orientation.magnitude();
    }

    fn integrate_angular_velocity(&mut self, dt: f32) {
        // Calculate delta
        let dq = self.orientation * self.orientation_prev.invert();
        
        // Update angular velocity
        self.angular_velocity = 2.0 * Vec3::new(dq.v.x, dq.v.y, dq.v.z) / dt;

        // Check if > 0
        self.angular_velocity = if Vec3::new(dq.v.x, dq.v.y, dq.v.z).dot(self.angular_velocity) > 0.0 { 
            self.angular_velocity 
        } else { 
            -self.angular_velocity 
        };
    }
}

pub struct PositionalConstraint {
    pub index_a: usize,
    pub index_b: usize,
    pub constraint: XpbdPositionalConstraint,
}