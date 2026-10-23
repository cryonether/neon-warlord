//! Extended Position Based Dynamics
//! https://matthias-research.github.io/pages/publications/PBDBodies.pdf
//! 

pub mod xpbd_positional_constraint;
pub mod xpbd_drawer;

#[cfg(test)]
mod tests;

use cgmath::{InnerSpace, One, Rotation, SquareMatrix, Zero};

use crate::physics::{rigid_body::RigidBody, xpdb::xpbd_positional_constraint::XpbdPositionalConstraint};

type Vec3 = cgmath::Vector3<f32>;
type Quad = cgmath::Quaternion<f32>;
type Mat3 = cgmath::Matrix3<f32>;

const GRAVITY: Vec3 = Vec3::new(0.0, 0.0, -10.0);

#[derive(Clone)]
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


    pub fn add_rigid_body(&mut self, pos: Vec3, mass: f32) -> usize {
        assert!(
            mass.is_finite() && mass >= 0.0,
            "Mass must be finite and non-negative"
        );
        
        let index = self.rigid_bodies.len();

        let inv_mass = if mass > 0.0 {
            1.0 / mass
        } else {
            0.0
        };

        let inertia_tensor = Mat3::identity();
        let inverse_inertia_tensor = if mass > 0.0 {
            Mat3::identity()
        } else {
            Mat3::zero()
        };

        let orientation= Quad::one();

        let obj = XpbdRigidBody {
            p: PositionalQuantities {
                pos: pos,
                pos_prev: pos,
                vel: Vec3::zero(),
                mass,
                inv_mass,
            },
            r: RotationalQuantities {
                orientation,
                orientation_prev: orientation,
                angular_velocity: Vec3::zero(),
                inertia_tensor,
                inverse_inertia_tensor,
            },
        };

        self.rigid_bodies.push(obj);
        
        index
    }

    pub fn add_constraint_position(&mut self, index_a: usize, index_b: usize, compliance: f32) -> usize {
        let index = self.position_constraints.len();
        
        let constraint = PositionalConstraint{
            index_a,
            index_b,
            constraint: XpbdPositionalConstraint {
                relative_position_0: Vec3::zero(),
                relative_position_1: Vec3::zero(),
                compliance,
                lagrange_multiplier: 1.0,
            },
        };

        self.position_constraints.push(constraint);

        index
    }
}

#[derive(Clone)]
pub struct XpbdParticle {
    p: PositionalQuantities,
}

#[derive(Clone)]
pub struct XpbdRigidBody {
    p: PositionalQuantities,
    r: RotationalQuantities,
}

#[derive(Clone)]
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

#[derive(Clone)]
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

#[derive(Clone)]
pub struct PositionalConstraint {
    pub index_a: usize,
    pub index_b: usize,
    pub constraint: XpbdPositionalConstraint,
}