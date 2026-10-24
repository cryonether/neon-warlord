//! Extended Position Based Dynamics
//! https://matthias-research.github.io/pages/publications/PBDBodies.pdf
//! https://github.com/matthias-research/pages/blob/master/tenMinutePhysics/25-joints.html#L752
//! 

pub mod xpbd_positional_constraint;
pub mod xpbd_angular_constraint;
pub mod xpbd_drawer;
pub mod xpbd_rigid_body;

#[cfg(test)]
mod tests;

use cgmath::{InnerSpace, One, Rad, Rotation, SquareMatrix, Zero};

use crate::physics::{rigid_body::RigidBody, xpdb::{xpbd_positional_constraint::XpbdPositionalConstraint, xpbd_rigid_body::XpbdRigidBody}};

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

    pub fn step(&mut self, dt: f32) {
        // const DT: f32 = 1.0 / 60.0;
        const SUB_STEPS: usize = 4;
        let sub_dt = dt / SUB_STEPS as f32;

        for constraint in &mut self.position_constraints {
            constraint.constraint.reset_lagrange_multiplier();
        }


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
                particle.p.update_velocity(sub_dt);
            }

            for rigid_body in &mut self.rigid_bodies {
                rigid_body.p.update_velocity(sub_dt);
                rigid_body.r.update_angular_velocity(sub_dt);
            }

            // for constraint in &self.velocity_constraints {
            //     // SolveVelocities(v_1, . . . v_n, w_1, . . . w_n)
            // }
        }
    }

    pub fn add_rigid_body(&mut self, rigid_body: &XpbdRigidBody) -> usize {
        let index = self.rigid_bodies.len();
        self.rigid_bodies.push(rigid_body.clone());
        index
    }

    pub fn add_rigid_body_2(&mut self, pos: Vec3, mass: f32) -> usize {
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

        let orientation= Quad::one();

        let obj = XpbdRigidBody {
            p: PositionalQuantities {
                pos: pos,
                prev_pos: pos,
                vel: Vec3::zero(),
                // mass,
                inv_mass,
            },
            r: RotationalQuantities::new(Vec3::zero(), mass),
            particles: [Vec3::zero()].to_vec()
        };

        self.rigid_bodies.push(obj);
        
        index
    }

    pub fn add_constraint_position(&mut self, index_a: usize, index_b: usize, constraint: XpbdPositionalConstraint) -> usize {
        let index = self.position_constraints.len();
        
        let constraint = PositionalConstraint{
            index_a,
            index_b,
            constraint,
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
pub struct PositionalQuantities {
    // Position
    pos: Vec3,

    // Previous position
    prev_pos: Vec3,

    // Velocity
    vel: Vec3,

    // mass
    // mass: f32,

    // Inverse mass
    inv_mass: f32,
}

impl PositionalQuantities {
    fn integrate_position(&mut self, dt: f32) {
        // Save previous position
        self.prev_pos = self.pos;

        // Apply external forces
        self.vel = self.vel + dt * GRAVITY;

        // Update position
        self.pos = self.pos + dt * self.vel;
    }

    fn update_velocity(&mut self, dt: f32) {
        self.vel = (self.pos - self.prev_pos) / dt;
    }
}

#[derive(Clone)]
pub struct RotationalQuantities {
    // Rotation
    pub rot: Quad,
    pub inv_rot: Quad,

    // Previous rotation
    pub prev_rot: Quad,

    // Angular velocity
    pub omega: Vec3,

    // // Inertia tensor
    // pub inertia_tensor: Mat3,

    // // Inverse inertia tensor
    pub inverse_inertia_tensor: Mat3,


    // pub inv_inertia: Vec3,
}

impl RotationalQuantities {

    pub fn new(angles: Vec3, mass: f32) -> Self {
        let rot = Quad::from(cgmath::Euler{
            x: Rad(angles.x),
            y: Rad(angles.y),
            z: Rad(angles.z),
        });
        let inv_rot = rot.invert();
        let prev_rot = rot.clone();
        let omega = Vec3::zero();
        let inv_inertia = Vec3::new(1.0, 1.0, 1.0);

        let inverse_inertia_tensor = if mass > 0.0 {
            Mat3::identity()
        } else {
            Mat3::zero()
        };

        Self {
            rot,
            inv_rot,
            prev_rot,
            omega,
            inverse_inertia_tensor,
            // inv_inertia,
        }
    }

    fn integrate_orientation(&mut self, dt: f32) {
        // Save previous rotation
        self.prev_rot = self.rot;

        // // Apply external torques
        // let external_torque = Vec3::zero();
        // self.omega = self.omega + 
        //     dt * self.inverse_inertia_tensor * 
        //     (external_torque - self.omega.cross(
        //         self.inertia_tensor * self.omega
        //     )
        // );

        // Update orientation
        self.rot = self.rot + dt * 0.5 * Quad::new(
            0.0,
            self.omega.x, 
            self.omega.y, 
            self.omega.z, 
        ) * self.rot;

        // Normalize
        self.rot = self.rot / self.rot.magnitude();

        // Calculate inverse rotation
        self.inv_rot = self.rot.invert();
    }

    fn update_angular_velocity(&mut self, dt: f32) {
        // Calculate delta
        let dq = self.rot * self.prev_rot.invert();
        
        // Update angular velocity
        self.omega = 2.0 * Vec3::new(dq.v.x, dq.v.y, dq.v.z) / dt;

        // Check if > 0
        if dq.s < 0.0 { 
            self.omega = -self.omega
        }
    }

    fn apply_inverse_inertia_world(&self,
        // rotation: &Quad,
        // inverse_rotation: &Quad,
        // inverse_inertia_local: &Mat3,
        v_world: &Vec3,
    ) -> Vec3 {
        // Transform the vector into local space.
        let v_local = self.inv_rot.rotate_vector(*v_world);

        // Apply the full local-space inverse inertia tensor.
        let result_local = self.inverse_inertia_tensor * v_local;

        // Transform the result back into world space.
        self.rot.rotate_vector(result_local)
    }
}

#[derive(Clone)]
pub struct PositionalConstraint {
    pub index_a: usize,
    pub index_b: usize,
    pub constraint: XpbdPositionalConstraint,
}