//! Extended Position Based Dynamics
//! https://matthias-research.github.io/pages/publications/PBDBodies.pdf
//! https://github.com/matthias-research/pages/blob/master/tenMinutePhysics/25-joints.html#L752
//! 

pub mod xpbd_positional_constraint;
pub mod xpbd_drawer;

#[cfg(test)]
mod tests;

use cgmath::{InnerSpace, One, Rad, Rotation, SquareMatrix, Zero};

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

impl XpbdRigidBody {
    pub fn get_velocity_at(&self, pos: &Vec3) -> Vec3 {
        let p = &self.p;
        let r = &self.r;
        
        if p.inv_mass <= 0.0 {
            return Vec3::zero();
        } 

        p.vel - (pos - p.pos).cross(r.omega)
    }

    pub fn local_to_world(&self, local_pos: &Vec3) -> Vec3 {
        let p = &self.p;
        let r = &self.r;

        r.rot * local_pos + p.pos
    }

    pub fn world_to_local(&self, world_pos: &Vec3) -> Vec3 {
        let p = &self.p;
        let r = &self.r;

        r.inv_rot * (*world_pos - p.pos)
    }


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
    // pub fn get_inverse_mass(&self, normal: &Vec3, pos: &Vec3) -> f32 {
    //     let p = &self.p;
    //     let r = &self.r;

    //     if p.inv_mass <= 0.0 {
    //         return 0.0;
    //     }

    //     let rn = r.inv_rot * (pos - p.pos).cross(*normal);

    //     let w = p.inv_mass +
    //         rn.x * rn.x * r.inv_inertia.x +
    //         rn.y * rn.y * r.inv_inertia.y +
    //         rn.z * rn.z * r.inv_inertia.z;
            
    //     w
    // }
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


    pub inv_inertia: Vec3,
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
            inv_inertia,
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
}

#[derive(Clone)]
pub struct PositionalConstraint {
    pub index_a: usize,
    pub index_b: usize,
    pub constraint: XpbdPositionalConstraint,
}