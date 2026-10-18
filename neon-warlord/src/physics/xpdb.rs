//! Extended Position Based Dynamics
//! https://matthias-research.github.io/pages/publications/PBDBodies.pdf
//! 

use cgmath::{InnerSpace, Rotation, Zero};

type Vec3 = cgmath::Vector3<f32>;
type Quad = cgmath::Quaternion<f32>;
type Mat3 = cgmath::Matrix3<f32>;

const GRAVITY: Vec3 = Vec3::new(0.0, 0.0, -10.0);

pub struct Xpbd {
    particles: Vec<XpbdParticle>,
    rigid_bodies: Vec<XpbdRigidBody>,

    position_constraints: Vec<f32>,
    velocity_constraints: Vec<f32>,
}

impl Xpbd {
    pub fn step(&mut self) {
        const DT: f32 = 1.0 / 60.0;
        const SUB_STEPS: usize = 4;
        let sub_dt = DT / SUB_STEPS as f32;

        for i in 0..SUB_STEPS {
            for particle in &mut self.particles {
                {
                    let p: &mut PositionalQuantities = &mut particle.pos;

                    // Save previous position
                    p.pos_prev = p.pos;

                    // Apply external forces
                    let force_external = GRAVITY * p.mass;
                    p.vel = p.vel + sub_dt * force_external * p.inv_mass;

                    // Update position
                    p.pos = p.pos + sub_dt * p.vel;
                }
            }

            for rigid_body in &mut self.rigid_bodies {
                {
                    let p: &mut PositionalQuantities = &mut rigid_body.pos;

                    // Save previous position
                    p.pos_prev = p.pos;

                    // Apply external forces
                    let force_external = GRAVITY * p.mass;
                    p.vel = p.vel + sub_dt * force_external * p.inv_mass;

                    // Update position
                    p.pos = p.pos + sub_dt * p.vel;
                }
                {
                    let r: &mut RotationalQuantities = &mut rigid_body.rot;

                    // Save previous rotation
                    r.orientation_prev = r.orientation;

                    // Apply external torques
                    let external_torque = Vec3::zero();
                    r.angular_velocity = r.angular_velocity + 
                        sub_dt * r.inverse_inertia_tensor * 
                        (external_torque - r.angular_velocity.cross(
                            r.inertia_tensor * r.angular_velocity
                        )
                    );

                    // Update orientation
                    r.orientation = r.orientation + sub_dt * 0.5 * Quad::new(
                        0.0,
                        r.angular_velocity.x, 
                        r.angular_velocity.y, 
                        r.angular_velocity.z, 
                    ) * r.orientation;

                    // Normalize
                    r.orientation = r.orientation / r.orientation.magnitude();
                }
            }

            for constraint in &self.position_constraints {
                // SolvePositions(x_1, . . . x_n, q_1, . . . q_n);
            }

            for particle in &mut self.particles {
                {
                    let p: &mut PositionalQuantities = &mut particle.pos;

                    // Update velocity
                    p.vel = (p.pos - p.pos_prev) / sub_dt;
                }
            }

            for rigid_body in &mut self.rigid_bodies {
                {
                    let p: &mut PositionalQuantities = &mut rigid_body.pos;

                    // Update velocity
                    p.vel = (p.pos - p.pos_prev) / sub_dt;
                }
                {
                    let r: &mut RotationalQuantities = &mut rigid_body.rot;

                    // Calculate delta
                    let dq = r.orientation * r.orientation_prev.invert();
                    
                    // Update angular velocity
                    r.angular_velocity = 2.0 * Vec3::new(dq.v.x, dq.v.y, dq.v.z) / sub_dt;

                    // Check if > 0
                    r.angular_velocity = if Vec3::new(dq.v.x, dq.v.y, dq.v.z).dot(r.angular_velocity) > 0.0 { 
                        r.angular_velocity 
                    } else { 
                        -r.angular_velocity 
                    };
                }
            }

            for constraint in &self.velocity_constraints {
                // SolveVelocities(v_1, . . . v_n, w_1, . . . w_n)
            }
        }


    }
}

pub struct XpbdParticle {
    pos: PositionalQuantities,
}

pub struct XpbdRigidBody {
    pos: PositionalQuantities,
    rot: RotationalQuantities,
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