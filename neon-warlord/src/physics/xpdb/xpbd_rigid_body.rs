//! RigidBody implementation using XPBD

use std::iter::zip;

use super::*;

// A rigid body composed of single particles
#[derive(Clone)]
pub struct XpbdRigidBody {
    pub p: PositionalQuantities,
    pub r: RotationalQuantities,

    pub particles: Vec<Vec3>,
}

impl XpbdRigidBody {
    pub fn new(pos: Vec3, particles: &[Vec3], mass: f32) -> Self {
        let prev_pos = pos;
        let overall_mass = particles.len() as f32 * mass;
        let inv_mass = if overall_mass > 0.0 {
            1.0 / overall_mass
        } else {
            0.0
        };

        let rot = Quad::one();
        let inv_rot = rot.invert();
        let prev_rot = rot;
        let omega = Vec3::zero();

        // Calculate the local-space inverse inertia tensor.
        //
        // First calculate the center of mass:
        //
        //             1
        // c =         ─── Σ rᵢ
        //             N
        //
        // Then shift each particle relative to the center of mass:
        //
        // rᵢ' = rᵢ - c
        //
        //              ⎡ Σ m(y² + z²)   -Σ mxy       -Σ mxz     ⎤
        // I =          ⎢ -Σ mxy         Σ m(x² + z²) -Σ myz     ⎥
        //              ⎣ -Σ mxz         -Σ myz       Σ m(x² + y²)⎦
        //
        // I⁻¹ = pseudo-inverse(I)
        //
        let particle_mass = mass;
        
        // Calculate center of mass
        let mut center_of_mass = Vec3::zero();
        for particle in particles {
            center_of_mass += *particle;
        }
        center_of_mass /= particles.len() as f32;

        // Calculate inertia tensor
        let mut inertia_tensor = Mat3::zero();

        for particle in particles {
            let r = *particle - center_of_mass;
            let r_squared = r.dot(r);

            inertia_tensor.x.x += particle_mass * (r_squared - r.x * r.x);
            inertia_tensor.y.y += particle_mass * (r_squared - r.y * r.y);
            inertia_tensor.z.z += particle_mass * (r_squared - r.z * r.z);

            inertia_tensor.x.y -= particle_mass * r.x * r.y;
            inertia_tensor.y.x -= particle_mass * r.x * r.y;

            inertia_tensor.x.z -= particle_mass * r.x * r.z;
            inertia_tensor.z.x -= particle_mass * r.x * r.z;

            inertia_tensor.y.z -= particle_mass * r.y * r.z;
            inertia_tensor.z.y -= particle_mass * r.y * r.z;
        }

        let inverse_inertia_tensor = inertia_tensor
            .invert()
            .unwrap_or_else(Mat3::zero);
            // .unwrap();

        // Move all particles by the center of mass
        let mut particles_centered = Vec::with_capacity(particles.len());
        for particle in particles {
                particles_centered.push(particle - center_of_mass);
        }

        Self {
            p:PositionalQuantities{
                pos,
                prev_pos: pos,
                vel: Vec3::zero(),
                inv_mass,
            },
            r: RotationalQuantities{
                rot,
                inv_rot,
                prev_rot,
                omega,
                inverse_inertia_tensor,
            },
            particles: particles_centered,
        }
    }


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
}