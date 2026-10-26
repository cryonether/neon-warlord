//! RigidBody implementation using XPBD


use cgmath::One;

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
        let _prev_pos = pos;
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

        // let inverse_inertia_tensor = inertia_tensor
        //     .invert()
        //     .unwrap_or_else(Mat3::zero);
        //     // .unwrap();

        let inverse_inertia_tensor = symmetric_pseudo_inverse(inertia_tensor);

        // Move all particles by the center of mass
        let mut particles_centered = Vec::with_capacity(particles.len());
        for particle in particles {
                particles_centered.push(particle - center_of_mass);
        }

        let pos = pos + center_of_mass;

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
                inertia_tensor,
                inverse_inertia_tensor,
                external_torque: Vec3::zero(),
            },
            particles: particles_centered,
        }
    }


    pub fn _get_velocity_at(&self, pos: &Vec3) -> Vec3 {
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

    pub fn _world_to_local(&self, world_pos: &Vec3) -> Vec3 {
        let p = &self.p;
        let r = &self.r;

        r.inv_rot * (*world_pos - p.pos)
    }
}


/// Computes the Moore-Penrose pseudoinverse of a symmetric 3x3 matrix
/// using Jacobi eigenvalue decomposition.
fn symmetric_pseudo_inverse(mut matrix: Mat3) -> Mat3 {
    // A = Q * D * Qᵀ
    // Q contains eigenvectors; D contains eigenvalues.

    let mut a = [
        [matrix.x.x, matrix.y.x, matrix.z.x],
        [matrix.x.y, matrix.y.y, matrix.z.y],
        [matrix.x.z, matrix.y.z, matrix.z.z],
    ];

    let mut q = [
        [1.0_f32, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];

    // Jacobi rotations diagonalize the symmetric matrix.
    for _ in 0..32 {
        // Find the largest off-diagonal element.
        let mut p = 0;
        let mut r = 1;
        let mut max_off_diag = a[0][1].abs();

        for (i, j) in [(0, 2), (1, 2)] {
            if a[i][j].abs() > max_off_diag {
                max_off_diag = a[i][j].abs();
                p = i;
                r = j;
            }
        }

        let scale = a[0][0].abs()
            .max(a[1][1].abs())
            .max(a[2][2].abs());

        if max_off_diag <= scale * 1e-7 || max_off_diag == 0.0 {
            break;
        }

        let app = a[p][p];
        let arr = a[r][r];
        let apr = a[p][r];

        // Stable Jacobi rotation.
        let tau = (arr - app) / (2.0 * apr);
        let t = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
        };

        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;

        // Update the diagonal elements.
        a[p][p] = app - t * apr;
        a[r][r] = arr + t * apr;
        a[p][r] = 0.0;
        a[r][p] = 0.0;

        // Update the remaining symmetric entries.
        #[allow(clippy::needless_range_loop)]
        for k in 0..3 {
            if k != p && k != r {
                let akp = a[k][p];
                let akr = a[k][r];

                a[k][p] = c * akp - s * akr;
                a[p][k] = a[k][p];

                a[k][r] = s * akp + c * akr;
                a[r][k] = a[k][r];
            }
        }

        // Accumulate eigenvectors: Q = Q * rotation.
        #[allow(clippy::needless_range_loop)]
        for k in 0..3 {
            let qkp = q[k][p];
            let qkr = q[k][r];

            q[k][p] = c * qkp - s * qkr;
            q[k][r] = s * qkp + c * qkr;
        }
    }

    // Invert only eigenvalues that are sufficiently far from zero.
    let max_eigenvalue = a[0][0]
        .abs()
        .max(a[1][1].abs())
        .max(a[2][2].abs());

    let tolerance = max_eigenvalue * 1e-6;

    let mut d_inv = [0.0_f32; 3];

    for i in 0..3 {
        let eigenvalue = a[i][i];

        if eigenvalue > tolerance {
            d_inv[i] = 1.0 / eigenvalue;
        }
    }

    // I⁺ = Q * D⁺ * Qᵀ
    //
    // Build the symmetric pseudoinverse directly.
    let mut result = [[0.0_f32; 3]; 3];

    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                result[i][j] += q[i][k] * d_inv[k] * q[j][k];
            }
        }
    }

    matrix = Mat3::from_cols(
        Vec3::new(result[0][0], result[1][0], result[2][0]),
        Vec3::new(result[0][1], result[1][1], result[2][1]),
        Vec3::new(result[0][2], result[1][2], result[2][2]),
    );

    matrix
}