//! Fixes the angle between three particles

use crate::verlet_physics_simd::{Vec3, verlet_particles::VerletParticles};

/// Fixes the angle between three particles
#[derive(Clone)]
pub struct AngleConstraints {
    /// Particle index a.
    pub a: Vec<u32>,
    /// Particle index b.
    pub b: Vec<u32>,
    /// Particle index c.
    pub c: Vec<u32>,

    /// Angle
    cos_alpha: Vec<f32>,

    /// Constraint stiffness in [0, 1].
    stiffness: Vec<f32>,
}

impl AngleConstraints {
    pub fn new() -> Self {
        let a = Vec::new();
        let b = Vec::new();
        let c = Vec::new();
        let cos_alpha = Vec::new();
        let stiffness = Vec::new();

        Self {
            a,
            b,
            c,
            cos_alpha,
            stiffness,
        }
    }

    pub fn solve(&mut self, particles: &mut VerletParticles) {
        self.assert_bounds_invariants();

        for (&a, &b, &c, cos_alpha, stiffness) in itertools::izip!(
            &self.a,
            &self.b,
            &self.c,
            &self.cos_alpha,
            &self.stiffness,
        ) {
            let a = a as usize;
            let b = b as usize;
            let c = c as usize;

            // Vectors from the central particle B.
            let ab_x = particles.x[a] - particles.x[b];
            let ab_y = particles.y[a] - particles.y[b];
            let ab_z = particles.z[a] - particles.z[b];

            let cb_x = particles.x[c] - particles.x[b];
            let cb_y = particles.y[c] - particles.y[b];
            let cb_z = particles.z[c] - particles.z[b];

            let len_ab_sq = ab_x * ab_x + ab_y * ab_y + ab_z * ab_z;
            let len_cb_sq = cb_x * cb_x + cb_y * cb_y + cb_z * cb_z;

            if len_ab_sq < 1e-12 || len_cb_sq < 1e-12 {
                continue;
            }

            let inv_len_ab = len_ab_sq.sqrt().recip();
            let inv_len_cb = len_cb_sq.sqrt().recip();

            // Normalized bond directions.
            let u_x = ab_x * inv_len_ab;
            let u_y = ab_y * inv_len_ab;
            let u_z = ab_z * inv_len_ab;

            let v_x = cb_x * inv_len_cb;
            let v_y = cb_y * inv_len_cb;
            let v_z = cb_z * inv_len_cb;

            // Constraint: dot(u, v) == cos_alpha.
            let dot = u_x * v_x + u_y * v_y + u_z * v_z;
            let error = dot - cos_alpha;

            // Gradients with respect to A and C.
            let ga_x = (v_x - dot * u_x) * inv_len_ab;
            let ga_y = (v_y - dot * u_y) * inv_len_ab;
            let ga_z = (v_z - dot * u_z) * inv_len_ab;

            let gc_x = (u_x - dot * v_x) * inv_len_cb;
            let gc_y = (u_y - dot * v_y) * inv_len_cb;
            let gc_z = (u_z - dot * v_z) * inv_len_cb;

            // Gradient with respect to B: gb = -ga - gc.
            let gb_x = -ga_x - gc_x;
            let gb_y = -ga_y - gc_y;
            let gb_z = -ga_z - gc_z;

            let denom =
                ga_x * ga_x + ga_y * ga_y + ga_z * ga_z
                + gb_x * gb_x + gb_y * gb_y + gb_z * gb_z
                + gc_x * gc_x + gc_y * gc_y + gc_z * gc_z;

            if denom < 1e-12 {
                continue;
            }

            // PBD correction multiplier.
            let lambda = -error * stiffness / denom;

            // Apply position corrections.
            particles.x[a] += lambda * ga_x;
            particles.y[a] += lambda * ga_y;
            particles.z[a] += lambda * ga_z;

            particles.x[b] += lambda * gb_x;
            particles.y[b] += lambda * gb_y;
            particles.z[b] += lambda * gb_z;

            particles.x[c] += lambda * gc_x;
            particles.y[c] += lambda * gc_y;
            particles.z[c] += lambda * gc_z;
        }
    }

    fn assert_bounds_invariants(&self) {
        let len = self.len();

        assert!(self.a.len() == len);
        assert!(self.b.len() == len);

        assert!(self.b.len() == len);
        assert!(self.b.len() == len);
        assert!(self.b.len() == len);

        assert!(self.c.len() == len);
        assert!(self.c.len() == len);
        assert!(self.c.len() == len);

        assert!(self.cos_alpha.len() == len);
        assert!(self.cos_alpha.len() == len);
        assert!(self.cos_alpha.len() == len);

        assert!(self.stiffness.len() == len);
    }

    pub fn len(&self) -> usize {
        self.a.len()
    }

    pub fn push(
        &mut self,
        a: usize,
        b: usize,
        c: usize,
        cos_alpha: f32,
        stiffness: f32,
    ) -> usize {
        self.assert_bounds_invariants();

        let index = self.len();

        self.a.push(a as u32);
        self.b.push(b as u32);
        self.c.push(c as u32);
        self.cos_alpha.push(cos_alpha);
        self.stiffness.push(stiffness);

        index
    }
}
