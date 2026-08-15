//! Forces between two particles

use crate::verlet_physics_simd::{Vec3, verlet_particles::VerletParticles};


#[derive(Clone)]
pub struct DistanceConstraints {
    /// Particle index for endpoint A.
    a: Vec<u32>,
    /// Particle index for endpoint B.
    b: Vec<u32>,

    /// Rest distance.
    rest_x: Vec<f32>,
    rest_y: Vec<f32>,
    rest_z: Vec<f32>,
    /// Constraint stiffness in [0, 1].
    stiffness: Vec<f32>,

    // The algorithm to apply the constraint
    constrain_kind: Vec<ConstraintKind>,
}


const LANES: usize = 16;

impl DistanceConstraints {
    pub fn new() -> Self {
        let a = Vec::new();
        let b = Vec::new();
        let rest_x = Vec::new();
        let rest_y = Vec::new();
        let rest_z = Vec::new();
        let stiffness = Vec::new();
        let constrain_kind = Vec::new();

        Self{
            a,
            b,
            rest_x,
            rest_y,
            rest_z,
            stiffness,
            constrain_kind,
        }
    }

    pub fn solve(&mut self, particles: &mut VerletParticles) {
        self.assert_bounds_invariants();

        for (a, b, rest_x, rest_y, rest_z, stiffness, constraint_kind) in itertools::izip!(
            &self.a, &self.b, &self.rest_x, &self.rest_y, &self.rest_z, &self.stiffness, &self.constrain_kind
        ) 
        {
            match constraint_kind{
                ConstraintKind::Distance => {
                    Self::solve_one_distance(*a as usize, *b as usize, *rest_x, *stiffness, particles);
                },
                ConstraintKind::Position => {
                    Self::solve_one_distance_position(*a as usize, *b as usize, *rest_x, *rest_y, *rest_z, *stiffness, particles);
                },
                ConstraintKind::None => {
                    // nothing to do
                },
            }
        }
    }

    fn solve_one_distance(
        a: usize,
        b: usize,
        rest: f32,
        stiffness: f32,
        particles: &mut VerletParticles,
    )
    {
        let dx = particles.x[b] - particles.x[a];
        let dy = particles.y[b] - particles.y[a];
        let dz = particles.z[b] - particles.z[a];

        let dist_sq = dx * dx + dy * dy + dz * dz;

        if dist_sq <= f32::EPSILON {
            return;
        }

        let dist = dist_sq.sqrt();


        let error = (dist - rest) / dist;
        let correction = error * stiffness;

        let wa = particles.inv_mass[a];
        let wb = particles.inv_mass[b];

        let weight = wa + wb;

        if weight <= f32::EPSILON {
            return;
        }

        let a_weight = wa / weight;
        let b_weight = wb / weight;

        let cx = dx * correction;
        let cy = dy * correction;
        let cz = dz * correction;

        particles.x[a] += cx * a_weight;
        particles.y[a] += cy * a_weight;
        particles.z[a] += cz * a_weight;

        particles.x[b] -= cx * b_weight;
        particles.y[b] -= cy * b_weight;
        particles.z[b] -= cz * b_weight;
    }

    fn solve_one_distance_position(
        a: usize,
        b: usize,
        rest_x: f32,
        rest_y: f32,
        rest_z: f32,
        stiffness: f32,
        particles: &mut VerletParticles,
    ) {
        // Desired position of B relative to A.
        let target_x = particles.x[a] + rest_x;
        let target_y = particles.y[a] + rest_y;
        let target_z = particles.z[a] + rest_z;

        // Error between current B and desired B.
        let dx = particles.x[b] - target_x;
        let dy = particles.y[b] - target_y;
        let dz = particles.z[b] - target_z;

        let wa = particles.inv_mass[a];
        let wb = particles.inv_mass[b];

        let weight = wa + wb;

        if weight <= f32::EPSILON {
            return;
        }

        // Move A and B proportionally to their inverse masses.
        let a_weight = wa / weight;
        let b_weight = wb / weight;

        let correction_x = dx * stiffness;
        let correction_y = dy * stiffness;
        let correction_z = dz * stiffness;

        particles.x[a] += correction_x * a_weight;
        particles.y[a] += correction_y * a_weight;
        particles.z[a] += correction_z * a_weight;

        particles.x[b] -= correction_x * b_weight;
        particles.y[b] -= correction_y * b_weight;
        particles.z[b] -= correction_z * b_weight;
    }

    fn assert_bounds_invariants(&self) {
        let len = self.len();

        assert!(self.a.len() == len);
        assert!(self.b.len() == len);

        assert!(self.rest_x.len() == len);
        assert!(self.rest_y.len() == len);
        assert!(self.rest_z.len() == len);

        assert!(self.stiffness.len() == len);
        assert!(self.constrain_kind.len() == len);
    }

    pub fn len(&self) -> usize {
        self.a.len()
    }

    pub fn push_constraint_distance(&mut self,
        a: usize,
        b: usize,
        rest: f32,
        stiffness: f32,
    ) -> usize {
        self.assert_bounds_invariants();

        let index = self.len();

        self.a.push(a as u32);
        self.b.push(b as u32);
        self.rest_x.push(rest);
        self.rest_y.push(rest);
        self.rest_z.push(rest);
        self.stiffness.push(stiffness);
        self.constrain_kind.push(ConstraintKind::Distance);

        index
    }

    pub fn push_constraint_position(&mut self,
        a: usize,
        b: usize,
        rest: Vec3,
        stiffness: f32,
    ) -> usize {
        self.assert_bounds_invariants();

        let index = self.len();

        self.a.push(a as u32);
        self.b.push(b as u32);
        self.rest_x.push(rest.x);
        self.rest_y.push(rest.y);
        self.rest_z.push(rest.z);
        self.stiffness.push(stiffness);
        self.constrain_kind.push(ConstraintKind::Distance);

        index
    }

    pub fn push_constraint_none(&mut self,
        a: usize,
        b: usize,
    ) -> usize {
        self.assert_bounds_invariants();

        let index = self.len();

        self.a.push(a as u32);
        self.b.push(b as u32);
        self.rest_x.push(0.0);
        self.rest_y.push(0.0);
        self.rest_z.push(0.0);
        self.stiffness.push(0.0);
        self.constrain_kind.push(ConstraintKind::None);

        index
    }
}

#[derive(Clone)]
enum ConstraintKind {
    Distance,
    Position,
    None,
}