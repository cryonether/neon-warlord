//! A distance constraint between two particles

use cgmath::InnerSpace;

use crate::physics::verlet_velocity::verlet_velocity_particles::VerletVelocityParticles;

#[derive(Clone)]
pub struct DistanceConstraints {
    pub a: Vec<usize>,
    pub b: Vec<usize>,

    pub distance: Vec<f32>,
    pub stiffness: Vec<f32>,
    pub damping: Vec<f32>,
}

impl DistanceConstraints {
    pub fn new() -> Self {
        Self {
            a: Vec::new(),
            b: Vec::new(),
            distance: Vec::new(),
            stiffness: Vec::new(),
            damping: Vec::new(),
        }
    }

    /// Models distance constraints as springs
    pub fn apply_distance_constraints(&mut self, particles: &mut VerletVelocityParticles) {
        for i in 0..self.len() {
            let index_a = self.a[i];
            let index_b = self.b[i];
            let rest_length = self.distance[i];
            let stiffness = self.stiffness[i];
            let damping = self.damping[i];

            let inv_mass_a = particles.inv_mass[index_a];
            let inv_mass_b = particles.inv_mass[index_b];

            // Both particles are static.
            if inv_mass_a == 0.0 && inv_mass_b == 0.0 {
                continue;
            }

            // Particle a
            let pos_a = particles.get_pos(index_a);
            let vel_a = particles.get_vel_half(index_a);
            let mut acc_a = particles.get_acc(index_a);

            // Particle b
            let pos_b = particles.get_pos(index_b);
            let vel_b = particles.get_vel_half(index_b);
            let mut acc_b = particles.get_acc(index_b);

            let delta = pos_b - pos_a;
            let distance = delta.magnitude();

            // Avoid division by zero
            if distance < 1e-6 {
                continue;
            }

            let direction = delta / distance;
            let extension = distance - rest_length;

            let relative_speed = (vel_b - vel_a).dot(direction); // half velocity is used as approximation for damping

            let force_magnitude = stiffness * extension + damping * relative_speed;

            let force = direction * force_magnitude;

            // Acceleration = force / mass.
            // inv_mass == 0 for static particles.
            acc_a += force * inv_mass_a;

            acc_b -= force * inv_mass_b;

            particles.set_acc(index_a, acc_a);
            particles.set_acc(index_b, acc_b);
        }
    }

    pub fn push(&mut self, a: usize, b: usize, rest: f32, stiffness: f32, damping: f32) -> usize {
        let index = self.len();

        self.a.push(a);
        self.b.push(b);
        self.distance.push(rest);
        self.stiffness.push(stiffness);
        self.damping.push(damping);

        index
    }

    pub fn len(&self) -> usize {
        self.assert_bounds_invariants();

        self.a.len()
    }

    fn assert_bounds_invariants(&self) {
        let len = self.a.len();

        assert!(self.a.len() == len);
        assert!(self.b.len() == len);
        assert!(self.distance.len() == len);
        assert!(self.stiffness.len() == len);
        assert!(self.damping.len() == len);
    }
}
