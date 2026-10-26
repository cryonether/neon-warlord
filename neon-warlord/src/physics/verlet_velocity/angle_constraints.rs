//! An angle constraint between three particles

use cgmath::InnerSpace;

use crate::physics::verlet_velocity::verlet_velocity_particles::VerletVelocityParticles;

#[derive(Clone)]
pub struct AngleConstraints {
    pub a: Vec<usize>,
    pub b: Vec<usize>,
    pub c: Vec<usize>,

    pub angle: Vec<f32>,
    pub stiffness: Vec<f32>,
    pub damping: Vec<f32>,
}

impl AngleConstraints {
    pub fn new() -> Self {
        Self {
            a: Vec::new(),
            b: Vec::new(),
            c: Vec::new(),
            angle: Vec::new(),
            stiffness: Vec::new(),
            damping: Vec::new(),
        }
    }

    /// Models an angle constraint between three particles
    pub fn apply_angle_constraints(&mut self, particles: &mut VerletVelocityParticles) {
        for i in 0..self.len() {
            let index_a = self.a[i];
            let index_b = self.b[i];
            let index_c = self.c[i];

            let target_angle = self.angle[i];
            let stiffness = self.stiffness[i];
            let damping = self.damping[i];

            let pos_a = particles.get_pos(index_a);
            let pos_b = particles.get_pos(index_b);
            let pos_c = particles.get_pos(index_c);

            let vel_a = particles.get_vel_half(index_a);
            let vel_b = particles.get_vel_half(index_b);
            let vel_c = particles.get_vel_half(index_c);

            let u = pos_a - pos_b;
            let v = pos_c - pos_b;

            let len_u = u.magnitude();
            let len_v = v.magnitude();

            if len_u < 1e-6 || len_v < 1e-6 {
                continue;
            }

            let u_hat = u / len_u;
            let v_hat = v / len_v;

            let cos_angle = u_hat.dot(v_hat).clamp(-1.0, 1.0);
            let sin_angle = (1.0 - cos_angle * cos_angle).sqrt();

            // The angle gradient is singular at 0 and PI.
            if sin_angle < 1e-5 {
                continue;
            }

            let angle = cos_angle.acos();

            // Gradients of cos(angle) with respect to each position.
            let grad_a = -(v_hat - cos_angle * u_hat) / (len_u * sin_angle);
            let grad_c = -(u_hat - cos_angle * v_hat) / (len_v * sin_angle);
            let grad_b = -(grad_a + grad_c);

            let angular_velocity = grad_a.dot(vel_a) + grad_b.dot(vel_b) + grad_c.dot(vel_c);
            let torque = -stiffness * (angle - target_angle) - damping * angular_velocity;

            let force_a = torque * grad_a;
            let force_b = torque * grad_b;
            let force_c = torque * grad_c;

            let mut acc_a = particles.get_acc(index_a);
            let mut acc_b = particles.get_acc(index_b);
            let mut acc_c = particles.get_acc(index_c);

            // Apply forces as accelerations using inverse masses.
            acc_a += force_a * particles.inv_mass[index_a];
            acc_b += force_b * particles.inv_mass[index_b];
            acc_c += force_c * particles.inv_mass[index_c];

            particles.set_acc(index_a, acc_a);
            particles.set_acc(index_b, acc_b);
            particles.set_acc(index_c, acc_c);
        }
    }

    pub fn push(
        &mut self,
        a: usize,
        b: usize,
        c: usize,
        angle: f32,
        stiffness: f32,
        damping: f32,
    ) -> usize {
        let index = self.len();

        self.a.push(a);
        self.b.push(b);
        self.c.push(c);

        self.angle.push(angle);
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
        assert!(self.c.len() == len);
        assert!(self.angle.len() == len);
        assert!(self.stiffness.len() == len);
        assert!(self.damping.len() == len);
    }
}
