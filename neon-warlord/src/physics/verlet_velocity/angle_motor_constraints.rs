//! A moveable angle spring

use cgmath::InnerSpace;

use crate::physics::verlet_velocity::verlet_velocity_particles::VerletVelocityParticles;


#[derive(Clone)]
pub struct AngleMotorConstraints {
    pub a: Vec<usize>,
    pub b: Vec<usize>,
    pub c: Vec<usize>,

    pub min_angle: Vec<f32>,
    pub max_angle: Vec<f32>,
    pub max_torque: Vec<f32>,

    pub stiffness: Vec<f32>,
    pub damping: Vec<f32>,

    // Externally commanded torque for each joint.
    pub external_torque: Vec<f32>,

    // status variables
    pub angle: Vec<f32>,
    pub angular_velocity: Vec<f32>,
}

impl AngleMotorConstraints {
    pub fn new() -> Self {
        Self {
            a:  Vec::new(),
            b:  Vec::new(),
            c:  Vec::new(),
            min_angle:  Vec::new(),
            max_angle:  Vec::new(),
            max_torque:  Vec::new(),
            stiffness:  Vec::new(),
            damping:  Vec::new(),
            external_torque:  Vec::new(),
            angle: Vec::new(),
            angular_velocity: Vec::new(),
        }
    }


    /// Models an angle constraint between three particles
    pub fn apply_angle_motor_constraints(&mut self,  particles: &mut VerletVelocityParticles) {
        for i in 0..self.len() {
            let index_a = self.a[i];
            let index_b = self.b[i];
            let index_c = self.c[i];

            let min_angle = self.min_angle[i];
            let max_angle = self.max_angle[i];
            let max_torque = self.max_torque[i];
            let stiffness = self.stiffness[i];
            let damping = self.damping[i];
            let external_torque = self.external_torque[i];

            let pos_a = particles.get_pos(index_a);
            let pos_b = particles.get_pos(index_b);
            let pos_c = particles.get_pos(index_c);

            let vel_a = particles.get_vel_half(index_a);
            let vel_b = particles.get_vel_half(index_b);
            let vel_c = particles.get_vel_half(index_c);

            // Vectors pointing away from the central particle.
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

            // The angle gradient is singular near 0 and PI.
            if sin_angle < 1e-5 {
                continue;
            }

            let angle = cos_angle.acos();

            // Gradients of the angle with respect to the particle positions.
            let grad_a = -(v_hat - cos_angle * u_hat) / (len_u * sin_angle);

            let grad_c = -(u_hat - cos_angle * v_hat) / (len_v * sin_angle);

            let grad_b = -(grad_a + grad_c);

            // Angular velocity, computed without storing it.
            let angular_velocity = grad_a.dot(vel_a) + grad_b.dot(vel_b) + grad_c.dot(vel_c);

            // Motor command, clamped to the permitted torque.
            let motor_torque = external_torque.clamp(-max_torque, max_torque);

            // Soft mechanical stops.
            let limit_torque = if angle < min_angle {
                stiffness * (min_angle - angle) - damping * angular_velocity.min(0.0)
            } else if angle > max_angle {
                -stiffness * (angle - max_angle) - damping * angular_velocity.max(0.0)
            } else {
                0.0
            };

            let torque = motor_torque + limit_torque;

            let force_a = torque * grad_a;
            let force_b = torque * grad_b;
            let force_c = torque * grad_c;

            let mut acc_a = particles.get_acc(index_a);
            let mut acc_b = particles.get_acc(index_b);
            let mut acc_c = particles.get_acc(index_c);

            acc_a += force_a * particles.inv_mass[index_a];
            acc_b += force_b * particles.inv_mass[index_b];
            acc_c += force_c * particles.inv_mass[index_c];

            particles.set_acc(index_a, acc_a);
            particles.set_acc(index_b, acc_b);
            particles.set_acc(index_c, acc_c);

            self.angle[i] = angle;
            self.angular_velocity[i] = angular_velocity;
        }
    }

    pub fn push(
        &mut self,
        a: usize,
        b: usize,
        c: usize,
        min_angle: f32,
        max_angle: f32,
        max_torque: f32,
        stiffness: f32,
        damping: f32,
    ) -> usize {
        let index = self.len();

        self.a.push(a);
        self.b.push(b);
        self.c.push(c);

        self.min_angle.push(min_angle);
        self.max_angle.push(max_angle);
        self.max_torque.push(max_torque);
        self.stiffness.push(stiffness);
        self.damping.push(damping);

        self.external_torque.push(0.0);
        self.angle.push(0.0);
        self.angular_velocity.push(0.0);

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

        assert!(self.min_angle.len() == len);
        assert!(self.max_angle.len() == len);
        assert!(self.max_torque.len() == len);

        assert!(self.stiffness.len() == len);
        assert!(self.damping.len() == len);

        assert!(self.external_torque.len() == len);

        assert!(self.angle.len() == len);
        assert!(self.angular_velocity.len() == len);
    }
}
