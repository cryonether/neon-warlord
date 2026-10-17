//! Velocity Verlet integration physics
//!
//! https://www.algorithm-archive.org/contents/verlet_integration/verlet_integration.html
//!

pub mod verlet_velocity_drawer;
pub mod verlet_velocity_particles;

use cgmath::{InnerSpace, MetricSpace, Zero};

use crate::physics::verlet_velocity::verlet_velocity_particles::VerletVelocityParticles;

type Vec3 = cgmath::Vector3<f32>;

/// Velocity Verlet integration physics
#[derive(Clone)]
pub struct VerletVelocity {
    pub particles: VerletVelocityParticles,

    // constraints
    distance_constraints: DistanceConstraints,
    angle_constraints: AngleConstraints,
    angle_motor_constraints: AngleMotorConstraints,
}

impl VerletVelocity {
    pub fn new(

    ) -> Self {
        let particles = VerletVelocityParticles::new();

        let distance_constraints = DistanceConstraints {
            a:  Vec::new(),
            b:  Vec::new(),
            distance:  Vec::new(),
            stiffness:  Vec::new(),
            damping:  Vec::new(),
        };
        let angle_constraints = AngleConstraints {
            a:  Vec::new(),
            b:  Vec::new(),
            c:  Vec::new(),
            angle: Vec::new(),
            stiffness: Vec::new(),
            damping: Vec::new(),
        };
        let angle_motor_constraints = AngleMotorConstraints {
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
        };

        Self {
            particles,
            distance_constraints,
            angle_constraints,
            angle_motor_constraints,
        }
    }

    pub fn push_constraint_distance(
        &mut self,
        a: usize,
        b: usize,
        rest: f32,
        stiffness: f32,
        damping: f32,
    ) -> usize {
        let index = self.distance_constraints.len();
        
        self.distance_constraints.a.push(a);
        self.distance_constraints.b.push(b);
        self.distance_constraints.distance.push(rest);
        self.distance_constraints.stiffness.push(stiffness);
        self.distance_constraints.damping.push(damping);

        index
    }

    pub fn push_constraint_distance_infer(
        &mut self,
        a: usize,
        b: usize,
        stiffness: f32,
        damping: f32,
    ) -> usize {
        let pos_a = self.particles.get_pos(a);
        let pos_b = self.particles.get_pos(b);
        let distance = pos_a.distance(pos_b);

        self.push_constraint_distance(a, b, distance, stiffness, damping)
    }

    pub fn push_constraint_angle(
        &mut self,
        a: usize,
        b: usize,
        c: usize,
        angle: f32,
        stiffness: f32,
        damping: f32,
    ) -> usize {
        let index = self.angle_constraints.len();

        self.angle_constraints.a.push(a);
        self.angle_constraints.b.push(b);
        self.angle_constraints.c.push(c);

        self.angle_constraints.angle.push(angle);
        self.angle_constraints.stiffness.push(stiffness);
        self.angle_constraints.damping.push(damping);

        index
    }

    pub fn push_constraint_angle_infer(
        &mut self,
        a: usize,
        b: usize,
        c: usize,
        stiffness: f32,
        damping: f32,
    ) -> usize {
        let pos_a = self.particles.get_pos(a);
        let pos_b = self.particles.get_pos(b);
        let pos_c = self.particles.get_pos(c);

        let ab = pos_a - pos_b;
        let cb = pos_c - pos_b;

        let len_ab = ab.magnitude();
        let len_cb = cb.magnitude();

        assert!(
            len_ab > 1e-6 && len_cb > 1e-6,
            "Cannot infer angle from coincident particles"
        );

        let cos_angle = (ab.dot(cb) / (len_ab * len_cb)).clamp(-1.0, 1.0);
        let angle = cos_angle.acos();

        self.push_constraint_angle(a, b, c, angle, stiffness, damping)
    }

    pub fn push_constraint_angle_motor(
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
        let index = self.angle_motor_constraints.len();

        self.angle_motor_constraints.a.push(a);
        self.angle_motor_constraints.b.push(b);
        self.angle_motor_constraints.c.push(c);

        self.angle_motor_constraints.min_angle.push(min_angle);
        self.angle_motor_constraints.max_angle.push(max_angle);
        self.angle_motor_constraints.max_torque.push(max_torque);
        self.angle_motor_constraints.stiffness.push(stiffness);
        self.angle_motor_constraints.damping.push(damping);

        self.angle_motor_constraints.external_torque.push(0.0);
        self.angle_motor_constraints.angle.push(0.0);
        self.angle_motor_constraints.angular_velocity.push(0.0);

        index
    }

    pub fn set_external_torque(&mut self, index: usize, value: f32) {
        self.angle_motor_constraints.external_torque[index] = value;
    }

    pub fn get_motor_angle(&mut self, index: usize) -> f32 {
        self.angle_motor_constraints.angle[index]
    }

    pub fn get_motor_angular_velocity(&mut self, index: usize) -> f32 {
        self.angle_motor_constraints.angular_velocity[index]
    }

    /// Run a single step
    pub fn step(&mut self) {
        // running at fixed step size
        const DT: f32 = 1.0 / 60.0;
        const SUB_STEPS: usize = 8;

        let sub_dt = DT / SUB_STEPS as f32;

        for i in 0..SUB_STEPS
        {
            // Update the position using the old velocity and acceleration
            self.particles.integrate_position(sub_dt);

            self.solve_ground_collision();

            self.particles.zero_acc();

            // Recalculate the acceleration at the new position
            self.apply_gravity();
            self.apply_distance_constraints();
            self.apply_angle_constraints();
            self.apply_angle_motor_constraints();

            // Update the position using the old velocity and acceleration
            self.particles.integrate_velocity(sub_dt);
        }
    }


    /// Apply gravity
    fn apply_gravity(&mut self) {
        const GRAVITY: Vec3 = Vec3::new(0.0, 0.0, -10.0);

        for i in 0..self.particles.len() {
            let mut acc = self.particles.get_acc(i);
            let inv_mass = self.particles.inv_mass[i];

            // static particles have inv_mass == 0
            if inv_mass > 0.0 {
                acc += GRAVITY;
            }

            self.particles.set_acc(i, acc);
        }
    }

    /// Models distance constraints as springs
    fn apply_distance_constraints(&mut self) {
        for i in 0..self.distance_constraints.len() {
            let index_a = self.distance_constraints.a[i];
            let index_b = self.distance_constraints.b[i];
            let rest_length = self.distance_constraints.distance[i];
            let stiffness = self.distance_constraints.stiffness[i];
            let damping = self.distance_constraints.damping[i];

            let inv_mass_a = self.particles.inv_mass[index_a];
            let inv_mass_b = self.particles.inv_mass[index_b];

            // Both particles are static.
            if inv_mass_a == 0.0 && inv_mass_b == 0.0 {
                continue;
            }

            // Particle a
            let pos_a = self.particles.get_pos(index_a);
            let vel_a = self.particles.get_vel_half(index_a);
            let mut acc_a = self.particles.get_acc(index_a);

            // Particle b
            let pos_b = self.particles.get_pos(index_b);
            let vel_b = self.particles.get_vel_half(index_b);
            let mut acc_b = self.particles.get_acc(index_b);

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

            self.particles.set_acc(index_a, acc_a);
            self.particles.set_acc(index_b, acc_b);
        }
    }

    /// Models an angle constraint between three particles
    fn apply_angle_constraints(&mut self) {
        for i in 0..self.angle_constraints.len() {
            let index_a = self.angle_constraints.a[i];
            let index_b = self.angle_constraints.b[i];
            let index_c = self.angle_constraints.c[i];

            let target_angle = self.angle_constraints.angle[i];
            let stiffness = self.angle_constraints.stiffness[i];
            let damping = self.angle_constraints.damping[i];

            let pos_a = self.particles.get_pos(index_a);
            let pos_b = self.particles.get_pos(index_b);
            let pos_c = self.particles.get_pos(index_c);

            let vel_a = self.particles.get_vel_half(index_a);
            let vel_b = self.particles.get_vel_half(index_b);
            let vel_c = self.particles.get_vel_half(index_c);

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

            let mut acc_a = self.particles.get_acc(index_a);
            let mut acc_b = self.particles.get_acc(index_b);
            let mut acc_c = self.particles.get_acc(index_c);

            // Apply forces as accelerations using inverse masses.
            acc_a += force_a * self.particles.inv_mass[index_a];
            acc_b += force_b * self.particles.inv_mass[index_b];
            acc_c += force_c * self.particles.inv_mass[index_c];

            self.particles.set_acc(index_a, acc_a);
            self.particles.set_acc(index_b, acc_b);
            self.particles.set_acc(index_c, acc_c);
        }
    }

    /// Models an angle constraint between three particles
    fn apply_angle_motor_constraints(&mut self) {
        for i in 0..self.angle_motor_constraints.len() {
            let index_a = self.angle_motor_constraints.a[i];
            let index_b = self.angle_motor_constraints.b[i];
            let index_c = self.angle_motor_constraints.c[i];

            let min_angle = self.angle_motor_constraints.min_angle[i];
            let max_angle = self.angle_motor_constraints.max_angle[i];
            let max_torque = self.angle_motor_constraints.max_torque[i];
            let stiffness = self.angle_motor_constraints.stiffness[i];
            let damping = self.angle_motor_constraints.damping[i];
            let external_torque = self.angle_motor_constraints.external_torque[i];

            let pos_a = self.particles.get_pos(index_a);
            let pos_b = self.particles.get_pos(index_b);
            let pos_c = self.particles.get_pos(index_c);

            let vel_a = self.particles.get_vel_half(index_a);
            let vel_b = self.particles.get_vel_half(index_b);
            let vel_c = self.particles.get_vel_half(index_c);

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

            let mut acc_a = self.particles.get_acc(index_a);
            let mut acc_b = self.particles.get_acc(index_b);
            let mut acc_c = self.particles.get_acc(index_c);

            acc_a += force_a * self.particles.inv_mass[index_a];
            acc_b += force_b * self.particles.inv_mass[index_b];
            acc_c += force_c * self.particles.inv_mass[index_c];

            self.particles.set_acc(index_a, acc_a);
            self.particles.set_acc(index_b, acc_b);
            self.particles.set_acc(index_c, acc_c);

            self.angle_motor_constraints.angle[i] = angle;
            self.angle_motor_constraints.angular_velocity[i] = angular_velocity;
        }
    }

    /// Collision with the ground plane at z == 0.0
    fn solve_ground_collision(&mut self) {
        const GROUND_HEIGHT_Z: f32 = 0.0;
        const RESTITUTION: f32 = 0.9;
        const FRICTION: f32 = 0.8;

        for i in 0..self.particles.len() {
            if self.particles.inv_mass[i] == 0.0 {
                // static particle
                continue;
            }

            let mut pos = self.particles.get_pos(i);
            let mut vel = self.particles.get_vel_half(i);

            let min_z = GROUND_HEIGHT_Z + self.particles.radius[i];

            if pos.z >= min_z {
                continue;
            }

            // Prevent penetration.
            pos.z = min_z;

            // Bounce only if moving into the ground.
            if vel.z < 0.0 {
                vel.x = vel.x + -RESTITUTION * vel.x;
                // vel.y = vel.x + -RESTITUTION * vel.x;
                vel.z = vel.z + 0.2;

                // vel.z = -RESTITUTION * vel.z;

                // // Reduce tangential velocity due to friction.
                // vel.x *= 1.0 - FRICTION;
                // vel.y *= 1.0 - FRICTION;
            }

            self.particles.set_pos(i, pos);
            self.particles.set_vel_half(i, vel);
        }
    }
}

#[derive(Clone)]
struct DistanceConstraints {
    pub a: Vec<usize>,
    pub b: Vec<usize>,

    pub distance: Vec<f32>,
    pub stiffness: Vec<f32>,
    pub damping: Vec<f32>,
}

impl DistanceConstraints {
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

#[derive(Clone)]
struct AngleConstraints {
    pub a: Vec<usize>,
    pub b: Vec<usize>,
    pub c: Vec<usize>,

    pub angle: Vec<f32>,
    pub stiffness: Vec<f32>,
    pub damping: Vec<f32>,
}

impl AngleConstraints {
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

#[derive(Clone)]
struct AngleMotorConstraints {
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
