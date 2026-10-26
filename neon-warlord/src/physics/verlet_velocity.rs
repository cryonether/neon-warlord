//! Velocity Verlet integration physics
//!
//! https://www.algorithm-archive.org/contents/verlet_integration/verlet_integration.html
//!

pub mod angle_constraints;
pub mod angle_motor_constraints;
pub mod distance_constraints;
pub mod verlet_velocity_drawer;
pub mod verlet_velocity_particles;

use cgmath::{InnerSpace, MetricSpace};

use crate::physics::verlet_velocity::{
    angle_constraints::AngleConstraints, angle_motor_constraints::AngleMotorConstraints,
    distance_constraints::DistanceConstraints, verlet_velocity_particles::VerletVelocityParticles,
};

type Vec3 = cgmath::Vector3<f32>;

/// Velocity Verlet integration physics
#[derive(Clone)]
pub struct VerletVelocityPhysics {
    pub particles: VerletVelocityParticles,

    // constraints
    distance_constraints: DistanceConstraints,
    angle_constraints: AngleConstraints,
    angle_motor_constraints: AngleMotorConstraints,
}

impl VerletVelocityPhysics {
    pub fn new() -> Self {
        let particles = VerletVelocityParticles::new();

        let distance_constraints = DistanceConstraints::new();

        let angle_constraints = AngleConstraints::new();

        let angle_motor_constraints = AngleMotorConstraints::new();

        Self {
            particles,
            distance_constraints,
            angle_constraints,
            angle_motor_constraints,
        }
    }

    /// Run a single step
    pub fn step(&mut self) {
        // running at fixed step size
        const DT: f32 = 1.0 / 60.0;
        const SUB_STEPS: usize = 16;

        let sub_dt = DT / SUB_STEPS as f32;

        for _i in 0..SUB_STEPS {
            // Update the position using the old velocity and acceleration
            self.particles.integrate_position(sub_dt);

            self.solve_ground_collision();

            self.particles.zero_acc();

            // apply_external_accelerations(&mut particles);
            self.apply_gravity();
            self.distance_constraints
                .apply_distance_constraints(&mut self.particles);
            self.angle_constraints
                .apply_angle_constraints(&mut self.particles);
            self.angle_motor_constraints
                .apply_angle_motor_constraints(&mut self.particles);

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

    /// Collision with the ground plane at z == 0.0
    fn solve_ground_collision(&mut self) {
        const GROUND_HEIGHT_Z: f32 = 0.0;
        const RESTITUTION: f32 = 0.9;
        // const FRICTION: f32 = 0.8;

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
                vel.z += 0.2;

                // vel.z = -RESTITUTION * vel.z;

                // // Reduce tangential velocity due to friction.
                // vel.x *= 1.0 - FRICTION;
                // vel.y *= 1.0 - FRICTION;
            }

            self.particles.set_pos(i, pos);
            self.particles.set_vel_half(i, vel);
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
        self.distance_constraints
            .push(a, b, rest, stiffness, damping)
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
        self.angle_constraints
            .push(a, b, c, angle, stiffness, damping)
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

    #[allow(clippy::too_many_arguments)]
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
        self.angle_motor_constraints.push(
            a, b, c, min_angle, max_angle, max_torque, stiffness, damping,
        )
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
}
