//! Rigid body extension for verlet velocity integrator

use std::iter::zip;

use cgmath::{InnerSpace, Matrix, Matrix3, Quaternion, Rad, Rotation, Rotation3, SquareMatrix, Vector3, Zero};

use crate::physics::verlet_velocity::verlet_velocity_particles::VerletVelocityParticles;

type Vec3 = Vector3<f32>;
type Mat3 = Matrix3<f32>;
type Quat = Quaternion<f32>;

/// A rigid body composed of particles in a Verlet physics world.
#[derive(Clone, Debug)]
pub struct RigidBody {
    /// Particle indices belonging to this body.
    pub particles: Vec<usize>,
    pub local_positions: Vec<Vec3>,

    /// Center of mass in world space.
    pub position: Vec3,

    /// Orientation relative to world space (normalized).
    pub orientation: Quat,

    /// Linear velocity of the center of mass.
    pub linear_velocity: Vec3,

    /// Angular velocity in world space (radians/sec).
    pub angular_velocity: Vec3,

    /// Total mass.
    pub mass: f32,

    /// Inverse mass. Zero means static.
    pub inv_mass: f32,

    /// Inertia tensor in body-local coordinates.
    pub inertia_local: Mat3,

    /// Inverse inertia tensor in body-local coordinates.
    pub inv_inertia_local: Mat3,

    /// Accumulated external force in world space.
    pub force: Vec3,

    /// Accumulated external torque in world space.
    pub torque: Vec3,
}

impl RigidBody {
    pub fn new(
        position: Vec3,
        mass: f32,
        inertia_local: Mat3,
    ) -> Self {
        assert!(mass >= 0.0);

        let inv_mass = if mass > 0.0 {
            1.0 / mass
        } else {
            0.0
        };

        let inv_inertia_local = if mass > 0.0 {
            // For a valid, nonsingular inertia tensor.
            inertia_local
                .invert()
                .unwrap_or_else(Mat3::zero)
        } else {
            Mat3::zero()
        };

        Self {
            particles: Vec::new(),
            local_positions: Vec::new(),
            position,
            orientation: Quat::new(1.0, 0.0, 0.0, 0.0),
            linear_velocity: Vec3::new(0.0, 0.0, 0.0),
            angular_velocity: Vec3::new(0.0, 0.0, 0.0),
            mass,
            inv_mass,
            inertia_local,
            inv_inertia_local,
            force: Vec3::zero(),
            torque: Vec3::zero(),
        }
    }

    /// Integrates the rigid body's linear and angular motion.
    ///
    /// Uses semi-implicit Euler integration.
    pub fn integrate(&mut self, dt: f32) {
        if self.is_static() {
            self.force = Vec3::zero();
            self.torque = Vec3::zero();
            return;
        }

        // Linear motion.
        let acceleration = self.force * self.inv_mass;

        self.linear_velocity += acceleration * dt;
        self.position += self.linear_velocity * dt;

        // World-space inertia tensor.
        let rotation = Mat3::from(self.orientation);
        let inertia_world =
            rotation * self.inertia_local * rotation.transpose();

        let inv_inertia_world =
            rotation * self.inv_inertia_local * rotation.transpose();

        // Euler's rigid-body rotational equation:
        // torque = I * alpha + omega x (I * omega)
        let angular_momentum =
            inertia_world * self.angular_velocity;

        let gyroscopic_torque =
            self.angular_velocity.cross(angular_momentum);

        let angular_acceleration =
            inv_inertia_world * (self.torque - gyroscopic_torque);

        self.angular_velocity += angular_acceleration * dt;

        self.update_orientation(dt);

        // Clear accumulators for the next simulation step.
        self.force = Vec3::zero();
        self.torque = Vec3::zero();
    }

    /// Integrates orientation using world-space angular velocity.
    pub fn update_orientation(&mut self, dt: f32) {
        let speed = self.angular_velocity.magnitude();

        if speed < 1e-7 {
            return;
        }

        let axis = self.angular_velocity / speed;
        let angle = Rad(speed * dt);

        let delta = Quat::from_axis_angle(axis, angle);

        // Angular velocity is expressed in world space.
        self.orientation =
            (delta * self.orientation).normalize();
    }

    pub fn update_particles(
        &self,
        particles: &mut VerletVelocityParticles,
    ) {
        assert_eq!(self.particles.len(), self.local_positions.len());

        for (&index, &local_pos) in
            self.particles.iter().zip(self.local_positions.iter())
        {
            let world_pos =
                self.position + self.orientation.rotate_vector(local_pos);

            particles.set_pos(index, world_pos);
        }
    }

    /// Accumulates forces and torques from the attached particles.
    ///
    /// Assumes particle accelerations represent external accelerations
    /// and that particle masses sum to the rigid body's mass.
    pub fn apply_forces(
        &mut self,
        particles: &VerletVelocityParticles,
    ) {
        self.force = Vec3::zero();
        self.torque = Vec3::zero();

        for (&index, &local_pos) in
            self.particles.iter().zip(self.local_positions.iter())
        {
            let mass = particles.mass[index];

            if mass <= 0.0 {
                continue;
            }

            // Convert particle acceleration into force.
            let particle_force =
                particles.get_acc(index) * mass;

            // Lever arm in world space.
            let world_offset =
                self.orientation.rotate_vector(local_pos);

            self.force += particle_force;
            self.torque += world_offset.cross(particle_force);
        }
    }

    pub fn attach_particle(
        &mut self,
        particles: &mut VerletVelocityParticles,
        index: usize,
    ) {
        let world_pos = particles.get_pos(index);
        let local_pos = self.orientation.conjugate()
            .rotate_vector(world_pos - self.position);

        self.particles.push(index);
        self.local_positions.push(local_pos);

        particles.is_rigid_body[index] = true;
    }

    pub fn is_static(&self) -> bool {
        self.inv_mass == 0.0
    }
}
