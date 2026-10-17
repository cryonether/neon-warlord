//! Rigid body extension for verlet velocity integrator

use cgmath::{Matrix3, Quaternion, SquareMatrix, Vector3, Zero};

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

    /// Orientation relative to world space.
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
        }
    }

    pub fn is_static(&self) -> bool {
        self.inv_mass == 0.0
    }

    pub fn update
}
