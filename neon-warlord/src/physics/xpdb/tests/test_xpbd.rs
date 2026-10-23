//! Test of xpbd physics

use super::super::*;
use cgmath::{Quaternion, SquareMatrix, Vector3};

const EPSILON: f32 = 1e-4;

/// Creates one rigid body without any constraints and
/// simulates multiple simulation steps.
#[test]
fn test_steps_one_rigid_body() {
    let initial_pos = Vec3::new(0.0, 0.0, 10.0);

    let body = XpbdRigidBody {
        p: PositionalQuantities {
            pos: initial_pos,
            pos_prev: initial_pos,
            vel: Vec3::zero(),
            mass: 1.0,
            inv_mass: 1.0,
            compliance: 0.0,
        },
        r: RotationalQuantities {
            orientation: Quad::new(1.0, 0.0, 0.0, 0.0),
            orientation_prev: Quad::new(1.0, 0.0, 0.0, 0.0),
            angular_velocity: Vec3::zero(),
            inertia_tensor: Mat3::from_value(1.0),
            inverse_inertia_tensor: Mat3::from_value(1.0),
        },
    };

    let mut xpbd = Xpbd::new();
    xpbd.rigid_bodies.push(body);

    let initial_orientation = xpbd.rigid_bodies[0].r.orientation;

    // Simulate one second (60 frames at 60 Hz).
    for _ in 0..60 {
        xpbd.step();
    }

    let body = &xpbd.rigid_bodies[0];

    // The body should fall due to gravity.
    assert!(
        body.p.pos.z < initial_pos.z,
        "Body should fall: initial z = {}, final z = {}",
        initial_pos.z,
        body.p.pos.z
    );

    // Horizontal position should remain unchanged.
    assert!(
        (body.p.pos.x - initial_pos.x).abs() < EPSILON
    );
    assert!(
        (body.p.pos.y - initial_pos.y).abs() < EPSILON
    );

    // Orientation should remain normalized.
    assert!(
        (body.r.orientation.magnitude() - 1.0).abs() < EPSILON
    );

    // No torque or initial angular velocity means no rotation.
    assert!(
        (body.r.angular_velocity - Vec3::zero()).magnitude() < EPSILON
    );

    assert!(
        (body.r.orientation.s - initial_orientation.s).abs() < EPSILON
    );
    assert!(
        (body.r.orientation.v - initial_orientation.v).magnitude() < EPSILON
    );
}