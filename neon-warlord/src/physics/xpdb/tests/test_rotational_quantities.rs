//! Test of RotationalQuantities

use cgmath::SquareMatrix;

use super::super::*;

const EPSILON: f32 = 1e-5;

#[test]
fn test_integrate_orientation() {
    let dt = 0.1;
    let angular_velocity = Vec3::new(0.0, 0.0, 1.0);

    let mut quantities = RotationalQuantities {
        orientation: Quad::new(1.0, 0.0, 0.0, 0.0),
        orientation_prev: Quad::new(1.0, 0.0, 0.0, 0.0),
        angular_velocity,
        inertia_tensor: Mat3::from_value(1.0),
        inverse_inertia_tensor: Mat3::from_value(1.0),
    };

    quantities.integrate_orientation(dt);

    // Previous orientation should be saved.
    assert!(
        (quantities.orientation_prev.v
            - Vec3::zero()).magnitude() < EPSILON
    );
    assert!((quantities.orientation_prev.s - 1.0).abs() < EPSILON);

    // With zero external torque and isotropic inertia,
    // angular velocity should remain constant.
    assert!(
        (quantities.angular_velocity - angular_velocity).magnitude() < EPSILON
    );

    // First-order quaternion integration:
    // q_new = normalize(q_old + 0.5 * dt * omega_quat * q_old)
    let expected = Quad::new(
        1.0,
        0.0,
        0.0,
        0.5 * dt,
    );
    let expected = expected / expected.magnitude();

    assert!(
        (quantities.orientation.s - expected.s).abs() < EPSILON
    );
    assert!(
        (quantities.orientation.v - expected.v).magnitude() < EPSILON
    );

    // Orientation should remain normalized.
    assert!((quantities.orientation.magnitude() - 1.0).abs() < EPSILON);
}

#[test]
fn test_integrate_angular_velocity() {
    let dt = 0.1;

    let orientation_prev = Quad::new(1.0, 0.0, 0.0, 0.0);
    let angle = 0.2_f32;
    let half_angle = angle / 2.0;

    // Rotation of angle radians around the Z axis.
    let orientation = Quad::new(
        half_angle.cos(),
        0.0,
        0.0,
        half_angle.sin(),
    );

    let mut quantities = RotationalQuantities {
        orientation,
        orientation_prev,
        angular_velocity: Vec3::zero(),
        inertia_tensor: Mat3::from_value(1.0),
        inverse_inertia_tensor: Mat3::from_value(1.0),
    };

    quantities.integrate_angular_velocity(dt);

    // The implementation approximates angular velocity as
    // 2 * quaternion_vector / dt.
    let expected = Vec3::new(
        0.0,
        0.0,
        2.0 * half_angle.sin() / dt,
    );

    assert!(
        (quantities.angular_velocity - expected).magnitude() < EPSILON
    );
}

#[test]
fn test_steps() {
    let dt = 0.01;
    let steps = 10;

    let initial_orientation = Quad::new(1.0, 0.0, 0.0, 0.0);
    let initial_angular_velocity = Vec3::new(0.0, 0.0, 1.0);

    let mut quantities = RotationalQuantities {
        orientation: initial_orientation,
        orientation_prev: initial_orientation,
        angular_velocity: initial_angular_velocity,
        inertia_tensor: Mat3::from_value(1.0),
        inverse_inertia_tensor: Mat3::from_value(1.0),
    };

    for _ in 0..steps {
        quantities.integrate_orientation(dt);
        quantities.integrate_angular_velocity(dt);
    }

    // Orientation should remain normalized after repeated steps.
    assert!(
        (quantities.orientation.magnitude() - 1.0).abs() < EPSILON
    );

    // For isotropic inertia and zero external torque, rotation
    // should remain around the Z axis.
    assert!(quantities.orientation.v.x.abs() < EPSILON);
    assert!(quantities.orientation.v.y.abs() < EPSILON);

    // Angular velocity should remain finite and point along Z.
    assert!(quantities.angular_velocity.x.abs() < EPSILON);
    assert!(quantities.angular_velocity.y.abs() < EPSILON);
    assert!(quantities.angular_velocity.z.is_finite());
    assert!(quantities.angular_velocity.z > 0.0);
}