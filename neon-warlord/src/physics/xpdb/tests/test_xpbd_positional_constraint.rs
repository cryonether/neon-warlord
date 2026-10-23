//! Tests for XpbdPositionalConstraint

use super::super::*;
use cgmath::{InnerSpace, SquareMatrix, Zero};

const EPSILON: f32 = 1e-5;

fn create_body(position: Vec3) -> XpbdRigidBody {
    XpbdRigidBody {
        p: PositionalQuantities {
            pos: position,
            pos_prev: position,
            vel: Vec3::zero(),
            mass: 1.0,
            inv_mass: 1.0,
            compliance: 0.0,
        },
        r: RotationalQuantities {
            orientation: Quad::new(1.0, 0.0, 0.0, 0.0),
            orientation_prev: Quad::new(1.0, 0.0, 0.0, 0.0),
            angular_velocity: Vec3::zero(),
            inertia_tensor: Mat3::from_diagonal(Vec3::new(1.0, 1.0, 1.0)),
            inverse_inertia_tensor: Mat3::from_diagonal(Vec3::new(1.0, 1.0, 1.0)),
        },
    }
}

#[test]
fn test_solve() {
    let mut body0 = create_body(Vec3::zero());
    let mut body1 = create_body(Vec3::new(2.0, 0.0, 0.0));

    let mut constraint = XpbdPositionalConstraint {
        relative_position_0: Vec3::zero(),
        relative_position_1: Vec3::new(2.0, 0.0, 0.0),
        compliance: 0.0,
        lagrange_multiplier: 0.0,
    };

    let dt = 1.0 / 60.0;

    constraint.solve(&mut body0, &mut body1, dt);

    // Equal inverse masses should split the positional correction equally.
    assert!(
        (body0.p.pos.x - 1.0).abs() < EPSILON,
        "body0 position: {:?}",
        body0.p.pos
    );

    assert!(
        (body1.p.pos.x - 1.0).abs() < EPSILON,
        "body1 position: {:?}",
        body1.p.pos
    );

    // The constraint should accumulate a nonzero Lagrange multiplier.
    assert!(constraint.lagrange_multiplier.abs() > 0.0);

    // No rotational correction should occur for lever arms parallel to
    // the correction direction.
    assert!(
        (body0.r.orientation.magnitude() - 1.0).abs() < EPSILON
    );
    assert!(
        (body1.r.orientation.magnitude() - 1.0).abs() < EPSILON
    );

    // The constraint force estimate should be finite.
    let force = constraint.get_force(dt);
    assert!(force.is_finite());
}

/// Verify that the lighter body moves farther than the heavier body.
#[test]
fn test_unequal_masses() {
    let mut body0 = create_body(Vec3::zero());
    let mut body1 = create_body(Vec3::new(2.0, 0.0, 0.0));

    // Body 0 is heavier than body 1.
    body0.p.mass = 2.0;
    body0.p.inv_mass = 0.5;

    body1.p.mass = 1.0;
    body1.p.inv_mass = 1.0;

    let initial_pos0 = body0.p.pos;
    let initial_pos1 = body1.p.pos;

    let mut constraint = XpbdPositionalConstraint {
        relative_position_0: Vec3::zero(),
        relative_position_1: Vec3::new(2.0, 0.0, 0.0),
        compliance: 0.0,
        lagrange_multiplier: 0.0,
    };

    constraint.solve(&mut body0, &mut body1, 1.0 / 60.0);

    let displacement0 = (body0.p.pos - initial_pos0).magnitude();
    let displacement1 = (body1.p.pos - initial_pos1).magnitude();

    assert!(
        displacement1 > displacement0,
        "Lighter body should move farther: heavy = {displacement0}, light = {displacement1}"
    );

    // Expected corrections are 2/3 for the heavier body
    // and 4/3 for the lighter body.
    assert!((displacement0 - 2.0 / 3.0).abs() < EPSILON);
    assert!((displacement1 - 4.0 / 3.0).abs() < EPSILON);
}

/// Verify that nonzero compliance reduces the positional correction.
#[test]
fn test_nonzero_compliance() {
    let mut body0 = create_body(Vec3::zero());
    let mut body1 = create_body(Vec3::new(2.0, 0.0, 0.0));

    let mut constraint = XpbdPositionalConstraint {
        relative_position_0: Vec3::zero(),
        relative_position_1: Vec3::new(2.0, 0.0, 0.0),
        compliance: 0.01,
        lagrange_multiplier: 0.0,
    };

    let initial_distance = (body1.p.pos - body0.p.pos).magnitude();

    constraint.solve(&mut body0, &mut body1, 1.0 / 60.0);

    let final_distance = (body1.p.pos - body0.p.pos).magnitude();

    assert!(
        final_distance < initial_distance,
        "Constraint should reduce the error"
    );

    // Unlike a rigid constraint, compliance should leave some error.
    assert!(
        final_distance > EPSILON,
        "Compliant constraint should not fully correct the error"
    );

    assert!(constraint.lagrange_multiplier.is_finite());
}

/// Ensure zero constraint error does not produce NaN values.
#[test]
fn test_zero_constraint_error() {
    let mut body0 = create_body(Vec3::zero());
    let mut body1 = create_body(Vec3::zero());

    let mut constraint = XpbdPositionalConstraint {
        relative_position_0: Vec3::zero(),
        relative_position_1: Vec3::zero(),
        compliance: 0.0,
        lagrange_multiplier: 0.0,
    };

    constraint.solve(&mut body0, &mut body1, 1.0 / 60.0);

    assert!(body0.p.pos.x.is_finite());
    assert!(body0.p.pos.y.is_finite());
    assert!(body0.p.pos.z.is_finite());

    assert!(body1.p.pos.x.is_finite());
    assert!(body1.p.pos.y.is_finite());
    assert!(body1.p.pos.z.is_finite());

    assert!(body0.r.orientation.s.is_finite());
    assert!(body1.r.orientation.s.is_finite());

    assert!(constraint.lagrange_multiplier.is_finite());
}

/// Verify that off-center constraint points produce rotational correction.
#[test]
fn test_nonzero_rotational_correction() {
    let mut body0 = create_body(Vec3::zero());
    let mut body1 = create_body(Vec3::new(2.0, 0.0, 0.0));

    let initial_orientation0 = body0.r.orientation;
    let initial_orientation1 = body1.r.orientation;

    let mut constraint = XpbdPositionalConstraint {
        relative_position_0: Vec3::new(0.0, 1.0, 0.0),
        relative_position_1: Vec3::new(2.0, 1.0, 0.0),
        compliance: 0.0,
        lagrange_multiplier: 0.0,
    };

    constraint.solve(&mut body0, &mut body1, 1.0 / 60.0);

    assert!(
        (body0.r.orientation - initial_orientation0).magnitude() > EPSILON,
        "Body 0 should rotate"
    );

    assert!(
        (body1.r.orientation - initial_orientation1).magnitude() > EPSILON,
        "Body 1 should rotate"
    );
}