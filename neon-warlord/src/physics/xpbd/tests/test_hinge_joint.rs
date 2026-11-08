use crate::physics::xpbd::{
    xpbd_angular_constraint::XpbdAngularConstraint, xpbd_hinge_joint::RotationAxis,
};

use super::super::*;

use cgmath::{InnerSpace, Rad, Rotation3};
use std::f32::consts::PI;

type Quat = cgmath::Quaternion<f32>;

fn create_body(position: Vec3) -> XpbdRigidBody {
    // Asymmetric particle distribution to avoid degenerate inertia.
    let particles = [
        Vec3::new(-0.5, -0.5, -0.5),
        Vec3::new(0.5, -0.5, -0.5),
        Vec3::new(-0.5, 0.5, -0.5),
        Vec3::new(-0.5, -0.5, 0.5),
        Vec3::new(0.5, 0.5, 0.5),
    ];

    XpbdRigidBody::new(position, &particles, 1.0, 0.1)
}

fn create_joint(lower_limit: f32, upper_limit: f32, rotation_axis: RotationAxis) -> XpbdHingeJoint {
    XpbdHingeJoint::new(
        Vec3::zero(),
        Vec3::zero(),
        lower_limit,
        upper_limit,
        rotation_axis.clone(),
        rotation_axis,
        0.002,
        0.002,
    )
}

fn create_joint_fix(
    lower_limit: f32,
    upper_limit: f32,
    rotation_axis: RotationAxis,
) -> XpbdHingeJoint {
    XpbdHingeJoint::new(
        Vec3::zero(),
        Vec3::zero(),
        lower_limit,
        upper_limit,
        rotation_axis.clone(),
        rotation_axis,
        0.00,
        0.00,
    )
}

// Run the constraint solver for multiple iterations.
// fn simulate_joint(
//     joint: &mut XpbdHingeJoint,
//     body0: &mut XpbdRigidBody,
//     body1: &mut XpbdRigidBody,
//     dt: f32,
//     iterations: usize,
// ) {
//     for _ in 0..iterations {
//         joint.solve(body0, body1, dt);
//     }
// }

fn simulate_joint(
    joint: &mut XpbdHingeJoint,
    body0: &mut XpbdRigidBody,
    body1: &mut XpbdRigidBody,
    dt: f32,
    iterations: usize,
) {
    for i in 0..iterations {
        joint.solve(body0, body1, dt);

        let rotations = [
            body0.r.rot.s,
            body0.r.rot.v.x,
            body0.r.rot.v.y,
            body0.r.rot.v.z,
            body1.r.rot.s,
            body1.r.rot.v.x,
            body1.r.rot.v.y,
            body1.r.rot.v.z,
        ];

        // println!("body0: {:?}, body1: {:?}", body0.r.rot, body1.r.rot);

        assert!(
            rotations.iter().all(|x| x.is_finite()),
            "Invalid rotation at solver iteration {}",
            i,
        );

        let q0_length = body0.r.rot.magnitude();
        let q1_length = body1.r.rot.magnitude();

        assert!(
            (q0_length - 1.0).abs() < 1e-3,
            "Body 0 quaternion is not normalized: {}",
            q0_length,
        );

        assert!(
            (q1_length - 1.0).abs() < 1e-3,
            "Body 1 quaternion is not normalized: {}",
            q1_length,
        );
    }
}

fn assert_angle_near(actual: f32, expected: f32, tolerance: f32) {
    assert!(
        (actual - expected).abs() < tolerance,
        "Expected angle {}, got {} (tolerance {})",
        expected,
        actual,
        tolerance,
    );
}

// ------------------------------------------------------------
// Test 1: Initial configuration
// ------------------------------------------------------------

#[test]
fn test_initial_configuration() {
    let mut body0 = create_body(Vec3::zero());
    let mut body1 = create_body(Vec3::zero());

    let mut joint = create_joint(-PI / 4.0, PI / 4.0, RotationAxis::Y);

    let initial_angle = joint.get_angle();

    simulate_joint(&mut joint, &mut body0, &mut body1, 1.0 / 60.0, 10);

    assert_angle_near(joint.get_angle(), initial_angle, 1e-3);
}

// ------------------------------------------------------------
// Test 2: Free rotation around hinge axis
// ------------------------------------------------------------

#[test]
fn test_free_rotation_within_limits() {
    let mut body0 = create_body(Vec3::zero());
    let mut body1 = create_body(Vec3::zero());

    let mut joint = create_joint(-PI / 4.0, PI / 4.0, RotationAxis::Y);

    body1.r.rot = Quat::from_axis_angle(Vec3::unit_y(), Rad(PI / 6.0));

    // Initialize the joint's measured angle.
    joint.solve(&mut body0, &mut body1, 1.0 / 60.0);

    let initial_angle = joint.get_angle();

    simulate_joint(&mut joint, &mut body0, &mut body1, 1.0 / 60.0, 10);

    assert_angle_near(joint.get_angle(), initial_angle, 1e-3);
}

// ------------------------------------------------------------
// Test 3: Lower angular limit
// ------------------------------------------------------------

#[test]
fn test_rigid_body_inertia_is_finite() {
    let body = create_body(Vec3::zero());

    let i = body.r.inertia_tensor;
    let inv_i = body.r.inverse_inertia_tensor;

    for value in [
        i.x.x, i.x.y, i.x.z, i.y.x, i.y.y, i.y.z, i.z.x, i.z.y, i.z.z, inv_i.x.x, inv_i.x.y,
        inv_i.x.z, inv_i.y.x, inv_i.y.y, inv_i.y.z, inv_i.z.x, inv_i.z.y, inv_i.z.z,
    ] {
        assert!(value.is_finite(), "Non-finite inertia: {value}");
    }
}

#[test]
fn test_angular_constraint_is_finite() {
    let mut body0 = create_body(Vec3::zero());
    let mut body1 = create_body(Vec3::zero());

    let mut constraint = XpbdAngularConstraint::new(0.0);

    constraint.solve(&mut body0, &mut body1, &Vec3::unit_x(), 0.1, 1.0 / 60.0);

    for value in [
        body0.r.rot.s,
        body0.r.rot.v.x,
        body0.r.rot.v.y,
        body0.r.rot.v.z,
        body1.r.rot.s,
        body1.r.rot.v.x,
        body1.r.rot.v.y,
        body1.r.rot.v.z,
    ] {
        assert!(
            value.is_finite(),
            "Angular constraint produced NaN: {value}",
        );
    }
}

#[test]
fn test_hinge_axis_misalignment_produces_finite_rotation() {
    let mut body0 = create_body(Vec3::zero());
    let mut body1 = create_body(Vec3::zero());

    let mut joint = create_joint(-PI, PI, RotationAxis::Y);

    body1.r.rot = Quat::from_axis_angle(Vec3::unit_x(), Rad(PI / 6.0));

    joint.solve(&mut body0, &mut body1, 1.0 / 60.0);

    assert!(body0.r.rot.s.is_finite());
    assert!(body1.r.rot.s.is_finite());

    assert!(body0.r.rot.v.x.is_finite());
    assert!(body0.r.rot.v.y.is_finite());
    assert!(body0.r.rot.v.z.is_finite());

    assert!(body1.r.rot.v.x.is_finite());
    assert!(body1.r.rot.v.y.is_finite());
    assert!(body1.r.rot.v.z.is_finite());
}

#[test]
fn test_lower_angular_limit() {
    let mut body0 = create_body(Vec3::zero());
    let mut body1 = create_body(Vec3::zero());

    let lower_limit = -PI / 4.0;
    let upper_limit = PI / 4.0;

    let mut joint = create_joint_fix(lower_limit, upper_limit, RotationAxis::Y);

    // Rotate body 1 by -90 degrees.
    body1.r.rot = Quat::from_axis_angle(
        Vec3::unit_y(),
        Rad(-PI / 2.0),
        // Rad(PI),
    );

    let dt = 1.0 / 60.0;

    simulate_joint(&mut joint, &mut body0, &mut body1, dt * 10.0, 100);

    assert_angle_near(joint.get_angle(), lower_limit, 0.02);
}

// ------------------------------------------------------------
// Test 4: Upper angular limit
// ------------------------------------------------------------

#[test]
fn test_upper_angular_limit_y() {
    let mut body0 = create_body(Vec3::zero());
    let mut body1 = create_body(Vec3::zero());

    let lower_limit = -PI / 4.0;
    let upper_limit = PI / 4.0;

    let mut joint = create_joint_fix(lower_limit, upper_limit, RotationAxis::Y);

    // Rotate body 1 by +90 degrees.
    body1.r.rot = Quat::from_axis_angle(Vec3::unit_y(), Rad(PI / 2.0));

    simulate_joint(&mut joint, &mut body0, &mut body1, 1.0 / 60.0, 100);

    assert_angle_near(joint.get_angle(), upper_limit, 1e-2);

    // Rotate body 1 by +90 degrees.
    body1.r.rot = Quat::from_axis_angle(Vec3::unit_y(), Rad(-PI / 2.0));

    simulate_joint(&mut joint, &mut body0, &mut body1, 1.0 / 60.0, 100);

    assert_angle_near(joint.get_angle(), lower_limit, 1e-2);
}

#[test]
fn test_upper_angular_limit_x() {
    let mut body0 = create_body(Vec3::zero());
    let mut body1 = create_body(Vec3::zero());

    let lower_limit = -PI / 4.0;
    let upper_limit = PI / 4.0;

    let mut joint = create_joint_fix(lower_limit, upper_limit, RotationAxis::X);

    // Rotate body 1 by +90 degrees.
    body1.r.rot = Quat::from_axis_angle(Vec3::unit_x(), Rad(PI / 2.0));

    simulate_joint(&mut joint, &mut body0, &mut body1, 1.0 / 60.0, 100);

    assert_angle_near(joint.get_angle(), upper_limit, 1e-2);

    // Rotate body 1 by +90 degrees.
    body1.r.rot = Quat::from_axis_angle(Vec3::unit_x(), Rad(-PI / 2.0));

    simulate_joint(&mut joint, &mut body0, &mut body1, 1.0 / 60.0, 100);

    assert_angle_near(joint.get_angle(), lower_limit, 1e-2);
}

#[test]
fn test_upper_angular_limit_z() {
    let mut body0 = create_body(Vec3::zero());
    let mut body1 = create_body(Vec3::zero());

    let lower_limit = -PI / 4.0;
    let upper_limit = PI / 4.0;

    let mut joint = create_joint_fix(lower_limit, upper_limit, RotationAxis::Z);

    // Rotate body 1 by +90 degrees.
    body1.r.rot = Quat::from_axis_angle(Vec3::unit_z(), Rad(PI / 2.0));

    simulate_joint(&mut joint, &mut body0, &mut body1, 1.0 / 60.0, 100);

    assert_angle_near(joint.get_angle(), upper_limit, 1e-2);

    // Rotate body 1 by +90 degrees.
    body1.r.rot = Quat::from_axis_angle(Vec3::unit_z(), Rad(-PI / 2.0));

    simulate_joint(&mut joint, &mut body0, &mut body1, 1.0 / 60.0, 100);

    assert_angle_near(joint.get_angle(), lower_limit, 1e-2);
}

// ------------------------------------------------------------
// Test 5: Hinge axis alignment
// ------------------------------------------------------------

#[test]
fn test_hinge_axis_alignment() {
    let mut body0 = create_body(Vec3::zero());
    let mut body1 = create_body(Vec3::zero());

    let mut joint = create_joint_fix(-PI, PI, RotationAxis::Y);

    // Rotate body 1 by 30 degrees around X.
    // This misaligns its local Y hinge axis.
    body1.r.rot = Quat::from_axis_angle(Vec3::unit_x(), Rad(PI / 6.0));

    simulate_joint(&mut joint, &mut body0, &mut body1, 1.0 / 60.0, 100);

    let a0 = body0.r.rot.rotate_vector(joint.a0);
    let a1 = body1.r.rot.rotate_vector(joint.a1);

    assert!(
        a0.x.is_finite() && a0.y.is_finite() && a0.z.is_finite(),
        "Body 0 hinge axis is NaN: {:?}",
        a0,
    );

    assert!(
        a1.x.is_finite() && a1.y.is_finite() && a1.z.is_finite(),
        "Body 1 hinge axis is NaN: {:?}",
        a1,
    );

    let alignment = a0.dot(a1).clamp(-1.0, 1.0);

    assert!(
        alignment > 0.99,
        "Hinge axes are not aligned: dot = {}",
        alignment,
    );
}
