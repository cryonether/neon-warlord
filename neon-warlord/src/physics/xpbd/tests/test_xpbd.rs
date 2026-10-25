// //! Test of xpbd physics

// use crate::physics::xpbd::xpbd_rigid_body::XpbdRigidBody;

// use super::super::*;
// use cgmath::{Quaternion, SquareMatrix, Vector3};

// const EPSILON: f32 = 1e-4;

// /// Creates one rigid body without any constraints and
// /// simulates multiple simulation steps.
// #[test]
// fn test_steps_one_rigid_body() {
//     let initial_pos = Vec3::new(0.0, 0.0, 10.0);
//     let rot = Quad::new(1.0, 0.0, 0.0, 0.0);

//     let body = XpbdRigidBody {
//         p: PositionalQuantities {
//             pos: initial_pos,
//             prev_pos: initial_pos,
//             vel: Vec3::zero(),
//             // mass: 1.0,
//             inv_mass: 1.0,
//         },
//         r: RotationalQuantities {
//             rot: Quad::new(1.0, 0.0, 0.0, 0.0),
//             prev_rot: Quad::new(1.0, 0.0, 0.0, 0.0),
//             omega: Vec3::zero(),
//             // inertia_tensor: Mat3::from_value(1.0),
//             inverse_inertia_tensor: Mat3::from_value(1.0),
//             inv_rot: rot.invert(),
//             // inv_inertia: Vec3::new(1.0, 1.0, 1.0),
//         },
//         particles: [Vec3::zero()].to_vec(),
//     };

//     let mut xpbd = Xpbd::new();
//     xpbd.rigid_bodies.push(body);

//     let initial_orientation = xpbd.rigid_bodies[0].r.rot;

//     // Simulate one second (60 frames at 60 Hz).
//     for _ in 0..60 {
//         xpbd.step(1.0 / 60.0);
//     }

//     let body = &xpbd.rigid_bodies[0];

//     // The body should fall due to gravity.
//     assert!(
//         body.p.pos.z < initial_pos.z,
//         "Body should fall: initial z = {}, final z = {}",
//         initial_pos.z,
//         body.p.pos.z
//     );

//     // Horizontal position should remain unchanged.
//     assert!(
//         (body.p.pos.x - initial_pos.x).abs() < EPSILON
//     );
//     assert!(
//         (body.p.pos.y - initial_pos.y).abs() < EPSILON
//     );

//     // Orientation should remain normalized.
//     assert!(
//         (body.r.rot.magnitude() - 1.0).abs() < EPSILON
//     );

//     // No torque or initial angular velocity means no rotation.
//     assert!(
//         (body.r.omega - Vec3::zero()).magnitude() < EPSILON
//     );

//     assert!(
//         (body.r.rot.s - initial_orientation.s).abs() < EPSILON
//     );
//     assert!(
//         (body.r.rot.v - initial_orientation.v).magnitude() < EPSILON
//     );
// }