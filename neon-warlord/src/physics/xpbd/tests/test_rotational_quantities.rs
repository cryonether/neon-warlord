// //! Test of RotationalQuantities

// use cgmath::SquareMatrix;

// use super::super::*;

// const EPSILON: f32 = 1e-5;

// #[test]
// fn test_integrate_orientation() {
//     let dt = 0.1;
//     let angular_velocity = Vec3::new(0.0, 0.0, 1.0);
//     let rot = Quad::new(1.0, 0.0, 0.0, 0.0);

//     let mut quantities = RotationalQuantities {
//         rot,
//         prev_rot: Quad::new(1.0, 0.0, 0.0, 0.0),
//         omega: angular_velocity,
//         // inertia_tensor: Mat3::from_value(1.0),
//         inverse_inertia_tensor: Mat3::from_value(1.0),
//         inv_rot: rot.invert(),
//         // inv_inertia: Vec3::new(1.0, 1.0, 1.0),
//     };

//     quantities.integrate_orientation(dt);

//     // Previous orientation should be saved.
//     assert!(
//         (quantities.prev_rot.v
//             - Vec3::zero()).magnitude() < EPSILON
//     );
//     assert!((quantities.prev_rot.s - 1.0).abs() < EPSILON);

//     // With zero external torque and isotropic inertia,
//     // angular velocity should remain constant.
//     assert!(
//         (quantities.omega - angular_velocity).magnitude() < EPSILON
//     );

//     // First-order quaternion integration:
//     // q_new = normalize(q_old + 0.5 * dt * omega_quat * q_old)
//     let expected = Quad::new(
//         1.0,
//         0.0,
//         0.0,
//         0.5 * dt,
//     );
//     let expected = expected / expected.magnitude();

//     assert!(
//         (quantities.rot.s - expected.s).abs() < EPSILON
//     );
//     assert!(
//         (quantities.rot.v - expected.v).magnitude() < EPSILON
//     );

//     // Orientation should remain normalized.
//     assert!((quantities.rot.magnitude() - 1.0).abs() < EPSILON);
// }

// #[test]
// fn test_integrate_angular_velocity() {
//     let dt = 0.1;

//     let orientation_prev = Quad::new(1.0, 0.0, 0.0, 0.0);
//     let angle = 0.2_f32;
//     let half_angle = angle / 2.0;

//     // Rotation of angle radians around the Z axis.
//     let orientation = Quad::new(
//         half_angle.cos(),
//         0.0,
//         0.0,
//         half_angle.sin(),
//     );

//     let mut quantities = RotationalQuantities {
//         rot: orientation,
//         prev_rot: orientation_prev,
//         omega: Vec3::zero(),
//         // inertia_tensor: Mat3::from_value(1.0),
//         inverse_inertia_tensor: Mat3::from_value(1.0),
//         inv_rot: orientation.invert(),
//         // inv_inertia: Vec3::new(1.0, 1.0, 1.0),
//     };

//     quantities.update_angular_velocity(dt);

//     // The implementation approximates angular velocity as
//     // 2 * quaternion_vector / dt.
//     let expected = Vec3::new(
//         0.0,
//         0.0,
//         2.0 * half_angle.sin() / dt,
//     );

//     assert!(
//         (quantities.omega - expected).magnitude() < EPSILON
//     );
// }

// #[test]
// fn test_steps() {
//     let dt = 0.01;
//     let steps = 10;

//     let initial_orientation = Quad::new(1.0, 0.0, 0.0, 0.0);
//     let initial_angular_velocity = Vec3::new(0.0, 0.0, 1.0);

//     let mut quantities = RotationalQuantities {
//         rot: initial_orientation,
//         prev_rot: initial_orientation,
//         omega: initial_angular_velocity,
//         // inertia_tensor: Mat3::from_value(1.0),
//         inverse_inertia_tensor: Mat3::from_value(1.0),
//         inv_rot: initial_orientation.invert(),
//         // inv_inertia: Vec3::new(1.0, 1.0, 1.0),
//     };

//     for _ in 0..steps {
//         quantities.integrate_orientation(dt);
//         quantities.update_angular_velocity(dt);
//     }

//     // Orientation should remain normalized after repeated steps.
//     assert!(
//         (quantities.rot.magnitude() - 1.0).abs() < EPSILON
//     );

//     // For isotropic inertia and zero external torque, rotation
//     // should remain around the Z axis.
//     assert!(quantities.rot.v.x.abs() < EPSILON);
//     assert!(quantities.rot.v.y.abs() < EPSILON);

//     // Angular velocity should remain finite and point along Z.
//     assert!(quantities.omega.x.abs() < EPSILON);
//     assert!(quantities.omega.y.abs() < EPSILON);
//     assert!(quantities.omega.z.is_finite());
//     assert!(quantities.omega.z > 0.0);
// }