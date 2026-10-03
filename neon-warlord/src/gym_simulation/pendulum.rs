// //! A pendulum on a cart

// use cgmath::InnerSpace;

// use crate::{
//     advanced_composition::{motor_linear::MotorLinear, motor_rotation::MotorRotation}, pendulum_simulation::Vec3, verlet_physics_simd::VerletPhysicsSimd,
// };

// #[derive(Clone)]
// pub struct Pendulum {
//     pub verlet_physics: VerletPhysicsSimd,

//     particles_static: [usize; 2],
//     particles_static_pos: [Vec3; 2],
//     particle_pendulum: usize,

//     motor_rotation: MotorRotation,

//     // variables
//     previous_sin_alpha: f32,
//     previous_cos_alpha: f32,

//     pendulum_state: PendulumState,
// }

// impl Pendulum {
//     pub fn new() -> Self {
//         let mut verlet_physics = VerletPhysicsSimd::new();

//         let radius = 0.1;
//         let mass = 0.1;

//         let particles_static_pos_0 = Vec3::new(0.0, 0.0, 0.0);
//         let particles_static_0 = verlet_physics.push_particle(particles_static_pos_0, radius, mass);

//         let particles_static_pos_1 = Vec3::new(0.0, 1.0, 0.0);
//         let particles_static_1 = verlet_physics.push_particle(particles_static_pos_1, radius, mass);

//         let particle_pendulum =
//             verlet_physics.push_particle(Vec3::new(0.0, 0.0, -1.0), radius, mass);

//         verlet_physics.push_constraint_distance(particles_static_0, particle_pendulum, 1.0, 0.8);
//         verlet_physics.push_constraint_none(particles_static_0, particles_static_1);

//         let motor_rotation = MotorRotation::new(particle_pendulum, particles_static_0, particles_static_1);

//         let pendulum_state = PendulumState {
//             sin_theta: 0.0,
//             angular_velocity: 0.0,
//             cos_theta: 0.0,
//             cos_alpha_v: 0.0,
//         };

//         let mut obj = Self {
//             verlet_physics,
//             particles_static: [particles_static_0, particles_static_1],
//             particles_static_pos: [particles_static_pos_0, particles_static_pos_1],
//             particle_pendulum,
//             motor_rotation,
//             previous_sin_alpha: 0.0,
//             previous_cos_alpha: 0.0,
//             pendulum_state,
//         };

//         obj.update(0.0, 0.0);

//         obj
//     }

//     pub fn state(&self) -> PendulumState {
//         self.pendulum_state.clone()
//     }

//     pub fn update(&mut self, acc: f32, dt: f32) -> PendulumState {
//         self.apply_static_constraint();

//         self.motor_rotation.accelerate(acc);
//         self.motor_rotation.update(&mut self.verlet_physics.particles);

//         let sin_alpha = self.motor_rotation.get_sin_alpha();
//         let cos_alpha = self.motor_rotation.get_cos_alpha();
//         let sin_alpha_v = (sin_alpha - self.previous_sin_alpha) * dt;
//         let cos_alpha_v = (cos_alpha - self.previous_cos_alpha) * dt;

//         self.pendulum_state = PendulumState {
//             sin_theta: sin_alpha,
//             angular_velocity: sin_alpha_v,
//             cos_theta: cos_alpha,
//             cos_alpha_v,
//         };

//         self.pendulum_state.clone()
//     }

//     fn apply_static_constraint(&mut self) {
//         self.verlet_physics
//             .particles
//             .set_position(self.particles_static[0], self.particles_static_pos[0]);

//         self.verlet_physics
//             .particles
//             .set_position(self.particles_static[1], self.particles_static_pos[1]);
//     }

//     pub fn update_verlet_physics(&mut self, dt: f32) {
//         self.verlet_physics.update(dt);
//     }

//     pub(crate) fn rand_pos(&mut self) {
//         self.verlet_physics.particles.reset_position(self.particle_pendulum,
//             Vec3::new(
//                 fastrand::f32() * 2.0 - 1.0,
//                 0.0,
//                 fastrand::f32() * 2.0 - 1.0,
//             ).normalize()
//         );
//     }

//     pub fn reward(&self) -> f32 {
//         0.0
//     }
// }

// #[derive(Clone)]
// pub struct PendulumState {
//     pub sin_theta: f32,
//     pub cos_theta: f32,
//     pub angular_velocity: f32,
// }
