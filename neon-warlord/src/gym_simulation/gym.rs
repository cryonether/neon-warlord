//! Environment for simulations

pub mod gym_cart;
pub mod gym_pendulum;
pub mod gym_pendulum_on_cart;

use crate::verlet_physics_simd::VerletPhysicsSimd;

pub trait Gym<const OBSERVATIONS: usize, const ACTIONS: usize> {
    fn get_verlet_physics(&self) -> &VerletPhysicsSimd;
    fn update_verlet_physics(&mut self, dt: f32);

    fn get_state(&self) -> [f32; OBSERVATIONS];
    fn update(&mut self, actions: &[f32; ACTIONS], dt: f32);
    fn get_reward(&self) -> f32;

    fn reset(&mut self);
}
