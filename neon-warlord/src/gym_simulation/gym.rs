//! Environment for simulations

pub mod gym_cart;
pub mod gym_cart_pole;
pub mod gym_half_cheetah;
pub mod gym_half_cheetah_2;
pub mod gym_pole;

use forward_renderer::{particle_shader, particle_shader_two_point};

use crate::gym_simulation::Vec3;

pub trait Gym<const OBSERVATIONS: usize, const ACTIONS: usize> {
    fn set_drawer_position(&mut self, pos: &Vec3);
    fn set_drawer_radius(&mut self, radius: f32);

    fn draw(
        &self,
        producer_nodes: &mut Vec<particle_shader::Instance>,
        producer_edges: &mut Vec<particle_shader_two_point::Instance>,
    );

    fn update_verlet_physics(&mut self, dt: f32);

    fn get_state(&self) -> [f32; OBSERVATIONS];
    fn update(&mut self, actions: &[f32; ACTIONS], dt: f32);
    fn get_reward(&self) -> (f32, bool);

    fn reset(&mut self);
}
