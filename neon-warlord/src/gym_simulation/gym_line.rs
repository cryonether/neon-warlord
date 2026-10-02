//! A pendulum on a cart

use crate::{
    advanced_composition::motor_linear::MotorLinear, gym_simulation::{Vec3, gym::Gym}, verlet_physics_simd::VerletPhysicsSimd,
};

const OBSERVATIONS: usize = 2;
const ACTIONS: usize = 1;

#[derive(Clone)]
pub struct GymLine {
    pub verlet_physics: VerletPhysicsSimd,

    particles_static: [usize; 2],
    particles_static_pos: [Vec3; 2],
    particle_cart: usize,

    motor_linear: MotorLinear,

    // variables
    previous_cart_position: f32,

    state: State,
}

impl GymLine {
    pub fn new() -> Self {
        let mut verlet_physics = VerletPhysicsSimd::new();

        let radius = 0.1;
        let mass = 0.1;

        let particles_static_pos_0 = Vec3::new(-10.0, 0.0, 0.0);
        let particles_static_0 = verlet_physics.push_particle(particles_static_pos_0, radius, mass);

        let particle_cart = verlet_physics.push_particle(Vec3::new(0.0, 0.0, 0.0), radius, mass);

        let particles_static_pos_1 = Vec3::new(10.0, 0.0, 0.0);
        let particles_static_1 = verlet_physics.push_particle(particles_static_pos_1, radius, mass);

        verlet_physics.push_constraint_none(particles_static_0, particle_cart);
        verlet_physics.push_constraint_none(particles_static_1, particle_cart);

        let motor_linear = MotorLinear::new(particle_cart, particles_static_0, particles_static_1);

        let state = State {
            cart_pos: 0.0,
            cart_velocity: 0.0,
        };

        let mut obj = Self {
            verlet_physics,
            particles_static: [particles_static_0, particles_static_1],
            particles_static_pos: [particles_static_pos_0, particles_static_pos_1],
            particle_cart,
            motor_linear,
            previous_cart_position: 0.0,
            state,
        };

        // obj.update(PendulumAction::Left0, 0.0);

        obj
    }

    // Calculates the position of the cart ranging from -1.0 to 1.0 and the velocity
    fn calculate_cart_position(&mut self, dt: f32) -> (f32, f32) {
        let cart_index: usize = self.particle_cart;
        let lef_index: usize = self.particles_static[0];
        let right_index: usize = self.particles_static[1];
        let cart: cgmath::Vector3<f32> = self.verlet_physics.particles.position(cart_index);
        let left: cgmath::Vector3<f32> = self.verlet_physics.particles.position(lef_index);
        let right: cgmath::Vector3<f32> = self.verlet_physics.particles.position(right_index);

        // Normalize cart position from [left.x, right.x] to [-1.0, 1.0].
        let position = 2.0 * (cart.x - left.x) / (right.x - left.x) - 1.0;

        // Calculate velocity.
        let velocity = if dt > 0.0 {
            (position - self.previous_cart_position) / dt
        } else {
            0.0
        };

        self.previous_cart_position = position;

        (position, velocity)
    }

    fn apply_static_constraint(&mut self) {
        self.verlet_physics
            .particles
            .set_position(self.particles_static[0], self.particles_static_pos[0]);

        self.verlet_physics
            .particles
            .set_position(self.particles_static[1], self.particles_static_pos[1]);
    }
}

impl Gym<OBSERVATIONS, ACTIONS> for GymLine  {
    fn get_verlet_physics(&self) -> &VerletPhysicsSimd {
        &self.verlet_physics
    }

    fn update_verlet_physics(&mut self, dt: f32) {
        self.verlet_physics.update(dt);
    }

    fn get_state(&self) -> [f32; OBSERVATIONS] {
        let state = &self.state;
        [state.cart_pos, state.cart_velocity]
    }

    fn update(&mut self, actions: &[f32; ACTIONS], dt: f32) {
        self.apply_static_constraint();
        self.motor_linear.accelerate(actions[0]);
        self.motor_linear.update(&mut self.verlet_physics.particles);

        let (cart_pos, cart_velocity) = self.calculate_cart_position(dt);

        self.state = State { cart_pos, cart_velocity };
    }

    fn get_reward(&self) -> f32 {
        self.state.cart_pos * self.state.cart_pos
    }

    fn reset(&mut self) {
        self.verlet_physics.particles.reset_position(self.particle_cart, 
            Vec3::new(
                (fastrand::f32() - 0.5) * 2.0 * 0.7, 
                0.0, 
                0.0,
            )
        );
    }
}

#[derive(Clone)]
pub struct State {
    pub cart_pos: f32,
    pub cart_velocity: f32,
}


