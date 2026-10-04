//! A pendulum mounted on a movable cart.

use cgmath::InnerSpace;

use crate::{
    advanced_composition::motor_linear::MotorLinear,
    gym_simulation::{Vec3, gym::Gym},
    verlet_physics_simd::VerletPhysicsSimd,
};

const OBSERVATIONS: usize = 5;
const ACTIONS: usize = 1;

const CART_MASS: f32 = 0.1;
const POLE_MASS: f32 = 0.1;

const CART_LEFT_X: f32 = -2.0;
const CART_RIGHT_X: f32 = 2.0;
const POLE_LENGTH: f32 = 1.0;

#[derive(Clone)]
pub struct GymPendulumOnCart {
    pub verlet_physics: VerletPhysicsSimd,

    // Cart
    particles_static: [usize; 2],
    particles_static_pos: [Vec3; 2],
    particle_cart: usize,

    motor_linear: MotorLinear,

    // Pendulum
    particle_pole: usize,

    // Variables
    previous_cart_position: f32,
    state: State,
}

impl GymPendulumOnCart {
    pub fn new() -> Self {
        let mut verlet_physics = VerletPhysicsSimd::new();

        let radius = 0.1;

        // Fixed cart endpoints.
        let particles_static_pos_0 = Vec3::new(CART_LEFT_X, 0.0, 0.0);
        let particles_static_0 =
            verlet_physics.push_particle(particles_static_pos_0, radius, CART_MASS);

        let particles_static_pos_1 = Vec3::new(CART_RIGHT_X, 0.0, 0.0);
        let particles_static_1 =
            verlet_physics.push_particle(particles_static_pos_1, radius, CART_MASS);

        // Cart.
        let particle_cart =
            verlet_physics.push_particle(Vec3::new(0.0, 0.0, 0.0), radius, CART_MASS);

        // These constraints provide the cart's connection to the two endpoints.
        verlet_physics.push_constraint_none(particles_static_0, particle_cart);
        verlet_physics.push_constraint_none(particles_static_1, particle_cart);

        // Linear motor controlling the cart.
        let motor_linear =
            MotorLinear::new(particle_cart, particles_static_0, particles_static_1);

        // Pendulum starts pointing straight down.
        let particle_pole = verlet_physics.push_particle(
            Vec3::new(0.0, 0.0, -POLE_LENGTH),
            radius,
            POLE_MASS,
        );

        // Pendulum is attached to the cart.
        verlet_physics.push_constraint_distance(
            particle_cart,
            particle_pole,
            POLE_LENGTH,
            0.8,
        );

        let state = State {
            cart_pos: 0.0,
            cart_velocity: 0.0,
            pol_sin_alpha: -1.0,
            pol_cos_alpha: 0.0,
            angular_velocity: 0.0,
        };

        Self {
            verlet_physics,

            particles_static: [particles_static_0, particles_static_1],
            particles_static_pos: [particles_static_pos_0, particles_static_pos_1],
            particle_cart,

            motor_linear,

            particle_pole,

            previous_cart_position: 0.0,
            state,
        }
    }

    /// Calculate cart position in [-1, 1] and cart velocity.
    fn calculate_cart_state(&mut self, dt: f32) -> (f32, f32) {
        let cart = self
            .verlet_physics
            .particles
            .position(self.particle_cart);

        let left = self
            .verlet_physics
            .particles
            .position(self.particles_static[0]);

        let right = self
            .verlet_physics
            .particles
            .position(self.particles_static[1]);

        let position = 2.0 * (cart.x - left.x) / (right.x - left.x) - 1.0;

        let velocity = if dt > 0.0 {
            (position - self.previous_cart_position) / dt
        } else {
            0.0
        };

        self.previous_cart_position = position;

        (position, velocity)
    }

    /// Calculate pendulum sin/cos and angular velocity.
    fn calculate_pendulum_state(&self, dt: f32) -> State {
        let pole = self
            .verlet_physics
            .particles
            .position(self.particle_pole);

        let cart = self
            .verlet_physics
            .particles
            .position(self.particle_cart);

        // Pendulum position relative to cart.
        let direction = (pole - cart).normalize();

        let pol_sin_alpha = direction.z;
        let pol_cos_alpha = direction.x;

        let pol_sin_alpha_v =
            if dt > 0.0 {
                (pol_sin_alpha - self.state.pol_sin_alpha) / dt
            } else {
                0.0
            };

        let pol_cos_alpha_v =
            if dt > 0.0 {
                (pol_cos_alpha - self.state.pol_cos_alpha) / dt
            } else {
                0.0
            };

        //
        // d sin(α)/dt = cos(α) α'
        // d cos(α)/dt = -sin(α) α'
        //
        // cos(α) · (cos(α) α')
        // - sin(α) · (-sin(α) α')
        //
        // = α'
        //
        let angular_velocity =
            pol_cos_alpha * pol_sin_alpha_v
                - pol_sin_alpha * pol_cos_alpha_v;

        State {
            cart_pos: self.state.cart_pos,
            cart_velocity: self.state.cart_velocity,
            pol_sin_alpha,
            pol_cos_alpha,
            angular_velocity,
        }
    }

    fn apply_static_constraint(&mut self) {
        self.verlet_physics
            .particles
            .set_position(
                self.particles_static[0],
                self.particles_static_pos[0],
            );

        self.verlet_physics
            .particles
            .set_position(
                self.particles_static[1],
                self.particles_static_pos[1],
            );
    }
}

impl Gym<OBSERVATIONS, ACTIONS> for GymPendulumOnCart {
    fn get_verlet_physics(&self) -> &VerletPhysicsSimd {
        &self.verlet_physics
    }

    fn update_verlet_physics(&mut self, dt: f32) {
        self.verlet_physics.update(dt);
    }

    fn get_state(&self) -> [f32; OBSERVATIONS] {
        [
            self.state.cart_pos,
            self.state.cart_velocity,
            self.state.pol_sin_alpha,
            self.state.pol_cos_alpha,
            self.state.angular_velocity,
        ]
    }

    fn update(&mut self, actions: &[f32; ACTIONS], dt: f32) {
        self.apply_static_constraint();

        // Cart control.
        let force = actions[0] * 20.0;

        self.motor_linear.accelerate(force);
        self.motor_linear
            .update(&mut self.verlet_physics.particles);

        // Calculate cart state.
        let (cart_pos, cart_velocity) =
            self.calculate_cart_state(dt);

        // Calculate pendulum state.
        let pendulum_state =
            self.calculate_pendulum_state(dt);

        self.state = State {
            cart_pos,
            cart_velocity,
            pol_sin_alpha: pendulum_state.pol_sin_alpha,
            pol_cos_alpha: pendulum_state.pol_cos_alpha,
            angular_velocity: pendulum_state.angular_velocity,
        };
    }

    fn get_reward(&self) -> f32 {
        let cart_pos = self.state.cart_pos;
        let angular_velocity = self.state.angular_velocity;

        // sin(alpha) = -1 when the pendulum is pointing straight down.
        // sin(alpha) =  0 when horizontal.
        // sin(alpha) =  1 when pointing straight up.
        let upright_reward =
            (self.state.pol_sin_alpha + 1.0) / 2.0;

        let cart_penalty = 0.001 * cart_pos * cart_pos;
        let velocity_penalty = 0.0001 * angular_velocity * angular_velocity;


        upright_reward - cart_penalty - velocity_penalty
    }

    fn reset(&mut self) {
        // Random cart position.
        let cart_pos =
            (fastrand::f32() - 0.5) * 2.0 * 1.8;

        self.verlet_physics
            .particles
            .reset_position(
                self.particle_cart,
                Vec3::new(cart_pos, 0.0, 0.0),
            );

        // Random pendulum angle.
        let alpha =
            (fastrand::f32() * 2.0 - 1.0)
                * std::f32::consts::PI;

        let pol_sin_alpha = alpha.sin();
        let pol_cos_alpha = alpha.cos();

        self.verlet_physics
            .particles
            .reset_position(
                self.particle_pole,
                Vec3::new(
                    cart_pos + pol_cos_alpha * POLE_LENGTH,
                    0.0,
                    pol_sin_alpha * POLE_LENGTH,
                ),
            );

        // Reset velocity/state tracking.
        self.previous_cart_position = cart_pos / 2.0;

        self.state = State {
            cart_pos: cart_pos / 2.0,
            cart_velocity: 0.0,
            pol_sin_alpha,
            pol_cos_alpha,
            angular_velocity: 0.0,
        };
    }
}

#[derive(Clone)]
pub struct State {
    pub cart_pos: f32,
    pub cart_velocity: f32,
    pub pol_sin_alpha: f32,
    pub pol_cos_alpha: f32,
    pub angular_velocity: f32,
}
