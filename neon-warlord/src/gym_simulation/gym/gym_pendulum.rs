//! A pendulum, try to balance it

use cgmath::InnerSpace;

use crate::{
    gym_simulation::{Vec3, gym::Gym},
    verlet_physics_simd::VerletPhysicsSimd,
};

const OBSERVATIONS: usize = 3;
const ACTIONS: usize = 1;

const MASS: f32 = 0.1;

#[derive(Clone)]
pub struct GymPendulum {
    pub verlet_physics: VerletPhysicsSimd,

    particles_static: usize,
    particles_static_pos: Vec3,
    particle_pole: usize,

    state: State,
}

impl GymPendulum {
    pub fn new() -> Self {
        let mut verlet_physics = VerletPhysicsSimd::new();

        let radius = 0.1;

        let particles_static_pos = Vec3::new(0.0, 0.0, 0.0);
        let particles_static = verlet_physics.push_particle(particles_static_pos, radius, MASS);

        let particle_pole = verlet_physics.push_particle(Vec3::new(0.0, 0.0, -1.0), radius, MASS);

        verlet_physics.push_constraint_distance(particles_static, particle_pole, 1.0, 0.8);

        let state = State {
            pol_sin_alpha: -1.0,
            pol_cos_alpha: 0.0,
            angular_velocity: 0.0,
        };

        Self {
            verlet_physics,
            particles_static,
            particles_static_pos,
            particle_pole,
            state,
        }
    }

    // Calculates the sin_x and cos_x of the pole and the velocities
    fn update(&mut self, force: f32, dt: f32) -> State {
        let pole_index: usize = self.particle_pole;
        let static_index: usize = self.particles_static;
        let pos_pole: cgmath::Vector3<f32> = self.verlet_physics.particles.position(pole_index);
        let pos_static: cgmath::Vector3<f32> = self.verlet_physics.particles.position(static_index);

        // move to the center
        let pos_pole = pos_pole - pos_static;
        // let _pos_static = pos_static - pos_static;

        // normalize
        let pos_pole = pos_pole.normalize();

        // tangent
        let tangent = cgmath::Vector3::new(-pos_pole.z, 0.0, pos_pole.x);

        // apply force
        let acc = tangent * force / MASS;
        self.verlet_physics.particles.accelerate(pole_index, acc);

        // get sin and cos
        let pol_sin_alpha = pos_pole.z;
        let pol_cos_alpha = pos_pole.x;

        // get velocities
        let pol_sin_alpha_v = (pol_sin_alpha - self.state.pol_sin_alpha) / dt;
        let pol_cos_alpha_v = (pol_cos_alpha - self.state.pol_cos_alpha) / dt;

        //
        // d sin(α)/dt = cos(α) α'
        // d cos(α)/dt = -sin(α) α'
        //
        // cos(α) · (cos(α) α') - sin(α) · (-sin(α) α') = α'
        //
        let angular_velocity = pol_cos_alpha * pol_sin_alpha_v - pol_sin_alpha * pol_cos_alpha_v;

        State {
            pol_sin_alpha,
            pol_cos_alpha,
            angular_velocity,
        }
    }

    fn apply_static_constraint(&mut self) {
        self.verlet_physics
            .particles
            .set_position(self.particles_static, self.particles_static_pos);
    }
}

impl Gym<OBSERVATIONS, ACTIONS> for GymPendulum {
    fn get_verlet_physics(&self) -> &VerletPhysicsSimd {
        &self.verlet_physics
    }

    fn update_verlet_physics(&mut self, dt: f32) {
        self.verlet_physics.update(dt);
    }

    fn get_state(&self) -> [f32; OBSERVATIONS] {
        let state = &self.state;
        [
            state.pol_sin_alpha,
            state.pol_cos_alpha,
            state.angular_velocity,
        ]
    }

    fn update(&mut self, actions: &[f32; ACTIONS], dt: f32) {
        self.apply_static_constraint();

        let force = actions[0] * 0.7;
        self.state = self.update(force, dt);
    }

    fn get_reward(&self) -> (f32, bool)  {
        let sin = self.state.pol_sin_alpha;
        let vel = self.state.angular_velocity;

        let reward = (sin + 1.0) / 2.0 - 0.01 * vel * vel;

        (reward, false)
    }

    fn reset(&mut self) {
        let alpha = (fastrand::f32() * 2.0 - 1.0) * std::f32::consts::PI;
        // let alpha = -std::f32::consts::PI * 0.5  + (fastrand::f32() * 2.0 - 1.0) * 0.3;

        let pol_sin_alpha = alpha.sin();
        let pol_cos_alpha = alpha.cos();

        self.verlet_physics.particles.reset_position(
            self.particle_pole,
            Vec3::new(pol_cos_alpha, 0.0, pol_sin_alpha),
        );

        self.state = State {
            pol_sin_alpha,
            pol_cos_alpha,
            angular_velocity: 0.0,
        }
    }
}

#[derive(Clone)]
pub struct State {
    pol_sin_alpha: f32,
    pol_cos_alpha: f32,
    angular_velocity: f32,
}
