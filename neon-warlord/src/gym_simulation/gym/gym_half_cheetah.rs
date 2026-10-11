//! A pendulum, try to balance it

use std::iter::zip;

use cgmath::InnerSpace;
use wgpu::rwh;

use crate::{
    gym_simulation::{Vec3, gym::Gym},
    verlet_physics_simd::VerletPhysicsSimd,
};

const OBSERVATIONS: usize = 15;
const ACTIONS: usize = 4;

const MASS: f32 = 0.1;

#[derive(Clone)]
pub struct GymHalfCheetah {
    pub verlet_physics: VerletPhysicsSimd,

    particles_body:  [usize; 4],
    particles_leg_0: [usize; 2],
    particles_leg_1: [usize; 2],

    state: State,
}

impl GymHalfCheetah {
    pub fn new() -> Self {
        let mut verlet_physics = VerletPhysicsSimd::new();

        let radius = 0.1;

        // body
        let particles_body = [
            verlet_physics.push_particle(Vec3::new(-1.0, 0.0, 1.5), radius, MASS),
            verlet_physics.push_particle(Vec3::new(0.0, 0.0, 1.5), radius, MASS),
            verlet_physics.push_particle(Vec3::new(1.0, 0.0, 1.5), radius, MASS),
            verlet_physics.push_particle(Vec3::new(1.3, 0.0, 1.8), radius, MASS),
        ];

        verlet_physics.push_constraint_distance_infer(particles_body[0], particles_body[1], 0.8);
        verlet_physics.push_constraint_distance_infer(particles_body[1], particles_body[2], 0.8);
        verlet_physics.push_constraint_distance_infer(particles_body[2], particles_body[3], 0.8);

        verlet_physics.push_constraint_angle_infer(particles_body[0], particles_body[1], particles_body[2], 0.8);
        verlet_physics.push_constraint_angle_infer(particles_body[1], particles_body[2], particles_body[3], 0.8);

        // leg 0
        let particles_leg_0 = [
            verlet_physics.push_particle(Vec3::new(-1.0, 0.0, 1.0), radius, MASS),
            verlet_physics.push_particle(Vec3::new(-1.0, 0.0, 0.5), radius, MASS),
        ];

        verlet_physics.push_constraint_distance_infer(particles_body[0], particles_leg_0[0], 0.8);
        verlet_physics.push_constraint_distance_infer(particles_leg_0[0], particles_leg_0[1], 0.8);

        verlet_physics.push_constraint_angle_min_max(particles_body[1], particles_body[0], particles_leg_0[0], 
        f32::cos(f32::to_radians(160.0)),
        f32::cos(f32::to_radians(20.0)),
        0.8
        );
        verlet_physics.push_constraint_angle_min_max(particles_body[0], particles_leg_0[0], particles_leg_0[1], 
        f32::cos(f32::to_radians(180.0)),
        f32::cos(f32::to_radians(40.0)),
        0.8
        );


        // leg 1
        let particles_leg_1 = [
            verlet_physics.push_particle(Vec3::new(1.0, 0.0, 1.0), radius, MASS),
            verlet_physics.push_particle(Vec3::new(1.0, 0.0, 0.5), radius, MASS),
        ];
        
        verlet_physics.push_constraint_distance_infer(particles_body[2], particles_leg_1[0], 0.8);
        verlet_physics.push_constraint_distance_infer(particles_leg_1[0], particles_leg_1[1], 0.8);


        verlet_physics.push_constraint_angle_min_max(particles_body[1], particles_body[2], particles_leg_1[0], 
        f32::cos(f32::to_radians(160.0)),
        f32::cos(f32::to_radians(20.0)),
        0.8
        );
        verlet_physics.push_constraint_angle_min_max(particles_body[2], particles_leg_1[0], particles_leg_1[1], 
        f32::cos(f32::to_radians(180.0)),
        f32::cos(f32::to_radians(40.0)),
        0.8
        );




        let state = State {
            pos_z: 2.3,
            vel_x: 0.0,
            vel_z: 0.0,
            leg_0: StateLeg { sin_alpha: 0.0, cos_alpha: -1.0, angular_velocity: 0.0 },
            leg_1: StateLeg { sin_alpha: 0.0, cos_alpha: -1.0, angular_velocity: 0.0 },
            leg_2: StateLeg { sin_alpha: 0.0, cos_alpha: -1.0, angular_velocity: 0.0 },
            leg_3: StateLeg { sin_alpha: 0.0, cos_alpha: -1.0, angular_velocity: 0.0 },
        };

        Self {
            verlet_physics,
            particles_body,
            particles_leg_0,
            particles_leg_1,
            state,
        }
    }

    fn apply_angle_force(&mut self, index_base: usize, index_pole: usize, force: f32, state_leg: StateLeg, dt: f32) -> StateLeg {
        let index_pole: usize = index_pole;
        let index_base: usize = index_base;
        let pos_pole: cgmath::Vector3<f32> = self.verlet_physics.particles.position(index_pole);
        let pos_base: cgmath::Vector3<f32> = self.verlet_physics.particles.position(index_base);

        // move to the center
        let pos_pole = pos_pole - pos_base;
        // let _pos_static = pos_static - pos_static;

        // normalize
        let pos_pole = pos_pole.normalize();

        // tangent
        let tangent = cgmath::Vector3::new(-pos_pole.z, 0.0, pos_pole.x);

        // apply force
        let acc = tangent * force / MASS;
        self.verlet_physics.particles.accelerate(index_pole, acc);

        // get sin and cos
        let pol_sin_alpha = pos_pole.z;
        let pol_cos_alpha = pos_pole.x;

        // get velocities
        let pol_sin_alpha_v = (pol_sin_alpha - state_leg.sin_alpha) / dt;
        let pol_cos_alpha_v = (pol_cos_alpha - state_leg.cos_alpha) / dt;

        //
        // d sin(α)/dt = cos(α) α'
        // d cos(α)/dt = -sin(α) α'
        //
        // cos(α) · (cos(α) α') - sin(α) · (-sin(α) α') = α'
        //
        let angular_velocity = pol_cos_alpha * pol_sin_alpha_v - pol_sin_alpha * pol_cos_alpha_v;


        StateLeg {
            sin_alpha: pol_sin_alpha,
            cos_alpha: pol_cos_alpha,
            angular_velocity,
        }

    }

    // Calculates the sin_x and cos_x of the pole and the velocities
    fn update(&mut self, force: [f32; 4], dt: f32) -> State {

        let leg_0 = self.apply_angle_force(self.particles_body[0], self.particles_leg_0[0], force[0], self.state.leg_0.clone(), dt);
        let leg_1 = self.apply_angle_force(self.particles_leg_0[0], self.particles_leg_0[1], force[1], self.state.leg_1.clone(), dt);

        let leg_2 = self.apply_angle_force(self.particles_body[2], self.particles_leg_1[0], force[2], self.state.leg_2.clone(), dt);
        let leg_3 = self.apply_angle_force(self.particles_leg_1[0], self.particles_leg_1[1], force[3], self.state.leg_3.clone(), dt);

        let front_tip = self.particles_body[3];
        let pos = self.verlet_physics.particles.position(front_tip);
        let vel = self.verlet_physics.particles.velocity(front_tip);

        State {
            pos_z: pos.z,
            vel_x: vel.x,
            vel_z: vel.z,

            leg_0,
            leg_1,
            leg_2,
            leg_3,
        }
    }

    fn apply_static_constraint(&mut self) {
        let friction = 1.00; // 0.0 = no friction, 1.0 = full stop

        let particles = &mut self.verlet_physics.particles;
        let len = particles.len();

        for i in 0..len {
            let radius = particles.radius[i];

            if particles.z[i] < radius {
                // Resolve collision with the floor
                particles.z[i] = radius;
                particles.prev_z[i] = radius;

                // Apply friction to horizontal velocity
                let vx = particles.x[i] - particles.prev_x[i];
                // particles.prev_x[i] = particles.x[i] - vx * (1.0 - friction);
                particles.prev_x[i] = particles.x[i];
            }
        }
    }
}

impl Gym<OBSERVATIONS, ACTIONS> for GymHalfCheetah {
    fn get_verlet_physics(&self) -> &VerletPhysicsSimd {
        &self.verlet_physics
    }

    fn update_verlet_physics(&mut self, dt: f32) {
        // self.verlet_physics.update(dt * 0.4);
        self.verlet_physics.update(dt * 1.0);
    }

    fn get_state(&self) -> [f32; OBSERVATIONS] {
        let state = &self.state;
        [
            state.pos_z,
            state.vel_x,
            state.vel_z,

            state.leg_0.sin_alpha,
            state.leg_0.cos_alpha,
            state.leg_0.angular_velocity,

            state.leg_1.sin_alpha,
            state.leg_1.cos_alpha,
            state.leg_1.angular_velocity,

            state.leg_2.sin_alpha,
            state.leg_2.cos_alpha,
            state.leg_2.angular_velocity,

            state.leg_3.sin_alpha,
            state.leg_3.cos_alpha,
            state.leg_3.angular_velocity,
        ]
    }

    fn update(&mut self, actions: &[f32; ACTIONS], dt: f32) {
        self.apply_static_constraint();

        let force = [
            actions[0] * 4.0,
            actions[1] * 4.0,
            actions[2] * 4.0,
            actions[3] * 4.0,
        ];

        // let force = actions[0] * 0.7;
        self.state = self.update(force, dt);
    }

    fn get_reward(&self) -> (f32, bool) {
        let vel_x = self.state.vel_x;

        let reward = vel_x.clamp(0.0, 2.0);

        // let reward = vel_x * (2.0 - vel_x);

        // let reward = reward.max(0.0);

        (reward, false)
    }

    fn reset(&mut self) {
        let alpha = (fastrand::f32() * 2.0 - 1.0) * std::f32::consts::PI;
        // let alpha = -std::f32::consts::PI * 0.5  + (fastrand::f32() * 2.0 - 1.0) * 0.3;

        let pol_sin_alpha = alpha.sin();
        let pol_cos_alpha = alpha.cos();

        // self.verlet_physics.particles.reset_position(
        //     self.particle_pole[0],
        //     Vec3::new(pol_cos_alpha, 0.0, pol_sin_alpha),
        // );

        // self.state = State {
        //     pol_sin_alpha,
        //     pol_cos_alpha,
        //     angular_velocity: 0.0,
        // }

        *self = Self::new();
    }
}


#[derive(Clone)]
pub struct StateLeg {
    sin_alpha: f32,
    cos_alpha: f32,
    angular_velocity: f32,

//     leg_0_sin_alpha, leg_0_cos_alpha, log_0_angular_velocity
// leg_1_sin_alpha, leg_1_cos_alpha, log_1_angular_velocity
// leg_2_sin_alpha, leg_2_cos_alpha, log_2_angular_velocity
// leg_3_sin_alpha, leg_3_cos_alpha, log_3_angular_velocity
}


#[derive(Clone)]
pub struct State {
    
    pos_z: f32,
    vel_x: f32,
    vel_z: f32,

    leg_0 : StateLeg,
    leg_1 : StateLeg,
    leg_2 : StateLeg,
    leg_3 : StateLeg,

    // pol_sin_alpha: f32,
    // pol_cos_alpha: f32,
    // angular_velocity: f32,
}
