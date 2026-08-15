//! Solves collision for verlet physics

use cgmath::InnerSpace;
use forward_renderer::height_map::HeightMapInterface;
use noise::NoiseFn;
use wgpu_renderer::performance_monitor::watch::Watch;

use crate::{
    verlet_physics::{self, Vec3, VerletObject, verlet_composition::VerletComposition},
};

pub struct Solver {
    perlin: noise::Perlin,
    ticks: u64,
}

impl Solver {
    pub fn new() -> Self {
        let perlin: noise::Perlin = noise::Perlin::new(1);

        Self { perlin, ticks: 0 }
    }

    pub fn update(
        &mut self,
        verlet_objects: &mut [VerletObject],
        links: &[verlet_physics::link::Link],
        fixed_links: &[verlet_physics::fixed_link::FixedLink],
        fixed: &[verlet_physics::fixed::Fixed],
        dt: f32,
    ) {
        let sub_steps = 1;
        let sub_dt = dt / sub_steps as f32;

        for _i in 0..sub_steps {
            Self::apply_gravity(verlet_objects);
            self.apply_wind(verlet_objects);
            // Self::apply_constraint(verlet_objects);
            for elem in fixed {
                elem.apply(verlet_objects);
            }
            for link in links {
                link.apply(verlet_objects);
            }
            for link in fixed_links {
                link.apply(verlet_objects);
            }
            // Self::solve_collisions(verlet_objects);
            Self::update_positions(verlet_objects, sub_dt);
        }

        self.ticks += 1;
    }

    pub fn _update_composites(
        &self,
        verlet_compositions: &mut [VerletComposition],
        height_map: &impl HeightMapInterface,
        dt: f32,
    ) {
        for composition in verlet_compositions {
            let verlet_objects = &mut composition.verlet_objects;

            // gravity
            Self::apply_gravity(verlet_objects);
            Self::apply_map_constraint(verlet_objects, height_map);

            // constraints
            for elem in &composition.fixed {
                elem.apply(verlet_objects);
            }
            for elem in &composition.links {
                elem.apply(verlet_objects);
            }
            for elem in &composition.fixed_links {
                elem.apply(verlet_objects);
            }
            for elem in &composition.sticky_links {
                elem.apply(verlet_objects);
            }

            // physics equation
            Self::update_positions(verlet_objects, dt);
        }
    }

    fn update_positions(verlet_objects: &mut [VerletObject], dt: f32) {
        for elem in verlet_objects {
            elem.update_position(dt);
        }
    }

    fn apply_gravity(verlet_objects: &mut [VerletObject]) {
        const GRAVITY: Vec3 = Vec3::new(0.0, 0.0, -1.0);

        for elem in verlet_objects {
            elem.accelerate(GRAVITY);
        }
    }

    fn apply_wind(&mut self, verlet_objects: &mut [VerletObject]) {
        use std::f64::consts::PI;
        let ticks_for_half_cycle: u64 = 1000000;
        let ticks: u64 = self.ticks;
        let seed = (ticks % ticks_for_half_cycle) as f64 / ticks_for_half_cycle as f64 * PI;
        let seed_cos = seed.cos() * 1000.0;
        let seed_sin = seed.sin() * 1000.0;

        for elem in verlet_objects {
            let x = elem.position().x as f64 / 2.0 + seed_cos;
            let y = elem.position().y as f64 / 2.0 + seed_sin;

            let force = Vec3::new(
                self.perlin.get([x, y]) as f32,
                self.perlin.get([x + 1000.0, y + 1000.0]) as f32,
                0.0,
            );

            elem.accelerate(force * 40.0);
        }
    }

    fn _apply_constraint(verlet_objects: &mut [VerletObject]) {
        const POSITION: Vec3 = Vec3::new(-3.0, 0.0, 10.0);
        const RADIUS: f32 = 10.0;

        for elem in verlet_objects {
            let to_obj = elem.position() - POSITION;
            let dist = to_obj.magnitude();

            if dist > RADIUS - elem.radius() {
                let n = to_obj / dist;
                let new_pos = POSITION + n * (RADIUS - elem.radius());

                elem.set_position(new_pos);
            }
        }
    }

    fn apply_map_constraint(
        verlet_objects: &mut [VerletObject],
        height_map: &impl HeightMapInterface,
    ) {
        for elem in verlet_objects {
            let pos = elem.position();
            let radius = elem.radius();
            let height = height_map.get_height(&pos);

            if pos.z - radius < height {
                let mut new_pos = pos;
                new_pos.z = height + radius;

                elem.set_position(new_pos);
            }
        }
    }

    fn _solve_collisions(verlet_objects: &mut [VerletObject]) {
        let object_count = verlet_objects.len();

        for i in 0..object_count {
            for k in i + 1..object_count {
                let (left, right) = verlet_objects.split_at_mut(k);

                let object_1 = &mut left[i];
                let object_2 = &mut right[0];

                let collision_axis = object_1.position() - object_2.position();
                let dist = collision_axis.magnitude();
                let min_dist = object_1.radius + object_2.radius;
                if dist < min_dist {
                    let n = collision_axis / dist;
                    let delta = min_dist - dist;
                    object_1.set_position(object_1.position() + 0.5 * delta * n);
                    object_2.set_position(object_2.position() - 0.5 * delta * n);
                }
            }
        }
    }
}
