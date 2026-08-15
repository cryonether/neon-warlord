//! Verlet physics particles

use wide::f32x16;

use crate::verlet_physics_simd::Vec3;

#[derive(Clone)]
pub struct VerletParticles {
    pub x: Vec<f32>,
    pub y: Vec<f32>,
    pub z: Vec<f32>,

    prev_x: Vec<f32>,
    prev_y: Vec<f32>,
    prev_z: Vec<f32>,

    pub acc_x: Vec<f32>,
    pub acc_y: Vec<f32>,
    pub acc_z: Vec<f32>,

    pub inv_mass: Vec<f32>,
    pub radius: Vec<f32>,
}

const LANES: usize = 16;

impl VerletParticles {

    pub fn new() -> Self {

        let x = Vec::new();
        let y = Vec::new();
        let z = Vec::new();
        let prev_x = Vec::new();
        let prev_y = Vec::new();
        let prev_z = Vec::new();
        let acc_x = Vec::new();
        let acc_y = Vec::new();
        let acc_z = Vec::new();
        let inv_mass = Vec::new();
        let radius = Vec::new();

        Self {
            x,
            y,
            z,
            prev_x,
            prev_y,
            prev_z,
            acc_x,
            acc_y,
            acc_z,
            inv_mass,
            radius,
        }

    }

    pub fn integrate(&mut self, dt: f32) {
        self.assert_bounds_invariants();

        let dt2_ = f32x16::splat(dt * dt);

        for(x, y, z, prev_x, prev_y, prev_z, acc_x, acc_y, acc_z) in itertools::izip!(
            self.x.chunks_exact_mut(LANES),
            self.y.chunks_exact_mut(LANES),
            self.z.chunks_exact_mut(LANES),
            self.prev_x.chunks_exact_mut(LANES),
            self.prev_y.chunks_exact_mut(LANES),
            self.prev_z.chunks_exact_mut(LANES),
            self.acc_x.chunks_exact_mut(LANES),
            self.acc_y.chunks_exact_mut(LANES),
            self.acc_z.chunks_exact_mut(LANES),
        ) {
            let x_ = f32x16::from(&*x);
            let y_ = f32x16::from(&*y);
            let z_ = f32x16::from(&*z);

            let prev_x_ = f32x16::from(&*prev_x);
            let prev_y_ = f32x16::from(&*prev_y);
            let prev_z_ = f32x16::from(&*prev_z);

            let acc_x_ = f32x16::from(&*acc_x);
            let acc_y_ = f32x16::from(&*acc_y);
            let acc_z_ = f32x16::from(&*acc_z);

            let new_x_ = x_ + (x_ - prev_x_) + acc_x_ * dt2_;
            let new_y_ = y_ + (y_ - prev_y_) + acc_y_ * dt2_;
            let new_z_ = z_ + (z_ - prev_z_) + acc_z_ * dt2_;

            // Current position becomes the previous position.
            prev_x.copy_from_slice(x);
            prev_y.copy_from_slice(y);
            prev_z.copy_from_slice(z);

            // Then advance to the new position.
            x.copy_from_slice(new_x_.as_array());
            y.copy_from_slice(new_y_.as_array());
            z.copy_from_slice(new_z_.as_array());

            // Clear acceleration
            acc_x.fill(0.0);
            acc_y.fill(0.0);
            acc_z.fill(0.0);
        }


        // Scalar remainder.
        let remainder_start = self.len() - (self.len() % LANES);
        let dt2 = dt * dt;

        for (x, y, z, prev_x, prev_y, prev_z, acc_x, acc_y, acc_z) in itertools::izip!(
            self.x[remainder_start..].iter_mut(),
            self.y[remainder_start..].iter_mut(),
            self.z[remainder_start..].iter_mut(),
            self.prev_x[remainder_start..].iter_mut(),
            self.prev_y[remainder_start..].iter_mut(),
            self.prev_z[remainder_start..].iter_mut(),
            self.acc_x[remainder_start..].iter_mut(),
            self.acc_y[remainder_start..].iter_mut(),
            self.acc_z[remainder_start..].iter_mut(),
        ) {
            let new_x = *x + (*x - *prev_x) + *acc_x * dt2;
            let new_y = *y + (*y - *prev_y) + *acc_y * dt2;
            let new_z = *z + (*z - *prev_z) + *acc_z * dt2;

            // Current position becomes the previous position.
            *prev_x = *x;
            *prev_y = *y;
            *prev_z = *z;

            // Then advance to the new position.
            *x = new_x;
            *y = new_y;
            *z = new_z;

            // Clear acceleration
            *acc_x = 0.0;
            *acc_y = 0.0;
            *acc_z = 0.0;
        }
    }

    pub fn apply_gravity(&mut self, gravity: &Vec3) {
        self.assert_bounds_invariants();

        let gravity_x_ = f32x16::splat(gravity.x);
        let gravity_y_ = f32x16::splat(gravity.y);
        let gravity_z_ = f32x16::splat(gravity.z);
        let zero_ = f32x16::splat(0.0);

        for (acc_x, acc_y, acc_z, inv_mass) in itertools::izip!(
            self.acc_x.chunks_exact_mut(LANES),
            self.acc_y.chunks_exact_mut(LANES),
            self.acc_z.chunks_exact_mut(LANES),
            self.inv_mass.chunks_exact(LANES),
        ) {
            let acc_x_ = f32x16::from(&*acc_x);
            let acc_y_ = f32x16::from(&*acc_y);
            let acc_z_ = f32x16::from(&*acc_z);
            let inv_mass_ = f32x16::from(&*inv_mass);

            // Static particles have inv_mass == 0.
            let movable_ = inv_mass_.simd_gt(zero_);

            let new_acc_x = acc_x_ + gravity_x_ * movable_;
            let new_acc_y = acc_y_ + gravity_y_ * movable_;
            let new_acc_z = acc_z_ + gravity_z_ * movable_;

            acc_x.copy_from_slice(new_acc_x.as_array());
            acc_y.copy_from_slice(new_acc_y.as_array());
            acc_z.copy_from_slice(new_acc_z.as_array());
        }

        // Scalar remainder.
        let len = self.len();
        let remainder_start = len - (len % LANES);
        for (acc_x, acc_y, acc_z, &inv_mass) in itertools::izip!(
            self.acc_x[remainder_start..].iter_mut(),
            self.acc_y[remainder_start..].iter_mut(),
            self.acc_z[remainder_start..].iter_mut(),
            self.inv_mass[remainder_start..].iter(),
        ) {
            if inv_mass > 0.0 {
                *acc_x += gravity.x;
                *acc_y += gravity.y;
                *acc_z += gravity.z;
            }
        }
    }

    fn assert_bounds_invariants(&self) {
        let len = self.len();

        assert!(self.x.len() == len);
        assert!(self.y.len() == len);
        assert!(self.z.len() == len);

        assert!(self.prev_x.len() == len);
        assert!(self.prev_y.len() == len);
        assert!(self.prev_z.len() == len);

        assert!(self.acc_x.len() == len);
        assert!(self.acc_y.len() == len);
        assert!(self.acc_z.len() == len);

        assert!(self.inv_mass.len() == len);
        assert!(self.radius.len() == len);
    }

    pub fn len(&self) -> usize {
        self.x.len()
    }

    pub fn push(&mut self, pos: Vec3, radius: f32, mass: f32) -> usize {
        let index = self.len();

        self.x.push(pos.x);
        self.y.push(pos.y);
        self.z.push(pos.z);

        self.prev_x.push(pos.x);
        self.prev_y.push(pos.y);
        self.prev_z.push(pos.z);

        self.acc_x.push(0.0);
        self.acc_y.push(0.0);
        self.acc_z.push(0.0);

        self.inv_mass.push(1.0 / mass);
        self.radius.push(radius);

        index
    }
    
    pub fn position(&self, index: usize) -> Vec3 {
        let x = self.x[index];
        let y = self.y[index];
        let z = self.z[index];
        Vec3::new(x, y, z)
    }

    pub fn set_position(&mut self, index: usize, pos: Vec3) {
        self.x[index] = pos.x;
        self.y[index] = pos.y;
        self.z[index] = pos.z;
    }

    pub fn accelerate(&mut self, index: usize, acc: Vec3) {
        self.acc_x[index] = acc.x;
        self.acc_y[index] = acc.y;
        self.acc_z[index] = acc.z;
    }

}