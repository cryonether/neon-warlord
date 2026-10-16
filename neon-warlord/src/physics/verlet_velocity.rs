//! Velocity Verlet integration physics
//! 
//! https://www.algorithm-archive.org/contents/verlet_integration/verlet_integration.html
//! 

use cgmath::{InnerSpace, Zero};

type Vec3 = cgmath::Vector3<f32>;

/// Velocity Verlet integration physics
pub struct VerletVelocity {
    pos_x: Vec<f32>,
    pos_y: Vec<f32>,
    pos_z: Vec<f32>,

    vel_half_x: Vec<f32>,
    vel_half_y: Vec<f32>,
    vel_half_z: Vec<f32>,

    vel_x: Vec<f32>,
    vel_y: Vec<f32>,
    vel_z: Vec<f32>,

    acc_x: Vec<f32>,
    acc_y: Vec<f32>,
    acc_z: Vec<f32>,

    mass: Vec<f32>,
    inv_mass: Vec<f32>,
    radius: Vec<f32>,

    // constraints
    distance_constraints: DistanceConstraints,
    angle_constraints: AngleConstraints,
}

impl VerletVelocity {

    /// Run a single step
    pub fn step(&mut self) {
        // running at fixed step size
        const DT: f32 = 1.0 / 60.0;

        // Update the position using the old velocity and acceleration
        self.integrate_position(DT);        

        self.solve_ground_collision();

        self.zero_acc();

        // Recalculate the acceleration at the new position
        self.apply_gravity();
        self.apply_distance_constraints();
        self.apply_angle_constraints();
        
        // Update the position using the old velocity and acceleration
        self.integrate_velocity(DT);
    }

    /// Velocity Verlet integration
    fn integrate_position(&mut self, dt: f32) {
        for i in 0..self.len() {
            if self.inv_mass[i] == 0.0 {
                // static particle
                continue;
            }

            let mut pos = self.get_pos(i);
            let vel = self.get_vel(i);
            let acc = self.get_acc(i);

            let vel_half = vel + 0.5 * acc * dt;
            pos += vel_half * dt;

            self.set_pos(i, pos);
            self.set_vel_half(i, vel_half);
        }
    }

    fn zero_acc(&mut self) {
        for i in 0..self.len() {
            self.set_acc(i,  Vec3::zero());
        }
    }

    /// Velocity Verlet integration
    fn integrate_velocity(&mut self, dt: f32) {
        for i in 0..self.len() {
            if self.inv_mass[i] == 0.0 {
                // static particle
                continue;
            }

            let vel_half = self.get_vel_half(i);
            let acc = self.get_acc(i);

            let vel = vel_half + 0.5 * acc * dt;

            self.set_vel(i, vel);
        }
    }

    /// Apply gravity
    fn apply_gravity(&mut self) {
        const GRAVITY: Vec3 = Vec3::new(0.0, 0.0, -10.0);
    
        for i in 0..self.len() {
            let mut acc = self.get_acc(i);
            let inv_mass = self.inv_mass[i];

            // static particles have inv_mass == 0
            if inv_mass > 0.0 {
                acc += GRAVITY;
            }

            self.set_acc(i, acc);
        }
    }

    /// Models distance constraints as springs
    fn apply_distance_constraints(&mut self) {
        for i in 0..self.distance_constraints.len() {
            let index_a = self.distance_constraints.a[i];
            let index_b = self.distance_constraints.b[i];
            let rest_length = self.distance_constraints.distance[i];
            let stiffness = self.distance_constraints.stiffness[i];
            let damping = self.distance_constraints.damping[i];

            let inv_mass_a = self.inv_mass[index_a];
            let inv_mass_b = self.inv_mass[index_b];

            // Both particles are static.
            if inv_mass_a == 0.0 && inv_mass_b == 0.0 {
                continue;
            }

            // Particle a
            let pos_a = self.get_pos(index_a);
            let vel_a = self.get_vel_half(index_a);
            let mut acc_a = self.get_acc(index_a);

            // Particle b
            let pos_b = self.get_pos(index_b);
            let vel_b = self.get_vel_half(index_b);
            let mut acc_b = self.get_acc(index_b);

            let delta = pos_b - pos_a;
            let distance = delta.magnitude();

            // Avoid division by zero
            if distance < 1e-6 {
                continue;
            }

            let direction = delta / distance;
            let extension = distance - rest_length;

            let relative_speed = (vel_b - vel_a).dot(direction);    // half velocity is used as approximation for damping

            let force_magnitude = stiffness * extension + damping * relative_speed;

            let force = direction * force_magnitude;

            // Acceleration = force / mass.
            // inv_mass == 0 for static particles.
            acc_a += force * inv_mass_a;

            acc_b -= force * inv_mass_b;

            self.set_acc(index_a, acc_a);
            self.set_acc(index_b, acc_b);
        }
    }

    /// Models an angle constraint between three particles
    fn apply_angle_constraints(&mut self) {
        for i in 0..self.angle_constraints.len() {
            let index_a = self.angle_constraints.a[i];
            let index_b = self.angle_constraints.b[i];
            let index_c = self.angle_constraints.c[i];

            let target_angle = self.angle_constraints.angle[i];
            let stiffness = self.angle_constraints.stiffness[i];
            let damping = self.angle_constraints.damping[i];

            let pos_a = self.get_pos(index_a);
            let pos_b = self.get_pos(index_b);
            let pos_c = self.get_pos(index_c);

            let vel_a = self.get_vel_half(index_a);
            let vel_b = self.get_vel_half(index_b);
            let vel_c = self.get_vel_half(index_c);
            
            let u = pos_a - pos_b;
            let v = pos_c - pos_b;

            let len_u = u.magnitude();
            let len_v = v.magnitude();

            if len_u < 1e-6 || len_v < 1e-6 {
                continue;
            }

            let u_hat = u / len_u;
            let v_hat = v / len_v;

            let cos_angle = u_hat.dot(v_hat).clamp(-1.0, 1.0);
            let sin_angle = (1.0 - cos_angle * cos_angle).sqrt();

            // The angle gradient is singular at 0 and PI.
            if sin_angle < 1e-5 {
                continue;
            }

            let angle = cos_angle.acos();

            // Gradients of cos(angle) with respect to each position.
            let grad_a = -(v_hat - cos_angle * u_hat) / (len_u * sin_angle);
            let grad_c = -(u_hat - cos_angle * v_hat) / (len_v * sin_angle);
            let grad_b = -(grad_a + grad_c);

            let angular_velocity = grad_a.dot(vel_a) + grad_b.dot(vel_b) + grad_c.dot(vel_c);
            let torque = -stiffness * (angle - target_angle) -damping * angular_velocity;

            let force_a = torque * grad_a;
            let force_b = torque * grad_b;
            let force_c = torque * grad_c;

            let mut acc_a = self.get_acc(index_a);
            let mut acc_b = self.get_acc(index_b);
            let mut acc_c = self.get_acc(index_c);

            // Apply forces as accelerations using inverse masses.
            acc_a += force_a * self.inv_mass[index_a];
            acc_b += force_b * self.inv_mass[index_b];
            acc_c += force_c * self.inv_mass[index_c];

            self.set_acc(index_a, acc_a);
            self.set_acc(index_b, acc_b);
            self.set_acc(index_c, acc_c);
        }
    }

    /// Collision with the ground plane at z == 0.0
    fn solve_ground_collision(&mut self) {
        const GROUND_HEIGHT_Z: f32 = 0.0;
        const RESTITUTION: f32 = 0.9;
        const FRICTION: f32 = 0.1;

        for i in 0..self.len() {
            if self.inv_mass[i] == 0.0 {
                // static particle
                continue;
            }

            let mut pos = self.get_pos(i);
            let mut vel = self.get_vel_half(i);

            let min_z = GROUND_HEIGHT_Z + self.radius[i];

            if pos.z >= min_z {
                continue;
            }

            // Prevent penetration.
            pos.z = min_z;

            // Bounce only if moving into the ground.
            if vel.z < 0.0 {
                vel.z = -RESTITUTION * vel.z;

                // Reduce tangential velocity due to friction.
                vel.x *= 1.0 - FRICTION;
                vel.y *= 1.0 - FRICTION;
            }

            self.set_pos(i, pos);
            self.set_vel_half(i, vel);
        }
    }


    pub fn get_pos(&self, i: usize) -> Vec3 {
        Vec3::new(
            self.pos_x[i], 
            self.pos_y[i], 
            self.pos_z[i]
        )
    }

    pub fn get_vel_half(&self, i: usize) -> Vec3 {
        Vec3::new(
            self.vel_half_x[i], 
            self.vel_half_y[i], 
            self.vel_half_z[i]
        )
    }

    pub fn get_vel(&self, i: usize) -> Vec3 {
        Vec3::new(
            self.vel_x[i], 
            self.vel_y[i], 
            self.vel_z[i]
        )
    }

    pub fn get_acc(&self, i: usize) -> Vec3 {
        Vec3::new(
            self.acc_x[i], 
            self.acc_y[i], 
            self.acc_z[i]
        )
    }

    pub fn set_pos(&mut self, i: usize, val: Vec3) {
        self.pos_x[i] = val.x;
        self.pos_y[i] = val.y;
        self.pos_z[i] = val.z;
    }

    pub fn set_vel_half(&mut self, i: usize, val: Vec3) {
        self.vel_half_x[i] = val.x;
        self.vel_half_y[i] = val.y;
        self.vel_half_z[i] = val.z;
    }

    pub fn set_vel(&mut self, i: usize, val: Vec3) {
        self.vel_x[i] = val.x;
        self.vel_y[i] = val.y;
        self.vel_z[i] = val.z;
    }

    pub fn set_acc(&mut self, i: usize, val: Vec3) {
        self.acc_x[i] = val.x;
        self.acc_y[i] = val.y;
        self.acc_z[i] = val.z;
    }

    fn assert_bounds_invariants(&self) {
        let len = self.pos_x.len();

        assert!(self.pos_x.len() == len);
        assert!(self.pos_y.len() == len);
        assert!(self.pos_z.len() == len);

        assert!(self.vel_half_x.len() == len);
        assert!(self.vel_half_y.len() == len);
        assert!(self.vel_half_z.len() == len);

        assert!(self.vel_x.len() == len);
        assert!(self.vel_y.len() == len);
        assert!(self.vel_z.len() == len);

        assert!(self.acc_x.len() == len);
        assert!(self.acc_y.len() == len);
        assert!(self.acc_z.len() == len);

        assert!(self.inv_mass.len() == len);
        assert!(self.radius.len() == len);
    }

    pub fn len(&self) -> usize {
        self.assert_bounds_invariants();

        self.pos_x.len()
    }

}


struct DistanceConstraints {
    pub a: Vec<usize>,
    pub b: Vec<usize>,

    pub distance: Vec<f32>,
    pub stiffness: Vec<f32>,
    pub damping: Vec<f32>,
}

impl DistanceConstraints {
    pub fn len(&self) -> usize {
        self.assert_bounds_invariants();

        self.a.len()
    }

    fn assert_bounds_invariants(&self) {
        let len = self.a.len();

        assert!(self.a.len() == len);
        assert!(self.b.len() == len);
        assert!(self.distance.len() == len);
        assert!(self.stiffness.len() == len);
        assert!(self.damping.len() == len);
    }
}


struct AngleConstraints {
    pub a: Vec<usize>,
    pub b: Vec<usize>,
    pub c: Vec<usize>,

    pub angle: Vec<f32>,
    pub stiffness: Vec<f32>,
    pub damping: Vec<f32>,
}

impl AngleConstraints {
    pub fn len(&self) -> usize {
        self.assert_bounds_invariants();

        self.a.len()
    }

    fn assert_bounds_invariants(&self) {
        let len = self.a.len();

        assert!(self.a.len() == len);
        assert!(self.b.len() == len);
        assert!(self.c.len() == len);
        assert!(self.angle.len() == len);
        assert!(self.stiffness.len() == len);
        assert!(self.damping.len() == len);
    }
}