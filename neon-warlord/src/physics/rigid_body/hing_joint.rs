//! A joint connecting two rigid bodies

use cgmath::{InnerSpace, Rotation, Vector3};

use crate::physics::rigid_body::RigidBody;

type Vec3 = Vector3<f32>;

#[derive(Clone, Debug)]
pub struct HingeJoint {
    pub body_a: usize,
    pub body_b: usize,

    pub local_anchor_a: Vec3,
    pub local_anchor_b: Vec3,

    pub local_axis_a: Vec3,
    pub local_axis_b: Vec3,

    /// Baumgarte stabilization factor, typically 0.1–0.2.
    pub beta: f32,
}

impl HingeJoint {
    pub fn new(
        body_a: usize,
        body_b: usize,
        local_anchor_a: Vec3,
        local_anchor_b: Vec3,
        local_axis_a: Vec3,
        local_axis_b: Vec3,
    ) -> Self {
        assert_ne!(body_a, body_b);
        assert!(local_axis_a.magnitude2() > 1e-12);
        assert!(local_axis_b.magnitude2() > 1e-12);

        Self {
            body_a,
            body_b,
            local_anchor_a,
            local_anchor_b,
            local_axis_a: local_axis_a.normalize(),
            local_axis_b: local_axis_b.normalize(),
            beta: 0.2,
        }
    }

    /// Solve the five bilateral hinge constraints.
    ///
    /// Call after external forces have updated velocities
    /// and before advancing body positions.
    pub fn solve(
        &self,
        bodies: &mut [RigidBody],
        dt: f32,
        iterations: usize,
    ) {
        assert!(dt > 0.0);
        assert!(self.body_a < bodies.len());
        assert!(self.body_b < bodies.len());

        let (a, b) = get_two_mut(
            bodies,
            self.body_a,
            self.body_b,
        );

        for _ in 0..iterations {
            // 1. Three translational constraints:
            // the two anchor points must coincide.
            let anchor_a = a.position
                + a.orientation.rotate_vector(self.local_anchor_a);

            let anchor_b = b.position
                + b.orientation.rotate_vector(self.local_anchor_b);

            let error = anchor_b - anchor_a;

            solve_anchor_axis(
                a, b, anchor_a, anchor_b,
                Vec3::unit_x(), error, dt, self.beta,
            );

            solve_anchor_axis(
                a, b, anchor_a, anchor_b,
                Vec3::unit_y(), error, dt, self.beta,
            );

            solve_anchor_axis(
                a, b, anchor_a, anchor_b,
                Vec3::unit_z(), error, dt, self.beta,
            );

            // 2. Two rotational constraints:
            // the hinge axes must align.
            let axis_a = a.orientation
                .rotate_vector(self.local_axis_a)
                .normalize();

            let axis_b = b.orientation
                .rotate_vector(self.local_axis_b)
                .normalize();

            let angular_error = axis_a.cross(axis_b);

            // Construct a tangent basis perpendicular to the
            // average hinge axis.
            let sum = axis_a + axis_b;

            let hinge_axis = if sum.magnitude2() > 1e-10 {
                sum.normalize()
            } else {
                axis_a
            };

            let helper = if hinge_axis.x.abs() < 0.8 {
                Vec3::unit_x()
            } else {
                Vec3::unit_y()
            };

            let tangent_1 = hinge_axis
                .cross(helper)
                .normalize();

            let tangent_2 = hinge_axis
                .cross(tangent_1)
                .normalize();

            solve_angular_axis(
                a, b, tangent_1,
                angular_error.dot(tangent_1),
                dt, self.beta,
            );

            solve_angular_axis(
                a, b, tangent_2,
                angular_error.dot(tangent_2),
                dt, self.beta,
            );
        }
    }
}

fn solve_anchor_axis(
    a: &mut RigidBody,
    b: &mut RigidBody,
    anchor_a: Vec3,
    anchor_b: Vec3,
    axis: Vec3,
    error: Vec3,
    dt: f32,
    beta: f32,
) {
    let r_a = anchor_a - a.position;
    let r_b = anchor_b - b.position;

    let v_a = a.linear_velocity
        + a.angular_velocity.cross(r_a);

    let v_b = b.linear_velocity
        + b.angular_velocity.cross(r_b);

    let relative_velocity = (v_b - v_a).dot(axis);

    let ra_cross_n = r_a.cross(axis);
    let rb_cross_n = r_b.cross(axis);

    let inv_i_a = a.inv_inertia_world();
    let inv_i_b = b.inv_inertia_world();

    let effective_mass =
        a.inv_mass
        + b.inv_mass
        + ra_cross_n.dot(inv_i_a * ra_cross_n)
        + rb_cross_n.dot(inv_i_b * rb_cross_n);

    if effective_mass < 1e-10 {
        return;
    }

    // Velocity-level constraint with positional stabilization.
    let bias = beta / dt * error.dot(axis);

    let lambda =
        -(relative_velocity + bias) / effective_mass;

    let impulse = axis * lambda;

    a.apply_impulse(-impulse, anchor_a);
    b.apply_impulse(impulse, anchor_b);
}

fn solve_angular_axis(
    a: &mut RigidBody,
    b: &mut RigidBody,
    axis: Vec3,
    error: f32,
    dt: f32,
    beta: f32,
) {
    let relative_velocity =
        (b.angular_velocity - a.angular_velocity).dot(axis);

    let inv_i_a = a.inv_inertia_world();
    let inv_i_b = b.inv_inertia_world();

    let effective_mass =
        axis.dot(inv_i_a * axis)
        + axis.dot(inv_i_b * axis);

    if effective_mass < 1e-10 {
        return;
    }

    let bias = beta / dt * error;

    let lambda =
        -(relative_velocity + bias) / effective_mass;

    let impulse = axis * lambda;

    a.apply_angular_impulse(-impulse);
    b.apply_angular_impulse(impulse);
}

fn get_two_mut<T>(
    items: &mut [T],
    i: usize,
    j: usize,
) -> (&mut T, &mut T) {
    assert_ne!(i, j);

    if i < j {
        let (left, right) = items.split_at_mut(j);
        (&mut left[i], &mut right[0])
    } else {
        let (left, right) = items.split_at_mut(i);
        (&mut right[0], &mut left[j])
    }
}