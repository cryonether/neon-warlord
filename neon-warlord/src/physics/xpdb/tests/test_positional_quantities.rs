//! Test of RotationalQuantities

use super::super::*;

#[test]
fn test_integrate_position() {
   let dt = 0.1;

    let mut quantities = PositionalQuantities {
        pos: Vec3::new(0.0, 10.0, 0.0),
        prev_pos: Vec3::new(0.0, 10.0, 0.0),
        vel: Vec3::new(1.0, 0.0, 0.0),
        // mass: 2.0,
        inv_mass: 0.5,
    };

    quantities.integrate_position(dt);

    // Previous position should be saved.
    assert_eq!(
        quantities.prev_pos,
        Vec3::new(0.0, 10.0, 0.0)
    );

    // Gravity should affect velocity.
    let expected_vel = Vec3::new(1.0, 0.0, 0.0)
        + dt * GRAVITY;

    assert!(
        (quantities.vel - expected_vel).magnitude() < 1e-5
    );

    // Position should be updated using the new velocity.
    let expected_pos = Vec3::new(0.0, 10.0, 0.0)
        + dt * expected_vel;

    assert!(
        (quantities.pos - expected_pos).magnitude() < 1e-5
    );
}

#[test]
fn test_integrate_velocity() {
    let dt = 0.5;

    let mut quantities = PositionalQuantities {
        pos: Vec3::new(3.0, 4.0, 5.0),
        prev_pos: Vec3::new(1.0, 2.0, 3.0),
        vel: Vec3::new(0.0, 0.0, 0.0),
        // mass: 1.0,
        inv_mass: 1.0,
    };

    quantities.update_velocity(dt);

    // Velocity = (current position - previous position) / dt.
    let expected_vel = Vec3::new(4.0, 4.0, 4.0);

    assert!(
        (quantities.vel - expected_vel).magnitude() < 1e-5
    );
}

#[test]
fn test_steps() {
    let dt = 0.1;
    let steps = 10;

    let initial_pos = Vec3::new(0.0, 10.0, 0.0);
    let initial_vel = Vec3::new(1.0, 0.0, 0.0);

    let mut quantities = PositionalQuantities {
        pos: initial_pos,
        prev_pos: initial_pos,
        vel: initial_vel,
        // mass: 1.0,
        inv_mass: 1.0,
    };

    for _ in 0..steps {
        quantities.integrate_position(dt);
        quantities.update_velocity(dt);
    }

    // Expected velocity after constant gravity acceleration.
    let expected_vel = initial_vel + GRAVITY * (dt * steps as f32);

    assert!(
        (quantities.vel - expected_vel).magnitude() < 1e-4,
        "Expected velocity {:?}, got {:?}",
        expected_vel,
        quantities.vel
    );

    // Semi-implicit Euler integration:
    // p(n) = p(0) + n*dt*v(0) + dt²*a*n*(n+1)/2
    let n = steps as f32;
    let expected_pos = initial_pos
        + initial_vel * (n * dt)
        + GRAVITY * (dt * dt * n * (n + 1.0) / 2.0);

    assert!(
        (quantities.pos - expected_pos).magnitude() < 1e-4,
        "Expected position {:?}, got {:?}",
        expected_pos,
        quantities.pos
    );
}