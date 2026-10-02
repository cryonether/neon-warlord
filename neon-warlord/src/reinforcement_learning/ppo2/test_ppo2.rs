

use super::*;

#[test]
fn test_one_armed_bandit() {
    const EPISODES: usize = 10_000;
    const BATCH_SIZE: usize = 32;

    let mut ppo = Ppo2::<1, 1, 64, 1, false>::new(42);

    let observation = [1.0];

    for episode in 0..EPISODES {
        let (action, _mean, log_probability) = ppo.get_action(&observation);

        // Optimal action is 2.0.
        // Maximum reward is therefore 0.0.
        let reward = -(action[0] - 2.0).powi(2);

        ppo.save_reward(
            observation,
            action,
            log_probability,
            reward,
            true,
        );

        if (episode + 1) % BATCH_SIZE == 0 {
            ppo.learn();
        }
    }

    // Evaluate without training.
    let mut mean_error = 0.0;

    for _ in 0..100 {
        let (_action, mean, _log_probability) = ppo.get_action(&observation);
        mean_error += (mean[0] - 2.0).abs();
    }

    mean_error /= 100.0;

    println!("mean error: {mean_error}");

    assert!(
        mean_error < 0.5,
        "policy failed to learn target action: mean error = {mean_error}"
    );
}

#[test]
fn test_two_armed_bandit() {
    const EPISODES: usize = 10_000;
    const BATCH_SIZE: usize = 32;

    let mut ppo = Ppo2::<1, 1, 64, 1, false>::new(42);

    let observation = [1.0];

    for episode in 0..EPISODES {
        let (action, _mean, log_probability) =
            ppo.get_action(&observation);

        let reward = if action[0] >= 0.0 {
            2.0 - (action[0] - 2.0).powi(2)
        } else {
            1.0 - (action[0] + 2.0).powi(2)
        };

        ppo.save_reward(
            observation,
            action,
            log_probability,
            reward,
            true,
        );

        if (episode + 1) % BATCH_SIZE == 0 {
            ppo.learn();
        }
    }

    let mut mean = 0.0;

    for _ in 0..100 {
        let (_action, action_mean, _log_probability) =
            ppo.get_action(&observation);

        mean += action_mean[0];
    }

    mean /= 100.0;

    println!("learned mean: {mean}");

    let mut better_arm_count = 0;

    for _ in 0..5000 {
        let (action, _mean, _log_probability) =
            ppo.get_action(&observation);

        if action[0] >= 0.0 {
            better_arm_count += 1;
        }
    }

    let better_arm_fraction =
        better_arm_count as f32 / 5000.0;

    println!("learned mean: {mean}");
    println!("better arm fraction: {better_arm_fraction:.3}");

    assert!(
        better_arm_fraction > 0.70,
        "policy failed to prefer the better arm: {better_arm_fraction:.3}"
    );

}
