use super::*;

/// actor learns a continuous optimum
#[test]
fn test_one_armed_bandit() {
    const EPISODES: usize = 2_000;
    const BATCH_SIZE: usize = 32;

    let mut ppo = Ppo2::<1, 1, 64, 1, false>::new(42);

    let observation = [1.0];

    for episode in 0..EPISODES {
        let (action, _mean, log_probability) = ppo.get_action(&observation);

        // Optimal action is 2.0.
        // Maximum reward is therefore 0.0.
        let reward = -(action[0] - 2.0).powi(2);

        ppo.save_reward(observation, action, log_probability, reward, true);

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

/// Actor learns to prefer the higher-value region.
#[test]
fn test_two_armed_bandit() {
    const EPISODES: usize = 2_000;
    const BATCH_SIZE: usize = 32;
    const EVALUATION_SAMPLES: usize = 5_000;

    let mut ppo = Ppo2::<1, 1, 64, 1, false>::new(42);

    let observation = [1.0];

    for episode in 0..EPISODES {
        let (action, _mean, log_probability) = ppo.get_action(&observation);

        let x = action[0];

        let reward = if x >= 0.0 {
            2.0 - (x - 2.0).powi(2)
        } else {
            1.0 - (x + 2.0).powi(2)
        };

        ppo.save_reward(observation, action, log_probability, reward, true);

        if (episode + 1) % BATCH_SIZE == 0 {
            ppo.learn();
        }
    }

    let mean = ppo.actor.forward(&observation)[0];

    println!("learned mean: {mean}");

    let mut better_arm_count = 0;
    let mut total_reward = 0.0;

    for _ in 0..EVALUATION_SAMPLES {
        let (action, _mean, _log_probability) = ppo.get_action(&observation);

        let x = action[0];

        let reward = if x >= 0.0 {
            better_arm_count += 1;

            2.0 - (x - 2.0).powi(2)
        } else {
            1.0 - (x + 2.0).powi(2)
        };

        total_reward += reward;
    }

    let better_arm_fraction = better_arm_count as f32 / EVALUATION_SAMPLES as f32;

    let average_reward = total_reward / EVALUATION_SAMPLES as f32;

    println!("better arm fraction: {better_arm_fraction:.3}");

    println!("average reward: {average_reward:.3}");

    assert!(
        better_arm_fraction > 0.70,
        "policy failed to prefer the better arm: {better_arm_fraction:.3}"
    );

    assert!(
        average_reward > 1.0,
        "policy reward too low: {average_reward:.3}"
    );
}

/// delayed reward propagates backward
#[test]
fn test_two_step_mdp() {
    const EPISODES: usize = 4_000;
    const BATCH_SIZE: usize = 32;

    let mut ppo = Ppo2::<1, 1, 64, 1, false>::new(42);

    // State 0: the agent must choose a positive action
    // to reach the rewarding state.
    let start_state = [0.0];

    // State 1: positive action gives +1.
    let good_state = [1.0];

    // State 2: negative action from the start state
    // leads here and eventually gives -1.
    let bad_state = [-1.0];

    for episode in 0..EPISODES {
        //
        // STEP 1
        //
        let (action, _mean, log_probability) = ppo.get_action(&start_state);

        // No immediate reward.
        let reward = 0.0;

        // Positive action leads to the good state.
        // Negative action leads to the bad state.
        let next_state = if action[0] >= 0.0 {
            good_state
        } else {
            bad_state
        };

        ppo.save_reward(start_state, action, log_probability, reward, false);

        //
        // STEP 2
        //
        let (action, _mean, log_probability) = ppo.get_action(&next_state);

        let reward = if next_state == good_state {
            if action[0] >= 0.0 { 1.0 } else { -1.0 }
        } else {
            -1.0
        };

        ppo.save_reward(next_state, action, log_probability, reward, true);

        //
        // Update PPO using the complete episode.
        //
        if (episode + 1) % BATCH_SIZE == 0 {
            ppo.learn();
        }
    }

    // Evaluate the first decision.
    let mut good_path_count = 0;

    for _ in 0..5_000 {
        let (action, _mean, _log_probability) = ppo.get_action(&start_state);

        if action[0] >= 0.0 {
            good_path_count += 1;
        }
    }

    let good_path_fraction = good_path_count as f32 / 5_000.0;

    let start_mean = ppo.actor.forward(&start_state)[0];

    println!("start state mean: {start_mean}");
    println!("good path fraction: {good_path_fraction:.3}");

    // Evaluate the second decision in the good state.
    let mut good_action_count = 0;

    for _ in 0..5_000 {
        let (action, _mean, _log_probability) = ppo.get_action(&good_state);

        if action[0] >= 0.0 {
            good_action_count += 1;
        }
    }

    let good_action_fraction = good_action_count as f32 / 5_000.0;

    let good_state_mean = ppo.actor.forward(&good_state)[0];

    println!("good state mean: {good_state_mean}");
    println!(
        "good state positive action fraction: \
         {good_action_fraction:.3}"
    );

    // Critic evaluation.
    //
    // The good state gives +1 on the next step:
    //
    //     V(good_state) ~= 1
    //
    // The start state receives zero immediately and
    // then transitions to the good state:
    //
    //     V(start_state) ~= gamma * 1
    //                    ~= 0.95
    //
    let start_value = ppo.critic.forward(&start_state)[0];

    let good_state_value = ppo.critic.forward(&good_state)[0];

    println!("start state value: {start_value}");
    println!("good state value: {good_state_value}");

    assert!(
        good_path_fraction > 0.70,
        "policy failed to learn the good path: \
         fraction = {good_path_fraction:.3}"
    );

    assert!(
        good_action_fraction > 0.70,
        "policy failed to learn the final action: \
         fraction = {good_action_fraction:.3}"
    );

    assert!(
        (start_value - 0.95).abs() < 0.20,
        "critic failed to learn start-state value: \
         value = {start_value}"
    );

    assert!(
        (good_state_value - 1.0).abs() < 0.20,
        "critic failed to learn good-state value: \
         value = {good_state_value}"
    );
}

/// done/reset handling works
#[test]
fn test_episode_boundaries() {
    const EPISODES: usize = 4_000;
    const BATCH_SIZE: usize = 32;

    let mut ppo = Ppo2::<1, 1, 64, 1, false>::new(42);

    let observation = [0.0];

    for episode in 0..EPISODES {
        let (action, _mean, log_probability) = ppo.get_action(&observation);

        // Each transition is a complete one-step episode.
        //
        // Correct return calculation:
        //
        //   episode 0: reward = +1 -> return = +1
        //   episode 1: reward = -1 -> return = -1
        //   episode 2: reward = +1 -> return = +1
        //   ...
        //
        // Therefore the critic should learn V(s) ~= 0.
        let reward = if episode % 2 == 0 { 1.0 } else { -1.0 };

        ppo.save_reward(observation, action, log_probability, reward, true);

        if (episode + 1) % BATCH_SIZE == 0 {
            ppo.learn();
        }
    }

    let learned_value = ppo.critic.forward(&observation)[0];

    println!("learned value: {learned_value}");

    // Because positive and negative episodes occur equally often,
    // the expected value of this state is zero.
    assert!(
        learned_value.abs() < 0.1,
        "episode boundary leakage detected: \
         learned value = {learned_value}"
    );
}

/// Delayed reward propagates backward through multiple steps.
#[test]
fn test_multi_step_mdp() {
    const EPISODES: usize = 6_000;
    const BATCH_SIZE: usize = 32;
    const EVALUATION_SAMPLES: usize = 5_000;

    let mut ppo = Ppo2::<1, 1, 64, 1, false>::new(42);

    // State 0: choosing positive leads toward the reward.
    let start_state = [0.0];

    // State 1: choosing positive continues toward the reward.
    let middle_state = [1.0];

    // State 2: choosing positive receives +1.
    let good_state = [2.0];

    // Bad states terminate with -1.
    let bad_state = [-1.0];

    for episode in 0..EPISODES {
        //
        // STEP 1
        //
        let (action, _mean, log_probability) = ppo.get_action(&start_state);

        let next_state = if action[0] >= 0.0 {
            middle_state
        } else {
            bad_state
        };

        ppo.save_reward(start_state, action, log_probability, 0.0, false);

        //
        // STEP 2
        //
        let (action, _mean, log_probability) = ppo.get_action(&next_state);

        if next_state == bad_state {
            ppo.save_reward(next_state, action, log_probability, -1.0, true);

            if (episode + 1) % BATCH_SIZE == 0 {
                ppo.learn();
            }

            continue;
        }

        let next_state = if action[0] >= 0.0 {
            good_state
        } else {
            bad_state
        };

        ppo.save_reward(middle_state, action, log_probability, 0.0, false);

        //
        // STEP 3
        //
        let (action, _mean, log_probability) = ppo.get_action(&next_state);

        let reward = if next_state == good_state {
            if action[0] >= 0.0 { 1.0 } else { -1.0 }
        } else {
            -1.0
        };

        ppo.save_reward(next_state, action, log_probability, reward, true);

        //
        // Update PPO using the complete episode.
        //
        if (episode + 1) % BATCH_SIZE == 0 {
            ppo.learn();
        }
    }

    //
    // Actor evaluation.
    //

    let mut start_good_count = 0;
    let mut middle_good_count = 0;
    let mut final_good_count = 0;

    for _ in 0..EVALUATION_SAMPLES {
        let (action, _mean, _log_probability) = ppo.get_action(&start_state);

        if action[0] >= 0.0 {
            start_good_count += 1;
        }

        let (action, _mean, _log_probability) = ppo.get_action(&middle_state);

        if action[0] >= 0.0 {
            middle_good_count += 1;
        }

        let (action, _mean, _log_probability) = ppo.get_action(&good_state);

        if action[0] >= 0.0 {
            final_good_count += 1;
        }
    }

    let start_good_fraction = start_good_count as f32 / EVALUATION_SAMPLES as f32;

    let middle_good_fraction = middle_good_count as f32 / EVALUATION_SAMPLES as f32;

    let final_good_fraction = final_good_count as f32 / EVALUATION_SAMPLES as f32;

    let start_mean = ppo.actor.forward(&start_state)[0];

    let middle_mean = ppo.actor.forward(&middle_state)[0];

    let good_mean = ppo.actor.forward(&good_state)[0];

    println!("start mean: {start_mean}");
    println!("start good fraction: {start_good_fraction:.3}");

    println!("middle mean: {middle_mean}");
    println!("middle good fraction: {middle_good_fraction:.3}");

    println!("good mean: {good_mean}");
    println!("good action fraction: {final_good_fraction:.3}");

    //
    // Critic evaluation.
    //
    let start_value = ppo.critic.forward(&start_state)[0];

    let middle_value = ppo.critic.forward(&middle_state)[0];

    let good_value = ppo.critic.forward(&good_state)[0];

    println!("start value: {start_value}");
    println!("middle value: {middle_value}");
    println!("good value: {good_value}");

    //
    // Expected values:
    //
    // V(good)   ~= 1
    // V(middle) ~= gamma * 1
    //            ~= 0.95
    // V(start)  ~= gamma² * 1
    //            ~= 0.9025
    //

    assert!(
        start_good_fraction > 0.70,
        "policy failed at start: {start_good_fraction:.3}"
    );

    assert!(
        middle_good_fraction > 0.70,
        "policy failed in middle state: {middle_good_fraction:.3}"
    );

    assert!(
        final_good_fraction > 0.70,
        "policy failed in good state: {final_good_fraction:.3}"
    );

    assert!(
        (start_value - 0.9025).abs() < 0.25,
        "critic failed at start: value = {start_value}"
    );

    assert!(
        (middle_value - 0.95).abs() < 0.25,
        "critic failed in middle: value = {middle_value}"
    );

    assert!(
        (good_value - 1.0).abs() < 0.25,
        "critic failed in good state: value = {good_value}"
    );

    assert!(
        start_value < middle_value,
        "critic failed to propagate value from middle to start: \
        start = {start_value}, middle = {middle_value}"
    );

    assert!(
        middle_value < good_value,
        "critic failed to propagate value from good to middle: \
        middle = {middle_value}, good = {good_value}"
    );
}

#[test]
fn test_ppo_clipping() {
    const CLIP: f32 = 0.2;
    const EPSILON: f32 = 1e-6;

    //
    // Positive advantage.
    //
    {
        let advantage = 1.0;

        let mut loss = PpoSurrogateLossClipped::new();
        let value = loss.calc(0.7, advantage, CLIP);
        let derivative = loss.derivative();

        assert!((value - (-0.7)).abs() < EPSILON);
        assert!((derivative + 1.0).abs() < EPSILON);

        let mut loss = PpoSurrogateLossClipped::new();
        let value = loss.calc(1.1, advantage, CLIP);
        let derivative = loss.derivative();

        assert!((value - (-1.1)).abs() < EPSILON);
        assert!((derivative + 1.0).abs() < EPSILON);

        // Exact upper clipping boundary.
        let mut loss = PpoSurrogateLossClipped::new();
        let value = loss.calc(1.2, advantage, CLIP);

        assert!((value - (-1.2)).abs() < EPSILON);

        // Beyond upper boundary -> clipped.
        let mut loss = PpoSurrogateLossClipped::new();
        let value = loss.calc(1.3, advantage, CLIP);
        let derivative = loss.derivative();

        assert!((value - (-1.2)).abs() < EPSILON);
        assert!(derivative.abs() < EPSILON);
    }

    //
    // Negative advantage.
    //
    {
        let advantage = -1.0;

        // Below lower clipping boundary -> clipped.
        let mut loss = PpoSurrogateLossClipped::new();
        let value = loss.calc(0.7, advantage, CLIP);
        let derivative = loss.derivative();

        assert!((value - 0.8).abs() < EPSILON);
        assert!(derivative.abs() < EPSILON);

        // Inside clipping range.
        let mut loss = PpoSurrogateLossClipped::new();
        let value = loss.calc(0.9, advantage, CLIP);
        let derivative = loss.derivative();

        assert!((value - 0.9).abs() < EPSILON);
        assert!((derivative - 1.0).abs() < EPSILON);

        // Exact lower clipping boundary.
        let mut loss = PpoSurrogateLossClipped::new();
        let value = loss.calc(0.8, advantage, CLIP);

        assert!((value - 0.8).abs() < EPSILON);

        // Above upper boundary remains unclipped for A < 0.
        let mut loss = PpoSurrogateLossClipped::new();
        let value = loss.calc(1.3, advantage, CLIP);
        let derivative = loss.derivative();

        assert!((value - 1.3).abs() < EPSILON);
        assert!((derivative - 1.0).abs() < EPSILON);
    }
}

#[test]
fn test_ppo_actor_ratio() {
    const EPSILON: f32 = 1e-6;

    let mut ratio = PpoActorRatio::new();

    let value = ratio.calc(0.0, 0.0);
    let derivative = ratio.derivative();

    assert!(
        (value - 1.0).abs() < EPSILON,
        "ratio(0, 0) should be 1, got {value}"
    );

    assert!(
        (derivative - 1.0).abs() < EPSILON,
        "d ratio / d current_log_probability should be 1 at ratio=1, got {derivative}"
    );

    let mut ratio = PpoActorRatio::new();

    let value = ratio.calc(1.0, 0.0);
    let derivative = ratio.derivative();

    assert!(
        (value - std::f32::consts::E).abs() < EPSILON,
        "ratio(1, 0) should be e, got {value}"
    );

    assert!(
        (derivative - std::f32::consts::E).abs() < EPSILON,
        "derivative should equal ratio, got {derivative}"
    );

    let mut ratio = PpoActorRatio::new();

    let value = ratio.calc(0.0, 1.0);
    let derivative = ratio.derivative();

    assert!(
        (value - (-1.0f32).exp()).abs() < EPSILON,
        "ratio(0, 1) should be exp(-1), got {value}"
    );

    assert!(
        (derivative - (-1.0f32).exp()).abs() < EPSILON,
        "derivative should equal ratio, got {derivative}"
    );
}

#[test]
fn test_gaussian_log_probability() {
    const EPSILON: f32 = 1e-5;

    let action = [1.0];
    let mean = [0.0];
    let std_dev = 1.0;

    let mut glp = GaussianLogProbability::new();

    let value = glp.calc(&action, &mean, std_dev);
    let derivative = glp.derivative();

    // log N(1 | 0, 1)
    //
    // = -0.5 * (1^2)
    //   - 0.5 * ln(2*pi)
    //
    // ≈ -1.4189385
    let expected = -1.4189385;

    assert!(
        (value - expected).abs() < EPSILON,
        "unexpected Gaussian log probability: {value}"
    );

    // d log p / d mean
    //
    // = (action - mean) / variance
    // = 1 / 1
    // = 1
    assert_eq!(derivative.len(), 1);

    assert!(
        (derivative[0] - 1.0).abs() < EPSILON,
        "unexpected Gaussian log probability derivative: {}",
        derivative[0]
    );
}

#[test]
fn test_ppo_ratio_identity() {
    const EPSILON: f32 = 1e-6;

    let log_probability = -1.2345;

    let mut ratio = PpoActorRatio::new();

    let value = ratio.calc(log_probability, log_probability);

    let derivative = ratio.derivative();

    // If the old and current policies are identical:
    //
    // exp(log_pi - log_pi_old) = exp(0) = 1
    assert!(
        (value - 1.0).abs() < EPSILON,
        "identical policies must have ratio 1, got {value}"
    );

    // d exp(x) / dx = exp(x)
    //
    // At x = 0:
    //
    // derivative = 1
    assert!(
        (derivative - 1.0).abs() < EPSILON,
        "ratio derivative should be 1 at ratio=1, got {derivative}"
    );
}

#[test]
fn test_ppo_surrogate_identity() {
    const CLIP: f32 = 0.2;
    const EPSILON: f32 = 1e-6;

    // When ratio == 1, clipping should do nothing.
    for advantage in [-2.0, -1.0, 0.5, 1.0, 2.0] {
        let mut loss = PpoSurrogateLossClipped::new();

        let value = loss.calc(1.0, advantage, CLIP);
        let derivative = loss.derivative();

        // Your implementation is minimizing the negative surrogate:
        //
        // loss = -(ratio * advantage)
        //
        // ratio = 1
        //
        // => loss = -advantage
        assert!(
            (value + advantage).abs() < EPSILON,
            "unexpected surrogate at ratio=1: \
             advantage={advantage}, value={value}"
        );

        // d loss / d ratio = -advantage
        assert!(
            (derivative + advantage).abs() < EPSILON,
            "unexpected surrogate derivative: \
             advantage={advantage}, derivative={derivative}"
        );
    }
}

#[test]
fn test_actor_gradient_checking() {
    const EPSILON: f32 = 1e-3;
    const TOLERANCE: f32 = 5e-3;

    const INPUTS: usize = 1;
    const OUTPUTS: usize = 1;
    const NEURONS: usize = 4;
    const LAYERS: usize = 1;

    let observation = [0.7_f32];
    let action = [1.2_f32];

    // This is the log probability under the OLD policy.
    // We deliberately choose a fixed value so the PPO ratio
    // is non-trivial.
    let old_log_probability = -1.0_f32;

    let advantage = 1.0_f32;
    let clip = 0.2_f32;
    let std_dev = 0.70710677_f32;

    //
    // Create deterministic initial network.
    //
    let mut actor = NeuralNetworkLayered::<INPUTS, OUTPUTS, NEURONS, LAYERS, false>::new_rand(42);

    //
    // Forward pass.
    //
    let mean = actor.forward(&observation);

    //
    // Calculate the PPO loss.
    //
    let mut glp = GaussianLogProbability::new();

    let current_log_probability = glp.calc(&action, &mean, std_dev);

    let mut ratio = PpoActorRatio::new();

    let current_ratio = ratio.calc(current_log_probability, old_log_probability);

    let mut surrogate = PpoSurrogateLossClipped::new();

    let loss = surrogate.calc(current_ratio, advantage, clip);

    let surrogate_derivative = surrogate.derivative();

    let ratio_derivative = ratio.derivative();

    let log_probability_derivative = glp.derivative();

    //
    // Chain rule:
    //
    // dL/dmean
    //
    let d_loss_d_mean = surrogate_derivative * ratio_derivative * log_probability_derivative[0];

    //
    // Backprop through the actor.
    //
    actor.backward(&[d_loss_d_mean]);

    //
    // Grab one parameter's analytical gradient.
    //
    let analytical_gradient = actor.output.w[0][0];

    //
    // IMPORTANT:
    //
    // The value stored in output.w is the weight itself,
    // not its gradient.
    //
    // The gradient is stored in:
    //
    //     actor.output.dl_dw[0][0]
    //
    let analytical_gradient = actor.output.dl_dw[0][0];

    //
    // Numerical gradient.
    //
    let original_weight = actor.output.w[0][0];

    //
    // L(theta + epsilon)
    //
    actor.output.w[0][0] = original_weight + EPSILON;

    let mean_plus = actor.forward(&observation);

    let mut glp_plus = GaussianLogProbability::new();

    let log_probability_plus = glp_plus.calc(&action, &mean_plus, std_dev);

    let mut ratio_plus = PpoActorRatio::new();

    let ratio_plus_value = ratio_plus.calc(log_probability_plus, old_log_probability);

    let mut surrogate_plus = PpoSurrogateLossClipped::new();

    let loss_plus = surrogate_plus.calc(ratio_plus_value, advantage, clip);

    //
    // L(theta - epsilon)
    //
    actor.output.w[0][0] = original_weight - EPSILON;

    let mean_minus = actor.forward(&observation);

    let mut glp_minus = GaussianLogProbability::new();

    let log_probability_minus = glp_minus.calc(&action, &mean_minus, std_dev);

    let mut ratio_minus = PpoActorRatio::new();

    let ratio_minus_value = ratio_minus.calc(log_probability_minus, old_log_probability);

    let mut surrogate_minus = PpoSurrogateLossClipped::new();

    let loss_minus = surrogate_minus.calc(ratio_minus_value, advantage, clip);

    //
    // Restore parameter.
    //
    actor.output.w[0][0] = original_weight;

    //
    // Central finite difference.
    //
    let numerical_gradient = (loss_plus - loss_minus) / (2.0 * EPSILON);

    println!("loss:                {loss}");
    println!("analytical gradient: {analytical_gradient}");
    println!("numerical gradient:  {numerical_gradient}");

    let absolute_error = (analytical_gradient - numerical_gradient).abs();

    let relative_error = absolute_error / numerical_gradient.abs().max(1e-6);

    println!("absolute error: {absolute_error}");
    println!("relative error: {relative_error}");

    assert!(
        relative_error < TOLERANCE,
        "actor gradient mismatch: \
         analytical={analytical_gradient}, \
         numerical={numerical_gradient}, \
         relative_error={relative_error}"
    );
}

/// PPO should learn a useful policy in a longer-horizon environment
/// with noisy rewards.
///
/// Environment:
///
///     start -> middle_1 -> middle_2 -> middle_3 -> goal
///
/// At every state the agent must choose a positive action to continue
/// toward the goal. A negative action sends it to a terminal failure.
///
/// The final reward is noisy:
///
///     reward = +2.0 + N(0, 0.5)
///
/// Intermediate rewards are noisy as well:
///
///     reward = N(0, 0.1)
///
/// This test exercises:
/// - multi-step credit assignment
/// - GAE
/// - noisy returns
/// - stochastic policy learning
/// - critic value estimation
#[test]
fn test_long_noisy_mdp() {
    const EPISODES: usize = 2_000;
    const BATCH_SIZE: usize = 32;
    const EVALUATION_EPISODES: usize = 1_000;

    let mut ppo = Ppo2::<1, 1, 64, 1, false>::new(42);

    let start_state = [0.0];
    let state_1 = [1.0];
    let state_2 = [2.0];
    let state_3 = [3.0];
    let goal_state = [4.0];
    let bad_state = [-1.0];

    // Deterministic RNG for reproducible environment noise.
    let mut rng = fastrand::Rng::with_seed(12345);

    for episode in 0..EPISODES {
        //
        // STEP 1
        //
        let (action, _mean, log_probability) = ppo.get_action(&start_state);

        let next_state = if action[0] >= 0.0 { state_1 } else { bad_state };

        let reward = rng.f32() * 0.2 - 0.1;

        ppo.save_reward(start_state, action, log_probability, reward, false);

        //
        // If the agent chose the bad path, terminate.
        //
        if next_state == bad_state {
            let (action, _mean, log_probability) = ppo.get_action(&bad_state);

            let reward = -2.0 + rng.f32() * 0.4 - 0.2;

            ppo.save_reward(bad_state, action, log_probability, reward, true);
        } else {
            //
            // STEP 2
            //
            let (action, _mean, log_probability) = ppo.get_action(&state_1);

            let next_state = if action[0] >= 0.0 { state_2 } else { bad_state };

            let reward = rng.f32() * 0.2 - 0.1;

            ppo.save_reward(state_1, action, log_probability, reward, false);

            if next_state == bad_state {
                let (action, _mean, log_probability) = ppo.get_action(&bad_state);

                let reward = -2.0 + rng.f32() * 0.4 - 0.2;

                ppo.save_reward(bad_state, action, log_probability, reward, true);
            } else {
                //
                // STEP 3
                //
                let (action, _mean, log_probability) = ppo.get_action(&state_2);

                let next_state = if action[0] >= 0.0 { state_3 } else { bad_state };

                let reward = rng.f32() * 0.2 - 0.1;

                ppo.save_reward(state_2, action, log_probability, reward, false);

                if next_state == bad_state {
                    let (action, _mean, log_probability) = ppo.get_action(&bad_state);

                    let reward = -2.0 + rng.f32() * 0.4 - 0.2;

                    ppo.save_reward(bad_state, action, log_probability, reward, true);
                } else {
                    //
                    // STEP 4
                    //
                    let (action, _mean, log_probability) = ppo.get_action(&state_3);

                    let next_state = if action[0] >= 0.0 {
                        goal_state
                    } else {
                        bad_state
                    };

                    let reward = rng.f32() * 0.2 - 0.1;

                    ppo.save_reward(state_3, action, log_probability, reward, false);

                    //
                    // STEP 5
                    //
                    let (action, _mean, log_probability) = ppo.get_action(&next_state);

                    let reward = if next_state == goal_state {
                        // Large but noisy terminal reward.
                        2.0 + rng.f32() * 1.0 - 0.5
                    } else {
                        -2.0 + rng.f32() * 0.4 - 0.2
                    };

                    ppo.save_reward(next_state, action, log_probability, reward, true);
                }
            }
        }

        if (episode + 1) % BATCH_SIZE == 0 {
            ppo.learn();
        }
    }

    //
    // Evaluate the learned policy.
    //
    let mut successful_episodes = 0;
    let mut total_reward = 0.0;

    for _ in 0..EVALUATION_EPISODES {
        let mut state = start_state;
        let mut episode_reward = 0.0;
        let mut success = true;

        for _ in 0..5 {
            let (action, _mean, _log_probability) = ppo.get_action(&state);

            let x = action[0];

            if state == goal_state {
                break;
            }

            if state == bad_state {
                success = false;
                episode_reward -= 2.0;
                break;
            }

            if state == start_state {
                if x >= 0.0 {
                    state = state_1;
                } else {
                    state = bad_state;
                    success = false;
                    episode_reward -= 2.0;
                    break;
                }

                episode_reward += 0.0;
            } else if state == state_1 {
                if x >= 0.0 {
                    state = state_2;
                } else {
                    state = bad_state;
                    success = false;
                    episode_reward -= 2.0;
                    break;
                }
            } else if state == state_2 {
                if x >= 0.0 {
                    state = state_3;
                } else {
                    state = bad_state;
                    success = false;
                    episode_reward -= 2.0;
                    break;
                }
            } else if state == state_3 {
                if x >= 0.0 {
                    state = goal_state;
                } else {
                    state = bad_state;
                    success = false;
                    episode_reward -= 2.0;
                    break;
                }
            }
        }

        if state == goal_state {
            successful_episodes += 1;
            episode_reward += 2.0;
        }

        total_reward += episode_reward;

        // Keep `success` explicit so this remains easy to extend
        // with additional terminal states.
        let _ = success;
    }

    let success_fraction = successful_episodes as f32 / EVALUATION_EPISODES as f32;

    let average_reward = total_reward / EVALUATION_EPISODES as f32;

    println!("success fraction: {success_fraction:.3}");

    println!("average reward: {average_reward:.3}");

    //
    // Every state should prefer continuing toward the goal.
    //
    let start_mean = ppo.actor.forward(&start_state)[0];
    let state_1_mean = ppo.actor.forward(&state_1)[0];
    let state_2_mean = ppo.actor.forward(&state_2)[0];
    let state_3_mean = ppo.actor.forward(&state_3)[0];

    println!("start mean:   {start_mean}");
    println!("state 1 mean: {state_1_mean}");
    println!("state 2 mean: {state_2_mean}");
    println!("state 3 mean: {state_3_mean}");

    assert!(
        success_fraction > 0.80,
        "policy failed to learn the long noisy path: \
         success fraction = {success_fraction:.3}"
    );

    assert!(
        average_reward > 1.0,
        "policy reward too low: {average_reward:.3}"
    );

    assert!(
        start_mean > 0.0,
        "start-state policy should prefer positive actions: \
         mean = {start_mean}"
    );

    assert!(
        state_1_mean > 0.0,
        "state-1 policy should prefer positive actions: \
         mean = {state_1_mean}"
    );

    assert!(
        state_2_mean > 0.0,
        "state-2 policy should prefer positive actions: \
         mean = {state_2_mean}"
    );

    assert!(
        state_3_mean > 0.0,
        "state-3 policy should prefer positive actions: \
         mean = {state_3_mean}"
    );

    assert!(
        start_mean.abs() < 10.0,
        "start-state policy mean exploded: {start_mean}"
    );

    assert!(
        state_1_mean.abs() < 10.0,
        "state-1 policy mean exploded: {state_1_mean}"
    );

    assert!(
        state_2_mean.abs() < 10.0,
        "state-2 policy mean exploded: {state_2_mean}"
    );

    assert!(
        state_3_mean.abs() < 10.0,
        "state-3 policy mean exploded: {state_3_mean}"
    );

    let start_value = ppo.critic.forward(&start_state)[0];
    let state_1_value = ppo.critic.forward(&state_1)[0];
    let state_2_value = ppo.critic.forward(&state_2)[0];
    let state_3_value = ppo.critic.forward(&state_3)[0];

    println!("start value:   {start_value}");
    println!("state 1 value: {state_1_value}");
    println!("state 2 value: {state_2_value}");
    println!("state 3 value: {state_3_value}");

    for (name, value) in [
        ("start", start_value),
        ("state 1", state_1_value),
        ("state 2", state_2_value),
        ("state 3", state_3_value),
    ] {
        assert!(
            value.is_finite(),
            "{name} critic value is not finite: {value}"
        );

        assert!(value.abs() < 10.0, "{name} critic value exploded: {value}");
    }
}

#[test]
fn test_gae_arithmetic() {
    const GAMMA: f32 = 0.95;
    const LAMBDA: f32 = 0.90;
    const EPSILON: f32 = 1e-4;

    let mut ppo = Ppo2::<1, 1, 64, 1, false>::new(42);

    //
    // We want known critic values so that we can verify the
    // GAE arithmetic exactly.
    //
    // Instead of relying on random network initialization,
    // overwrite the critic to produce the desired values.
    //
    // s0 -> s1 -> s2 -> terminal
    //
    // rewards:
    //
    //   s0: +0.3
    //   s1: -0.2
    //   s2: +1.0
    //
    // desired V:
    //
    //   V(s0) = 0.8
    //   V(s1) = 0.7
    //   V(s2) = 0.6
    //
    // If your network cannot conveniently be configured to
    // produce exact constants, use the actual values produced
    // by the critic below and calculate the expected GAE from
    // those values.
    //

    let states = [[0.0], [1.0], [2.0]];

    let rewards = [0.3, -0.2, 1.0];

    let dones = [false, false, true];

    //
    // Collect the trajectory through the real PPO API.
    //
    for i in 0..3 {
        let observation = states[i];

        let (action, _mean, log_probability) = ppo.get_action(&observation);

        ppo.save_reward(observation, action, log_probability, rewards[i], dones[i]);
    }

    //
    // Calculate GAE using the values that PPO actually stored
    // in its transitions.
    //
    let values: Vec<f32> = ppo
        .transitions
        .iter()
        .map(|transition| transition.value)
        .collect();

    assert_eq!(values.len(), 3);

    let (advantages, returns) = ppo.calculate_gae();

    assert_eq!(advantages.len(), 3);
    assert_eq!(returns.len(), 3);

    //
    // Independently calculate GAE from the stored values.
    //
    let delta_2 = rewards[2] + GAMMA * 0.0 - values[2];

    let delta_1 = rewards[1] + GAMMA * values[2] - values[1];

    let delta_0 = rewards[0] + GAMMA * values[1] - values[0];

    let expected_2 = delta_2;

    let expected_1 = delta_1 + GAMMA * LAMBDA * expected_2;

    let expected_0 = delta_0 + GAMMA * LAMBDA * expected_1;

    //
    // Verify the GAE values produced by Ppo2.
    //
    assert!(
        (advantages[0] - expected_0).abs() < EPSILON,
        "A0 mismatch: actual={}, expected={}",
        advantages[0],
        expected_0
    );

    assert!(
        (advantages[1] - expected_1).abs() < EPSILON,
        "A1 mismatch: actual={}, expected={}",
        advantages[1],
        expected_1
    );

    assert!(
        (advantages[2] - expected_2).abs() < EPSILON,
        "A2 mismatch: actual={}, expected={}",
        advantages[2],
        expected_2
    );

    //
    // Verify critic targets:
    //
    // target_t = V(s_t) + A_t
    //
    let expected_return_0 = values[0] + expected_0;

    let expected_return_1 = values[1] + expected_1;

    let expected_return_2 = values[2] + expected_2;

    assert!(
        (returns[0] - expected_return_0).abs() < EPSILON,
        "return0 mismatch: actual={}, expected={}",
        returns[0],
        expected_return_0
    );

    assert!(
        (returns[1] - expected_return_1).abs() < EPSILON,
        "return1 mismatch: actual={}, expected={}",
        returns[1],
        expected_return_1
    );

    assert!(
        (returns[2] - expected_return_2).abs() < EPSILON,
        "return2 mismatch: actual={}, expected={}",
        returns[2],
        expected_return_2
    );

    println!("values:");
    for value in &values {
        println!("  {value}");
    }

    println!("advantages:");
    for advantage in &advantages {
        println!("  {advantage}");
    }

    println!("returns:");
    for value_target in &returns {
        println!("  {value_target}");
    }
}

#[test]
fn test_gae_episode_boundary() {
    let mut ppo = Ppo2::<1, 1, 16, 1, false>::new(42);

    //
    // Manually construct:
    //
    // Episode A:
    //   reward = 0, terminal
    //
    // Episode B:
    //   reward = 10, terminal
    //
    // Both values are zero so the expected GAE is obvious.
    //

    ppo.transitions.push(Transition {
        observation: [0.0],
        action: [0.0],
        log_probability: 0.0,
        reward: 0.0,
        done: true,
        value: 0.0,
    });

    ppo.transitions.push(Transition {
        observation: [1.0],
        action: [0.0],
        log_probability: 0.0,
        reward: 10.0,
        done: true,
        value: 0.0,
    });

    let (advantages, returns) = ppo.calculate_gae();

    println!("advantages: {advantages:?}");
    println!("returns:    {returns:?}");

    //
    // Episode B gets its own reward.
    //
    assert!((advantages[1] - 10.0).abs() < 1e-6);
    assert!((returns[1] - 10.0).abs() < 1e-6);

    //
    // Episode A must NOT receive Episode B's reward.
    //
    assert!(advantages[0].abs() < 1e-6);
    assert!(returns[0].abs() < 1e-6);
}

#[test]
fn test_gae_propagation() {
    let mut ppo = Ppo2::<1, 1, 16, 1, false>::new(42);

    ppo.transitions.push(Transition {
        observation: [0.0],
        action: [0.0],
        log_probability: 0.0,
        reward: 0.0,
        done: false,
        value: 0.0,
    });

    ppo.transitions.push(Transition {
        observation: [1.0],
        action: [0.0],
        log_probability: 0.0,
        reward: 1.0,
        done: true,
        value: 0.0,
    });

    let (advantages, returns) = ppo.calculate_gae();

    println!("advantages: {advantages:?}");
    println!("returns:    {returns:?}");

    let expected_a1 = 1.0;
    let expected_a0 = ppo.gamma * ppo.gae_lambda * expected_a1;

    assert!(
        (advantages[1] - expected_a1).abs() < 1e-6,
        "A1 = {}, expected {}",
        advantages[1],
        expected_a1
    );

    assert!(
        (advantages[0] - expected_a0).abs() < 1e-6,
        "A0 = {}, expected {}",
        advantages[0],
        expected_a0
    );

    assert!((returns[0] - expected_a0).abs() < 1e-6);

    assert!((returns[1] - 1.0).abs() < 1e-6);
}

#[test]
fn test_gae_bootstrap_value() {
    let mut ppo = Ppo2::<1, 1, 16, 1, false>::new(42);

    ppo.transitions.push(Transition {
        observation: [0.0],
        action: [0.0],
        log_probability: 0.0,
        reward: 0.0,
        done: false,
        value: 0.2,
    });

    ppo.transitions.push(Transition {
        observation: [1.0],
        action: [0.0],
        log_probability: 0.0,
        reward: 1.0,
        done: true,
        value: 0.8,
    });

    let (advantages, returns) = ppo.calculate_gae();

    println!("advantages: {advantages:?}");
    println!("returns:    {returns:?}");

    let delta_1 = 1.0 - 0.8;
    let expected_a1 = delta_1;

    let delta_0 = 0.0 + ppo.gamma * 0.8 - 0.2;

    let expected_a0 = delta_0 + ppo.gamma * ppo.gae_lambda * expected_a1;

    let expected_return_0 = expected_a0 + 0.2;
    let expected_return_1 = expected_a1 + 0.8;

    assert!(
        (advantages[1] - expected_a1).abs() < 1e-6,
        "A1 = {}, expected {}",
        advantages[1],
        expected_a1
    );

    assert!(
        (advantages[0] - expected_a0).abs() < 1e-6,
        "A0 = {}, expected {}",
        advantages[0],
        expected_a0
    );

    assert!(
        (returns[0] - expected_return_0).abs() < 1e-6,
        "return0 = {}, expected {}",
        returns[0],
        expected_return_0
    );

    assert!(
        (returns[1] - expected_return_1).abs() < 1e-6,
        "return1 = {}, expected {}",
        returns[1],
        expected_return_1
    );
}

#[test]
fn test_continuous_noisy_mdp() {
    const EPISODES: usize = 2_000;
    const BATCH_SIZE: usize = 64;
    const HORIZON: usize = 8;

    let mut ppo = Ppo2::<1, 1, 64, 2, false>::new(42);

    for episode in 0..EPISODES {
        for step in 0..HORIZON {
            let state = [step as f32 / HORIZON as f32];

            let (action, _, log_probability) = ppo.get_action(&state);

            let target = 1.0;

            let error = action[0] - target;

            let noise = fastrand::f32() * 0.2 - 0.1;

            let reward = 1.0 - 0.5 * error * error + noise;

            let done = step + 1 == HORIZON;

            ppo.save_reward(state, action, log_probability, reward, done);
        }

        if (episode + 1) % BATCH_SIZE == 0 {
            ppo.learn();
        }
    }

    //
    // Evaluate without caring about the sampled action.
    //
    let mut mean_squared_error = 0.0;

    for step in 0..HORIZON {
        let state = [step as f32 / HORIZON as f32];

        let mean = ppo.actor.forward(&state)[0];

        println!("state {step}: mean = {mean}");

        let error = mean - 1.0;

        mean_squared_error += error * error;
    }

    mean_squared_error /= HORIZON as f32;

    println!("mean squared policy error: {mean_squared_error}");

    assert!(
        mean_squared_error < 0.25,
        "policy did not learn the target action: \
         mse = {mean_squared_error}"
    );
}


#[test]
fn test_surrogate_loss() {
    let mut ppo = PpoSurrogateLossClipped::new();

    let loss = ppo.calc(1.0, 1.0, 0.2);
    let derivative = ppo.derivative();

    println!("loss = {loss}, derivative = {derivative}");
}