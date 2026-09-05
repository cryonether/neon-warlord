//! Algorithms for reinforcement learning

#[allow(dead_code)]
pub mod dqn;
#[allow(dead_code)]
pub mod dqn_dfdx;
pub mod neat;
#[allow(dead_code)]
pub mod neural_network;
#[allow(dead_code)]
pub mod neural_network_dfdx;
#[allow(dead_code)]
pub mod neural_network_simd;
#[allow(dead_code)]
pub mod ppo;
#[allow(dead_code)]
pub mod q_learning;

pub fn _assert_f32_eq(a: f32, b: f32, epsilon: f32) {
    assert!(
        (a - b).abs() < epsilon,
        "expected {a} ≈ {b}, difference = {}",
        (a - b).abs()
    );
}

pub fn _assert_f32_eq_(a: f32, b: f32) {
    _assert_f32_eq(a, b, 0.1);
}


#[allow(dead_code)]
pub mod console_color {
    pub const RESET: &str = "\x1b[0m";

    // Standard colors
    pub const BLACK: &str = "\x1b[30m";
    pub const RED: &str = "\x1b[31m";
    pub const GREEN: &str = "\x1b[32m";
    pub const YELLOW: &str = "\x1b[33m";
    pub const BLUE: &str = "\x1b[34m";
    pub const MAGENTA: &str = "\x1b[35m";
    pub const CYAN: &str = "\x1b[36m";
    pub const WHITE: &str = "\x1b[37m";

    // Bright colors
    pub const BRIGHT_BLACK: &str = "\x1b[90m";
    pub const BRIGHT_RED: &str = "\x1b[91m";
    pub const BRIGHT_GREEN: &str = "\x1b[92m";
    pub const BRIGHT_YELLOW: &str = "\x1b[93m";
    pub const BRIGHT_BLUE: &str = "\x1b[94m";
    pub const BRIGHT_MAGENTA: &str = "\x1b[95m";
    pub const BRIGHT_CYAN: &str = "\x1b[96m";
    pub const BRIGHT_WHITE: &str = "\x1b[97m";

    // Useful 256-color shades
    pub const GRAY: &str = "\x1b[38;5;245m";
    pub const DARK_GRAY: &str = "\x1b[38;5;238m";

    pub const ORANGE: &str = "\x1b[38;5;208m";
    pub const RUST: &str = "\x1b[38;5;130m";
    pub const GOLD: &str = "\x1b[38;5;220m";

    pub const PINK: &str = "\x1b[38;5;205m";
    pub const PURPLE: &str = "\x1b[38;5;129m";
    pub const TEAL: &str = "\x1b[38;5;30m";
    pub const LIME: &str = "\x1b[38;5;118m";
}


