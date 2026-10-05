//! Prints with a color gradient to the console

use crate::print_color::color::{PrintColor, gradient};

pub mod color;

#[allow(dead_code)]
pub fn print_color(val: f32, min: f32, max: f32, color: PrintColor) {
    let t = if max > min {
        ((val - min) / (max - min)).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let (r, g, b) = gradient(t, color.stops());

    print!("\x1b[38;2;{};{};{}m{:.3}\x1b[0m ", r, g, b, val);
}
