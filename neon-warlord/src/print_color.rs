//! Prints with a color gradient to the console

#[derive(Clone, Copy)]
pub enum PrintColor {
    RedYellowGreen,     // intuitive for probabilities / scores
    GreenYellowRed, 
    BlueCyanGreen,      // nice for heatmaps
    GreenCyanBlue,  
    BluePurpleRed,      // good for low → high intensity
    PurplePinkYellow,   // visually bright
    BlackWhite,         // grayscale
    WhiteBlack,
    Cool,               // blue
    Warm,               // red
    Rainbow,            // maximum visual distinction
}

fn lerp(a: u8, b: u8, t: f32) -> u8 {
    (a as f32 + (b as f32 - a as f32) * t) as u8
}

fn gradient(t: f32, colors: &[(u8, u8, u8)]) -> (u8, u8, u8) {
    let t = t.clamp(0.0, 1.0);

    if colors.len() == 1 {
        return colors[0];
    }

    let x = t * (colors.len() - 1) as f32;
    let i = (x.floor() as usize).min(colors.len() - 2);
    let local_t = x - i as f32;

    let (r1, g1, b1) = colors[i];
    let (r2, g2, b2) = colors[i + 1];

    (
        lerp(r1, r2, local_t),
        lerp(g1, g2, local_t),
        lerp(b1, b2, local_t),
    )
}

pub fn print_color(val: f32, min: f32, max: f32, color: PrintColor) {
    let t = if max > min {
        ((val - min) / (max - min)).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let colors = match color {
        PrintColor::RedYellowGreen => &[
            (255, 0, 0),
            (255, 255, 0),
            (0, 255, 0),
        ][..],

        PrintColor::GreenYellowRed => &[
            (0, 255, 0),
            (255, 255, 0),
            (255, 0, 0),
        ][..],

        PrintColor::BlueCyanGreen => &[
            (0, 0, 255),
            (0, 255, 255),
            (0, 255, 0),
        ][..],

        PrintColor::GreenCyanBlue => &[
            (0, 255, 0),
            (0, 255, 255),
            (0, 0, 255),
        ][..],

        PrintColor::BluePurpleRed => &[
            (0, 0, 255),
            (128, 0, 255),
            (255, 0, 0),
        ][..],

        PrintColor::PurplePinkYellow => &[
            (128, 0, 255),
            (255, 0, 128),
            (255, 255, 0),
        ][..],

        PrintColor::BlackWhite => &[
            (0, 0, 0),
            (255, 255, 255),
        ][..],

        PrintColor::WhiteBlack => &[
            (255, 255, 255),
            (0, 0, 0),
        ][..],

        PrintColor::Cool => &[
            (0, 255, 255),
            (0, 0, 255),
        ][..],

        PrintColor::Warm => &[
            (255, 255, 0),
            (255, 0, 0),
        ][..],

        PrintColor::Rainbow => &[
            (255, 0, 0),
            (255, 255, 0),
            (0, 255, 0),
            (0, 255, 255),
            (0, 0, 255),
            (128, 0, 255),
        ][..],
    };

    let (r, g, b) = gradient(t, colors);

    print!(
        "\x1b[38;2;{};{};{}m{:.3}\x1b[0m ",
        r, g, b, val
    );
}
