
type Vec3 = cgmath::Vector3<f32>;


#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
pub enum PrintColor {
    RedYellowGreen,
    GreenYellowRed,
    BlueCyanGreen,
    GreenCyanBlue,
    BluePurpleRed,
    PurplePinkYellow,
    BlackWhite,
    WhiteBlack,
    Cool,
    Warm,
    Rainbow,
}

impl PrintColor {
    pub const fn stops(self) -> &'static [(u8, u8, u8)] {
        match self {
            Self::RedYellowGreen => &[
                (255, 0, 0),
                (255, 255, 0),
                (0, 255, 0),
            ],

            Self::GreenYellowRed => &[
                (0, 255, 0),
                (255, 255, 0),
                (255, 0, 0),
            ],

            Self::BlueCyanGreen => &[
                (0, 0, 255),
                (0, 255, 255),
                (0, 255, 0),
            ],

            Self::GreenCyanBlue => &[
                (0, 255, 0),
                (0, 255, 255),
                (0, 0, 255),
            ],

            Self::BluePurpleRed => &[
                (0, 0, 255),
                (128, 0, 255),
                (255, 0, 0),
            ],

            Self::PurplePinkYellow => &[
                (128, 0, 255),
                (255, 0, 128),
                (255, 255, 0),
            ],

            Self::BlackWhite => &[
                (0, 0, 0),
                (255, 255, 255),
            ],

            Self::WhiteBlack => &[
                (255, 255, 255),
                (0, 0, 0),
            ],

            Self::Cool => &[
                (0, 255, 255),
                (0, 0, 255),
            ],

            Self::Warm => &[
                (255, 255, 0),
                (255, 0, 0),
            ],

            Self::Rainbow => &[
                (255, 0, 0),
                (255, 255, 0),
                (0, 255, 0),
                (0, 255, 255),
                (0, 0, 255),
                (128, 0, 255),
            ],
        }
    }

    pub fn at(self, t: f32) -> Vec3 {
        let (r, g, b) = gradient(t, self.stops());

        Vec3::new(
            r as f32 / 255.0,
            g as f32 / 255.0,
            b as f32 / 255.0,
        )
    }

    pub fn into_vec<const N: usize>(self) -> [Vec3; N] {
        std::array::from_fn(|i| {
            let t = if N <= 1 {
                0.0
            } else {
                i as f32 / (N - 1) as f32
            };

            self.at(t)
        })
    }
}

fn lerp(a: u8, b: u8, t: f32) -> u8 {
    (a as f32 + (b as f32 - a as f32) * t).round() as u8
}

pub fn gradient(t: f32, colors: &[(u8, u8, u8)]) -> (u8, u8, u8) {
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
