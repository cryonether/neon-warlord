//! Gait, like walk or trot

#[derive(Clone)]
pub struct Gait {
    pub sequence: [[f32; 4]; 100],

    pub index: usize,
    pub duration: f32,
    pub t: f32,
}

impl Gait {
    pub fn new(gait_kind: GaitKind, duration: f32) -> Self {
        let (offsets, duty_factor) = match gait_kind {
            GaitKind::Trot => (
                [0.0, 0.5, 0.5, 0.0],
                0.5,
            ),
            GaitKind::Walk => (
                [0.0, 0.5, 0.75, 0.25],
                0.75,
            ),
        };

        Self {
            sequence: Self::generate(offsets, duty_factor),
            index: 0,
            duration,
            t: 0.0,
        }
    }

    // fn generate(
    //     offsets: [f32; 4],
    //     duty_factor: f32,
    // ) -> [[f32; 4]; 10] {
    //     std::array::from_fn(|i| {
    //         std::array::from_fn(|leg| {
    //             let phase = (
    //                 i as f32 / 10.0 - offsets[leg]
    //             ).rem_euclid(1.0);

    //             if phase < duty_factor {
    //                 1.0
    //             } else {
    //                 0.0
    //             }
    //         })
    //     })
    // }

    fn generate(
        offsets: [f32; 4],
        duty_factor: f32,
    ) -> [[f32; 4]; 100] {
        std::array::from_fn(|i| {
            std::array::from_fn(|leg| {
                let phase = (
                    i as f32 / 100.0 - offsets[leg]
                ).rem_euclid(1.0);

                if phase < duty_factor {
                    // Stance: foot on the ground
                    0.0
                } else {
                    // Swing: smooth foot lift and landing
                    let swing_phase =
                        (phase - duty_factor) / (1.0 - duty_factor);

                    (std::f32::consts::PI * swing_phase).sin()
                }
            })
        })
    }
    
    pub fn update(&mut self, dt: f32) {
        self.t = (self.t + dt).rem_euclid(self.duration);

        let steps = self.sequence.len() as f32;

        self.index = (self.t / self.duration * steps) as usize;
        self.index = self.index.min(self.sequence.len() - 1);
    }

    pub fn get_current(&self) -> [f32; 4] {
        self.sequence[self.index]
    }
}

pub enum GaitKind {
    Trot,
    Walk,
}

#[test]
fn print_movement() {
    // LF, left fore; LH, left hind; RF, right fore; RH, right hind
    
    println!("Walk: ");
    let gait = Gait::new(GaitKind::Walk, 10.0);
    println!("LF   LH   RF   RH");
    for elem in &gait.sequence {
        println!("{:.2}  {:.2}  {:.2}  {:.2}", elem[0], elem[1], elem[2], elem[3]);
    }
    println!("");

    println!("Trot: ");
    println!("LF   LH   RF   RH");
    let gait = Gait::new(GaitKind::Trot, 10.0);
    for elem in &gait.sequence {
        println!("{:.2}  {:.2}  {:.2}  {:.2}", elem[0], elem[1], elem[2], elem[3]);
    }
    println!("");
}
