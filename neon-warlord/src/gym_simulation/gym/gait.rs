//! Gait, like walk or trot

/// Gait generator for a quadruped robot.
#[derive(Clone)]
pub struct Gait {
    /// Normalized gait phase [0.0, 1.0).
    pub phase: f32,

    /// Duration of one complete gait cycle in seconds.
    pub duration: f32,

    /// Phase offsets: LF, LH, RF, RH.
    offsets: [f32; 4],

    /// Fraction of the gait cycle spent in stance.
    duty_factor: f32,
}

impl Gait {
    pub fn new(gait_kind: GaitKind, duration: f32) -> Self {
        assert!(duration.is_finite() && duration > 0.0);

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
            phase: 0.0,
            duration,
            offsets,
            duty_factor,
        }
    }

    /// Advance the gait by dt seconds.
    pub fn update(&mut self, dt: f32) {
        assert!(dt.is_finite() && dt >= 0.0);

        self.phase = (
            self.phase + dt / self.duration
        ).rem_euclid(1.0);
    }

    /// Calculate the current normalized foot heights.
    /// Order: LF, LH, RF, RH.
    pub fn get_current(&self) -> [f32; 4] {
        std::array::from_fn(|leg| {
        let phase = (
            self.phase - self.offsets[leg]
        ).rem_euclid(1.0);

        if phase < self.duty_factor {
            0.0
        } else {
            let swing_phase =
                (phase - self.duty_factor)
                / (1.0 - self.duty_factor);

            (std::f32::consts::PI * swing_phase).sin()
        }
    })
}
}

#[derive(Clone, Copy)]
pub enum GaitKind {
    Trot,
    Walk,
}

#[test]
fn print_movement() {
    // LF, left fore; LH, left hind; RF, right fore; RH, right hind
    
    println!("Walk: ");
    let mut gait = Gait::new(GaitKind::Walk, 10.0);
    println!("LF   LH   RF   RH");
    for i in 0..100 {
        let elem = gait.get_current();
        println!("{:.2}  {:.2}  {:.2}  {:.2}", elem[0], elem[1], elem[2], elem[3]);
        gait.update(10.0 / 100.0);
    }
    println!("");

    println!("Trot: ");
    println!("LF   LH   RF   RH");
    let mut gait = Gait::new(GaitKind::Trot, 10.0);
    for i in 0..100 {
        let elem = gait.get_current();
        println!("{:.2}  {:.2}  {:.2}  {:.2}", elem[0], elem[1], elem[2], elem[3]);
        gait.update(10.0 / 100.0);
    }
    println!("");
}
