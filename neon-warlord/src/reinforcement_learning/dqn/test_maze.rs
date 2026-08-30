//! Trying to learn a maze

use super::*;

const W: usize = 3;
const H: usize = 3;

// const MAZE: [[u8; W]; H] = [
//     [0, 0, 0, 1, 0],
//     [1, 1, 0, 1, 0],
//     [0, 0, 0, 0, 0],
//     [0, 1, 1, 1, 1],
//     [0, 0, 0, 0, 0],
// ];

const MAZE: [[u8; W]; H] = [
    [0, 0, 0],
    [1, 1, 0],
    [0, 0, 0],
];

const START: (usize, usize) = (0, 0);
const GOAL:  (usize, usize) = (2, 1);

const UP: usize    = 0;
const RIGHT: usize = 1;
const DOWN: usize  = 2;
const LEFT: usize  = 3;



const REWARD_GOAL: f32 = 10.0;
const REWARD_STEP: f32 = -0.1;
const REWARD_WALL: f32 = -1.0;

struct Maze {
    position: (usize, usize)
}

impl Maze {
    fn new() -> Self {
        let position  = START;

        Self { position }
    }

    fn step(&mut self, action: usize) -> (f32, bool) {
        let (x, y) = self.position;

        let (nx, ny) = match action {
            UP    if y > 0 => (x, y - 1),
            RIGHT if x + 1 < W => (x + 1, y),
            DOWN  if y + 1 < H => (x, y + 1),
            LEFT  if x > 0 => (x - 1, y),
            _ => return (REWARD_WALL, false),
        };

        if MAZE[ny][nx] == 1 {
            return (REWARD_WALL, false);
        }

        self.position = (nx, ny);

        if self.position == GOAL {
            return (REWARD_GOAL, true);
        }

        (REWARD_STEP, false)
    }
    
    fn reset(&mut self) {
        self.position = START;
    }

    fn state(&self) -> [f32; 2] {
        [
            self.position.0 as f32 / (W - 1) as f32,
            self.position.1 as f32 / (H - 1) as f32,
        ]
    }

}



#[test]
fn test_maze_0() {
    const INPUTS: usize = 2;
    const OUTPUTS: usize = 4;

    let mut dqn = Dqn::<INPUTS, OUTPUTS>::new();
    let mut maze = Maze::new();


    for episode in 0..100 {
        maze.reset();

        let mut finished = false;
        for _step in 0..5 {
            let s = maze.state();

            let prediction = dqn.predict(&s);
            
            let (reward, done) = maze.step(prediction.action);
            println!("state: {:?} act: {:?}, pred: {}, reward. {}", s, prediction.action, prediction.q_values[prediction.action], reward);
            
            dqn.remember(reward);

            if done {
                finished = true;
                break;
            }
        }

        let loss = dqn.adjust();

        // if episode % 100 == 0 {
            println!(
                "episode={episode}, loss={loss}, finished={finished}",
            );
        // }
    }
}