//! Learns the path through the maze

use std::{os::unix::thread, time::Duration};

use crate::reinforcement_learning::q_learning::QLearning;

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

struct Maze<const W: usize, const H: usize> {
    maze: [[u8; W]; H],

    start: (usize, usize),
    goal: (usize, usize),

    position: (usize, usize),
}

impl<const W: usize, const H: usize> Maze<W, H> 
{
    fn new(maze: [[u8; W]; H], start: (usize, usize), goal: (usize, usize), ) -> Self {

        let position: (usize, usize) = (0, 0);

        Self { maze, start, goal, position }
    }

    fn step(&mut self, action: Action) -> f32 {
        let position = self.position;
        let mut x = position.0 as usize;
        let mut y = position.1;       
        
        match action {
            Action::Up => {
                if y == 0 {
                    return -1.0;
                }
                y = y-1;                
            },
            Action::Down => {
                if y == H-1 {
                    return -1.0;
                }
                y = y+1;   
            },
            Action::Left => {
                if x== 0 {
                    return -1.0;
                }
                x = x-1;   
            },
            Action::Right => {
                if x == W-1 {
                    return -1.0;
                }
                x = x+1;   
            },

        }

        if self.maze[y][x] != 0 {
            -1.0
        } else {
            self.position = (x, y);
            if self.position == self.goal {
                10.0
            } else {
                -0.1
            }
        }
    }

    fn reset(&mut self) {
        self.position = self.start;
    }

    fn get_position(&self) -> (usize, usize) {
        self.position
    }

    fn finished(&self) -> bool {
        self.position == self.goal
    }

    pub fn is_wall(&self, position: &(usize, usize)) -> bool {
        self.maze[position.1][position.0] != 0
    }

    pub fn is_start(&self, position: &(usize, usize)) -> bool {
        *position == self.start
    }

    pub fn is_goal(&self, position: &(usize, usize)) -> bool {
        *position == self.goal
    }

    pub fn is_agent_position(&self, position: &(usize, usize)) -> bool {
        *position == self.position
    }
}


// const S: u8 = 2;
// const G: u8 = 3;

#[repr(usize)]
enum Action {
    Up = 0,
    Down = 1,
    Left = 2,
    Right = 3,
}

impl TryFrom<usize> for Action {
    type Error = ();

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Action::Up),
            1 => Ok(Action::Down),
            2 => Ok(Action::Left),
            3 => Ok(Action::Right),
            _ => Err(()),
        }
    }
}

fn encode_position<const W: usize, const H: usize, const WH: usize>
    (position: &(usize, usize)) -> [u8; WH]
{
    assert!(WH >= W + H);
    let mut res: [u8; WH] = [0; WH];

    res[position.0] = 1;
    res[W + position.1] = 1;

    res
}
fn print_maze<const W: usize, const H: usize, const WH: usize>
    (maze: &Maze<W, H>, agent: &mut QLearning<WH, 4>)
{
    // let position_agent = maze.get_position();

    for y in 0..H {
        for x in 0..W {
            print!("-------------");
        }
        println!("-");


        for i in 0..3 {
            for x in 0..W {
                let position = (x, y);
                let position_encoded = encode_position::<W, H, WH>(&position);
                let (_action, q_values) = agent.choose_action(&position_encoded);

                let marker = if maze.is_wall(&position) {
                    print!("{}", BRIGHT_BLACK);
                    '#'
                }
                else if maze.is_goal(&position) {
                    'G'
                }
                else if maze.is_start(&position) {
                    'S'
                }
                else {
                    ' '
                };

                let marker_player = if maze.is_agent_position(&position) {
                    '*'
                }
                else {
                    ' '
                };

                match i {
                    0 => {
                        print!("|{}  {:5.2}    ", marker, q_values[0]);
                    },
                    1 => {
                        print!("|{:5.2}{}{:5.2} ", q_values[2], marker_player, q_values[3]);
                    },
                    2 => {
                        print!("|   {:5.2}    ", q_values[1]);
                    },
                    _ => {}
                }
                print!("{}", RESET);
            }
            println!("|");

        }
    }

    for x in 0..W {
            print!("-------------");
    }
    println!("-");
    println!("");
}

#[test]
fn test_solve_maze() {
    const W: usize = 8;
    const H: usize = 8;
    const WH: usize = W+H;
    let maze: [[u8; W]; H] = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 1, 1, 0, 1, 1],
        [0, 0, 0, 0, 1, 0, 1, 0],
        [1, 0, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 1, 1, 1, 0, 1, 0, 0],
        [1, 1, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
    ];
    let start = (0, 0);
    let goal = (0, 7);

    let mut maze = Maze::new(maze, start, goal);


    let mut agent: QLearning<WH, 4> = QLearning::new(); 

    for _episode in 0..60 {
        maze.reset();
        for _steps in 0..30 {

            let position = maze.get_position();
            let position_encoded: [u8; WH] = encode_position::<W, H, WH>(&position);

            let (action, _q_values) = agent.choose_action(&position_encoded);

            let action_ = Action::try_from(action).unwrap();
            let reward = maze.step(action_);
            let next_position = maze.get_position();
            let next_position_encoded = encode_position::<W, H, WH>(&next_position);
            agent.set_reward(position_encoded, action, reward, next_position_encoded);

            print_maze(&maze, &mut agent);
            std::thread::sleep(Duration::from_millis(100));

            if maze.finished() {
                break;
            }
        }
        agent.learn();
        print_maze(&maze, &mut agent);
        // std::thread::sleep(Duration::from_secs(1));
    }

    assert!(maze.finished());
}