//! A maze for testing reinforcement learning algorithms

use crate::reinforcement_learning::console_color::{BRIGHT_BLACK, RESET};

pub struct Maze<const W: usize, const H: usize> {
    maze: [[u8; W]; H],

    start: (usize, usize),
    goal: (usize, usize),

    position: (usize, usize),
}

impl<const W: usize, const H: usize> Maze<W, H> 
{
    pub fn new(maze: [[u8; W]; H], start: (usize, usize), goal: (usize, usize), ) -> Self {

        let position: (usize, usize) = (0, 0);

        Self { maze, start, goal, position }
    }

    pub fn step(&mut self, action: Action) -> f32 {
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

    pub fn reset(&mut self) {
        self.position = self.start;
    }

    pub fn get_position(&self) -> (usize, usize) {
        self.position
    }

    pub fn finished(&self) -> bool {
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

#[repr(usize)]
pub enum Action {
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

pub fn encode_position<const W: usize, const H: usize, const WH: usize>
    (position: &(usize, usize)) -> [u8; WH]
{
    assert!(WH >= W + H);
    let mut res: [u8; WH] = [0; WH];

    res[position.0] = 1;
    res[W + position.1] = 1;

    res
}

pub fn print_maze<const W: usize, const H: usize, const WH: usize>
    (maze: &Maze<W, H>, agent: &mut impl Agent<WH>)
{
    for y in 0..H {
        for _x in 0..W {
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

    for _x in 0..W {
            print!("-------------");
    }
    println!("-");
    println!("");
}

pub trait Agent<const WH: usize> {
    fn choose_action(&mut self, inputs: &[u8; WH]) -> (usize, [f32; 4]);
}