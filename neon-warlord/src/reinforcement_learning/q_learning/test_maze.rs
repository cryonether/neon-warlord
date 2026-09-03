//! Learns the path through the maze

use crate::reinforcement_learning::q_learning::QLearning;

const WALL: usize = 1;

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

    fn get_position<const WH:usize>(&self) -> [u8; WH] 
    {
        assert!(WH >= W + H);
        let mut res: [u8; WH] = [0; WH];

        res[self.position.0] = 1;
        res[self.position.1] = 1;

        res
    }
    
    fn finished(&self) -> bool {
        self.position == self.goal
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

#[test]
fn test_solve_maze() {
    const W: usize = 3;
    const H: usize = 3;
    const WH: usize = W+H;
    let maze: [[u8; W]; H] = [
        [0, 0, 0],
        [1, 1, 0],
        [0, 0, 0],
    ];
    let start = (0, 0);
    let goal = (2, 1);

    let mut maze = Maze::new(maze, start, goal);


    let mut agent: QLearning<WH, 4> = QLearning::new(); 

    for _episode in 0..10 {
        maze.reset();
        for _steps in 0..10 {

            let position: [u8; WH] = maze.get_position();

            let (action, _q_values) = agent.choose_action(&position);

            let action_ = Action::try_from(action).unwrap();
            let reward = maze.step(action_);
            agent.set_reward(&position, action, reward);

            if maze.finished() {
                break;
            }
        }
        agent.learn();
    }

    assert!(maze.finished());
}