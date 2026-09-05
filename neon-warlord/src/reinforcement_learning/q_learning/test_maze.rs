//! Learns the path through the maze

use std::{os::unix::thread, time::Duration};

use crate::reinforcement_learning::{
    console_color::{BRIGHT_BLACK, RESET},
    q_learning::{
        QLearning,
        maze::{Action, Agent, Maze, encode_position, print_maze},
    },
};

const W: usize = 8;
const H: usize = 8;
const WH: usize = W + H;

impl Agent<WH> for QLearning<WH, 4> {
    fn choose_action(&mut self, inputs: &[u8; WH]) -> (usize, [f32; 4]) {
        QLearning::choose_action(self, inputs)
    }
}

#[test]
#[ignore = "too expensive"]
fn test_solve_maze() {
    let maze: [[u8; W]; H] = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 1, 0],
        [1, 0, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 1, 1, 1, 0, 1, 0, 0],
        [1, 1, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
    ];
    let start = (0, 0);
    let goal = (0, 7);

    let mut maze = Maze::new(maze, start, goal);

    let mut agent: QLearning<WH, 4> = QLearning::new();

    for episode in 0..200 {
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

            // print_maze(&maze, &mut agent);
            print_maze(&maze, &mut agent);
            print!("episode: {}", episode);
            if maze.finished() {
                println!(" * ");
            } else {
                println!("   ");
            }

            std::thread::sleep(Duration::from_millis(10));

            if maze.finished() {
                break;
            }
        }
        agent.learn();
        // print_maze(&maze, &mut agent);
    }

    assert!(maze.finished());
}
