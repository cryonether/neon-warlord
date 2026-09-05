//! Learns the path through the maze

use std::time::Duration;

use instant::Instant;

use crate::reinforcement_learning::{
    dqn::Dqn,
    q_learning::maze::{Action, Agent, Maze, encode_position, print_maze},
};

const W: usize = 8;
const H: usize = 8;
const WH: usize = W + H;

impl Agent<WH> for Dqn<WH, 4> {
    fn choose_action(&mut self, inputs: &[u8; WH]) -> (usize, [f32; 4]) {
        Dqn::choose_action_u8(self, inputs)
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

    // let maze: [[u8; W]; H] = [
    //     [0, 0, 0],
    //     [1, 1, 0],
    //     [0, 0, 0],
    // ];

    let start = (0, 0);
    let goal = (0, 7);

    let mut maze = Maze::new(maze, start, goal);

    let mut agent: Dqn<WH, 4> = Dqn::new();

    let mut last_print = Instant::now();
    for episode in 0..100_000 {
        maze.reset();
        let nr_steps = 128;
        for step in 0..nr_steps {
            // random sampling
            // let mut position = (
            //     fastrand::usize(0..W),
            //     fastrand::usize(0..H),
            // );
            // while maze.is_wall(&position) {
            //     position = (
            //         fastrand::usize(0..W),
            //         fastrand::usize(0..H),
            //     );
            // }
            // maze.set_position(position);
            //

            let position = maze.get_position();
            let position_encoded: [u8; WH] = encode_position::<W, H, WH>(&position);

            let (action, _q_values) = agent.choose_action_u8(&position_encoded);

            let action_ = Action::try_from(action).unwrap();
            let reward = maze.step(action_);
            let next_position = maze.get_position();
            let next_position_encoded = encode_position::<W, H, WH>(&next_position);

            let finished = step >= nr_steps - 1 || maze.finished() || position == next_position;

            agent.set_reward_u8(
                position_encoded,
                action,
                reward,
                next_position_encoded,
                finished,
            );

            if maze.finished() {
                break;
            }
        }
        // loss = agent.learn();
        let loss = agent.learn_replay();

        if last_print.elapsed() >= Duration::from_millis(20) {
            last_print = Instant::now();
            print_maze(&maze, &mut agent);
            print!("episode: {}, loss: {}", episode, loss);
            if maze.finished() {
                println!(" * ");
            } else {
                println!("   ");
            }
        }
    }

    assert!(maze.finished());
}
