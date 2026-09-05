//! Learns the path through the maze

use std::time::Duration;

use crate::reinforcement_learning::{dqn_dfdx::DqnDfdx, q_learning::maze::{Action, Agent, Maze, encode_position, print_maze}};


const W: usize = 8;
const H: usize = 8;
const WH: usize = W+H;
const HIDDEN: usize = 256;


impl Agent<WH> for DqnDfdx<WH, HIDDEN, 4>
{
    fn choose_action(&mut self, inputs: &[u8; WH]) -> (usize, [f32; 4]) {
        DqnDfdx::choose_action_u8(self, inputs)
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

    let mut agent: DqnDfdx<WH, HIDDEN, 4> = DqnDfdx::new(); 

    let mut loss = 0.0;
    for episode in 0..200000 {
        maze.reset();
        let nr_steps = 100;
        for step in 0..nr_steps {

            // random sampling
            let mut position = (
                fastrand::usize(0..W), 
                fastrand::usize(0..H),
            );
            while maze.is_wall(&position) {
                position = (
                    fastrand::usize(0..W), 
                    fastrand::usize(0..H),
                );
            }
            maze.set_position(position);
            //

            let position = maze.get_position();
            let position_encoded: [u8; WH] = encode_position::<W, H, WH>(&position);

            let (action, _q_values) = agent.choose_action_u8(&position_encoded);

            let action_ = Action::try_from(action).unwrap();
            let reward = maze.step(action_);
            let next_position = maze.get_position();
            let next_position_encoded = encode_position::<W, H, WH>(&next_position);

            let finished = step >= nr_steps -1 || maze.finished() || position == next_position;

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
        loss = agent.learn();
        // agent.learn_backlog();
        std::thread::sleep(Duration::from_millis(2));
        print_maze(&maze, &mut agent);
        println!("episode: {}, loss: {}", episode, loss);
    }

    assert!(maze.finished());
}

