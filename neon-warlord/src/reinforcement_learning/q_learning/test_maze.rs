//! Learns the path through the maze

use std::{os::unix::thread, time::Duration};

use crate::reinforcement_learning::{console_color::{BRIGHT_BLACK, RESET}, q_learning::{QLearning, maze::{Action, Maze, encode_position}}};


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