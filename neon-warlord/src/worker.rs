//! Runs the physics simulation

pub mod interpolated_position;

use std::{sync::mpsc, time::Duration};

use forward_renderer::{HeightMap, TerrainTextureDetails};
use instant::Instant;
use wgpu_renderer::performance_monitor::{
    Fps,
    watch::{self, Watch},
};

use crate::{
    ant_ai::AntAi,
    ant_controller::{AntActionStruct, AntController},
    game_board::{Faction, GameBoard},
    heightmap_generator::HeightMapGenerator,
};

const WATCH_POINTS_SIZE: usize = 10;

pub enum MainMessage {
    #[allow(unused)]
    GetTerrain(TerrainTextureDetails), // Requests the terrain heightmap
}

pub enum WorkerMessage {
    UpdateWatchPoints(Box<watch::WatchViewerData<10>>), // all the data for a point of the performance monitor
    Ups(u32),
    TerrainData(Box<HeightMap>),
    Snapshot(Snapshot),
}

// #[derive(Clone)]
pub struct Snapshot {
    pub ant_actions: Vec<AntActionStruct>,
}

// impl Snapshot {
//     pub fn new() -> Self {
//         Self { ant_actions: Vec::new() }
//     }

//     // pub fn lerp(&self, next: &Snapshot, time_stamp: Instant) -> Snapshot {
//     //     let mut ants = [AnimationPosition::zero(); 16];

//     //     let amount = 1.0 /  (next.time_stamp - self.time_stamp).as_nanos() as f32 * (time_stamp - self.time_stamp).as_nanos() as f32;

//     //     for i in 0..ants.len() {
//     //         ants[i].pos = self.ants[i].pos.lerp(next.ants[i].pos, amount);
//     //         // ants[i].look_at = self.ants[i].look_at.lerp(next.ants[i].look_at, amount);
//     //         ants[i].look_at = next.ants[i].look_at;
//     //     }

//     //     Snapshot { ants, time_stamp }
//     // }
// }

// // Input |   |  |
// // Render  |######| |#######| |#######|
// // Tick          |#######|   |######|
// //
// #[derive(Clone, Copy)]
// pub struct AnimationPosition {
//     pub pos: cgmath::Vector3<f32>,
//     pub look_at: cgmath::Vector3<f32>,
//     pub animation: u32,
// }

// impl AnimationPosition {
//     pub fn zero() -> Self {
//         Self { pos: cgmath::Vector3::zero(), look_at: cgmath::Vector3::zero(), animation: 0 }
//     }
// }

pub struct Worker {
    channel_0_rx: mpsc::Receiver<MainMessage>,
    channel_1_tx: mpsc::Sender<WorkerMessage>,

    // Debug
    ups: Fps,

    watch_ups: Watch<WATCH_POINTS_SIZE>,

    // Terrain
    terrain_generator: HeightMapGenerator,

    // Game board
    game_board: GameBoard,

    // Ant
    ant_state: AntController,
    ant_ai: AntAi,
}

impl Worker {
    pub fn new(
        channel_0_rx: mpsc::Receiver<MainMessage>,
        channel_1_tx: mpsc::Sender<WorkerMessage>,
    ) -> Self {
        // Debug
        let ups = Fps::new();
        let watch_ups = Watch::new();

        // Terrain
        let terrain_generator = HeightMapGenerator::new();

        // Game board
        let game_board = GameBoard::new();

        // Ant
        let ant_state = AntController::new(cgmath::Vector2 { x: 0.0, y: 0.0 }, 0);
        let ant_ai = AntAi::new(Faction::Blue);

        Self {
            channel_0_rx,
            channel_1_tx,

            ups,
            watch_ups,

            terrain_generator,

            game_board,

            ant_state,
            ant_ai,
        }
    }

    pub fn update(&mut self, _tick: u64, dt: Duration) {
        let time_stamp = Instant::now();

        let main = &self.channel_1_tx;
        let messages = &self.channel_0_rx;

        // Update watch
        self.watch_ups.update();
        let watch_ups_data = self.watch_ups.get_viewer_data();
        let _ = main.send(WorkerMessage::UpdateWatchPoints(Box::new(watch_ups_data)));
        let mut watch_index = 0;

        // update ups
        self.ups.update(dt);
        let _ = main.send(WorkerMessage::Ups(self.ups.get()));

        // Process messages
        let mut terrain_detail = Vec::new();
        self.watch_ups.start(watch_index, "Messages");
        {
            for message in messages.try_iter() {
                match message {
                    // ##########################################################
                    MainMessage::GetTerrain(terrain_texture_details) => {
                        terrain_detail.push(terrain_texture_details);
                    }
                }
            }
        }
        self.watch_ups.stop(watch_index);

        // Terrain
        watch_index += 1;
        self.watch_ups.start(watch_index, "Update Terrain");
        {
            for elem in terrain_detail {
                let terrain_part = self.terrain_generator.generate(&elem);
                let _ = main.send(WorkerMessage::TerrainData(Box::new(terrain_part)));
            }
        }
        self.watch_ups.stop(watch_index);

        // Ant
        watch_index += 1;
        self.watch_ups.start(watch_index, "Update Ant");
        {
            let ant_state = &mut self.ant_state;
            let game_board = &mut self.game_board;

            // update ant_state by ant_ai
            self.ant_ai.update(ant_state, game_board);

            // update state
            let mut actions: Vec<AntActionStruct> = Vec::new();
            self.ant_state.update(&time_stamp, &mut actions);

            // // send to main
            // let mut ants = [AnimationPosition::zero(); 16];
            // ants[0].animation = 0;
            // ants[0].pos = ant_pos.pos;
            // ants[0].look_at = ant_pos.look_at;

            let _ = main.send(WorkerMessage::Snapshot(Snapshot {
                ant_actions: actions,
            }));
        }
        self.watch_ups.stop(watch_index);
    }
}
