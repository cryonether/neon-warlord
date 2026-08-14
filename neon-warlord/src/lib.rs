//! Creates the Neon-Warlord application

mod advanced_composition;
mod agents;
mod ant_ai;
mod ant_controller;
mod ant_generator;
mod ant_state;
mod ant_storage;
mod camera_controller;
mod debug_overlay;
mod game_board;
mod heightmap_generator;
mod orb_controller;
mod orb_storage;
mod physics_simulation_v2;
mod physics_simulation_v3;
mod procedural_tree;
mod reinforcement_learning;
mod settings;
mod simple_physics_simulation;
mod sun_storage;
mod verlet_physics;
mod worker;
mod worker_instance;
mod triple_buffer;
mod physics_simulation_v3_drawer;

use forward_renderer::{
    AnimatedObjectStorage, ForwardRenderer, PerformanceMonitor, glow_storage::GlowStorage,
    height_map::height_map_drawer::HeightMapDrawer, particle_storage::ParticleStorage,
    plasma_orb_storage::PlasmaOrbStorage,
};
use instant::Instant;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
use wgpu_renderer::{
    default_application::{
        self,
        default_application_interface::{
            DefaultApplicationInterfaceCreate, DefaultApplicationInterfaceRuntime, RenderError,
        },
    },
    performance_monitor::{Fps, watch::Watch},
    wgpu_renderer::WgpuRendererInterface,
};
use winit::event::{ElementState, WindowEvent};

use crate::{
    ant_controller::AntPosition, ant_generator::AntGenerator, ant_storage::AntStorage, camera_controller::CameraController, debug_overlay::DebugOverlay, physics_simulation_v3::PhysicsSimulationV3, physics_simulation_v3_drawer::PhysicsSimulationV3Drawer, simple_physics_simulation::SimplePhysicsSimulation, sun_storage::SunStorage, worker_instance::WorkerInstance,
};

const WATCH_POINTS_SIZE: usize = 10;
const DEBUG_OVERLAY_SIZE: usize = 10;

struct ObjectSettings {
    pub max_nr_ants: usize,
}

struct CameraSettings {
    speed: f32,
    sensitivity: f32,
    sensitivity_scroll: f32,
}

const HEIGHT_MAP_INNER_WIDTH: usize = 32;
const HEIGHT_MAP_INNER_HEIGHT: usize = 32;

const HEIGHT_MAP_TILE_WIDTH: usize = HEIGHT_MAP_INNER_WIDTH + 2;
const HEIGHT_MAP_TILE_HEIGHT: usize = HEIGHT_MAP_INNER_HEIGHT + 2;

const HEIGHT_MAP_NR_TILES_X: usize = 4;
const HEIGHT_MAP_NR_TILES_Y: usize = 4;

const HEIGHT_MAP_WIDTH: usize = HEIGHT_MAP_INNER_WIDTH * HEIGHT_MAP_NR_TILES_X;
const HEIGHT_MAP_HEIGHT: usize = HEIGHT_MAP_INNER_HEIGHT * HEIGHT_MAP_NR_TILES_Y;

struct NeonWarlord {
    _settings: settings::Settings,

    // Render engine
    size: winit::dpi::PhysicalSize<u32>,
    scale_factor: f32,

    renderer: ForwardRenderer,
    font: rusttype::Font<'static>,

    camera_controller: CameraController,

    // Debug Utilities
    fps: Fps,
    ups: u32,
    watch_fps: Watch<WATCH_POINTS_SIZE>,
    performance_monitor_fps: PerformanceMonitor<WATCH_POINTS_SIZE>,
    performance_monitor_ups: PerformanceMonitor<WATCH_POINTS_SIZE>,
    debug_overlay: DebugOverlay<DEBUG_OVERLAY_SIZE>,
    mouse_pos_y: u32,
    mouse_pos_x: u32,

    // debug
    device_id: String,
    phase: String,
    location: String,
    force: String,
    id: String,

    // Terrain
    // terrain: TerrainStorage,
    height_map: forward_renderer::height_map::HeightMap<
        HEIGHT_MAP_WIDTH,
        HEIGHT_MAP_HEIGHT,
        HEIGHT_MAP_TILE_WIDTH,
        HEIGHT_MAP_TILE_HEIGHT,
    >,
    height_map_drawer: HeightMapDrawer,

    // Ants
    ants: AntStorage,
    _ant_generator: AntGenerator,

    // Sun
    sun: SunStorage,

    // Particles
    particles: ParticleStorage,
    plasma_orbs: PlasmaOrbStorage,
    glows: GlowStorage,

    // Orbs
    // orbs: OrbStorage,

    // Simple physics simulation
    simple_physics_simulation: SimplePhysicsSimulation,

    // agent physics simulation
    // physics_simulation: PhysicsSimulationV2,
    physics_simulation_v3: PhysicsSimulationV3,
    physics_simulation_v3_drawer: PhysicsSimulationV3Drawer,
    // Worker
    worker: WorkerInstance,
    ant_positions: [AntPosition; 1],
}

impl NeonWarlord {
    pub fn new(
        renderer_interface: &mut dyn WgpuRendererInterface,
        size: winit::dpi::PhysicalSize<u32>,
        scale_factor: f32,
    ) -> Self {
        let settings = settings::Settings::new();

        let renderer = ForwardRenderer::new(renderer_interface, settings.get_renderer_settings());

        // font
        let font = wgpu_renderer::freefont::create_font_free_mono();

        // Camera
        let camera_controller = CameraController::new(
            settings.get_camera_settings().speed,
            settings.get_camera_settings().sensitivity,
            settings.get_camera_settings().sensitivity_scroll,
        );

        // performance monitor
        let watch_fps = Watch::new();
        let performance_monitor_fps = PerformanceMonitor::new(
            renderer_interface,
            &renderer.texture_bind_group_layout,
            &font,
            colorous::RAINBOW,
            true,
            "60 fps / 16.66 ms",
            scale_factor,
        );
        let performance_monitor_ups = PerformanceMonitor::new(
            renderer_interface,
            &renderer.texture_bind_group_layout,
            &font,
            colorous::PLASMA,
            false,
            "60 ups / 16.66 ms",
            scale_factor,
        );
        let fps = Fps::new();
        let debug_overlay = DebugOverlay::new(
            renderer_interface,
            &renderer.texture_bind_group_layout,
            &font,
            size.height,
            size.width,
            scale_factor,
        );

        // Mouse position
        let _mouse_pos_y = 0;
        let _mouse_pos_x = 0;

        // create ant
        // let glb_bin = include_bytes!("../res/wiggle_tower2.glb");
        let glb_bin = include_bytes!("../res/ant_0_10.glb");
        let animated_object_storage_ant = AnimatedObjectStorage::create_from_glb(
            renderer_interface,
            &renderer.animation_bind_group_layout,
            glb_bin,
            settings.get_object_settings().max_nr_ants,
        );

        let mut ants = AntStorage::new(
            // point_light_storage_ant,
            animated_object_storage_ant,
            settings.get_object_settings().max_nr_ants,
        );

        let ant_generator = AntGenerator::new(settings.get_object_settings().max_nr_ants);
        for elem in &ant_generator.ants {
            ants.set_ant(elem);
        }

        // terrain
        // let mut terrain_settings = settings.get_terrain_settings();
        // terrain_settings.nr_tiles = HEIGHT_MAP_INNER_WIDTH;
        // terrain_settings.max_depth = 1;

        // let terrain = TerrainStorage::new(
        //     terrain_settings,
        //     renderer_interface,
        //     &renderer.texture_bind_group_layout,
        //     include_bytes!("../res/tile.png"),
        // );

        let mut height_map: forward_renderer::height_map::HeightMap<
            HEIGHT_MAP_WIDTH,
            HEIGHT_MAP_HEIGHT,
            HEIGHT_MAP_TILE_WIDTH,
            HEIGHT_MAP_TILE_HEIGHT,
        > = forward_renderer::height_map::HeightMap::new();

        let mut data = Vec::new();
        for _i in 0..HEIGHT_MAP_INNER_HEIGHT * HEIGHT_MAP_INNER_WIDTH {
            // data.push(i as f32 * 0.1);
            data.push(1.0);
        }
        // height_map.set_tile(1, 1, &data);

        // height_map.set_tile(0, 0, &data);
        // height_map.set_tile(0, 1, &data);
        // height_map.set_tile(1, 0, &data);

        // height_map.set_tile(1, 2, &data);
        // height_map.set_tile(2, 1, &data);
        // height_map.set_tile(2, 2, &data);

        // height_map.set_tile(0, 2, &data);
        // height_map.set_tile(2, 0, &data);

        height_map.set_tile(0, 3, &data);
        height_map.set_tile(0, 0, &data);

        height_map.set_tile(3, 3, &data);
        height_map.set_tile(3, 0, &data);

        let height_map_inner_width = HEIGHT_MAP_INNER_WIDTH;
        let height_map_inner_height = HEIGHT_MAP_INNER_WIDTH;
        let _height_map_nr_tiles_x = HEIGHT_MAP_NR_TILES_X;
        let _height_map_nr_tiles_y = HEIGHT_MAP_NR_TILES_Y;
        let mut height_map_drawer = HeightMapDrawer::new(
            renderer_interface,
            &renderer.texture_bind_group_layout,
            include_bytes!("../res/tile.png"),
            height_map_inner_width,
            height_map_inner_height,
            _height_map_nr_tiles_x,
            _height_map_nr_tiles_y,
        );

        height_map_drawer.update(
            renderer_interface,
            &renderer.heightmap_bind_group_layout,
            &height_map,
        );

        // sun
        let sun = SunStorage::new(renderer_interface);

        // Particles
        let particles = ParticleStorage::new(renderer_interface, 1);
        let plasma_orbs = PlasmaOrbStorage::new(renderer_interface, 1);
        let glows = GlowStorage::new(renderer_interface, 1);

        // Simple physics simulation
        let simple_physics_simulation = SimplePhysicsSimulation::new(renderer_interface);

        // physics simulation
        // let mut physics_simulation = PhysicsSimulationV2::new();
        // physics_simulation.create_agent_0(renderer_interface);
        // physics_simulation.create_pendulum(renderer_interface);

        let (producer, consumer) = triple_buffer::create(physics_simulation_v3_drawer::DrawerObjects::new());
        let physics_simulation_v3 = PhysicsSimulationV3::new(producer);
        let physics_simulation_v3_drawer = PhysicsSimulationV3Drawer::new(renderer_interface, consumer);

        // Worker
        let worker = WorkerInstance::new();

        let ant_positions = [AntPosition::new(); 1];

        Self {
            _settings: settings,
            size,
            scale_factor,
            renderer,
            font,
            watch_fps,
            performance_monitor_fps,
            performance_monitor_ups,
            mouse_pos_y: 0,
            mouse_pos_x: 0,
            fps,
            debug_overlay,
            // terrain,
            height_map,
            height_map_drawer,
            // terrain_generator,
            ants,
            _ant_generator: ant_generator,
            camera_controller,
            device_id: String::new(),
            phase: String::new(),
            location: String::new(),
            force: String::new(),
            id: String::new(),
            sun,
            particles,
            plasma_orbs,
            glows,
            worker,
            ups: 0,
            ant_positions,
            simple_physics_simulation,
            // physics_simulation,
            physics_simulation_v3,
            physics_simulation_v3_drawer,
        }
    }
}

#[allow(unused)]
fn apply_scale_factor(
    position: winit::dpi::PhysicalPosition<f64>,
    scale_factor: f32,
) -> winit::dpi::PhysicalPosition<f64> {
    cfg_if::cfg_if! {
        // apply scale factor for the web
        if #[cfg(target_arch = "wasm32")] {
            let mut res = position;
            res.x = res.x / scale_factor as f64;
            res.y = res.y / scale_factor as f64;
            res
        }
        else {
            position
        }
    }
}

impl DefaultApplicationInterfaceCreate for NeonWarlord {
    fn create(
        renderer_interface: &mut dyn wgpu_renderer::wgpu_renderer::WgpuRendererInterface,
        size: winit::dpi::PhysicalSize<u32>,
        scale_factor: f32,
    ) -> Self {
        Self::new(renderer_interface, size, scale_factor)
    }
}

impl DefaultApplicationInterfaceRuntime for NeonWarlord {
    fn get_size(&self) -> winit::dpi::PhysicalSize<u32> {
        self.size
    }

    fn resize(
        &mut self,
        renderer_interface: &mut dyn wgpu_renderer::wgpu_renderer::WgpuRendererInterface,
        new_size: winit::dpi::PhysicalSize<u32>,
    ) {
        println!("resize: {:?}", new_size);

        self.size = new_size;
        self.renderer.resize(renderer_interface, new_size);
        self.debug_overlay.resize(
            renderer_interface,
            &self.renderer.texture_bind_group_layout,
            &self.font,
            new_size.height,
            new_size.width,
        );
    }

    fn update_scale_factor(
        &mut self,
        renderer_interface: &mut dyn wgpu_renderer::wgpu_renderer::WgpuRendererInterface,
        scale_factor: f32,
    ) {
        println!("new scale factor {}", scale_factor);

        // let scale_factor = 2.0;
        self.scale_factor = scale_factor;
        self.performance_monitor_fps.rescale(
            renderer_interface,
            &self.renderer.texture_bind_group_layout,
            &self.font,
            scale_factor,
        );
        self.performance_monitor_ups.rescale(
            renderer_interface,
            &self.renderer.texture_bind_group_layout,
            &self.font,
            scale_factor,
        );

        self.debug_overlay.rescale(
            renderer_interface,
            &self.renderer.texture_bind_group_layout,
            &self.font,
            scale_factor,
        );
    }

    fn update(
        &mut self,
        renderer_interface: &mut dyn wgpu_renderer::wgpu_renderer::WgpuRendererInterface,
        dt: instant::Duration,
    ) {
        let time_stamp = Instant::now();

        self.watch_fps.stop(WATCH_POINTS_SIZE - 1);
        self.watch_fps.update();
        let watch_fps_data = self.watch_fps.get_viewer_data();

        // Render engine
        self.camera_controller
            .update_camera(&mut self.renderer.camera, dt);
        self.renderer.update(renderer_interface, dt);

        // Worker
        let mut watch_index = 0;
        self.watch_fps.start(watch_index, "Update Worker");
        {
            let messages = self.worker.receive();
            for message in messages.try_iter() {
                match message {
                    // ##########################################################
                    worker::WorkerMessage::Ups(ups) => {
                        self.ups = ups;
                    }
                    // ##########################################################
                    worker::WorkerMessage::UpdateWatchPoints(watch_ups_data) => {
                        self.performance_monitor_ups.update_from_data(
                            renderer_interface,
                            &self.font,
                            &watch_ups_data,
                        );
                    }
                    // ##########################################################
                    worker::WorkerMessage::TerrainData(_terrain_part) => {

                        // // update physics
                        // let max_depth = 1;

                        // let mut data = Vec::new();
                        // for y in 0..terrain_part.details.nr_tiles {
                        //     for x in 0..terrain_part.details.nr_tiles {
                        //         let size_0 = terrain_part.details.size_0;
                        //         let val = terrain_part.heights[y * size_0 + x];
                        //         data.push(val);
                        //     }
                        // }

                        // if max_depth - terrain_part.details.depth == 1 {
                        //     let world_y = (terrain_part.details.pos_1.y+HEIGHT_MAP_INNER_HEIGHT as isize/2) as usize;
                        //     let world_x = (terrain_part.details.pos_1.x+HEIGHT_MAP_INNER_WIDTH as isize/2) as usize;

                        //     self.height_map.set_tile(world_y, world_x, &data);
                        // }

                        // // update gpu
                        // self.terrain.update_height_map(
                        //     renderer_interface,
                        //     &self.renderer.heightmap_bind_group_layout,
                        //     *terrain_part,
                        // );
                    }
                    // ##########################################################
                    worker::WorkerMessage::Snapshot(snapshot) => {
                        // transmitting the whole state ensures that a complete tick of the physics thread has been completed
                        for elem in snapshot.ant_actions {
                            let index = elem.index;
                            match elem.action {
                                // ##########################################################
                                ant_controller::AntAction::UpdatePosition(snapshot) => {
                                    self.ant_positions[index]
                                        .pos
                                        .add(snapshot.pos, snapshot.time_stamp);
                                    self.ant_positions[index].look_at = snapshot.look_at;
                                    self.ant_positions[index].is_final = false
                                }
                                // ##########################################################
                                ant_controller::AntAction::FinalPosition(snapshot) => {
                                    self.ant_positions[index]
                                        .pos
                                        .add(snapshot.pos, snapshot.time_stamp);
                                    self.ant_positions[index].look_at = snapshot.look_at;
                                    self.ant_positions[index].is_final = true
                                }
                                // ##########################################################
                                ant_controller::AntAction::SetAnimation(ant_animation) => {
                                    let animation_index = match ant_animation {
                                        ant_controller::AntAnimation::Idle => 0,
                                        ant_controller::AntAnimation::Walk => 1,
                                        ant_controller::AntAnimation::_ChargeShot => 0,
                                    };

                                    self.ants.set_animation(index, animation_index);
                                }
                                // ##########################################################
                                ant_controller::AntAction::SetAnimationSpeed(speed) => {
                                    self.ants.set_animation_speed(index, speed);
                                }
                            }
                        }
                    }
                }
            }

            self.worker.update(dt);
        }
        let _worker = self.worker.send();
        self.watch_fps.stop(watch_index);

        // Calculate current Snapshot
        for (i, elem) in self.ant_positions.iter().enumerate() {
            let pos = if elem.is_final {
                elem.pos.pos
            } else {
                elem.pos.lerp(time_stamp)
            };
            self.ants.set_position(i, pos, elem.look_at);
        }

        // let snapshot = self.last_snapshot.lerp(&self.snapshot, time_stamp);

        // let ant = snapshot.ants[0];
        // self.ants.set_position(0, ant.pos, ant.look_at);

        // Particles
        watch_index += 1;
        self.watch_fps.start(watch_index, "Update Particles");
        {
            // self.particles.update(renderer_interface, dt);
            // self.plasma_orbs.update(renderer_interface, dt);
            // self.glows.update(renderer_interface, dt);

            self.simple_physics_simulation.update(renderer_interface);

            // self.physics_simulation.update_physics(&self.height_map);
            // self.physics_simulation.update_device(renderer_interface);

            self.physics_simulation_v3.update_physics(&self.height_map);
            self.physics_simulation_v3.update_drawer();
            self.physics_simulation_v3_drawer.update(renderer_interface);
        }
        self.watch_fps.stop(watch_index);

        // Animations
        watch_index += 1;
        self.watch_fps.start(watch_index, "Update Animations");
        {
            self.ants.animated_object_storage.update_animations(&dt);

            self.ants
                .animated_object_storage
                .update_device_data(renderer_interface);
        }
        self.watch_fps.stop(watch_index);

        // Terrain
        // watch_index += 1;
        // self.watch_fps.start(watch_index, "Update Terrain");
        {
            // // set terrain view position
            // self.terrain
            //     .set_view_position(&self.renderer.get_view_position());

            // // generate map
            // let requests = self.terrain.get_requests().clone();
            // for request in requests {
            //     let _ = worker.send(MainMessage::GetTerrain(request));
            // }
            // self.terrain.clear_requests();
        }
        // self.watch_fps.stop(watch_index);

        // Debug utilities
        watch_index += 1;
        self.watch_fps.start(watch_index, "Update Debug");
        {
            self.fps.update(dt);

            self.debug_overlay.update_val(
                renderer_interface,
                &self.font,
                0,
                "fps",
                self.fps.get() as f32,
            );

            self.debug_overlay.update_val(
                renderer_interface,
                &self.font,
                1,
                "ups",
                self.ups as f32,
            );

            self.debug_overlay.update_val(
                renderer_interface,
                &self.font,
                2,
                "x",
                self.mouse_pos_x as f32,
            );

            self.debug_overlay.update_val(
                renderer_interface,
                &self.font,
                3,
                "y",
                self.mouse_pos_y as f32,
            );

            self.debug_overlay
                .update_str(renderer_interface, &self.font, 4, &self.device_id);

            self.debug_overlay
                .update_str(renderer_interface, &self.font, 5, &self.phase);

            self.debug_overlay
                .update_str(renderer_interface, &self.font, 6, &self.location);

            self.debug_overlay
                .update_str(renderer_interface, &self.font, 7, &self.force);

            self.debug_overlay
                .update_str(renderer_interface, &self.font, 8, &self.id);

            self.performance_monitor_fps.update_from_data(
                renderer_interface,
                &self.font,
                &watch_fps_data,
            );
        }
        self.watch_fps.stop(watch_index);
    }

    fn input(&mut self, event: &winit::event::WindowEvent) -> bool {
        match event {
            // #########################################################
            WindowEvent::KeyboardInput {
                event:
                    winit::event::KeyEvent {
                        physical_key:
                            winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::F1),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => {
                self.performance_monitor_ups.show = false;
                self.performance_monitor_fps.show = !self.performance_monitor_fps.show;
                true
            }
            // #########################################################
            WindowEvent::KeyboardInput {
                event:
                    winit::event::KeyEvent {
                        physical_key:
                            winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::F2),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => {
                self.performance_monitor_fps.show = false;
                self.performance_monitor_ups.show = !self.performance_monitor_ups.show;
                true
            }
            // #########################################################
            WindowEvent::KeyboardInput {
                event:
                    winit::event::KeyEvent {
                        // virtual_keycode: Some(key),
                        physical_key: winit::keyboard::PhysicalKey::Code(key),
                        state,
                        ..
                    },
                ..
            } => {
                // self.renderer.process_keyboard(*key, *state),
                self.camera_controller.process_keyboard(key, state)
            }
            // #########################################################
            WindowEvent::MouseWheel { delta, .. } => {
                // self.renderer.process_scroll(delta);
                self.camera_controller.process_scroll(delta);
                true
            }
            // #########################################################
            WindowEvent::CursorMoved { position, .. } => {
                let pos = apply_scale_factor(*position, self.scale_factor);

                self.mouse_pos_y = self.size.height.saturating_sub(pos.y as u32);
                self.mouse_pos_x = pos.x as u32;
                true
            }
            // #########################################################
            winit::event::WindowEvent::Touch(touch) => {
                self.device_id = format!("{:?}", touch.device_id);
                self.phase = format!("{:?}", touch.phase);
                self.location = format!("{:?}", touch.location);
                self.force = format!("{:?}", touch.force);
                self.id = format!("{:?}", touch.id);
                true
            }

            _ => false,
        }
    }

    fn render(
        &mut self,
        renderer_interface: &mut dyn wgpu_renderer::wgpu_renderer::WgpuRendererInterface,
    ) -> Result<(), RenderError> {
        // render current frame
        let res;
        {
            res = self.renderer.render(
                renderer_interface,
                &mut self.height_map_drawer,
                &[&self.ants.animated_object_storage],
                &[
                    &self.performance_monitor_fps,
                    &self.performance_monitor_ups,
                    &self.debug_overlay,
                ],
                &[
                    &self.sun,
                    &self.simple_physics_simulation,
                    // &self.physics_simulation,
                    // &self.physics_simulation_v3,
                ],
                &[
                    &self.simple_physics_simulation,
                    // &self.physics_simulation,
                    // &self.physics_simulation_v3,
                ],
                &[&self.particles],
                &[&self.plasma_orbs],
                &[&self.glows],
                &[&self.physics_simulation_v3_drawer],
                &[&self.physics_simulation_v3_drawer],
                &mut self.watch_fps,
            )
        }
        self.watch_fps
            .start(WATCH_POINTS_SIZE - 1, "Wait for Window Event");

        res
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub fn run() {
    default_application::init_env_logger();
    let event_loop = default_application::create_event_loop();

    default_application::run_app::<NeonWarlord>(event_loop);
}
