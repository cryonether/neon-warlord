//! Library to instantiate a forward rendering pipeline

mod animated_object;
pub mod animation_shader;
mod draw_gui;
mod forward_renderer;
pub mod geometry;
pub mod glow_storage;
pub mod height_map;
pub mod lod_heightmap_shader;
pub mod particle_shader;
pub mod particle_storage;
mod performance_monitor;
pub mod plasma_orb_storage;
mod terrain_storage;

pub use animated_object::animated_object_storage::AnimatedObjectStorage;
pub use draw_gui::DrawGui;
pub use forward_renderer::ForwardRenderer;
pub use forward_renderer::RendererSettings;
pub use performance_monitor::PerformanceMonitor;
pub use terrain_storage::HeightMap;
pub use terrain_storage::TerrainSettings;
pub use terrain_storage::TerrainStorage;
pub use terrain_storage::terrain_texture_details::TerrainTextureDetails;

// Some random useful utility functions

pub fn to_rgb(hex: &str) -> [f32; 3] {
    let hex = hex.trim_start_matches('#');

    assert!(hex.len() == 6, "Expected a 6-digit hex color like #RRGGBB");

    let r = u8::from_str_radix(&hex[0..2], 16).unwrap() as f32 / 255.0;
    let g = u8::from_str_radix(&hex[2..4], 16).unwrap() as f32 / 255.0;
    let b = u8::from_str_radix(&hex[4..6], 16).unwrap() as f32 / 255.0;

    [r, g, b]
}
