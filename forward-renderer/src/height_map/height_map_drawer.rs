//! Draws a heightmap

use cgmath::Zero;
use wgpu_renderer::{shape::{self, MeshDataInterface}, vertex_heightmap_shader, wgpu_renderer::WgpuRendererInterface};

use crate::{height_map::HeightMap, lod_heightmap_shader};
type Vec3 = cgmath::Vector3<f32>;


pub struct HeightMapDrawer {
    mesh: lod_heightmap_shader::Mesh,
    texture: lod_heightmap_shader::Texture,
    height_map_textures: Vec<lod_heightmap_shader::HeightmapTexture>,
    instances: Vec<lod_heightmap_shader::InstanceBuffer<lod_heightmap_shader::Instance>>,

    view_position: Vec3,

    _height_map_inner_width: usize,
    _height_map_inner_height: usize,
    _height_map_nr_tiles_x: usize,
    _height_map_nr_tiles_y: usize,
}

impl HeightMapDrawer {
    pub fn new(
        renderer: &mut dyn WgpuRendererInterface,
        texture_bind_group_layout: &lod_heightmap_shader::TextureBindGroupLayout,
        texture_bytes: &[u8],
        height_map_inner_width: usize,
        height_map_inner_height: usize,
        _height_map_nr_tiles_x: usize,
        _height_map_nr_tiles_y: usize,
        // heightmap_bind_group_layout: &deferred_heightmap_shader::HeightmapBindGroupLayout,
    ) -> Self {
        assert!(height_map_inner_width == height_map_inner_height);

        // mesh
        let grid = shape::Grid::new(1.0, height_map_inner_width+1, 0);
        let gird_triangles = grid.triangles();
        let mesh = lod_heightmap_shader::Mesh::from_shape(renderer.device(), gird_triangles);

        // texture
        // let texture_bytes = include_bytes!("../res/tile.png");
        let texture_image = image::load_from_memory(texture_bytes).unwrap();
        let texture_rgba = texture_image.to_rgba8();
        let texture = lod_heightmap_shader::Texture::new_with_mipmaps(
            renderer,
            texture_bind_group_layout,
            &texture_rgba,
            Some("tile.png"),
            9,
        )
        .unwrap();

        // heightmap_textures
        let height_map_textures = Vec::new();
        let instances = Vec::new();

        // view position
        let view_position = Vec3::zero();

        Self {
            mesh,
            texture,
            height_map_textures,
            instances,
            view_position,
            _height_map_inner_width: height_map_inner_width,
            _height_map_inner_height: height_map_inner_height,
            _height_map_nr_tiles_x,
            _height_map_nr_tiles_y,
        }
    }

    pub fn update<const WIDTH: usize, const HEIGHT: usize, const TILE_WIDTH: usize, const TILE_HEIGHT: usize>(
        &mut self,
        wgpu_renderer: &mut dyn WgpuRendererInterface,
        heightmap_bind_group_layout: &lod_heightmap_shader::HeightmapBindGroupLayout,
        height_map: &HeightMap<WIDTH, HEIGHT, TILE_WIDTH, TILE_HEIGHT>,
    )
    {
        let inner_width: usize = TILE_WIDTH - 2;
        let inner_height: usize = TILE_HEIGHT - 2;
        let inner_size: usize = inner_width * inner_height;

        let tiles_x: usize = WIDTH / inner_width;
        let tiles_y: usize = HEIGHT / inner_height;
        let _tile_count: usize = tiles_x * tiles_y;

        let data = &height_map.data;
        
        let mut height_map_textures = Vec::new();
        let mut instances = Vec::new();

        for tile_y in 0..tiles_y {
            for tile_x in 0..tiles_x {
                let tile = &data[tile_y * tiles_x + tile_x];

                // create host data
                let mut host_data: Vec<lod_heightmap_shader::Heightmap> = Vec::with_capacity(inner_size);
                for y in 1..TILE_HEIGHT {
                    for x in 1..TILE_WIDTH {
                        host_data.push(vertex_heightmap_shader::Heightmap { height: tile[y][x] });
                    }
                }

                // create device data
                let height_texture = lod_heightmap_shader::HeightmapTexture::new(
                    wgpu_renderer,
                    heightmap_bind_group_layout,
                    &host_data,
                    inner_width as u32 +1,
                    inner_height as u32 +1,
                    Some(&format!("terrain y={} x={}", tile_y, tile_x)),
                );  

                // create instance
                // let data_index = self.height_map_textures.len();
                let instance = lod_heightmap_shader::Instance {
                    position: [
                        (tile_x * inner_width) as f32,
                        (tile_y * inner_height) as f32,
                        0.0,
                    ],
                    color: [0.2, 0.2, 0.8],
                    distance: 1.0,
                };
                let instance_buffer =
                    lod_heightmap_shader::InstanceBuffer::new(wgpu_renderer.device(), &[instance]);


                height_map_textures.push(height_texture);
                instances.push(instance_buffer);
            }
        }

        self.height_map_textures = height_map_textures;
        self.instances = instances;
    }
}

impl lod_heightmap_shader::LodHeightMapShaderDraw for HeightMapDrawer {
    fn draw<'a>(&'a mut self, render_pass: &mut wgpu::RenderPass<'a>) {
        // mesh data
        let mesh = &self.mesh;
        let texture = &self.texture;
        let heightmap_textures = &self.height_map_textures;
        let instances = &self.instances;

        let _view_position = &self.view_position;

        // draw
        mesh.bind(render_pass);
        texture.bind(render_pass);

        assert!(self.height_map_textures.len() == self.instances.len());
        let size = self.instances.len();
        for i in 0..size {
            heightmap_textures[i].bind(render_pass);
            instances[i].bind(render_pass);
            mesh.draw_indexed(render_pass);
        }
    }
}

