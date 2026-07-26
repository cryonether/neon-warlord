//! Manages the data on the gpu of the terrain
//!

pub mod lod_quad_tree;
pub mod quad_tree_draw;
pub mod terrain_texture_details;

use cgmath::Zero;
use lod_quad_tree::LodQuadTree;
use quad_tree_draw::QuadTreeDraw;
use terrain_texture_details::TerrainTextureDetails;
use wgpu_renderer::{
    shape::{self, MeshDataInterface},
    vertex_heightmap_shader,
    wgpu_renderer::WgpuRendererInterface,
};

use crate::lod_heightmap_shader;

#[derive(Debug)]
pub struct HeightMap {
    pub heights: Vec<f32>,
    pub details: TerrainTextureDetails,
}

pub struct TerrainSettings {
    pub nr_tiles: usize,
    pub max_depth: usize,
}

pub struct TerrainStorage {
    _settings: TerrainSettings,

    mesh: lod_heightmap_shader::Mesh,
    texture: lod_heightmap_shader::Texture,
    heightmap_textures: Vec<lod_heightmap_shader::HeightmapTexture>,
    instances: Vec<lod_heightmap_shader::InstanceBuffer<lod_heightmap_shader::Instance>>,
    pub height_map_details: Vec<TerrainTextureDetails>,
    pub height_maps: Vec<Vec<f32>>,

    lod_quad_tree: lod_quad_tree::LodQuadTree,

    view_position: cgmath::Vector3<isize>,

    max_depth: usize,
    nr_tiles: usize,

    requests: Vec<TerrainTextureDetails>,
}

impl TerrainStorage {
    pub fn new(
        settings: TerrainSettings,
        renderer: &mut dyn WgpuRendererInterface,
        texture_bind_group_layout: &lod_heightmap_shader::TextureBindGroupLayout,
        texture_bytes: &[u8],
        // heightmap_bind_group_layout: &deferred_heightmap_shader::HeightmapBindGroupLayout,
    ) -> Self {
        let max_depth = settings.max_depth;
        let nr_tiles = settings.nr_tiles;
        let _size_0 = nr_tiles + 3;
        let size_1 = nr_tiles + 1;

        // mesh
        let grid = shape::Grid::new(1.0, size_1, 1);
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
        let heightmap_textures = Vec::new();
        let instances = Vec::new();
        let heightmap_details = Vec::new();
        let height_maps = Vec::new();

        // lod_quad_tree
        let lod_quad_tree = LodQuadTree::new(max_depth, nr_tiles);

        // view_position
        let view_position = cgmath::Vector3::zero();

        // requests
        let requests = Vec::new();

        Self {
            _settings: settings,
            mesh,
            texture,
            heightmap_textures,
            instances,
            height_map_details: heightmap_details,
            height_maps,
            lod_quad_tree,
            view_position,
            max_depth,
            nr_tiles,
            requests,
        }
    }

    pub fn update_height_map(
        &mut self,
        renderer: &mut dyn WgpuRendererInterface,
        heightmap_bind_group_layout: &lod_heightmap_shader::HeightmapBindGroupLayout,
        height_map: HeightMap,
    ) {
        let pos_0 = height_map.details.pos_0;
        let pos_1 = height_map.details.pos_1;
        let point_distance = height_map.details.point_distance;

        let size_0 = height_map.details.size_0;
        let size_1 = height_map.details.size_1;

        let nr_tiles = height_map.details.nr_tiles;

        let depth = height_map.details.depth;
        let node_index = height_map.details.node_index;

        assert_eq!(nr_tiles, self.nr_tiles);

        // create host data
        let mut heightmap: Vec<lod_heightmap_shader::Heightmap> =
            Vec::with_capacity(size_0 * size_0);
        assert_eq!(height_map.heights.len(), size_0 * size_0);
        for elem in &height_map.heights {
            heightmap.push(vertex_heightmap_shader::Heightmap { height: *elem });
            // heightmap.push(vertex_heightmap_shader::Heightmap { height: 0.0 });
        }

        let heightmap_details = TerrainTextureDetails {
            pos_0,
            pos_1,
            point_distance,
            size_0,
            size_1,
            nr_tiles,
            depth,
            node_index,
        };

        // create device data
        let height_texture = lod_heightmap_shader::HeightmapTexture::new(
            renderer,
            heightmap_bind_group_layout,
            &heightmap,
            size_0 as u32,
            size_0 as u32,
            Some("terrain"),
        );

        let data_index = self.heightmap_textures.len();
        let instance = lod_heightmap_shader::Instance {
            position: [
                (pos_1.x - point_distance as isize) as f32,
                (pos_1.y - point_distance as isize) as f32 + 0.0,
                0.0,
            ],
            color: [0.2, 0.2, 0.8],
            // entity: data_index as u32 | selector::ENTITY_TERRAIN_BIT,
            distance: point_distance as f32,
        };
        let instance_buffer =
            lod_heightmap_shader::InstanceBuffer::new(renderer.device(), &[instance]);

        // save device data
        assert_eq!(self.instances.len(), data_index);
        assert_eq!(self.height_map_details.len(), data_index);

        self.heightmap_textures.push(height_texture);
        self.instances.push(instance_buffer);
        self.height_map_details.push(heightmap_details);
        self.height_maps.push(height_map.heights);

        // make node data available
        self.lod_quad_tree.set_data_index(node_index, data_index);
    }

    pub fn set_view_position(&mut self, view_position: &cgmath::Vector3<f32>) {
        self.view_position = cgmath::Vector3::new(
            view_position.x as isize,
            view_position.y as isize,
            view_position.z as isize,
        );
    }

    // pub fn submit_requests(&mut self, sender: &mpsc::Sender<GameLogicMessageRequest>) {
    //     for elem in &self.requests {
    //         let _res = sender.send(GameLogicMessageRequest::GetTerrain(
    //             HeightMapDetails {
    //                 pos_0: elem.pos_0,
    //                 pos_1: elem.pos_1,
    //                 point_distance: elem.point_distance,
    //                 size_0: elem.size_0,
    //                 size_1: elem.size_1,
    //                 nr_tiles: elem.nr_tiles,
    //                 depth: elem.depth,
    //                 node_index: elem.node_index,
    //             },
    //         ));
    //     }

    //     self.requests.clear();
    // }

    pub fn get_requests(&self) -> &Vec<TerrainTextureDetails> {
        &self.requests
    }

    pub fn clear_requests(&mut self) {
        self.requests.clear();
    }
}

impl lod_heightmap_shader::LodHeightMapShaderDraw for TerrainStorage {
    fn draw<'a>(&'a mut self, render_pass: &mut wgpu::RenderPass<'a>) {
        // mesh data
        let mesh = &self.mesh;
        let texture = &self.texture;
        let heightmap_textures = &self.heightmap_textures;
        let instances = &self.instances;

        // quad tree data
        let lod_quad_tree = &mut self.lod_quad_tree;
        let view_position = &self.view_position;
        let max_depth = self.max_depth;
        let nr_tiles = self.nr_tiles;

        // draw
        mesh.bind(render_pass);
        texture.bind(render_pass);

        let quad_tree_draw = &mut QuadTreeDraw::new(max_depth, nr_tiles, |data_index| {
            heightmap_textures[data_index].bind(render_pass);
            instances[data_index].bind(render_pass);
            mesh.draw_indexed(render_pass);
        });

        lod_quad_tree.traverse_leaves(view_position, quad_tree_draw);

        self.requests = quad_tree_draw.requests.clone();
    }
}
