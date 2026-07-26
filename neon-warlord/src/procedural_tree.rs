//! Generates a tree with nodes an leaves

pub mod node;
pub mod tree;

use crate::{
    procedural_tree::tree::{Tree, TreeInterface},
    verlet_physics::{self, VerletObject, fixed_link::FixedLink},
};
use cgmath::{Rotation3, Zero};
use forward_renderer::{geometry, to_rgb};
use wgpu_renderer::{
    vertex_color_shader::{
        self, VertexColorShaderDraw, vertex_color_shader_draw::VertexColorShaderDrawLines,
    },
    wgpu_renderer::WgpuRendererInterface,
};

type Vec3 = cgmath::Vector3<f32>;

pub struct ProceduralTree {
    _tree: Tree,

    nodes_instances: Vec<vertex_color_shader::Instance>,
    nodes_mesh: vertex_color_shader::Mesh,

    leaves_instances: Vec<vertex_color_shader::Instance>,
    leaves_mesh: vertex_color_shader::Mesh,

    links_instances: Vec<vertex_color_shader::Instance>,
    links_lines: geometry::Lines,
    links_mesh: vertex_color_shader::Mesh,

    _index: usize,
    _size: usize,

    nodes_indices: Vec<usize>,
    leaves_indices: Vec<usize>,
    links_indices: Vec<(usize, usize)>,
}

impl ProceduralTree {
    pub fn new(
        wgpu_renderer: &mut dyn WgpuRendererInterface,
        verlet_objects: &mut Vec<VerletObject>,
        fixed_links: &mut Vec<verlet_physics::fixed_link::FixedLink>,
    ) -> Self {
        let leaves_color_0 = to_rgb("#4b964f");
        let leaves_color_1 = to_rgb("#0a340c");
        let nodes_color_0 = to_rgb("#836e37");
        let nodes_color_1 = to_rgb("#2b1f03");
        let links_color_0 = to_rgb("#98ae99");
        let links_color_1 = to_rgb("#2d2e27");

        let _tree = Tree::new(7, fastrand::u64(..));
        let nr_nodes = _tree.nr_nodes();
        let nr_leaves = _tree.nr_leaves();
        let nr_links = _tree.nr_links();

        let radius = 0.1;
        let nodes_circle =
            geometry::Circle::new_color_fade(radius, 32, nodes_color_0, nodes_color_1);
        let leaves_circle =
            geometry::Circle::new_color_fade(radius, 32, leaves_color_0, leaves_color_1);
        let links_lines = geometry::Lines::new_color_fade(nr_links, links_color_0, links_color_1);

        let instance = vertex_color_shader::Instance {
            position: cgmath::Vector3::new(0.0, 0.0, 0.0),
            rotation: cgmath::Quaternion::from_angle_x(cgmath::Deg(90.0)),
        };

        let mut leaves_instances = Vec::with_capacity(nr_leaves);
        for _i in 0..nr_leaves {
            leaves_instances.push(instance);
        }

        let mut nodes_instances = Vec::with_capacity(nr_nodes);
        for _i in 0..nr_nodes {
            nodes_instances.push(instance);
        }

        let links_instances = vec![vertex_color_shader::Instance::zero()];

        let nodes_mesh = vertex_color_shader::Mesh::new(
            wgpu_renderer.device(),
            &nodes_circle.vertices,
            &nodes_circle.colors,
            &nodes_circle.indices,
            &nodes_instances,
        );

        let leaves_mesh = vertex_color_shader::Mesh::new(
            wgpu_renderer.device(),
            &leaves_circle.vertices,
            &leaves_circle.colors,
            &leaves_circle.indices,
            &leaves_instances,
        );

        let links_mesh = vertex_color_shader::Mesh::new(
            wgpu_renderer.device(),
            &links_lines.vertices,
            &links_lines.colors,
            &links_lines.indices,
            &links_instances,
        );

        // create verlet objects
        let _size = _tree.size();
        let _index = verlet_objects.len();
        for _i in 0.._size {
            verlet_objects.push(VerletObject::new(Vec3::zero(), radius));
        }

        let mut nodes_indices = Vec::new();
        let mut leaves_indices = Vec::new();
        let mut links_indices = Vec::new();
        let mut create_links = CreateLinks {
            verlet_objects,
            fixed_links,
            index: _index,
            nodes_indices: &mut nodes_indices,
            leaves_indices: &mut leaves_indices,
            links_indices: &mut links_indices,
        };
        _tree.traverse_tree(&mut create_links);

        Self {
            _tree,
            nodes_instances,
            nodes_mesh,
            leaves_instances,
            leaves_mesh,
            _index,
            _size,

            nodes_indices,
            leaves_indices,
            links_lines,
            links_mesh,
            links_indices,
            links_instances,
        }
    }

    pub fn update(
        &mut self,
        wgpu_renderer: &mut dyn WgpuRendererInterface,
        verlet_objects: &[VerletObject],
    ) {
        let _dt = 1.0 / 60.0;

        // update nodes
        for (i, index) in self.nodes_indices.iter().enumerate() {
            let instance = &mut self.nodes_instances[i];
            let verlet_object = &verlet_objects[*index];

            instance.position = verlet_object.position();
        }

        // update leaves
        for (i, index) in self.leaves_indices.iter().enumerate() {
            let instance = &mut self.leaves_instances[i];
            let verlet_object = &verlet_objects[*index];

            instance.position = verlet_object.position();
        }

        // update links
        for (i, (index_0, index_1)) in self.links_indices.iter().enumerate() {
            let verlet_object_0 = &verlet_objects[*index_0];
            let verlet_object_1 = &verlet_objects[*index_1];

            self.links_lines.set_line_position(
                i,
                verlet_object_0.position(),
                verlet_object_1.position(),
            );
        }

        // copy data to device
        self.nodes_mesh
            .update_instance_buffer(wgpu_renderer.queue(), &self.nodes_instances);

        self.leaves_mesh
            .update_instance_buffer(wgpu_renderer.queue(), &self.leaves_instances);

        self.links_mesh
            .update_vertex_buffer(wgpu_renderer.queue(), &self.links_lines.vertices);
        self.links_mesh
            .update_instance_buffer(wgpu_renderer.queue(), &self.links_instances);
    }
}

impl VertexColorShaderDraw for ProceduralTree {
    fn draw<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        self.nodes_mesh.draw(render_pass);
        self.leaves_mesh.draw(render_pass);
    }
}

impl VertexColorShaderDrawLines for ProceduralTree {
    fn draw_lines<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        self.links_mesh.draw(render_pass);
    }
}

struct CreateLinks<'a> {
    verlet_objects: &'a mut Vec<VerletObject>,
    fixed_links: &'a mut Vec<verlet_physics::fixed_link::FixedLink>,
    index: usize,

    pub nodes_indices: &'a mut Vec<usize>,
    pub leaves_indices: &'a mut Vec<usize>,
    pub links_indices: &'a mut Vec<(usize, usize)>,
}

impl<'a> TreeInterface for CreateLinks<'a> {
    fn create_node(
        &mut self,
        node_index: usize,
        parent_index: Option<usize>,
        absolute_position: &Vec3,
        relative_position: &Vec3,
        depth: usize,
        is_leave: bool,
    ) {
        let node_index = self.index + node_index;
        // println!("node_index {node_index}, is_leave {is_leave}");

        self.verlet_objects[node_index].reset_position(*absolute_position);

        if let Some(parent_index) = parent_index {
            let parent_index = self.index + parent_index;
            self.links_indices.push((parent_index, node_index));

            let mut stiffness = 1.0 - 0.5 / depth as f32;
            let mut damping = 0.88 + 0.1 / depth as f32;

            if is_leave {
                stiffness = 0.2;
                damping = 0.90;
            }

            self.fixed_links.push(
                FixedLink::new(parent_index, node_index, *relative_position)
                    .stiffness(stiffness)
                    .damping(damping),
            );
        }

        if is_leave {
            self.leaves_indices.push(node_index);
        } else {
            self.nodes_indices.push(node_index);
        }
    }
}
