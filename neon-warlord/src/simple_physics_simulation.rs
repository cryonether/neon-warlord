//! Draws some objects using verlet physics

use cgmath::Rotation3;
use forward_renderer::geometry;
use wgpu_renderer::{
    vertex_color_shader::{
        self, VertexColorShaderDraw, vertex_color_shader_draw::VertexColorShaderDrawLines,
    },
    wgpu_renderer::WgpuRendererInterface,
};

use crate::{
    procedural_tree::ProceduralTree,
    physics::verlet_physics::{self, VerletObject},
};
type Vec3 = cgmath::Vector3<f32>;

pub struct SimplePhysicsSimulation {
    verlet_objects: Vec<VerletObject>,
    links: Vec<verlet_physics::link::Link>,
    fixed_links: Vec<verlet_physics::fixed_link::FixedLink>,
    fixed: Vec<verlet_physics::fixed::Fixed>,
    solver: verlet_physics::solver::Solver,

    mesh: vertex_color_shader::Mesh,
    instances: Vec<vertex_color_shader::Instance>,

    radius: f32,

    ticks: u64,

    procedural_tree: ProceduralTree,
}

impl SimplePhysicsSimulation {
    pub fn new(wgpu_renderer: &mut dyn WgpuRendererInterface) -> Self {
        let _nr_objects = 100;
        let radius = 0.1;

        let mut verlet_objects = Vec::with_capacity(1);

        // chain
        let nr_links = 40;
        let mut links = Vec::with_capacity(nr_links);
        for i in 0..nr_links {
            verlet_objects.push(VerletObject::new(
                Vec3::new(i as f32 * 0.2 - 5.0, 0.0, 15.0 - i as f32 * 0.2),
                radius,
            ));

            if i < nr_links - 1 {
                links.push(verlet_physics::link::Link::new(i, i + 1, 0.2));
            }
        }

        // fixed chain
        let nr_fixed_links = 8;
        let mut fixed_links = Vec::with_capacity(nr_fixed_links);
        for i in 0..nr_fixed_links {
            verlet_objects.push(VerletObject::new(
                Vec3::new(i as f32 * 0.2 - 6.0, 0.0, 15.0 - i as f32 * 0.2),
                radius,
            ));

            if i < nr_fixed_links - 1 {
                fixed_links.push(verlet_physics::fixed_link::FixedLink::new(
                    nr_links + i,
                    nr_links + i + 1,
                    Vec3::new(0.3, 0.0, 0.3 - i as f32 * 0.1),
                ));
            }
        }

        let nr_objects = verlet_objects.len();

        // tree
        let tree_root_index = verlet_objects.len();
        let procedural_tree =
            ProceduralTree::new(wgpu_renderer, &mut verlet_objects, &mut fixed_links);

        let fixed = vec![
            verlet_physics::fixed::Fixed::new(0, Vec3::new(-5.0, 0.0, 15.0)),
            verlet_physics::fixed::Fixed::new(nr_links, Vec3::new(-6.0, 0.0, 15.0)),
            verlet_physics::fixed::Fixed::new(tree_root_index, Vec3::new(-2.0, 15.0, 0.1)),
        ];

        let circle = geometry::Circle::new_color_fade(radius, 32, [0.0, 0.4, 0.4], [0.4, 0.0, 0.4]);

        let instance = vertex_color_shader::Instance {
            position: cgmath::Vector3::new(0.0, 0.0, 0.0),
            rotation: cgmath::Quaternion::from_angle_x(cgmath::Deg(90.0)),
        };

        let mut instances = Vec::with_capacity(nr_objects);
        for _i in 0..nr_objects {
            instances.push(instance);
        }

        let mesh = vertex_color_shader::Mesh::new(
            wgpu_renderer.device(),
            &circle.vertices,
            &circle.colors,
            &circle.indices,
            &instances,
        );

        let solver = verlet_physics::solver::Solver::new();

        Self {
            verlet_objects,
            links,
            fixed_links,
            fixed,
            solver,
            mesh,
            instances,
            radius,
            ticks: 0,

            procedural_tree,
        }
    }

    pub fn update(&mut self, wgpu_renderer: &mut dyn WgpuRendererInterface) {
        let dt = 1.0 / 60.0;
        self.ticks += 1;

        if self.verlet_objects.len() < self.instances.len() && self.ticks.is_multiple_of(8) {
            self.verlet_objects
                .push(VerletObject::new(Vec3::new(0.0, 0.0001, 15.0), self.radius));
        }

        self.solver.update(
            &mut self.verlet_objects,
            &self.links,
            &self.fixed_links,
            &self.fixed,
            dt,
        );

        for i in 0..self.instances.len() {
            let instance = &mut self.instances[i];
            let verlet_object = &mut self.verlet_objects[i];

            instance.position.x = verlet_object.position().x;
            instance.position.y = verlet_object.position().y;
            instance.position.z = verlet_object.position().z;
        }

        self.mesh
            .update_instance_buffer(wgpu_renderer.queue(), &self.instances);

        self.procedural_tree
            .update(wgpu_renderer, &self.verlet_objects);
    }
}

impl VertexColorShaderDraw for SimplePhysicsSimulation {
    fn draw<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        // self.mesh.draw(render_pass);
        self.procedural_tree.draw(render_pass);
    }
}

impl VertexColorShaderDrawLines for SimplePhysicsSimulation {
    fn draw_lines<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        self.procedural_tree.draw_lines(render_pass);
    }
}
