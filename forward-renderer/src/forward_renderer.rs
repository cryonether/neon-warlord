//! Renders everything
//!

use crate::animation_shader::AnimationShaderDraw;
use crate::particle_shader::{ParticleKind, ParticleShaderDraw};
use crate::particle_shader_two_point::ParticleShaderTwoPointDraw;
use crate::{animation_shader, particle_shader, particle_shader_two_point};
// use crate::animated_object_storage::AnimatedObjectStorage;
// use crate::deferred_color_shader::entity_buffer::MousePosition;
// use crate::deferred_color_shader::{self, DeferredShaderDraw, EntityBuffer, GBuffer};
// use crate::deferred_light_shader::DeferredLightShaderDraw;
// use crate::fxaa_shader::FxaaShaderDraw;
// use crate::performance_monitor::PerformanceMonitor;
// use crate::point_light_storage::PointLightStorage;
// use crate::terrain_storage::TerrainStorage;
use crate::lod_heightmap_shader::LodHeightMapShaderDraw;
use crate::{DrawGui, lod_heightmap_shader};
use cgmath::prelude::*;
use wgpu_renderer::default_application::default_application_interface::RenderError;
use wgpu_renderer::performance_monitor::watch;
use wgpu_renderer::vertex_color_shader::vertex_color_shader_draw::VertexColorShaderDrawLines;
use wgpu_renderer::vertex_color_shader::{self, VertexColorShaderDraw};
use wgpu_renderer::vertex_texture_shader;
use wgpu_renderer::wgpu_renderer::WgpuRendererInterface;
use wgpu_renderer::wgpu_renderer::camera::{Camera, Projection};
use wgpu_renderer::wgpu_renderer::depth_texture::DepthTexture;
use wgpu_renderer::wgpu_renderer::depth_texture_bind_group_layout::DepthTextureBindGroupLayout; // for Point3::from_vec
// use crate::{
//     deferred_animation_shader, deferred_heightmap_shader, deferred_light_shader,
//     deferred_light_sphere_shader, fxaa_shader,
// };

pub struct RendererSettings {
    pub wait_for_render_loop_to_finish: bool,
    pub enable_vertical_sync: bool,
    pub enable_fxaa: bool,
    pub window_resolution: (u32, u32),

    pub heightmap_lighting: lod_heightmap_shader::LightingModel,
    pub animation_lighting: animation_shader::LightingModel,
}

pub struct ForwardRenderer {
    settings: RendererSettings,

    depth_texture_bind_group_layout: DepthTextureBindGroupLayout,
    depth_texture: DepthTexture,
    shadow_map: DepthTexture,

    pipeline_color: vertex_color_shader::Pipeline,
    pipeline_lines: vertex_color_shader::Pipeline,

    pub texture_bind_group_layout: vertex_texture_shader::TextureBindGroupLayout,
    pipeline_texture_gui: vertex_texture_shader::Pipeline,

    // g_buffer_bind_group_layout: deferred_light_shader::GBufferBindGroupLayout,
    // g_buffer: deferred_color_shader::GBuffer,
    // entity_buffer: deferred_color_shader::EntityBuffer,
    // pipeline_deferred_color: deferred_color_shader::Pipeline,

    // pipeline_deferred_light: deferred_light_shader::Pipeline,
    // pipeline_deferred_light_ambient: deferred_light_shader::Pipeline,
    // pipeline_deferred_light_sphere: deferred_light_sphere_shader::Pipeline,
    pub animation_bind_group_layout: animation_shader::AnimationBindGroupLayout,
    pipeline_animated: animation_shader::Pipeline,

    pub heightmap_bind_group_layout: lod_heightmap_shader::HeightmapBindGroupLayout,
    pipeline_lod_heightmap: lod_heightmap_shader::Pipeline,

    pipeline_particle: particle_shader::PipelineParticle,
    pipeline_plasma: particle_shader::PipelineParticle,
    pipeline_glow: particle_shader::PipelineParticle,
    pipeline_billboard_sphere: particle_shader::PipelineParticle,
    pipeline_rectangle: particle_shader_two_point::PipelineParticleTwoPoint,

    // post_processing_bind_group_layout: fxaa_shader::PostProcessingTextureBindGroupLayout,
    // post_processing_texture: fxaa_shader::PostProcessingTexture,
    // pipeline_fxaa: fxaa_shader::Pipeline,

    // camera
    pub camera: Camera,
    pub projection: Projection,

    // global light projection
    camera_light: Camera,
    projection_light: Projection,

    camera_uniform: vertex_color_shader::CameraUniform,
    camera_light_uniform: vertex_color_shader::CameraUniform,
    camera_uniform_buffer: vertex_color_shader::CameraUniformBuffer,
    camera_uniform_shadow_map_buffer: vertex_color_shader::CameraUniformBuffer,

    camera_uniform_orthographic: vertex_color_shader::CameraUniform,
    camera_uniform_orthographic_buffer: vertex_color_shader::CameraUniformBuffer,
}

impl ForwardRenderer {
    pub fn new(wgpu_renderer: &mut dyn WgpuRendererInterface, settings: RendererSettings) -> Self {
        // enable vsync
        wgpu_renderer.enable_vsync(settings.enable_vertical_sync);
        wgpu_renderer
            .request_window_size(settings.window_resolution.0, settings.window_resolution.1);

        // wgpu renderer
        let _surface_width = wgpu_renderer.surface_width();
        let _surface_height = wgpu_renderer.surface_height();
        let surface_format: wgpu::TextureFormat = wgpu_renderer.surface_format();

        // dpeth texture and shadow map
        let depth_texture_bind_group_layout =
            DepthTextureBindGroupLayout::new(wgpu_renderer.device());
        let depth_texture = DepthTexture::create_depth_texture(
            wgpu_renderer,
            &depth_texture_bind_group_layout,
            "depth_texture",
        );

        let shadow_map = DepthTexture::create_depth_texture(
            wgpu_renderer,
            &depth_texture_bind_group_layout,
            "shadow_map_texture",
        );

        // pipeline color
        let camera_bind_group_layout =
            vertex_color_shader::CameraBindGroupLayout::new(wgpu_renderer.device());
        let pipeline_color = vertex_color_shader::Pipeline::new(
            wgpu_renderer.device(),
            &camera_bind_group_layout,
            surface_format,
        );

        // pipeline lines
        let pipeline_lines = vertex_color_shader::Pipeline::new_lines(
            wgpu_renderer.device(),
            &camera_bind_group_layout,
            surface_format,
        );

        // pipeline texture gui
        let texture_bind_group_layout =
            vertex_texture_shader::TextureBindGroupLayout::new(wgpu_renderer.device());
        let pipeline_texture_gui = vertex_texture_shader::Pipeline::new_gui(
            wgpu_renderer.device(),
            &camera_bind_group_layout,
            &texture_bind_group_layout,
            surface_format,
        );

        // g_buffer
        // let g_buffer_bind_group_layout =
        //     deferred_light_shader::GBufferBindGroupLayout::new(wgpu_renderer.device());
        // let g_buffer = deferred_color_shader::GBuffer::new(
        //     wgpu_renderer,
        //     &g_buffer_bind_group_layout,
        //     surface_width,
        //     surface_height,
        // );

        // entity_buffer
        // let entity_buffer = deferred_color_shader::EntityBuffer::new(
        //     wgpu_renderer,
        //     surface_width,
        //     surface_height,
        //     settings.enable_memory_mapped_read,
        // );

        // // pipeline deferred color
        // let pipeline_deferred_color = deferred_color_shader::Pipeline::new(
        //     wgpu_renderer.device(),
        //     &camera_bind_group_layout,
        //     surface_format,
        // );

        // // pipeline deferred light
        // let pipeline_deferred_light = deferred_light_shader::Pipeline::new(
        //     wgpu_renderer.device(),
        //     &camera_bind_group_layout,
        //     &g_buffer_bind_group_layout,
        //     surface_format,
        //     false,
        // );

        // let pipeline_deferred_light_ambient = deferred_light_shader::Pipeline::new(
        //     wgpu_renderer.device(),
        //     &camera_bind_group_layout,
        //     &g_buffer_bind_group_layout,
        //     surface_format,
        //     true,
        // );

        // let pipeline_deferred_light_sphere = deferred_light_sphere_shader::Pipeline::new(
        //     wgpu_renderer.device(),
        //     &camera_bind_group_layout,
        //     &g_buffer_bind_group_layout,
        //     surface_format,
        // );

        let animation_bind_group_layout =
            animation_shader::AnimationBindGroupLayout::new(wgpu_renderer.device());

        // pipeline animated
        let pipeline_animated = animation_shader::Pipeline::new(
            wgpu_renderer.device(),
            &camera_bind_group_layout,
            &animation_bind_group_layout,
            surface_format,
            &settings.animation_lighting,
        );

        // pipeline deferred heightmap
        let heightmap_bind_group_layout =
            lod_heightmap_shader::HeightmapBindGroupLayout::new(wgpu_renderer.device());
        let pipeline_lod_heightmap = lod_heightmap_shader::Pipeline::new(
            wgpu_renderer,
            &camera_bind_group_layout,
            &texture_bind_group_layout,
            &heightmap_bind_group_layout,
            &depth_texture_bind_group_layout,
            surface_format,
            &settings.heightmap_lighting,
        );

        // Particles
        let pipeline_particle = particle_shader::PipelineParticle::new(
            wgpu_renderer.device(),
            &camera_bind_group_layout,
            surface_format,
            ParticleKind::FloatToTheMiddle,
        );

        let pipeline_plasma = particle_shader::PipelineParticle::new(
            wgpu_renderer.device(),
            &camera_bind_group_layout,
            surface_format,
            ParticleKind::Plasma,
        );

        let pipeline_glow = particle_shader::PipelineParticle::new(
            wgpu_renderer.device(),
            &camera_bind_group_layout,
            surface_format,
            ParticleKind::Glow,
        );

        let pipeline_billboard_sphere = particle_shader::PipelineParticle::new(
            wgpu_renderer.device(),
            &camera_bind_group_layout,
            surface_format,
            ParticleKind::BillboardSphere,
        );

        let pipeline_rectangle = particle_shader_two_point::PipelineParticleTwoPoint::new(
            wgpu_renderer.device(),
            &camera_bind_group_layout,
            surface_format,
            particle_shader_two_point::ParticleKind::BillboardRectangle,
        );

        // // pipeline fxaa
        // let post_processing_bind_group_layout =
        //     fxaa_shader::PostProcessingTextureBindGroupLayout::new(wgpu_renderer.device());
        // let post_processing_texture = fxaa_shader::PostProcessingTexture::new(
        //     wgpu_renderer,
        //     &post_processing_bind_group_layout,
        //     surface_width,
        //     surface_height,
        //     surface_format,
        // );
        // let pipeline_fxaa = fxaa_shader::Pipeline::new(
        //     wgpu_renderer.device(),
        //     &camera_bind_group_layout,
        //     &post_processing_bind_group_layout,
        //     surface_format,
        // );

        // camera
        let position = cgmath::Point3::new(0.0, 0.0, 0.0);
        let yaw = cgmath::Deg(0.0);
        let pitch = cgmath::Deg(0.0);
        let mut camera = Camera::new(position, yaw, pitch);
        // Self::top_view_point(&mut camera);
        Self::side_view_point(&mut camera);

        // light camera
        let position = cgmath::Vector3::new(0.0, 80.0, 20.0);
        let look_at = cgmath::Vector3::new(0.0, 0.0, 0.0);
        let mut camera_light = Camera::new(cgmath::Point3::from_vec(position), yaw, pitch);
        camera_light.set_view_direction(look_at - position);
        // Self::side_view_point(&mut camera_light);

        let width = wgpu_renderer.surface_width();
        let height = wgpu_renderer.surface_height();
        let fovy = cgmath::Deg(45.0);
        let znear = 0.1;
        let zfar = 100.0;
        let projection = Projection::new(width, height, fovy, znear, zfar);

        // let fovy = cgmath::Deg(175.0);
        // let znear = 1.0;
        // let zfar = 10.0;
        let projection_light = Projection::new(width, height, fovy, znear, zfar);

        let camera_uniform = vertex_color_shader::CameraUniform::new();
        let camera_light_uniform = vertex_color_shader::CameraUniform::new();

        let camera_uniform_buffer = vertex_color_shader::CameraUniformBuffer::new(
            wgpu_renderer.device(),
            &camera_bind_group_layout,
        );

        let camera_uniform_orthographic: vertex_color_shader::CameraUniform =
            vertex_color_shader::CameraUniform::new_orthographic(width, height);
        let mut camera_uniform_orthographic_buffer = vertex_color_shader::CameraUniformBuffer::new(
            wgpu_renderer.device(),
            &camera_bind_group_layout,
        );

        camera_uniform_orthographic_buffer
            .update_camera(wgpu_renderer.queue(), camera_uniform_orthographic); // add uniform identity matrix

        let camera_uniform_shadow_map_buffer = vertex_color_shader::CameraUniformBuffer::new(
            wgpu_renderer.device(),
            &camera_bind_group_layout,
        );

        Self {
            settings,

            depth_texture_bind_group_layout,
            depth_texture,
            shadow_map,

            pipeline_color,
            pipeline_lines,

            texture_bind_group_layout,
            pipeline_texture_gui,

            // g_buffer_bind_group_layout,
            // g_buffer,
            // // entity_buffer,

            // pipeline_deferred_color,
            // pipeline_deferred_light,
            // pipeline_deferred_light_ambient,
            // pipeline_deferred_light_sphere,
            animation_bind_group_layout,
            pipeline_animated,

            heightmap_bind_group_layout,
            pipeline_lod_heightmap,

            pipeline_particle,
            pipeline_plasma,
            pipeline_glow,
            pipeline_billboard_sphere,
            pipeline_rectangle,

            // post_processing_bind_group_layout,
            // post_processing_texture,
            // pipeline_fxaa,
            camera,
            projection,

            camera_light,
            projection_light,

            camera_uniform,
            camera_light_uniform,
            camera_uniform_buffer,

            camera_uniform_orthographic,
            camera_uniform_orthographic_buffer,

            camera_uniform_shadow_map_buffer,
        }
    }

    fn _top_view_point(camera: &mut Camera) {
        let position = cgmath::Point3::new(0.0, 0.0, 10.0);
        let yaw = cgmath::Deg(-90.0).into();
        let pitch = cgmath::Deg(0.0).into();

        camera.position = position;
        camera.yaw = yaw;
        camera.pitch = pitch;
    }

    fn side_view_point(camera: &mut Camera) {
        // let position = cgmath::Point3::new(0.0, -5.0, 2.0);
        let position = cgmath::Point3::new(0.0, -8.0, 4.0);
        let yaw = cgmath::Deg(-90.0).into();
        let pitch = cgmath::Deg(80.0).into();

        camera.position = position;
        camera.yaw = yaw;
        camera.pitch = pitch;
    }

    pub fn resize(
        &mut self,
        renderer_interface: &mut dyn WgpuRendererInterface,
        new_size: winit::dpi::PhysicalSize<u32>,
    ) {
        // self.size = new_size;

        self.depth_texture = DepthTexture::create_depth_texture(
            renderer_interface,
            &self.depth_texture_bind_group_layout,
            "depth_texture",
        );

        self.shadow_map = DepthTexture::create_depth_texture(
            renderer_interface,
            &self.depth_texture_bind_group_layout,
            "shadow_map_texture",
        );

        self.projection.resize(new_size.width, new_size.height);
        self.projection_light
            .resize(new_size.width, new_size.height);
        // self.wgpu_renderer.resize(new_size);
        // self.g_buffer = GBuffer::new(
        //     renderer_interface,
        //     &self.g_buffer_bind_group_layout,
        //     new_size.width,
        //     new_size.height,
        // );

        // self.entity_buffer = EntityBuffer::new(
        //     renderer_interface,
        //     new_size.width,
        //     new_size.height,
        //     self.settings.enable_memory_mapped_read,
        // );

        let _surface_format = renderer_interface.surface_format();
        // self.post_processing_texture = fxaa_shader::PostProcessingTexture::new(
        //     renderer_interface,
        //     &self.post_processing_bind_group_layout,
        //     new_size.width,
        //     new_size.height,
        //     surface_format,
        // );

        self.camera_uniform_orthographic
            .resize_orthographic(new_size.width, new_size.height);
        self.camera_uniform_orthographic_buffer
            .update_camera(renderer_interface.queue(), self.camera_uniform_orthographic);
    }

    pub fn update(
        &mut self,
        renderer_interface: &mut dyn WgpuRendererInterface,
        _dt: instant::Duration,
    ) {
        // camera
        self.camera_uniform
            .update_view_proj(&self.camera, &self.projection);

        // camera light
        self.camera_light_uniform
            .update_view_proj(&self.camera_light, &self.projection);

        self.camera_uniform_buffer
            .update_camera(renderer_interface.queue(), self.camera_uniform);
        self.camera_uniform_buffer
            .update_light(renderer_interface.queue(), self.camera_light_uniform);

        self.camera_uniform_shadow_map_buffer
            .update_camera(renderer_interface.queue(), self.camera_light_uniform);
        self.camera_uniform_shadow_map_buffer
            .update_light(renderer_interface.queue(), self.camera_light_uniform);
    }

    pub fn get_view_position(&self) -> cgmath::Vector3<f32> {
        self.camera.get_view_position()
    }

    pub fn _get_view_direction(&self) -> cgmath::Vector3<f32> {
        self.camera.get_view_direction()
    }

    // fn render_fxaa(
    //     &self,
    //     view: &wgpu::TextureView,
    //     encoder: &mut wgpu::CommandEncoder,
    //     mesh: &dyn FxaaShaderDraw,
    // ) {
    //     let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
    //         label: Some("FXAA Render Pass"),
    //         color_attachments: &[Some(wgpu::RenderPassColorAttachment {
    //             view,
    //             resolve_target: None,
    //             ops: wgpu::Operations {
    //                 // load: wgpu::LoadOp::Load,
    //                 load: wgpu::LoadOp::Clear(wgpu::Color {
    //                     r: 0.00,
    //                     g: 0.00,
    //                     b: 0.00,
    //                     a: 1.0,
    //                 }),
    //                 store: wgpu::StoreOp::default(),
    //             },
    //         })],
    //         depth_stencil_attachment: None,
    //         timestamp_writes: None,
    //         occlusion_query_set: None,
    //     });

    //     self.pipeline_fxaa.draw(
    //         &mut render_pass,
    //         &self.camera_uniform_buffer,
    //         &self.post_processing_texture,
    //         mesh,
    //     );
    // }

    fn render_shadow_map(
        &self,
        view: &wgpu::TextureView,
        encoder: &mut wgpu::CommandEncoder,
        _lod_terrains: &mut dyn LodHeightMapShaderDraw,
        animations: &[&dyn AnimationShaderDraw],
        vertex_color_objects: &[&dyn VertexColorShaderDraw],
        vertex_color_objects_lines: &[&dyn VertexColorShaderDrawLines],
    ) {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Shadow Map Render Pass"),
            // color_attachments: &[], // no color target
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    // load: wgpu::LoadOp::Load,
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.01,
                        g: 0.01,
                        b: 0.01,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::default(),
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.shadow_map.view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    // load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::default(),
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        let camera = &self.camera_uniform_shadow_map_buffer;
        // let camera = &self.camera_uniform_buffer;

        // // lod heightmap
        // self.pipeline_lod_heightmap.draw(
        //     &mut render_pass,
        //     camera,
        //     &self.depth_texture,
        //     lod_terrains,
        // );

        // animations
        for elem in animations {
            self.pipeline_animated.draw(&mut render_pass, camera, *elem);
        }

        // vertex color shader
        for elem in vertex_color_objects {
            self.pipeline_color.draw(&mut render_pass, camera, *elem);
        }

        // vertex color shader lines
        for elem in vertex_color_objects_lines {
            self.pipeline_lines
                .draw_lines(&mut render_pass, camera, *elem);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn render_forward(
        &self,
        _renderer_interface: &mut dyn WgpuRendererInterface,
        view: &wgpu::TextureView,
        encoder: &mut wgpu::CommandEncoder,
        lod_terrains: &mut dyn LodHeightMapShaderDraw,
        animations: &[&dyn AnimationShaderDraw],
        // textured_meshes: &impl VertexTextureShaderDraw,
        gui_elements: &[&dyn DrawGui],
        vertex_color_objects: &[&dyn VertexColorShaderDraw],
        vertex_color_objects_lines: &[&dyn VertexColorShaderDrawLines],
        particles: &[&dyn ParticleShaderDraw],
        plasmas: &[&dyn ParticleShaderDraw],
        glow: &[&dyn ParticleShaderDraw],
        particles_bilboard_sphere: &[&dyn ParticleShaderDraw],
        particles_rectangle: &[&dyn ParticleShaderTwoPointDraw],
        // performance_monitors: &[&mut PerformanceMonitor<{ super::WATCH_POINTS_SIZE }>],
    ) {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Forward Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    // load: wgpu::LoadOp::Load,
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.01,
                        g: 0.01,
                        b: 0.01,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::default(),
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth_texture.view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    // load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::default(),
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        // lod heightmap
        self.pipeline_lod_heightmap.draw(
            &mut render_pass,
            &self.camera_uniform_buffer,
            &self.shadow_map,
            lod_terrains,
        );

        // animations
        for elem in animations {
            self.pipeline_animated
                .draw(&mut render_pass, &self.camera_uniform_buffer, *elem);
        }

        // vertex color shader
        for elem in vertex_color_objects {
            self.pipeline_color
                .draw(&mut render_pass, &self.camera_uniform_buffer, *elem);
        }

        // vertex color shader lines
        for elem in vertex_color_objects_lines {
            self.pipeline_lines
                .draw_lines(&mut render_pass, &self.camera_uniform_buffer, *elem);
        }

        // particle shader

        for elem in plasmas {
            self.pipeline_plasma
                .draw(&mut render_pass, &self.camera_uniform_buffer, *elem);
        }

        for elem in particles {
            self.pipeline_particle
                .draw(&mut render_pass, &self.camera_uniform_buffer, *elem);
        }
        for elem in glow {
            self.pipeline_glow
                .draw(&mut render_pass, &self.camera_uniform_buffer, *elem);
        }
        for elem in particles_bilboard_sphere {
            self.pipeline_billboard_sphere.draw(
                &mut render_pass,
                &self.camera_uniform_buffer,
                *elem,
            );
        }

        for elem in particles_rectangle {
            self.pipeline_rectangle
                .draw(&mut render_pass, &self.camera_uniform_buffer, *elem);
        }

        // gui lines
        for elem in gui_elements {
            self.pipeline_lines.draw_lines(
                &mut render_pass,
                &self.camera_uniform_orthographic_buffer,
                *elem,
            );
        }

        // gui color
        for elem in gui_elements {
            self.pipeline_color.draw(
                &mut render_pass,
                &self.camera_uniform_orthographic_buffer,
                *elem,
            );
        }

        // gui texture
        for elem in gui_elements {
            self.pipeline_texture_gui.draw(
                &mut render_pass,
                &self.camera_uniform_orthographic_buffer,
                *elem,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn render(
        &mut self,
        renderer_interface: &mut dyn WgpuRendererInterface,
        lod_terrains: &mut dyn LodHeightMapShaderDraw,
        animations: &[&dyn AnimationShaderDraw],
        gui_elements: &[&dyn DrawGui],
        vertex_color_objects: &[&dyn VertexColorShaderDraw],
        vertex_color_objects_lines: &[&dyn VertexColorShaderDrawLines],
        particles: &[&dyn ParticleShaderDraw],
        plasmas: &[&dyn ParticleShaderDraw],
        glow: &[&dyn ParticleShaderDraw],
        particles_bilboard_sphere: &[&dyn ParticleShaderDraw],
        particles_rectangle: &[&dyn ParticleShaderTwoPointDraw],
        watch_fps: &mut watch::Watch<10>,
    ) -> Result<(), RenderError> {
        let mut watch_index = 5;
        watch_fps.start(watch_index, "Get frame");

        let output = renderer_interface.get_current_texture();
        let surface_texture = match output {
            wgpu::CurrentSurfaceTexture::Success(surface_texture) => {
                // Successfully acquired a surface texture with no issues.
                surface_texture
            }
            wgpu::CurrentSurfaceTexture::Suboptimal(surface_texture) => {
                // Successfully acquired a surface texture, but texture no longer matches the properties of the underlying surface.
                // It's highly recommended to call [`Surface::configure`] again for optimal performance.
                log::warn!("wgpu::CurrentSurfaceTexture::Suboptimal");
                surface_texture
            }
            wgpu::CurrentSurfaceTexture::Timeout => {
                // A timeout was encountered while trying to acquire the next frame.
                //
                // Applications should skip the current frame and try again later.
                log::warn!("wgpu::CurrentSurfaceTexture::Timeout");
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Occluded => {
                // The window is occluded (e.g. minimized or behind another window).
                //
                // Applications should skip the current frame and try again once the window
                // is no longer occluded.
                log::warn!("wgpu::CurrentSurfaceTexture::Occluded");
                return Err(RenderError::SurfaceOccluded);
            }
            wgpu::CurrentSurfaceTexture::Outdated => {
                // The underlying surface has changed, and therefore the surface configuration is outdated.
                //
                // Call [`Surface::configure()`] and try again.
                log::warn!("wgpu::CurrentSurfaceTexture::Outdated");
                return Err(RenderError::SurfaceOutdated);
            }
            wgpu::CurrentSurfaceTexture::Lost => {
                // The surface has been lost and needs to be recreated.
                //
                // If the device as a whole is lost (see [`set_device_lost_callback()`][crate::Device::set_device_lost_callback]), then
                // you need to recreate the device and all resources.
                // Otherwise, call [`Instance::create_surface()`] to recreate the surface,
                // then [`Surface::configure()`], and try again.
                log::warn!("wgpu::CurrentSurfaceTexture::Lost");
                return Err(RenderError::SurfaceLost);
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                // A validation error inside [`Surface::get_current_texture()`] was raised
                // and caught by an [error scope](crate::Device::push_error_scope) or
                // [`on_uncaptured_error()`][crate::Device::on_uncaptured_error].
                //
                // Applications should attend to the validation error and try again.
                log::warn!("wgpu::CurrentSurfaceTexture::Validation");
                return Err(RenderError::SurfaceValidation);
            }
        };

        let format = renderer_interface.surface_format().add_srgb_suffix();
        let view: wgpu::TextureView =
            surface_texture
                .texture
                .create_view(&wgpu::TextureViewDescriptor {
                    // Without add_srgb_suffix() the image we will be working with
                    // might not be "gamma correct".
                    format: Some(format),
                    ..Default::default()
                });

        let mut encoder: wgpu::CommandEncoder =
            renderer_interface
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Render Encoder"),
                });

        watch_fps.stop(watch_index);

        // if self.settings.enable_fxaa {
        //     self.render_fxaa(&view, &mut encoder, ambient_light_quad);
        // }
        watch_index += 1;
        watch_fps.start(watch_index, "Draw Calls");

        self.render_shadow_map(
            &view,
            &mut encoder,
            lod_terrains,
            animations,
            vertex_color_objects,
            vertex_color_objects_lines,
        );

        self.render_forward(
            renderer_interface,
            &view,
            &mut encoder,
            lod_terrains,
            animations,
            gui_elements,
            vertex_color_objects,
            vertex_color_objects_lines,
            particles,
            plasmas,
            glow,
            particles_bilboard_sphere,
            particles_rectangle,
        );

        watch_fps.stop(watch_index);

        watch_index += 1;
        watch_fps.start(watch_index, "Present Surface");
        renderer_interface
            .queue()
            .submit(std::iter::once(encoder.finish()));
        renderer_interface.pre_present_notify();
        renderer_interface.queue().present(surface_texture);
        watch_fps.stop(watch_index);

        watch_index += 1;
        watch_fps.start(watch_index, "Wait Render Loop Finish");

        // wait to see how high the gpu load is
        if self.settings.wait_for_render_loop_to_finish {
            let _res = renderer_interface.device().poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            });
        } else {
            let _res = renderer_interface.device().poll(wgpu::PollType::Poll);
        }
        watch_fps.stop(watch_index);

        Ok(())
    }
}
