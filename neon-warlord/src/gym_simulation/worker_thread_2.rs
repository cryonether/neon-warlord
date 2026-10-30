//! Send data to a thread, run some function and receive data again

use std::{
    sync::mpsc::{Receiver, SyncSender, sync_channel},
    thread::{self, JoinHandle},
};

use crate::triple_buffer::{self};

/// Send data to a thread, run some function and receive data again
pub struct WorkerThread2<T>
where
    T: WorkerThread2Run,
{
    request_tx: SyncSender<Box<T>>,
    result_rx: Receiver<Box<T>>,
    pub drawer_consumer: triple_buffer::Consumer<Drawer>,

    thread: Thread,
}

impl<T> WorkerThread2<T>
where
    T: WorkerThread2Run + Send + 'static,
{
    pub fn new(name: String) -> Self {
        
        // Send data back and forth
        let (request_tx, request_rx) = sync_channel::<Box<T>>(1);
        let (result_tx, result_rx) = sync_channel::<Box<T>>(1);

        // Triple buffer for drawing
        let (mut drawer_producer, drawer_consumer) = triple_buffer::create(Drawer::new());

        #[allow(unused_mut)]
        #[allow(unused)]
        let mut single_threaded = false;
        #[cfg(target_arch = "wasm32")]
        {
            single_threaded = true;
        }

        let thread = if single_threaded {
            // Single threaded

            let lambda = move || {
                if let Ok(mut data) = request_rx.recv() {
                    while !data.done() {
                        data.run();

                        // draw
                        if drawer_producer.is_consumed() {
                            let buffer = drawer_producer.buffer();
                            data.draw(&mut buffer.nodes, &mut buffer.edges);
                            drawer_producer.publish();
                        }
                    }

                    result_tx.send(data).unwrap();
                }
            };

            Thread::SingleThread(Box::new(lambda))
        } else {
            // Multi threaded

            let builder = thread::Builder::new().name(name);
            let thread = builder
                .spawn(move || {
                    while let Ok(mut data) = request_rx.recv() {
                    data.init();
                    while !data.done() {
                        data.run();

                        // draw
                        if drawer_producer.is_consumed() {
                            let buffer = drawer_producer.buffer();
                            data.draw(&mut buffer.nodes, &mut buffer.edges);
                            drawer_producer.publish();
                        }
                    }

                    result_tx.send(data).unwrap();
                    }
                })
                .unwrap();

            Thread::MultiThread(thread)
        };

        Self {
            request_tx,
            result_rx,
            drawer_consumer,
            thread,
        }
    }

    pub fn send(&mut self, data: Box<T>) {
        self.request_tx.send(data).unwrap();

        if let Thread::SingleThread(thread) = &mut self.thread {
            thread();
        }
    }

    pub fn receive(&mut self) -> Box<T> {
        self.result_rx.recv().unwrap()
    }

    pub fn try_receive(&mut self) -> Option<Box<T>> {
        self.result_rx.try_recv().ok()
    }

}

/// Helper to distinguish between single and multithreaded run
enum Thread {
    SingleThread(Box<dyn FnMut() + Send + 'static>),
    #[allow(unused)]
    MultiThread(JoinHandle<()>),
}

pub trait WorkerThread2Run {
    fn init(&mut self);
    fn run(&mut self);
    fn draw(&self, 
        nodes: &mut Vec<forward_renderer::particle_shader::Instance>,
        edges: &mut Vec<forward_renderer::particle_shader_two_point::Instance>,
    );
    fn done(&self) -> bool;
}


#[derive(Clone)]
pub struct Drawer {
    pub nodes: Vec<forward_renderer::particle_shader::Instance>,
    pub edges: Vec<forward_renderer::particle_shader_two_point::Instance>,
}

impl Drawer {
    pub fn new() -> Self {
        Self { nodes: Vec::new(), edges: Vec::new() }
    }
}