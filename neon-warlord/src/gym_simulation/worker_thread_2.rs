//! Send data to a thread, run some function and receive data again

use std::{
    sync::mpsc::{Receiver, SyncSender, sync_channel}, thread::{self, JoinHandle},
};

/// Send data to a thread, run some function and receive data again
pub struct WorkerThread2<T> 
where T:
    WorkerThread2Run
{
    request_tx: SyncSender<Box<T>>,
    result_rx: Receiver<Box<T>>,

    thread: Thread,
}

impl<T> WorkerThread2<T> 
where T:
    WorkerThread2Run + Send + 'static
{
    pub fn new(
        name: String,
    ) -> Self {
        let (request_tx, request_rx) = sync_channel::<
            Box<T>,
        >(1);
        let (result_tx, result_rx) = sync_channel::<
            Box<T>,
        >(1);

        #[allow(unused_mut)]
        #[allow(unused)]
        let mut single_threaded = false;
        #[cfg(target_arch = "wasm32")]
        {
            single_threaded = true;
        }

        let thread = if single_threaded {
            // Single threaded

            let lambda = move || 
                {
                    if let Ok(mut data) = request_rx.recv() {
                        data.run();
                        result_tx.send(data).unwrap();
                    }
                };

            Thread::SingleThread(Box::new(lambda))
        }
        else {
            // Multi threaded

            let builder = thread::Builder::new().name(name);
            let thread = builder.spawn(move || 
                {
                    while let Ok(mut data) = request_rx.recv() {
                        data.run();
                        result_tx.send(data).unwrap();
                    }
                }
            ).unwrap();

            Thread::MultiThread(thread)
        };

        Self {
            request_tx,
            result_rx,
            thread,
        }
    }

    pub fn send(&mut self, data: Box<T>) {

        self.request_tx.send(data).unwrap();

        if let Thread::SingleThread(thread) = &self.thread 
        {
            thread();
        }
    }

    pub fn receive(&mut self) -> Box<T> {

        let res = self.result_rx.recv().unwrap();

        res
    }

}


/// Helper to distinguish between single and multithreaded run
enum Thread
{
    SingleThread(Box<dyn Fn() + Send + 'static>),
    #[allow(unused)]
    MultiThread(JoinHandle<()>),
}

pub trait WorkerThread2Run {
    fn run(&mut self);
}
