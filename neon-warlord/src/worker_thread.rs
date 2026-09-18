//! Creates a thread or uses a single threaded update function on wasm

use std::{sync::{Arc, atomic::{AtomicBool, Ordering}}, thread::JoinHandle, time::Duration};

use instant::Instant;

/// Creates a thread or uses a single threaded update function on wasm
pub struct WorkerThread<T>
where
    T: Update,
{
    thread: Thread<T>,

    limit_ups: Arc<AtomicBool>
}

impl<T> WorkerThread<T>
where
    T: Update,
    T: Send + 'static,
{
    /// Spawns a new thread, executing the update function from T
    /// Or just saves the object on wasm
    pub fn spawn(func_obj: T) -> Self {
        #[allow(unused_mut)]
        #[allow(unused)]
        let mut single_threaded = false;
        #[cfg(target_arch = "wasm32")]
        {
            single_threaded = true;
        }

        if single_threaded {
            let res = SingleThreadHandle { func_obj };
            WorkerThread {
                thread: Thread::SingleThread(res),
                limit_ups: Arc::new(AtomicBool::new(true))
            }
        } else {
            use std::thread;

            let limit_ups = Arc::new(AtomicBool::new(true));
            let limit_ups_thread = limit_ups.clone();
            let res = thread::spawn(move || {
                let mut func_obj = func_obj;
                loop {
                    let frame_start = Instant::now();

                    // update
                    func_obj.update();

                    // sleep until 16.6 ms have been reached
                    let frame_time = frame_start.elapsed();
                    let target_frame_time = Duration::from_micros(16_667);

                    if limit_ups_thread.load(Ordering::Relaxed)
                        && frame_time < target_frame_time {
                            thread::sleep(target_frame_time - frame_time);
                        }
                }
            });
            WorkerThread {
                thread: Thread::MultiThread(res),
                limit_ups
            }
        }
    }

    /// Runs the thread on wasm
    pub fn update(&mut self) {
        match &mut self.thread {
            Thread::SingleThread(single_thread_handle) => {
                single_thread_handle.update();
            }
            Thread::MultiThread(_join_handle) => {
                // nothing to do
            }
        }
    }

    pub fn _limit_ups(&mut self, val: bool) {
        self.limit_ups.store(val, Ordering::Relaxed);
    }

    pub fn limit_ups_toggle(&mut self) {
        let val = self.limit_ups.load(Ordering::Relaxed);
        self.limit_ups.store(!val, Ordering::Relaxed);
    }
}

/// Helper to distinguish between single and multithreaded run
enum Thread<T>
where
    T: Update,
{
    SingleThread(SingleThreadHandle<T>),
    MultiThread(JoinHandle<()>),
}

/// Holds an object for single threaded execution
struct SingleThreadHandle<T>
where
    T: Update,
{
    func_obj: T,
}

impl<T> SingleThreadHandle<T>
where
    T: Update,
{
    fn update(&mut self) {
        self.func_obj.update();
    }
}

pub trait Update {
    fn update(&mut self);
}
