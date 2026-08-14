//! Creates a thread or uses a single threaded update function on wasm

use std::thread::JoinHandle;

/// Creates a thread or uses a single threaded update function on wasm
pub struct WorkerThread<T> 
where
    T: Update,
{
    thread: Thread<T>,
}

impl<T> WorkerThread<T> 
where
    T: Update,
    T: Send + 'static,
{
    /// Spawns a new thread, executing the update function from T
    /// Or just saves the object on wasm
    pub fn spawn(func_obj: T) -> Self
    {   
        #[allow(unused_mut)]
        let mut single_threaded = false;
        #[cfg(target_arch = "wasm32")]
        {
            single_threaded = true;
        }

        if single_threaded
        {
            let res = SingleThreadHandle{func_obj};
            WorkerThread { thread: Thread::SingleThread(res) }
        }
        else {
            use std::thread;

            let res = thread::spawn(move || {
                let mut func_obj = func_obj;
                loop {
                    func_obj.update();
                }
            });
            WorkerThread { thread: Thread::MultiThread(res) }
        }
    }

    /// Runs the thread on wasm
    pub fn update(&mut self) {
        match &mut self.thread{
            Thread::SingleThread(single_thread_handle) => {
                single_thread_handle.update();
            },
            Thread::MultiThread(_join_handle) => {
                // nothing to do
            },
        }
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

pub trait Update{
    fn update(&mut self);
}
