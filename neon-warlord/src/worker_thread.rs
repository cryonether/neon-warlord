//! Creates a thread or using a single threaded update function on wasm

use std::thread::{Builder, JoinHandle};

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

enum Thread<T>
where
    T: Update,
{
    SingleThread(SingleThreadHandle<T>),
    MultiThread(JoinHandle<()>),
}

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
