//! Triple buffer synchronization

use std::cell::UnsafeCell;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

/// Triple buffer synchronization  
/// 
/// Details:  
///     Uses three shared buffers using UnsafeCell, front, middle and back.
///     The producer atomically swaps middle and back.
///     The consumer atomically swaps middle and front.
///     -> Producer and consumer never access the same buffer simultaneously.
///     -> No data races.
struct TripleBuffer<T> {
    buffers: [UnsafeCell<T>; 3],

    // Index of the buffer currently in the middle.
    middle: AtomicUsize,
    // Counts produced buffers
    count: AtomicUsize,
}

// We guarantee that producer and consumer never access
// the same buffer simultaneously.
unsafe impl<T: Send> Send for TripleBuffer<T> {}
unsafe impl<T: Send> Sync for TripleBuffer<T> {}

/// Adds data into the Triple Buffer
pub struct Producer<T> {
    inner: Arc<TripleBuffer<T>>,
    back: usize,
}

/// Gets data from the Triple Buffer
pub struct Consumer<T> {
    inner: Arc<TripleBuffer<T>>,
    front: usize,
    count: usize,
}

impl<T> TripleBuffer<T> {
    #[allow(clippy::mut_from_ref)]
    fn buffer(&self, index: usize) -> &mut T {
        // SAFETY:
        // The triple-buffer protocol guarantees exclusive ownership
        // of the buffer represented by `index`.
        unsafe { &mut *self.buffers[index].get() }
    }
}

impl<T> Producer<T> {
    pub fn buffer(&mut self) -> &mut T {
        self.inner.buffer(self.back)
    }

    pub fn publish(&mut self) {
        // Give our back buffer to the consumer.
        //
        // At the same time, take whatever was in the middle
        // as our new back buffer.
        self.back = self
            .inner
            .middle
            .swap(self.back, Ordering::AcqRel);

        self.inner.count.fetch_add(1, Ordering::Relaxed);
    }
}

impl<T> Consumer<T> {
    pub fn buffer(&self) -> &T {
        self.inner.buffer(self.front)
    }

    pub fn _buffer_mut(&mut self) -> &mut T {
        self.inner.buffer(self.front)
    }

    pub fn acquire_latest(&mut self) {
        let count: usize = self.inner.count.load(Ordering::Relaxed);
        if count == self.count {
            return;
        }
        self.count = count;

        // Take the newest published buffer.
        //
        // Our old front becomes the new middle buffer.
        self.front = self
            .inner
            .middle
            .swap(self.front, Ordering::AcqRel);
    }
}

/// Creates a new Triple Buffer consumer producer pair
pub fn create<T>(data: T) -> (Producer<T>, Consumer<T>) 
where T: Clone
{
    let inner = Arc::new(TripleBuffer {
        buffers: [
            UnsafeCell::new(data.clone()),
            UnsafeCell::new(data.clone()),
            UnsafeCell::new(data),
        ],
        middle: AtomicUsize::new(1),
        count: AtomicUsize::new(0),
    });

    let producer = Producer {
        inner: inner.clone(),
        back: 0,
    };

    let consumer = Consumer {
        inner,
        front: 2,
        count: 0,
    };

    (producer, consumer)
}
