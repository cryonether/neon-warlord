//! Tripple buffer synchronization
use std::cell::UnsafeCell;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

struct TrippleBuffer<T> {
    buffers: [UnsafeCell<Vec<T>>; 3],

    // Index of the buffer currently in the middle.
    middle: AtomicUsize,
    // Counts produced buffers
    count: AtomicUsize,
}

// We guarantee that producer and consumer never access
// the same buffer simultaneously.
unsafe impl<T: Send> Send for TrippleBuffer<T> {}
unsafe impl<T: Send> Sync for TrippleBuffer<T> {}

pub struct Producer<T> {
    inner: Arc<TrippleBuffer<T>>,
    back: usize,
}

pub struct Consumer<T> {
    inner: Arc<TrippleBuffer<T>>,
    front: usize,
    count: usize,
}

impl<T> TrippleBuffer<T> {
    pub fn create() -> (Producer<T>, Consumer<T>) {
        let inner = Arc::new(TrippleBuffer {
            buffers: [
                UnsafeCell::new(Vec::new()),
                UnsafeCell::new(Vec::new()),
                UnsafeCell::new(Vec::new()),
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

    fn buffer(&self, index: usize) -> &mut Vec<T> {
        // SAFETY:
        // The triple-buffer protocol guarantees exclusive ownership
        // of the buffer represented by `index`.
        unsafe { &mut *self.buffers[index].get() }
    }
}

impl<T> Producer<T> {
    pub fn buffer(&mut self) -> &mut Vec<T> {
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
    pub fn buffer(&self) -> &Vec<T> {
        self.inner.buffer(self.front)
    }

    pub fn buffer_mut(&mut self) -> &mut Vec<T> {
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
