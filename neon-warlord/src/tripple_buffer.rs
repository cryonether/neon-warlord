//! Tripple buffer synchronization

use std::sync::{Arc, RwLock};

struct Producer {

}

struct Consumer {

}

struct TrippleBuffer<T> {
    val_0: Arc<RwLock<T>>,
    val_1: Arc<RwLock<T>>,
    val_2: Arc<RwLock<T>>,
}



