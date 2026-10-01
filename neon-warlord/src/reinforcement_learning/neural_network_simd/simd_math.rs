//! Implements matrix multiplication with simd operations

pub mod simd_mat;
pub mod simd_vec;
pub mod simd_row_vec;

#[cfg(test)]
mod test_simd_mat;
#[cfg(test)]
mod test_simd_vec;
#[cfg(test)]
mod test_simd_row_vec;

use std::fmt;
use std::ops::{Add, AddAssign, Sub, SubAssign};
use std::{iter::zip, ops::Mul};

use wide::f32x16;

const LANES: usize = 16;

#[repr(align(64))]
#[derive(Debug, Copy, Clone)]
pub struct AlignedVec<const N: usize>(pub [f32; N]);

impl<const N: usize> AlignedVec<N> {
    pub fn simd_iter(&self) -> impl Iterator<Item = &f32x16> {
        const _: () = {
            assert!(std::mem::size_of::<f32x16>() == 16 * std::mem::size_of::<f32>());
            assert!(std::mem::align_of::<f32x16>() <= 64);
        };

        unsafe {
            std::slice::from_raw_parts(
                self.0.as_ptr() as *const f32x16,
                N / LANES,
            )
            .iter()
        }
    }

    pub fn simd_iter_mut(&mut self) -> impl Iterator<Item = &mut f32x16> {
        const _: () = {
            assert!(std::mem::size_of::<f32x16>() == 16 * std::mem::size_of::<f32>());
            assert!(std::mem::align_of::<f32x16>() <= 64);
        };

        unsafe {
            std::slice::from_raw_parts_mut(
                self.0.as_mut_ptr() as *mut f32x16,
                N / LANES,
            )
            .iter_mut()
        }
    }

    pub fn remainder(&self) -> &[f32] {
        &self.0[(N / LANES) * LANES..]
    }

    pub fn remainder_mut(&mut self) -> &mut [f32] {
        let start = (N / LANES) * LANES;
        &mut self.0[start..]
    }
}

// From

impl<const N: usize> From<[f32; N]> for AlignedVec<N> {
    fn from(value: [f32; N]) -> Self {
        Self(value)
    }
}

// Deref

impl<const N: usize> std::ops::Deref for AlignedVec<N> {
    type Target = [f32; N];

    fn deref(&self) -> &[f32; N] {
        &self.0
    }
}

impl<const N: usize> std::ops::DerefMut for AlignedVec<N> {
    fn deref_mut(&mut self) -> &mut [f32; N] {
        &mut self.0
    }
}

// Into Iterator

impl<const N: usize> IntoIterator for AlignedVec<N> {
    type Item = f32;
    type IntoIter = std::array::IntoIter<f32, N>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a, const N: usize> IntoIterator for &'a AlignedVec<N> {
    type Item = &'a f32;
    type IntoIter = std::slice::Iter<'a, f32>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl<'a, const N: usize> IntoIterator for &'a mut AlignedVec<N> {
    type Item = &'a mut f32;
    type IntoIter = std::slice::IterMut<'a, f32>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter_mut()
    }
}

// Print

impl<const N: usize> fmt::Display for AlignedVec<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;

        for (i, value) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{:?}", value)?;
        }

        write!(f, "]")
    }
}