//! Implements matrix multiplication with simd operations

pub mod simd_mat;
pub mod simd_vec;
pub mod simd_row_vec;

mod test_simd_mat;
mod test_simd_vec;
mod test_simd_row_vec;

use std::ops::{Add, AddAssign, Sub, SubAssign};
use std::{iter::zip, ops::Mul};

use wide::f32x16;

const LANES: usize = 16;

#[repr(align(64))]
#[derive(Debug, Copy, Clone)]
pub struct AlignedVec<const N: usize>(pub [f32; N]);

impl<const N: usize> AlignedVec<N> {
    pub fn as_simd<const L: usize>(&self) -> &[f32x16; L] {
        assert_eq!(N, L * LANES);
        assert_eq!(size_of_val(&self.0), size_of::<[f32x16; L]>());
        unsafe { &mut *(self.0.as_ptr() as *mut [f32x16; L]) }
    }

    pub fn as_mut_simd<const L: usize>(&self) -> &[f32x16; L] {
        assert_eq!(N, L * LANES);
        assert_eq!(size_of_val(&self.0), size_of::<[f32x16; L]>());
        unsafe { &mut *(self.0.as_ptr() as *mut [f32x16; L]) }
    }

    pub fn simd_iter(&self) -> impl Iterator<Item = &f32x16> {
        assert_eq!(N % LANES, 0);

        unsafe {
            std::slice::from_raw_parts(
                self.0.as_ptr() as *const f32x16,
                N / LANES,
            )
            .iter()
        }
    }

    pub fn simd_iter_mut(&mut self) -> impl Iterator<Item = &mut f32x16> {
        assert_eq!(N % LANES, 0);

        unsafe {
            std::slice::from_raw_parts_mut(
                self.0.as_mut_ptr() as *mut f32x16,
                N / LANES,
            )
            .iter_mut()
        }
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





// #[test]
// fn test_mul_mat_vec() {
//     const N: usize = 128;
//     const L: usize = 8;

//     // A[i][j] = i * N + j
//     let m: [[f32; N]; N] = std::array::from_fn(|i| std::array::from_fn(|j| (i * N + j) as f32));

//     // x = [1, 2, 3, ..., 128]
//     let a: [f32; N] = std::array::from_fn(|i| (i + 1) as f32);

//     let res: SVec16<N> = &SMat16::new(m) * &SVec16::new(a);

//     // Reference implementation.
//     let expected: [f32; N] = std::array::from_fn(|i| (0..N).map(|j| m[i][j] * a[j]).sum::<f32>());

//     // println!("res: {:?}", res);

//     let expected: [f32x16; L] = f32x16_from(expected);

//     for i in 0..L {
//         let got = res.a[i];
//         let expected = expected[i];

//         let diff = (got - expected).abs();
//         let tolerance = 1e-5 * expected.abs().max(f32x16::splat(1.0));

//         for (diff, tolerance) in zip(diff.as_array(), tolerance.as_array()) {
//             assert!(
//                 *diff <= *tolerance,
//                 "mismatch at index {i}: got {got}, expected {expected}, \
//                 diff {diff}, tolerance {tolerance}"
//             );
//         }
//     }
// }

// #[test]
// fn test_mul_vec_mat() {
//     const N: usize = 128;
//     const L: usize = 8;

//     // A[i][j] = i * N + j
//     let m = std::array::from_fn(|i| std::array::from_fn(|j| (i * N + j) as f32));

//     // x = [1, 2, 3, ..., 128]
//     let a = std::array::from_fn(|i| (i + 1) as f32);

//     let res: SRowVec16<N> = &SRowVec16::new(a) * &SMat16::new(m);

//     // Reference implementation:
//     //
//     // y[j] = sum_i x[i] * A[i][j]
//     let expected: [f32; N] = std::array::from_fn(|j| (0..N).map(|i| a[i] * m[i][j]).sum::<f32>());

//     let expected: [f32x16; L] = f32x16_from(expected);

//     for i in 0..L {
//         let got = res.a[i];
//         let expected = expected[i];

//         let diff = (got - expected).abs();
//         let tolerance = 1e-5 * expected.abs().max(f32x16::splat(1.0));

//         for (diff, tolerance) in zip(diff.as_array(), tolerance.as_array()) {
//             assert!(
//                 *diff <= *tolerance,
//                 "mismatch at index {i}: got {got}, expected {expected}, \
//                 diff {diff}, tolerance {tolerance}"
//             );
//         }
//     }
// }

// #[test]
// fn test_outer_product() {
//     const N: usize = 128;
//     const L: usize = 8;

//     // a = [1, 2, 3, ..., 128]
//     let a: [f32; N] = std::array::from_fn(|i| (i + 1) as f32);

//     // b = [129, 130, 131, ..., 256]
//     let b = std::array::from_fn(|i| (N + i + 1) as f32);

//     let res: SMat16<M, N> = &SVec16::new(a) * &SRowVec16::new(b);

//     // Reference implementation:
//     //
//     // M[i][j] = a[i] * b[j]
//     let expected: [[f32; N]; N] = std::array::from_fn(|i| std::array::from_fn(|j| a[i] * b[j]));

//     let expected: [[f32x16; L]; 128] = SMat16::new(expected).m;

//     for i in 0..N {
//         for j in 0..L {
//             let got = res.m[i][j];
//             let expected = expected[i][j];

//             let diff = (got - expected).abs();
//             let tolerance = 1e-5 * expected.abs().max(f32x16::splat(1.0));

//             for (diff, tolerance) in zip(diff.as_array(), tolerance.as_array()) {
//                 assert!(
//                     *diff <= *tolerance,
//                     "mismatch at index {i}: got {got}, expected {expected}, \
//                     diff {diff}, tolerance {tolerance}"
//                 );
//             }
//         }
//     }
// }

// #[test]
// fn test_mul_element_wise() {
//     const N: usize = 128;
//     const L: usize = 8;

//     // a = [1, 2, 3, ..., 128]
//     let a: [f32; N] = std::array::from_fn(|i| (i + 1) as f32);

//     // b = [129, 130, 131, ..., 256]
//     let b: [f32; N] = std::array::from_fn(|i| (N + i + 1) as f32);

//     let a = SVec16::<L>::new(a);
//     let b = SVec16::<L>::new(b);

//     let res = &a * &b;

//     // Scalar reference implementation.
//     let expected: [f32; N] = std::array::from_fn(|i| (i + 1) as f32 * (N + i + 1) as f32);

//     let expected: [f32x16; L] = f32x16_from(expected);

//     for i in 0..L {
//         let got = res.a[i];
//         let expected = expected[i];

//         let diff = (got - expected).abs();
//         let tolerance = 1e-5 * expected.abs().max(f32x16::splat(1.0));

//         for (diff, tolerance) in zip(diff.as_array(), tolerance.as_array()) {
//             assert!(
//                 *diff <= *tolerance,
//                 "mismatch at index {i}: got {got}, expected {expected}, \
//                 diff {diff}, tolerance {tolerance}"
//             );
//         }
//     }
// }

// #[cfg(test)]
// mod tests {
//     use super::*;

//     const N: usize = 128;
//     const L: usize = N / LANES;

//     #[test]
//     fn s_vec_as_array() {
//         let mut a = [f32x16::splat(0.0); L];

//         for (i, v) in a.iter_mut().enumerate() {
//             *v = f32x16::splat(i as f32);
//         }

//         let vec = SVec16::<L> { a };

//         let array: &[f32; N] = vec.as_array();

//         assert_eq!(array.len(), N);

//         for i in 0..N {
//             assert_eq!(array[i], (i / LANES) as f32);
//         }
//     }

//     #[test]
//     fn s_vec_as_mut_array() {
//         let a = [f32x16::splat(0.0); L];
//         let mut vec = SVec16::<L> { a };

//         {
//             let array: &mut [f32; N] = vec.as_mut_array();

//             assert_eq!(array.len(), N);

//             for (i, value) in array.iter_mut().enumerate() {
//                 *value = i as f32;
//             }
//         }

//         let array: &[f32; N] = vec.as_array();

//         for i in 0..N {
//             assert_eq!(array[i], i as f32);
//         }
//     }

//     #[test]
//     fn s_row_vec_as_array() {
//         let mut a = [f32x16::splat(0.0); L];

//         for (i, v) in a.iter_mut().enumerate() {
//             *v = f32x16::splat(i as f32);
//         }

//         let vec = SRowVec16::<N, L> { a };

//         let array = vec.as_array();

//         assert_eq!(array.len(), N);

//         for i in 0..N {
//             assert_eq!(array[i], (i / LANES) as f32);
//         }
//     }

//     #[test]
//     fn s_row_vec_as_mut_array() {
//         let a = [f32x16::splat(0.0); L];
//         let mut vec = SRowVec16::<N, L> { a };

//         {
//             let array = vec.as_mut_array();

//             assert_eq!(array.len(), N);

//             for (i, value) in array.iter_mut().enumerate() {
//                 *value = i as f32;
//             }
//         }

//         let array = vec.as_array();

//         for i in 0..N {
//             assert_eq!(array[i], i as f32);
//         }
//     }

//     #[test]
//     fn s_mat_as_array() {
//         let mut m = [[f32x16::splat(0.0); L]; N];

//         // Give every SIMD vector a unique value so that we can
//         // verify the complete memory layout.
//         for row in 0..N {
//             for lane in 0..L {
//                 m[row][lane] = f32x16::splat((row * L + lane) as f32);
//             }
//         }

//         let mat = SMat16::<N, L> { m };

//         let array = mat.as_array();

//         assert_eq!(array.len(), N);
//         assert_eq!(array[0].len(), N);

//         for row in 0..N {
//             for col in 0..N {
//                 let simd_index = col / LANES;
//                 let expected = (row * L + simd_index) as f32;

//                 assert_eq!(array[row][col], expected);
//             }
//         }
//     }

//     #[test]
//     fn s_mat_as_mut_array() {
//         let m = [[f32x16::splat(0.0); L]; N];
//         let mut mat = SMat16::<N, L> { m };

//         {
//             let array = mat.as_mut_array();

//             assert_eq!(array.len(), N);
//             assert_eq!(array[0].len(), N);

//             for row in 0..N {
//                 for col in 0..N {
//                     array[row][col] = (row * N + col) as f32;
//                 }
//             }
//         }

//         let array = mat.as_array();

//         for row in 0..N {
//             for col in 0..N {
//                 assert_eq!(array[row][col], (row * N + col) as f32);
//             }
//         }
//     }
// }

// fn assert_close(a: f32, b: f32, tolerance: f32) {
//     let diff = (a - b).abs();

//     assert!(diff <= tolerance, "expected {b}, got {a}, diff {diff}");
// }
