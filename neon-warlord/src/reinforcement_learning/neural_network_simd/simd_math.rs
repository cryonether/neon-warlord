//! Implements matrix multiplication with simd operations

use std::ops::{Add, AddAssign, Sub, SubAssign};
use std::{iter::zip, ops::Mul};

use wide::f32x16;

// 238 fps
// 169 fps
// 222 fps

// const N: usize = 128;
const LANES: usize = 16;
// const L: usize = N/LANES;

#[derive(Debug, Copy, Clone)]
pub struct SMat16<const N: usize, const L: usize> {
    pub m: [[f32x16; L]; N],
}

#[derive(Debug, Copy, Clone)]
pub struct SVec16<const L: usize> {
    pub a: [f32x16; L],
}

#[derive(Debug, Clone)]
pub struct SRowVec<const N: usize, const L: usize> {
    pub a: [f32x16; L],
}

impl<const N: usize, const L: usize> SMat16<N, L> {
    /// Returns an array reference containing the entire SIMD vector.
    // // #[inline]
    // #[must_use]
    pub fn as_array(&self) -> &[[f32; N]; N] {
        assert_eq!(N, L * 16);
        assert_eq!(size_of_val(&self.m), size_of::<[[f32; N]; N]>());

        // SAFETY:
        // - `self.a` contains exactly L * 16 f32 values.
        // - `f32x16` is a SIMD vector of 16 f32 values.
        // - The resulting reference has the same size as `self.a`.
        //
        // SAFETY: The input type has greater alignment than the output type,
        // and both pointed-at types have the same size, accept all bit-patterns
        // and only contain initialized memory.
        unsafe { &*(self.m.as_ptr() as *const [[f32; N]; N]) }
    }

    /// Returns an array reference containing the entire SIMD vector.
    // // #[inline]
    // #[must_use]
    pub fn as_mut_array(&mut self) -> &mut [[f32; N]; N] {
        assert_eq!(N, L * 16);
        assert_eq!(size_of_val(&self.m), size_of::<[[f32; N]; N]>());

        // SAFETY:
        // - `self.a` contains exactly L * 16 f32 values.
        // - `f32x16` is a SIMD vector of 16 f32 values.
        // - The resulting reference has the same size as `self.a`.
        //
        // SAFETY: The input type has greater alignment than the output type,
        // and both pointed-at types have the same size, accept all bit-patterns
        // and only contain initialized memory.
        unsafe { &mut *(self.m.as_ptr() as *mut [[f32; N]; N]) }
    }
}

impl<const L: usize> SVec16<L> {
    /// Returns an array reference containing the entire SIMD vector.
    // // #[inline]
    // #[must_use]
    pub fn as_array<const N: usize>(&self) -> &[f32; N] {
        assert_eq!(N, L * 16);
        assert_eq!(size_of_val(&self.a), size_of::<[f32; N]>());

        // SAFETY:
        // - `self.a` contains exactly L * 16 f32 values.
        // - `f32x16` is a SIMD vector of 16 f32 values.
        // - The resulting reference has the same size as `self.a`.
        //
        // SAFETY: The input type has greater alignment than the output type,
        // and both pointed-at types have the same size, accept all bit-patterns
        // and only contain initialized memory.
        unsafe { &*(self.a.as_ptr() as *const [f32; N]) }
    }

    /// Returns an array reference containing the entire SIMD vector.
    // // #[inline]
    // #[must_use]
    pub fn as_mut_array<const N: usize>(&mut self) -> &mut [f32; N] {
        assert_eq!(N, L * 16);
        assert_eq!(size_of_val(&self.a), size_of::<[f32; N]>());

        // SAFETY:
        // - `self.a` contains exactly L * 16 f32 values.
        // - `f32x16` is a SIMD vector of 16 f32 values.
        // - The resulting reference has the same size as `self.a`.
        //
        // SAFETY: The input type has greater alignment than the output type,
        // and both pointed-at types have the same size, accept all bit-patterns
        // and only contain initialized memory.
        unsafe { &mut *(self.a.as_ptr() as *mut [f32; N]) }
    }

    // // #[inline]
    pub fn as_row_vec<const N: usize>(self) -> SRowVec<N, L> {
        assert_eq!(N, L * 16);
        SRowVec { a: self.a }
    }
}

impl<const N: usize, const L: usize> SRowVec<N, L> {
    /// Returns an array reference containing the entire SIMD vector.
    // // #[inline]
    // #[must_use]
    pub fn as_array(&self) -> &[f32; N] {
        assert_eq!(N, L * 16);
        assert_eq!(size_of_val(&self.a), size_of::<[f32; N]>());

        // SAFETY:
        // - `self.a` contains exactly L * 16 f32 values.
        // - `f32x16` is a SIMD vector of 16 f32 values.
        // - The resulting reference has the same size as `self.a`.
        //
        // SAFETY: The input type has greater alignment than the output type,
        // and both pointed-at types have the same size, accept all bit-patterns
        // and only contain initialized memory.
        unsafe { &*(self.a.as_ptr() as *const [f32; N]) }
    }

    /// Returns an array reference containing the entire SIMD vector.
    // // #[inline]
    // #[must_use]
    pub fn as_mut_array(&mut self) -> &mut [f32; N] {
        assert_eq!(N, L * 16);
        assert_eq!(size_of_val(&self.a), size_of::<[f32; N]>());

        // SAFETY:
        // - `self.a` contains exactly L * 16 f32 values.
        // - `f32x16` is a SIMD vector of 16 f32 values.
        // - The resulting reference has the same size as `self.a`.
        //
        // SAFETY: The input type has greater alignment than the output type,
        // and both pointed-at types have the same size, accept all bit-patterns
        // and only contain initialized memory.
        unsafe { &mut *(self.a.as_ptr() as *mut [f32; N]) }
    }

    // #[inline]
    #[allow(clippy::wrong_self_convention)]
    pub fn as_column_vec(self) -> SVec16<L> {
        SVec16 { a: self.a }
    }
}

impl<const N: usize, const L: usize> SMat16<N, L> {
    // #[inline]
    pub fn new(m: [[f32; N]; N]) -> Self {
        let m_vec = std::array::from_fn(|i| f32x16_from(m[i]));

        Self { m: m_vec }
    }
}

impl<const L: usize> SVec16<L> {
    // #[inline]
    pub fn new<const N: usize>(a: [f32; N]) -> Self {
        let a_vec = f32x16_from(a);

        Self { a: a_vec }
    }
}

impl<const N: usize, const L: usize> SRowVec<N, L> {
    // #[inline]
    pub fn new(a: [f32; N]) -> Self {
        let a_vec = f32x16_from(a);

        Self { a: a_vec }
    }
}

impl<const N: usize, const L: usize> From<SVec16<L>> for [f32; N] {
    // #[inline]
    fn from(v: SVec16<L>) -> Self {
        f32x16_to(v.a)
    }
}

impl<const N: usize, const L: usize> From<SRowVec<N, L>> for [f32; N] {
    // #[inline]
    fn from(v: SRowVec<N, L>) -> Self {
        f32x16_to(v.a)
    }
}

impl<const N: usize, const L: usize> From<SMat16<N, L>> for [[f32; N]; N] {
    // #[inline]
    fn from(m: SMat16<N, L>) -> Self {
        std::array::from_fn(|i| f32x16_to(m.m[i]))
    }
}

impl<const N: usize, const L: usize> From<[f32; N]> for SVec16<L> {
    // #[inline]
    fn from(a: [f32; N]) -> Self {
        Self::new(a)
    }
}

impl<const N: usize, const L: usize> From<[f32; N]> for SRowVec<N, L> {
    // #[inline]
    fn from(a: [f32; N]) -> Self {
        Self::new(a)
    }
}

impl<const N: usize, const L: usize> From<[[f32; N]; N]> for SMat16<N, L> {
    // #[inline]
    fn from(m: [[f32; N]; N]) -> Self {
        Self::new(m)
    }
}

/// Matrix-vector multiplication: `y = A * x`.
///
/// (N×N)(N×1) → N×1
///
impl<const N: usize, const L: usize> Mul<&SVec16<L>> for &SMat16<N, L> {
    type Output = SVec16<L>;

    // 186 ups
    // #[inline]
    fn mul(self, rhs: &SVec16<L>) -> Self::Output {
        let a = rhs.a;
        let mut res = [0.0f32; N];

        for (res, m) in std::iter::zip(&mut res, &self.m) {
            let mut sum = f32x16::splat(0.0);

            for i in 0..L {
                sum += m[i] * a[i];
            }

            *res = sum.reduce_add();
        }

        SVec16::new(res)
    }
}

/// Vector-matrix multiplication: `y = x^T * A`.
///
/// (1×N)(N×N) → 1×N
///
impl<const N: usize, const L: usize> Mul<&SMat16<N, L>> for &SRowVec<N, L> {
    type Output = SRowVec<N, L>;

    // #[inline]
    fn mul(self, rhs: &SMat16<N, L>) -> Self::Output {
        let mut res = [f32x16::ZERO; L];

        let x = self.as_array();

        for (x, row) in zip(x, &rhs.m) {
            let x = f32x16::splat(*x);

            for i in 0..L {
                res[i] += x * row[i];
            }
        }

        SRowVec { a: res }
    }
}

///
/// Outer Product M = a * b^T
///
/// (N×1)(1×N) → N×N
///
impl<const N: usize, const L: usize> Mul<&SRowVec<N, L>> for &SVec16<L> {
    type Output = SMat16<N, L>;

    // #[inline]
    fn mul(self, rhs: &SRowVec<N, L>) -> Self::Output {
        let b = rhs.a;
        let a: [f32; N] = f32x16_to(self.a);

        let m = std::array::from_fn(|i| {
            let a = f32x16::splat(a[i]);

            std::array::from_fn(|j| a * b[j])
        });

        SMat16 { m }
    }
}

///
/// Element-wise multiplication: `c = a ⊙ b`.
///
/// (N×1) ⊙ (N×1) → N×1
///
/// Each element is multiplied independently:
/// `c[i] = a[i] * b[i]`.
///
impl<const L: usize> Mul<&SVec16<L>> for &SVec16<L> {
    type Output = SVec16<L>;

    // #[inline]
    fn mul(self, rhs: &SVec16<L>) -> Self::Output {
        let a = self.a;
        let b = rhs.a;

        let res = std::array::from_fn(|i| a[i] * b[i]);

        SVec16 { a: res }
    }
}

///
/// `a += b`
///
impl<const L: usize> AddAssign<&SVec16<L>> for SVec16<L> {
    // #[inline]
    fn add_assign(&mut self, rhs: &SVec16<L>) {
        for (a, b) in std::iter::zip(&mut self.a, &rhs.a) {
            *a += *b;
        }
    }
}

///
/// `a += b`
///
impl<const L: usize> SubAssign<&SVec16<L>> for SVec16<L> {
    // #[inline]
    fn sub_assign(&mut self, rhs: &SVec16<L>) {
        for (a, b) in std::iter::zip(&mut self.a, &rhs.a) {
            *a -= *b;
        }
    }
}

///
/// `a -= b`
///
impl<const N: usize, const L: usize> SubAssign<&SMat16<N, L>> for SMat16<N, L> {
    // #[inline]
    fn sub_assign(&mut self, rhs: &SMat16<N, L>) {
        for (a, b) in zip(&mut self.m, &rhs.m) {
            for (a, b) in zip(a, b) {
                *a -= *b;
            }
        }
    }
}

/// `c = a + b`
impl<const N: usize, const L: usize> Add<&SMat16<N, L>> for &SMat16<N, L> {
    type Output = SMat16<N, L>;

    // #[inline]
    fn add(self, rhs: &SMat16<N, L>) -> Self::Output {
        let a = &self.m;
        let b = &rhs.m;

        let c: [[f32x16; L]; N] =
            std::array::from_fn(|i| std::array::from_fn(|j| a[i][j] + b[i][j]));

        SMat16 { m: c }
    }
}

/// `c = a + b`
impl<const L: usize> Add<&SVec16<L>> for &SVec16<L> {
    type Output = SVec16<L>;

    // #[inline]
    fn add(self, rhs: &SVec16<L>) -> Self::Output {
        let a = &self.a;
        let b = &rhs.a;

        let c = std::array::from_fn(|i| a[i] + b[i]);

        SVec16 { a: c }
    }
}

/// `c = a + b`
impl<const L: usize> Add<&SVec16<L>> for SVec16<L> {
    type Output = SVec16<L>;

    // #[inline]
    fn add(self, rhs: &SVec16<L>) -> Self::Output {
        let a = &self.a;
        let b = &rhs.a;

        let c = std::array::from_fn(|i| a[i] + b[i]);

        SVec16 { a: c }
    }
}

///
/// `c = a - b`
///
impl<const N: usize, const L: usize> Sub<&SMat16<N, L>> for &SMat16<N, L> {
    type Output = SMat16<N, L>;

    // #[inline]
    fn sub(self, rhs: &SMat16<N, L>) -> Self::Output {
        let m = std::array::from_fn(|i| std::array::from_fn(|j| self.m[i][j] - rhs.m[i][j]));

        SMat16 { m }
    }
}

/// `c = a - b`
impl<const L: usize> Sub<&SVec16<L>> for &SVec16<L> {
    type Output = SVec16<L>;

    // #[inline]
    fn sub(self, rhs: &SVec16<L>) -> Self::Output {
        let a = &self.a;
        let b = &rhs.a;

        let c = std::array::from_fn(|i| a[i] - b[i]);

        SVec16 { a: c }
    }
}

// #[inline]
fn f32x16_from<const N: usize, const L: usize>(x: [f32; N]) -> [f32x16; L] {
    assert_eq!(N, L * LANES);

    let mut res = [f32x16::ZERO; L];

    for (x, res) in zip(x.as_chunks::<LANES>().0, &mut res) {
        *res = f32x16::from(*x);
    }

    res
}

// #[inline]
fn f32x16_to<const N: usize, const L: usize>(x: [f32x16; L]) -> [f32; N] {
    assert_eq!(N, L * LANES);

    let mut res = [0.0f32; N];

    for (src, dst) in zip(&x, res.as_chunks_mut::<LANES>().0) {
        *dst = (*src).into();
    }

    res
}

#[test]
fn test_mul_mat_vec() {
    const N: usize = 128;
    const L: usize = 8;

    // A[i][j] = i * N + j
    let m: [[f32; N]; N] = std::array::from_fn(|i| std::array::from_fn(|j| (i * N + j) as f32));

    // x = [1, 2, 3, ..., 128]
    let a: [f32; N] = std::array::from_fn(|i| (i + 1) as f32);

    let res: SVec16<L> = &SMat16::new(m) * &SVec16::new(a);

    // Reference implementation.
    let expected: [f32; N] = std::array::from_fn(|i| (0..N).map(|j| m[i][j] * a[j]).sum::<f32>());

    // println!("res: {:?}", res);

    let expected: [f32x16; L] = f32x16_from(expected);

    for i in 0..L {
        let got = res.a[i];
        let expected = expected[i];

        let diff = (got - expected).abs();
        let tolerance = 1e-5 * expected.abs().max(f32x16::splat(1.0));

        for (diff, tolerance) in zip(diff.as_array(), tolerance.as_array()) {
            assert!(
                *diff <= *tolerance,
                "mismatch at index {i}: got {got}, expected {expected}, \
                diff {diff}, tolerance {tolerance}"
            );
        }
    }
}

#[test]
fn test_mul_vec_mat() {
    const N: usize = 128;
    const L: usize = 8;

    // A[i][j] = i * N + j
    let m = std::array::from_fn(|i| std::array::from_fn(|j| (i * N + j) as f32));

    // x = [1, 2, 3, ..., 128]
    let a = std::array::from_fn(|i| (i + 1) as f32);

    let res: SRowVec<N, L> = &SRowVec::new(a) * &SMat16::new(m);

    // Reference implementation:
    //
    // y[j] = sum_i x[i] * A[i][j]
    let expected: [f32; N] = std::array::from_fn(|j| (0..N).map(|i| a[i] * m[i][j]).sum::<f32>());

    let expected: [f32x16; L] = f32x16_from(expected);

    for i in 0..L {
        let got = res.a[i];
        let expected = expected[i];

        let diff = (got - expected).abs();
        let tolerance = 1e-5 * expected.abs().max(f32x16::splat(1.0));

        for (diff, tolerance) in zip(diff.as_array(), tolerance.as_array()) {
            assert!(
                *diff <= *tolerance,
                "mismatch at index {i}: got {got}, expected {expected}, \
                diff {diff}, tolerance {tolerance}"
            );
        }
    }
}

#[test]
fn test_outer_product() {
    const N: usize = 128;
    const L: usize = 8;

    // a = [1, 2, 3, ..., 128]
    let a: [f32; N] = std::array::from_fn(|i| (i + 1) as f32);

    // b = [129, 130, 131, ..., 256]
    let b = std::array::from_fn(|i| (N + i + 1) as f32);

    let res: SMat16<N, L> = &SVec16::new(a) * &SRowVec::new(b);

    // Reference implementation:
    //
    // M[i][j] = a[i] * b[j]
    let expected: [[f32; N]; N] = std::array::from_fn(|i| std::array::from_fn(|j| a[i] * b[j]));

    let expected: [[f32x16; L]; 128] = SMat16::new(expected).m;

    for i in 0..N {
        for j in 0..L {
            let got = res.m[i][j];
            let expected = expected[i][j];

            let diff = (got - expected).abs();
            let tolerance = 1e-5 * expected.abs().max(f32x16::splat(1.0));

            for (diff, tolerance) in zip(diff.as_array(), tolerance.as_array()) {
                assert!(
                    *diff <= *tolerance,
                    "mismatch at index {i}: got {got}, expected {expected}, \
                    diff {diff}, tolerance {tolerance}"
                );
            }
        }
    }
}

#[test]
fn test_mul_element_wise() {
    const N: usize = 128;
    const L: usize = 8;

    // a = [1, 2, 3, ..., 128]
    let a: [f32; N] = std::array::from_fn(|i| (i + 1) as f32);

    // b = [129, 130, 131, ..., 256]
    let b: [f32; N] = std::array::from_fn(|i| (N + i + 1) as f32);

    let a = SVec16::<L>::new(a);
    let b = SVec16::<L>::new(b);

    let res = &a * &b;

    // Scalar reference implementation.
    let expected: [f32; N] = std::array::from_fn(|i| (i + 1) as f32 * (N + i + 1) as f32);

    let expected: [f32x16; L] = f32x16_from(expected);

    for i in 0..L {
        let got = res.a[i];
        let expected = expected[i];

        let diff = (got - expected).abs();
        let tolerance = 1e-5 * expected.abs().max(f32x16::splat(1.0));

        for (diff, tolerance) in zip(diff.as_array(), tolerance.as_array()) {
            assert!(
                *diff <= *tolerance,
                "mismatch at index {i}: got {got}, expected {expected}, \
                diff {diff}, tolerance {tolerance}"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const N: usize = 128;
    const L: usize = N / LANES;

    #[test]
    fn s_vec_as_array() {
        let mut a = [f32x16::splat(0.0); L];

        for (i, v) in a.iter_mut().enumerate() {
            *v = f32x16::splat(i as f32);
        }

        let vec = SVec16::<L> { a };

        let array: &[f32; N] = vec.as_array();

        assert_eq!(array.len(), N);

        for i in 0..N {
            assert_eq!(array[i], (i / LANES) as f32);
        }
    }

    #[test]
    fn s_vec_as_mut_array() {
        let a = [f32x16::splat(0.0); L];
        let mut vec = SVec16::<L> { a };

        {
            let array: &mut [f32; N] = vec.as_mut_array();

            assert_eq!(array.len(), N);

            for (i, value) in array.iter_mut().enumerate() {
                *value = i as f32;
            }
        }

        let array: &[f32; N] = vec.as_array();

        for i in 0..N {
            assert_eq!(array[i], i as f32);
        }
    }

    #[test]
    fn s_row_vec_as_array() {
        let mut a = [f32x16::splat(0.0); L];

        for (i, v) in a.iter_mut().enumerate() {
            *v = f32x16::splat(i as f32);
        }

        let vec = SRowVec::<N, L> { a };

        let array = vec.as_array();

        assert_eq!(array.len(), N);

        for i in 0..N {
            assert_eq!(array[i], (i / LANES) as f32);
        }
    }

    #[test]
    fn s_row_vec_as_mut_array() {
        let a = [f32x16::splat(0.0); L];
        let mut vec = SRowVec::<N, L> { a };

        {
            let array = vec.as_mut_array();

            assert_eq!(array.len(), N);

            for (i, value) in array.iter_mut().enumerate() {
                *value = i as f32;
            }
        }

        let array = vec.as_array();

        for i in 0..N {
            assert_eq!(array[i], i as f32);
        }
    }

    #[test]
    fn s_mat_as_array() {
        let mut m = [[f32x16::splat(0.0); L]; N];

        // Give every SIMD vector a unique value so that we can
        // verify the complete memory layout.
        for row in 0..N {
            for lane in 0..L {
                m[row][lane] = f32x16::splat((row * L + lane) as f32);
            }
        }

        let mat = SMat16::<N, L> { m };

        let array = mat.as_array();

        assert_eq!(array.len(), N);
        assert_eq!(array[0].len(), N);

        for row in 0..N {
            for col in 0..N {
                let simd_index = col / LANES;
                let expected = (row * L + simd_index) as f32;

                assert_eq!(array[row][col], expected);
            }
        }
    }

    #[test]
    fn s_mat_as_mut_array() {
        let m = [[f32x16::splat(0.0); L]; N];
        let mut mat = SMat16::<N, L> { m };

        {
            let array = mat.as_mut_array();

            assert_eq!(array.len(), N);
            assert_eq!(array[0].len(), N);

            for row in 0..N {
                for col in 0..N {
                    array[row][col] = (row * N + col) as f32;
                }
            }
        }

        let array = mat.as_array();

        for row in 0..N {
            for col in 0..N {
                assert_eq!(array[row][col], (row * N + col) as f32);
            }
        }
    }
}

fn assert_close(a: f32, b: f32, tolerance: f32) {
    let diff = (a - b).abs();

    assert!(diff <= tolerance, "expected {b}, got {a}, diff {diff}");
}
