//! Implements matrix multiplication with simd operations

use std::{iter::zip, ops::Mul};
use std::ops::{Add, AddAssign, Sub};

use wide::f32x16;

const N: usize = 128;
const LANES: usize = 16;
// const L: usize = N/LANES;

pub struct SMat<const N: usize, const L: usize> {
    pub m: [[f32; N]; N]
}

#[derive(Debug, Clone)]
pub struct SVec<const N: usize, const L: usize> {
    pub a: [f32; N]
}

pub struct SRowVec<const N: usize, const L: usize> {
    pub a: [f32; N]
}


impl<const N: usize, const L: usize> SMat<N, L> {
    #[inline]
    pub fn new(m: [[f32; N]; N]) -> Self {
        Self { m }
    }
}

impl<const N: usize, const L: usize> SVec<N, L> {
    #[inline]
    pub fn new(a: [f32; N]) -> Self {
        Self { a }
    }
}

impl<const N: usize, const L: usize> SRowVec<N, L> {
    #[inline]
    pub fn new(a: [f32; N]) -> Self {
        Self { a }
    }
}


impl<const N: usize, const L: usize> From<SVec<N, L>> for [f32; N] {
    #[inline]
    fn from(v: SVec<N, L>) -> Self {
        v.a
    }
}

impl<const N: usize, const L: usize> From<SRowVec<N, L>> for [f32; N] {
    #[inline]
    fn from(v: SRowVec<N, L>) -> Self {
        v.a
    }
}

impl<const N: usize, const L: usize> From<SMat<N, L>> for [[f32; N]; N] {
    #[inline]
    fn from(m: SMat<N, L>) -> Self {
        m.m
    }
}

impl<const N: usize, const L: usize> From<[f32; N]> for SVec<N, L> {
    #[inline]
    fn from(a: [f32; N]) -> Self {
        Self::new(a)
    }
}

impl<const N: usize, const L: usize> From<[f32; N]> for SRowVec<N, L> {
    #[inline]
    fn from(a: [f32; N]) -> Self {
        Self::new(a)
    }
}

impl<const N: usize, const L: usize> From<[[f32; N]; N]> for SMat<N, L> {
    #[inline]
    fn from(m: [[f32; N]; N]) -> Self {
        Self::new(m)
    }
}



/// Matrix-vector multiplication: `y = A * x`.
/// 
/// (N×N)(N×1) → N×1
///
impl<const N: usize, const L: usize> Mul<&SVec<N, L>> for SMat<N, L> {
    type Output = SVec<N, L>;

    // // 160 ups
    // #[inline]
    // fn mul(self, rhs: &SVec<N, L>) -> Self::Output {
    //     let a = f32x16_from::<N, L>(rhs.a);
    //     let mut res = [0.0f32; N];

    //     for (res, m) in std::iter::zip(&mut res, &self.m) {
    //         let m = f32x16_from::<N, L>(*m);

    //         let mut sum = 0.0f32;

    //         for i in 0..L {
    //             sum += (m[i] * a[i]).reduce_add();
    //         }

    //         *res = sum;
    //     }

    //     SVec { a: res }
    // }

    // 186 ups
    #[inline]
    fn mul(self, rhs: &SVec<N, L>) -> Self::Output {
        let a = f32x16_from::<N, L>(rhs.a);
        let mut res = [0.0f32; N];

        for (res, m) in std::iter::zip(&mut res, &self.m) {
            let m = f32x16_from::<N, L>(*m);

            let mut sum = f32x16::splat(0.0);

            for i in 0..L {
                sum += m[i] * a[i];
            }

            *res = sum.reduce_add();
        }

        SVec { a: res }
    }
}



/// Vector-matrix multiplication: `y = x^T * A`.
/// 
/// (1×N)(N×N) → 1×N
///
impl<const N: usize, const L: usize> Mul<SMat<N, L>> for SRowVec<N, L> {
    type Output = SRowVec<N, L>;

    #[inline]
    fn mul(self, rhs: SMat<N, L>) -> Self::Output {
        let mut res = [f32x16::ZERO; L];

        for (x, row) in zip(self.a, &rhs.m) {
            let x = f32x16::splat(x);
            let row = f32x16_from::<N, L>(*row);

            for i in 0..L {
                res[i] += x * row[i];
            }
        }

        SRowVec {
            a: f32x16_to::<N, L>(res),
        }
    }
}

///
/// Outer Product M = a * b^T
/// 
/// (N×1)(1×N) → N×N
///
impl<const N: usize, const L: usize> Mul<SRowVec<N, L>> for SVec<N, L> {
    type Output = SMat<N, L>;

    #[inline]
    fn mul(self, rhs: SRowVec<N, L>) -> Self::Output {
        let b = f32x16_from::<N, L>(rhs.a);

        let m = std::array::from_fn(|i| {
            let a = f32x16::splat(self.a[i]);

            let row = std::array::from_fn(|j| a * b[j]);

            f32x16_to::<N, L>(row)
        });

        SMat { m }
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
impl<const N: usize, const L: usize> Mul<&SVec<N, L>> for SVec<N, L> {
    type Output = SVec<N, L>;

    #[inline]
    fn mul(self, rhs: &SVec<N, L>) -> Self::Output {
        let a = f32x16_from::<N, L>(self.a);
        let b = f32x16_from::<N, L>(rhs.a);

        let res = std::array::from_fn(|i| a[i] * b[i]);

        SVec {
            a: f32x16_to::<N, L>(res),
        }
    }
}

/// 
/// `a += b`
///
impl<const N: usize, const L: usize> AddAssign<&SVec<N, L>> for SVec<N, L> {
    #[inline]
    fn add_assign(&mut self, rhs: &SVec<N, L>) {
        for (a, b) in std::iter::zip(&mut self.a, &rhs.a) {
            *a += *b;
        }
    }
}

// /// 
// /// `a -= b`
// ///
// impl<const N: usize, const L: usize> SubAssign<&SMat<N, L>> for SMat<N, L> {
//     #[inline]
//     fn sub_assign(&mut self, rhs: &SMat<N, L>) {
//         for (a_row, b_row) in zip(&mut self.m, &rhs.m) {
//             for (a, b) in zip(a_row, b_row) {
//                 *a -= *b;
//             }
//         }
//     }
// }

/// `c = a + b`
impl<const N: usize, const L: usize> Add<&SVec<N, L>> for SVec<N, L> {
    type Output = SVec<N, L>;

    #[inline]
    fn add(self, rhs: &SVec<N, L>) -> Self::Output {
        let a = f32x16_from::<N, L>(self.a);
        let b = f32x16_from::<N, L>(rhs.a);

        let c = std::array::from_fn(|i| a[i] + b[i]);

        SVec {
            a: f32x16_to::<N, L>(c),
        }
    }
}

/// 
/// `c = a - b`
///
impl<const N: usize, const L: usize> Sub<&SMat<N, L>> for SMat<N, L> {
    type Output = SMat<N, L>;

    #[inline]
    fn sub(self, rhs: &SMat<N, L>) -> Self::Output {
        let m = std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                self.m[i][j] - rhs.m[i][j]
            })
        });

        SMat { m }
    }
}

/// `c = a - b`
impl<const N: usize, const L: usize> Sub<&SVec<N, L>> for SVec<N, L> {
    type Output = SVec<N, L>;

    #[inline]
    fn sub(self, rhs: &SVec<N, L>) -> Self::Output {
        let a = f32x16_from::<N, L>(self.a);
        let b = f32x16_from::<N, L>(rhs.a);

        let c = std::array::from_fn(|i| a[i] - b[i]);

        SVec {
            a: f32x16_to::<N, L>(c),
        }
    }
}


fn f32x16_from<const N: usize, const L: usize>(x: [f32; N]) -> [f32x16; L] {
    assert_eq!(N, L * LANES);

    let mut res = [f32x16::ZERO; L];

    for (x, res) in zip( x.as_chunks::<LANES>().0, &mut res) {
        *res = f32x16::from(*x);
    }

    res
}

#[inline]
fn f32x16_to<const N: usize, const L: usize>(
    x: [f32x16; L],
) -> [f32; N] {
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
    let m = std::array::from_fn(|i| {
        std::array::from_fn(|j| (i * N + j) as f32)
    });

    // x = [1, 2, 3, ..., 128]
    let a = std::array::from_fn(|i| (i + 1) as f32);

    let res: SVec<N, L> = SMat { m } * &SVec { a };

    // Reference implementation.
    let expected: [f32; N] = std::array::from_fn(|i| {
        (0..N)
            .map(|j| m[i][j] * a[j])
            .sum::<f32>()
    });

    // println!("res: {:?}", res);


    for i in 0..N {
        let got = res.a[i];
        let expected = expected[i];

        let diff = (got - expected).abs();
        let tolerance = 1e-5 * expected.abs().max(1.0);

        assert!(
            diff <= tolerance,
            "mismatch at row {i}: got {got}, expected {expected}, diff {diff}, tolerance {tolerance}"
        );
    }
}


#[test]
fn test_mul_vec_mat() {
    const N: usize = 128;
    const L: usize = 8;

    // A[i][j] = i * N + j
    let m = std::array::from_fn(|i| {
        std::array::from_fn(|j| (i * N + j) as f32)
    });

    // x = [1, 2, 3, ..., 128]
    let a = std::array::from_fn(|i| (i + 1) as f32);

    let res: SRowVec<N, L> = SRowVec { a } * SMat { m };

    // Reference implementation:
    //
    // y[j] = sum_i x[i] * A[i][j]
    let expected: [f32; N] = std::array::from_fn(|j| {
        (0..N)
            .map(|i| a[i] * m[i][j])
            .sum::<f32>()
    });

    for i in 0..N {
        let got = res.a[i];
        let expected = expected[i];

        let diff = (got - expected).abs();
        let tolerance = 1e-5 * expected.abs().max(1.0);

        assert!(
            diff <= tolerance,
            "mismatch at column {i}: got {got}, expected {expected}, diff {diff}, tolerance {tolerance}"
        );
    }
}

#[test]
fn test_outer_product() {
    const N: usize = 128;
    const L: usize = 8;

    // a = [1, 2, 3, ..., 128]
    let a = std::array::from_fn(|i| (i + 1) as f32);

    // b = [129, 130, 131, ..., 256]
    let b = std::array::from_fn(|i| (N + i + 1) as f32);

    let res: SMat<N, L> = SVec { a } * SRowVec { a: b };

    // Reference implementation:
    //
    // M[i][j] = a[i] * b[j]
    let expected: [[f32; N]; N] = std::array::from_fn(|i| {
        std::array::from_fn(|j| a[i] * b[j])
    });

    for i in 0..N {
        for j in 0..N {
            let got = res.m[i][j];
            let expected = expected[i][j];

            let diff = (got - expected).abs();
            let tolerance = 1e-5 * expected.abs().max(1.0);

            assert!(
                diff <= tolerance,
                "mismatch at [{i}][{j}]: got {got}, expected {expected}, \
                 diff {diff}, tolerance {tolerance}"
            );
        }
    }
}


#[test]
fn test_mul_element_wise() {
    const N: usize = 128;
    const L: usize = 8;

    // a = [1, 2, 3, ..., 128]
    let a = std::array::from_fn(|i| (i + 1) as f32);

    // b = [129, 130, 131, ..., 256]
    let b = std::array::from_fn(|i| (N + i + 1) as f32);

    let a = SVec::<N, L>::new(a);
    let b = SVec::<N, L>::new(b);

    let res = a * &b;

    // Scalar reference implementation.
    let expected: [f32; N] = std::array::from_fn(|i| {
        (i + 1) as f32 * (N + i + 1) as f32
    });

    for i in 0..N {
        let got = res.a[i];
        let expected = expected[i];

        let diff = (got - expected).abs();
        let tolerance = 1e-5 * expected.abs().max(1.0);

        assert!(
            diff <= tolerance,
            "mismatch at index {i}: got {got}, expected {expected}, \
             diff {diff}, tolerance {tolerance}"
        );
    }
}



fn assert_close(a: f32, b: f32, tolerance: f32) {
    let diff = (a - b).abs();

    assert!(
        diff <= tolerance,
        "expected {b}, got {a}, diff {diff}"
    );
}