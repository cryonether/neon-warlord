//! MxN Matrix

use std::ops::MulAssign;

use itertools::izip;

use crate::reinforcement_learning::neural_network_simd::simd_math::simd_row_vec::SRowVec16;

use super::simd_vec::SVec16;
use super::*;

#[derive(Debug, Clone)]
pub struct SMat16<const M: usize, const N: usize>(pub [AlignedVec<N>; M]);

impl<const M: usize, const N: usize> SMat16<M, N> {
    pub fn new(m: [[f32; N]; M]) -> Self {
        let m_vec = std::array::from_fn(|i| AlignedVec(m[i]));

        Self(m_vec)
    }

    pub fn zero() -> Self {
        Self::new([[0.0; N]; M])
    }

    pub fn assert_not_nan(&self, name: &str) {
        for (i, row) in self.0.iter().enumerate() {
            for (j, &x) in row.0.iter().enumerate() {
                assert!(!x.is_nan(), "{name}[{i}][{j}] is NaN");
            }
        }
    }

    pub fn assert_finite(&self, name: &str) {
        for (i, row) in self.0.iter().enumerate() {
            for (j, &x) in row.0.iter().enumerate() {
                assert!(x.is_finite(), "{name}[{i}][{j}] is not finite: {x}");
            }
        }
    }
}

// From

impl<const M: usize, const N: usize> From<[[f32; N]; M]> for SMat16<M, N> {
    fn from(value: [[f32; N]; M]) -> Self {
        SMat16(value.map(AlignedVec))
    }
}

// Deref

impl<const M: usize, const N: usize> std::ops::Deref for SMat16<M, N> {
    type Target = [AlignedVec<N>; M];

    fn deref(&self) -> &[AlignedVec<N>; M] {
        &self.0
    }
}

impl<const M: usize, const N: usize> std::ops::DerefMut for SMat16<M, N> {
    fn deref_mut(&mut self) -> &mut [AlignedVec<N>; M] {
        &mut self.0
    }
}

// Into Iterator

impl<const M: usize, const N: usize> IntoIterator for SMat16<M, N> {
    type Item = AlignedVec<N>;
    type IntoIter = std::array::IntoIter<AlignedVec<N>, M>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a, const M: usize, const N: usize> IntoIterator for &'a SMat16<M, N> {
    type Item = &'a AlignedVec<N>;
    type IntoIter = std::slice::Iter<'a, AlignedVec<N>>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl<'a, const M: usize, const N: usize> IntoIterator for &'a mut SMat16<M, N> {
    type Item = &'a mut AlignedVec<N>;
    type IntoIter = std::slice::IterMut<'a, AlignedVec<N>>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter_mut()
    }
}

// Print

impl<const M: usize, const N: usize> fmt::Display for SMat16<M, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, row) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", row)?;
        }
        write!(f, "]")
    }
}

/// Matrix-vector multiplication: `y = A * x`.
///
/// (M×N)(N×1) → M×1
///
impl<const M: usize, const N: usize> Mul<&SVec16<N>> for &SMat16<M, N> {
    type Output = SVec16<M>;

    fn mul(self, rhs: &SVec16<N>) -> Self::Output {
        let a = rhs;
        let mut res = SVec16::zero();

        for (res, m) in zip(&mut res, self) {
            let mut sum = f32x16::splat(0.0);

            for (m, a) in zip(a.simd_iter(), m.simd_iter()) {
                sum += m * a;
            }

            *res = sum.reduce_add();

            for (m, a) in zip(a.remainder(), m.remainder()) {
                *res += m * a;
            }
        }

        res
    }
}

///
/// `a += b`
///
impl<const M: usize, const N: usize> AddAssign<&SMat16<M, N>> for SMat16<M, N> {
    fn add_assign(&mut self, rhs: &SMat16<M, N>) {
        for (a, b) in zip(self, rhs) {
            for (a, b) in zip(a.simd_iter_mut(), b.simd_iter()) {
                *a += *b;
            }

            for (a, b) in zip(a.remainder_mut(), b.remainder()) {
                *a += *b;
            }
        }
    }
}

///
/// `a -= b`
///
impl<const M: usize, const N: usize> SubAssign<&SMat16<M, N>> for SMat16<M, N> {
    fn sub_assign(&mut self, rhs: &SMat16<M, N>) {
        for (a, b) in zip(self, rhs) {
            for (a, b) in zip(a.simd_iter_mut(), b.simd_iter()) {
                *a -= *b;
            }

            for (a, b) in zip(a.remainder_mut(), b.remainder()) {
                *a -= *b;
            }
        }
    }
}

///
/// `a *= b`
///
impl<const M: usize, const N: usize> MulAssign<f32> for SMat16<M, N> {
    fn mul_assign(&mut self, rhs: f32) {
        let rhs_ = f32x16::splat(rhs);

        for a in self {
            for a in a.simd_iter_mut() {
                *a *= rhs_;
            }

            for a in a.remainder_mut() {
                *a *= rhs;
            }
        }
    }
}

/// `c = a + b`
impl<const M: usize, const N: usize> Add<&SMat16<M, N>> for &SMat16<M, N> {
    type Output = SMat16<M, N>;

    // #[inline]
    fn add(self, rhs: &SMat16<M, N>) -> Self::Output {
        let a = self;
        let b = rhs;
        let mut res = SMat16::zero();
        for (a, b, res) in izip!(a, b, &mut res) {
            for (a, b, res) in izip!(a.simd_iter(), b.simd_iter(), res.simd_iter_mut()) {
                *res = a + b;
            }

            for (a, b, res) in izip!(a.remainder(), b.remainder(), res.remainder_mut()) {
                *res = a + b;
            }
        }

        res
    }
}

///
/// `c = a - b`
///
impl<const M: usize, const N: usize> Sub<&SMat16<M, N>> for &SMat16<M, N> {
    type Output = SMat16<M, N>;

    fn sub(self, rhs: &SMat16<M, N>) -> Self::Output {
        let a = self;
        let b = rhs;
        let mut res = SMat16::zero();
        for (a, b, res) in izip!(a, b, &mut res) {
            for (a, b, res) in izip!(a.simd_iter(), b.simd_iter(), res.simd_iter_mut()) {
                *res = a - b;
            }

            for (a, b, res) in izip!(a.remainder(), b.remainder(), res.remainder_mut()) {
                *res = a - b;
            }
        }

        res
    }
}


// Special functions

// Outer Product a += b * c
///
/// (M×1)(1×N) → M×N
///
pub fn add_outer_product<const M: usize, const N: usize>(c: &mut SMat16<M, N>, a: &SVec16<M>, b: &SRowVec16<N>) {
    let res = c;

    for (a, res_row) in std::iter::zip(a,  res) {
        let a_ = f32x16::splat(*a);

        for (b, res) in std::iter::zip(b.simd_iter(), res_row.simd_iter_mut()) {
            *res += a_ * b;
        }

        for (b, res) in std::iter::zip(b.remainder(), res_row.remainder_mut()) {
            *res += a * b;
        }
    }   
}

