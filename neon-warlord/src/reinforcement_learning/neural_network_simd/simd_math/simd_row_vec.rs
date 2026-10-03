//! N row vector

use super::*;
use super::simd_vec::SVec16;
use super::simd_mat::SMat16;

#[derive(Debug, Clone)]
pub struct SRowVec16<const N: usize>(pub AlignedVec<N>);

impl<const N: usize> SRowVec16<N> {
    pub fn new(a: [f32; N]) -> Self {
        Self(AlignedVec(a))
    }

    pub fn zero() -> Self {
        Self::new([0.0; N])
    }

    #[allow(clippy::wrong_self_convention)]
    pub fn as_column_vec(self) -> SVec16<N> {
        SVec16(self.0)
    }
}

// Deref

impl<const N: usize> std::ops::Deref for SRowVec16<N> {
    type Target = AlignedVec<N>;

    fn deref(&self) -> &AlignedVec<N> {
        &self.0
    }
}

impl<const N: usize> std::ops::DerefMut for SRowVec16<N> {
    fn deref_mut(&mut self) -> &mut AlignedVec<N> {
        &mut self.0
    }
}

// Into Iterator

impl<const N: usize> IntoIterator for SRowVec16<N> {
    type Item = f32;
    type IntoIter = std::array::IntoIter<f32, N>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a, const N: usize> IntoIterator for &'a SRowVec16<N> {
    type Item = &'a f32;
    type IntoIter = std::slice::Iter<'a, f32>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl<'a, const N: usize> IntoIterator for &'a mut SRowVec16<N> {
    type Item = &'a mut f32;
    type IntoIter = std::slice::IterMut<'a, f32>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter_mut()
    }
}

/// Vector-matrix multiplication: `y = x^T * A`.
///
/// (1×M)(M×N) → 1×N
///
impl<const M: usize, const N: usize> Mul<&SMat16<M, N>> for &SRowVec16<M> {
    type Output = SRowVec16<N>;

    fn mul(self, rhs: &SMat16<M, N>) -> Self::Output {
        let mut res: SRowVec16<N> = SRowVec16::zero();

        let x = self;

        for (x, row) in zip(x, rhs) {
            let x_ = f32x16::splat(*x);

            for (res, row) in zip(res.simd_iter_mut(), row.simd_iter()) {
                *res += x_ * row;
            }

            for (res, row) in zip(res.remainder_mut(), row.remainder()) {
                *res += x * row;
            }
        }

        res
    }
}
