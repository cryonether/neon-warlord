//! MxN Matrix

use itertools::izip;
use winit::keyboard::KeyCode::Resume;

use super::*;
use super::simd_vec::SVec16;

#[derive(Debug, Copy, Clone)]
pub struct SMat16<const M: usize, const N: usize>(pub [AlignedVec<N>; M]);

impl<const M: usize, const N: usize> SMat16<M, N> {
    pub fn new(m: [[f32; N]; N]) -> Self {
        let m_vec = std::array::from_fn(|i| AlignedVec(m[i]));

        Self(m_vec)
    }

    pub fn zero() -> Self {
        Self::new([[0.0; N]; N])
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

            for (m, a) in zip(a.remainder(), m.remainder()) {
                sum += m * a;
            }

            *res = sum.reduce_add();
        }

        res
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

/// `c = a + b`
impl<const M: usize, const N: usize> Add<&SMat16<M, N>> for &SMat16<M, N> {
    type Output = SMat16<M, N>;

    // #[inline]
    fn add(self, rhs: &SMat16<M, N>) -> Self::Output {
        let a = self;
        let b = rhs;
        let mut res = SMat16::zero();
        for (a, b, res) in izip!(a, b, &mut res){
            for (a, b, res) in izip!(a.simd_iter(), b.simd_iter(), res.simd_iter_mut()){
                *res = a + b;
            }

            for (a, b, res) in izip!(a.remainder(), b.remainder(), res.remainder_mut()){
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
        for (a, b, res) in izip!(a, b, &mut res){
            for (a, b, res) in izip!(a.simd_iter(), b.simd_iter(), res.simd_iter_mut()){
                *res = a - b;
            }

            for (a, b, res) in izip!(a.remainder(), b.remainder(), res.remainder_mut()){
                *res = a - b;
            }
        }

        res
    }
}