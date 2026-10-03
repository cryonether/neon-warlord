// N column vector

use itertools::izip;

use super::*;
use super::simd_row_vec::SRowVec16;
use super::simd_mat::SMat16;

#[derive(Debug, Copy, Clone)]
pub struct SVec16<const N: usize>(pub  AlignedVec<N>);

impl<const N: usize> SVec16<N> {
    pub fn new(a: [f32; N]) -> Self {
        Self(AlignedVec(a))
    }

    pub fn zero() -> Self {
        Self::new([0.0; N])
    }
    
    pub fn as_row_vec(self) -> SRowVec16<N> {
        SRowVec16(self.0)
    }

    pub fn assert_not_nan(&self, name: &str) {
        for (i, &x) in self.0.iter().enumerate() {
            assert!(
                !x.is_nan(),
                "{name}[{i}] is NaN"
            );
        }
    }

    pub fn assert_finite(&self, name: &str) {
        for (i, &x) in self.0.iter().enumerate() {
            assert!(
                x.is_finite(),
                "{name}[{i}] is not finite: {x}"
            );
        }
    }
}

// From

impl<const N: usize> From<[f32; N]> for SVec16<N> {
    fn from(value: [f32; N]) -> Self {
        SVec16(AlignedVec(value))
    }
}

// Deref

impl<const N: usize> std::ops::Deref for SVec16<N> {
    type Target = AlignedVec<N>;

    fn deref(&self) -> &AlignedVec<N> {
        &self.0
    }
}

impl<const N: usize> std::ops::DerefMut for SVec16<N> {
    fn deref_mut(&mut self) -> &mut AlignedVec<N> {
        &mut self.0
    }
}

// Into Iterator

impl<const N: usize> IntoIterator for SVec16<N> {
    type Item = f32;
    type IntoIter = std::array::IntoIter<f32, N>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a, const N: usize> IntoIterator for &'a SVec16<N> {
    type Item = &'a f32;
    type IntoIter = std::slice::Iter<'a, f32>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl<'a, const N: usize> IntoIterator for &'a mut SVec16<N> {
    type Item = &'a mut f32;
    type IntoIter = std::slice::IterMut<'a, f32>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter_mut()
    }
}

// Print

impl<const N: usize> fmt::Display for SVec16<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

///
/// Outer Product M = a * b^T
///
/// (M×1)(1×N) → M×N
///
impl<const M: usize, const N: usize> Mul<&SRowVec16<N>> for &SVec16<M> {
    type Output = SMat16<M, N>;

    fn mul(self, rhs: &SRowVec16<N>) -> Self::Output {
        let mut res = SMat16::zero();

        for (a, res_row) in std::iter::zip(self, &mut res) {
            let a_ = f32x16::splat(*a);

            for (b, res) in std::iter::zip(rhs.simd_iter(), res_row.simd_iter_mut()) {
                *res = a_ * b;
            }

            for (b, res) in std::iter::zip(rhs.remainder(), res_row.remainder_mut()) {
                *res = a * b;
            }
        }

        res
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
impl<const N: usize> Mul<&SVec16<N>> for &SVec16<N> {
    type Output = SVec16<N>;

    fn mul(self, rhs: &SVec16<N>) -> Self::Output {
        let a = self;
        let b = rhs;

        let mut res = SVec16::zero();
        for(a, b, res) in izip!(
            a.simd_iter(), 
            b.simd_iter(), 
            res.simd_iter_mut())
        {
            *res = a * b
        }

        for(a, b, res) in izip!(
            a.remainder(), 
            b.remainder(), 
            res.remainder_mut())
        {
            *res = a * b
        }

        res
    }
}

///
/// `a += b`
///
impl<const N: usize> AddAssign<&SVec16<N>> for SVec16<N> {
    fn add_assign(&mut self, rhs: &SVec16<N>) {
        for (a, b) in std::iter::zip(self.simd_iter_mut(), rhs.simd_iter()) {
            *a += *b;
        }

        for (a, b) in std::iter::zip(self.remainder_mut(), rhs.remainder()) {
            *a += *b;
        }
    }
}

///
/// `a -= b`
///
impl<const N: usize> SubAssign<&SVec16<N>> for SVec16<N> {
    fn sub_assign(&mut self, rhs: &SVec16<N>) {
        for (a, b) in std::iter::zip(self.simd_iter_mut(), rhs.simd_iter()) {
            *a -= *b;
        }

        for (a, b) in std::iter::zip(self.remainder_mut(), rhs.remainder()) {
            *a -= *b;
        }
    }
}

/// `c = a + b`
impl<const N: usize> Add<&SVec16<N>> for &SVec16<N> {
    type Output = SVec16<N>;

    fn add(self, rhs: &SVec16<N>) -> Self::Output {
        let a = self;
        let b = rhs;
        let mut res = SVec16::zero();

        for( a, b, res ) in izip!(a.simd_iter(), b.simd_iter(), res.simd_iter_mut()) {
            *res = a + b;
        }

        for( a, b, res ) in izip!(a.remainder(), b.remainder(), res.remainder_mut()) {
            *res = a + b;
        }

        res
    }
}

/// `c = a - b`
impl<const N: usize> Sub<&SVec16<N>> for &SVec16<N> {
    type Output = SVec16<N>;

    fn sub(self, rhs: &SVec16<N>) -> Self::Output {
        let a = self;
        let b = rhs;
        let mut res = SVec16::zero();

        for( a, b, res ) in izip!(a.simd_iter(), b.simd_iter(), res.simd_iter_mut()) {
            *res = a - b;
        }

        for( a, b, res ) in izip!(a.remainder(), b.remainder(), res.remainder_mut()) {
            *res = a - b;
        }

        res
    }
}