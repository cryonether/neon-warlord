//! Tests for simd_vec

use super::simd_mat::SMat16;
use super::simd_row_vec::SRowVec16;
use super::simd_vec::SVec16;

const EPS: f32 = 1e-5;

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

fn assert_vec_eq<const N: usize>(
    a: &SVec16<N>,
    b: &SVec16<N>,
) {
    for (x, y) in a.iter().zip(b.iter()) {
        assert!(
            (x - y).abs() < EPS,
            "vector values differ: left={x}, right={y}"
        );
    }
}

fn assert_row_vec_eq<const N: usize>(
    a: &SRowVec16<N>,
    b: &SRowVec16<N>,
) {
    for (x, y) in a.iter().zip(b.iter()) {
        assert!(
            (x - y).abs() < EPS,
            "row vector values differ: left={x}, right={y}"
        );
    }
}

fn assert_matrix_eq<const M: usize, const N: usize>(
    a: &SMat16<M, N>,
    b: &SMat16<M, N>,
) {
    for (row_a, row_b) in a.iter().zip(b.iter()) {
        for (x, y) in row_a.iter().zip(row_b.iter()) {
            assert!(
                (x - y).abs() < EPS,
                "matrix values differ: left={x}, right={y}"
            );
        }
    }
}

// -----------------------------------------------------------------------------
// new()
// -----------------------------------------------------------------------------

#[test]
fn test_new() {
    let values = std::array::from_fn(|i| (i + 1) as f32);

    let vec = SVec16::<16>::new(values);

    for i in 0..16 {
        assert_eq!(vec[i], values[i]);
    }
}

// -----------------------------------------------------------------------------
// zero()
// -----------------------------------------------------------------------------

#[test]
fn test_zero() {
    let vec = SVec16::<16>::zero();

    for value in &vec {
        assert_eq!(*value, 0.0);
    }
}

// -----------------------------------------------------------------------------
// as_row_vec()
// -----------------------------------------------------------------------------

#[test]
fn test_as_row_vec() {
    let values = std::array::from_fn(|i| (i + 1) as f32);

    let vec = SVec16::<16>::new(values);
    let row = vec.as_row_vec();

    let expected = SRowVec16::<16>::new(values);

    assert_row_vec_eq(&row, &expected);
}

// -----------------------------------------------------------------------------
// Deref
// -----------------------------------------------------------------------------

#[test]
fn test_deref() {
    let mut vec = SVec16::<16>::zero();

    vec[0] = 42.0;
    vec[15] = 99.0;

    assert_eq!(vec[0], 42.0);
    assert_eq!(vec[15], 99.0);
}

// -----------------------------------------------------------------------------
// DerefMut
// -----------------------------------------------------------------------------

#[test]
fn test_deref_mut() {
    let mut vec = SVec16::<16>::zero();

    for i in 0..16 {
        vec[i] = (i * 2) as f32;
    }

    for i in 0..16 {
        assert_eq!(vec[i], (i * 2) as f32);
    }
}

// -----------------------------------------------------------------------------
// IntoIterator
// -----------------------------------------------------------------------------

#[test]
fn test_into_iterator_owned() {
    let values = std::array::from_fn(|i| (i + 1) as f32);

    let vec = SVec16::<16>::new(values);

    let result: Vec<f32> = vec.into_iter().collect();

    assert_eq!(result.len(), 16);

    for i in 0..16 {
        assert_eq!(result[i], values[i]);
    }
}

#[test]
fn test_into_iterator_shared() {
    let values = std::array::from_fn(|i| (i + 1) as f32);

    let vec = SVec16::<16>::new(values);

    let mut count = 0;

    for (i, value) in (&vec).into_iter().enumerate() {
        assert_eq!(*value, values[i]);
        count += 1;
    }

    assert_eq!(count, 16);
}

#[test]
fn test_into_iterator_mut() {
    let mut vec = SVec16::<16>::zero();

    for (i, value) in (&mut vec).into_iter().enumerate() {
        *value = (i + 1) as f32;
    }

    for i in 0..16 {
        assert_eq!(vec[i], (i + 1) as f32);
    }
}

// -----------------------------------------------------------------------------
// Outer product
// -----------------------------------------------------------------------------

#[test]
fn test_outer_product() {
    let a_values = std::array::from_fn(|i| (i + 1) as f32);

    let b_values = std::array::from_fn(|j| (j + 1) as f32);

    let a = SVec16::<16>::new(a_values);
    let b = SRowVec16::<16>::new(b_values);

    let result = &a * &b;

    let expected = SMat16::<16, 16>::new(
        std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                a_values[i] * b_values[j]
            })
        })
    );

    assert_matrix_eq(&result, &expected);
}

#[test]
fn test_outer_product_zero_vector() {
    let a = SVec16::<16>::zero();

    let b = SRowVec16::<16>::new(
        std::array::from_fn(|i| (i + 1) as f32)
    );

    let result = &a * &b;

    let expected = SMat16::<16, 16>::zero();

    assert_matrix_eq(&result, &expected);
}

#[test]
fn test_outer_product_zero_row_vector() {
    let a = SVec16::<16>::new(
        std::array::from_fn(|i| (i + 1) as f32)
    );

    let b = SRowVec16::<16>::zero();

    let result = &a * &b;

    let expected = SMat16::<16, 16>::zero();

    assert_matrix_eq(&result, &expected);
}

#[test]
fn test_outer_product_single_nonzero() {
    let mut a_values = [0.0f32; 16];
    let mut b_values = [0.0f32; 16];

    a_values[3] = 5.0;
    b_values[7] = 7.0;

    let a = SVec16::<16>::new(a_values);
    let b = SRowVec16::<16>::new(b_values);

    let result = &a * &b;

    for i in 0..16 {
        for j in 0..16 {
            let expected = if i == 3 && j == 7 {
                35.0
            } else {
                0.0
            };

            assert!(
                (result[i][j] - expected).abs() < EPS,
                "unexpected value at [{i}][{j}]: {}",
                result[i][j]
            );
        }
    }
}

// -----------------------------------------------------------------------------
// Element-wise multiplication
// -----------------------------------------------------------------------------

#[test]
fn test_elementwise_mul() {
    let a_values =
        std::array::from_fn(|i| (i + 1) as f32);

    let b_values =
        std::array::from_fn(|i| (i * 2 + 1) as f32);

    let a = SVec16::<16>::new(a_values);
    let b = SVec16::<16>::new(b_values);

    let result = &a * &b;

    let expected_values =
        std::array::from_fn(|i| a_values[i] * b_values[i]);

    let expected = SVec16::<16>::new(expected_values);

    assert_vec_eq(&result, &expected);
}

#[test]
fn test_elementwise_mul_zero() {
    let a = SVec16::<16>::new(
        std::array::from_fn(|i| (i + 1) as f32)
    );

    let b = SVec16::<16>::zero();

    let result = &a * &b;

    assert_vec_eq(
        &result,
        &SVec16::<16>::zero(),
    );
}

#[test]
fn test_elementwise_mul_one() {
    let a = SVec16::<16>::new(
        std::array::from_fn(|i| (i * 3 + 1) as f32)
    );

    let b = SVec16::<16>::new([1.0; 16]);

    let result = &a * &b;

    assert_vec_eq(&result, &a);
}

#[test]
fn test_elementwise_mul_commutative() {
    let a = SVec16::<16>::new(
        std::array::from_fn(|i| (i + 2) as f32)
    );

    let b = SVec16::<16>::new(
        std::array::from_fn(|i| (i * 2 + 1) as f32)
    );

    let ab = &a * &b;
    let ba = &b * &a;

    assert_vec_eq(&ab, &ba);
}

// -----------------------------------------------------------------------------
// Addition
// -----------------------------------------------------------------------------

#[test]
fn test_add() {
    let a_values =
        std::array::from_fn(|i| (i + 1) as f32);

    let b_values =
        std::array::from_fn(|i| (i * 2) as f32);

    let a = SVec16::<16>::new(a_values);
    let b = SVec16::<16>::new(b_values);

    let result = &a + &b;

    let expected_values =
        std::array::from_fn(|i| a_values[i] + b_values[i]);

    let expected = SVec16::<16>::new(expected_values);

    assert_vec_eq(&result, &expected);
}

#[test]
fn test_add_zero() {
    let a = SVec16::<16>::new(
        std::array::from_fn(|i| (i + 1) as f32)
    );

    let zero = SVec16::<16>::zero();

    let result = &a + &zero;

    assert_vec_eq(&result, &a);
}

#[test]
fn test_add_commutative() {
    let a = SVec16::<16>::new(
        std::array::from_fn(|i| (i + 1) as f32)
    );

    let b = SVec16::<16>::new(
        std::array::from_fn(|i| (i * 2) as f32)
    );

    let ab = &a + &b;
    let ba = &b + &a;

    assert_vec_eq(&ab, &ba);
}

// -----------------------------------------------------------------------------
// Subtraction
// -----------------------------------------------------------------------------

#[test]
fn test_sub() {
    let a_values =
        std::array::from_fn(|i| (i * 3 + 10) as f32);

    let b_values =
        std::array::from_fn(|i| (i + 2) as f32);

    let a = SVec16::<16>::new(a_values);
    let b = SVec16::<16>::new(b_values);

    let result = &a - &b;

    let expected_values =
        std::array::from_fn(|i| a_values[i] - b_values[i]);

    let expected = SVec16::<16>::new(expected_values);

    assert_vec_eq(&result, &expected);
}

#[test]
fn test_sub_zero() {
    let a = SVec16::<16>::new(
        std::array::from_fn(|i| (i + 1) as f32)
    );

    let zero = SVec16::<16>::zero();

    let result = &a - &zero;

    assert_vec_eq(&result, &a);
}

#[test]
fn test_sub_self() {
    let a = SVec16::<16>::new(
        std::array::from_fn(|i| (i + 1) as f32)
    );

    let result = &a - &a;

    assert_vec_eq(
        &result,
        &SVec16::<16>::zero(),
    );
}

// -----------------------------------------------------------------------------
// AddAssign
// -----------------------------------------------------------------------------

#[test]
fn test_add_assign() {
    let a_values =
        std::array::from_fn(|i| (i + 1) as f32);

    let b_values =
        std::array::from_fn(|i| (i * 2) as f32);

    let mut actual = SVec16::<16>::new(a_values);

    let b = SVec16::<16>::new(b_values);

    actual += &b;

    let expected_values =
        std::array::from_fn(|i| a_values[i] + b_values[i]);

    let expected = SVec16::<16>::new(expected_values);

    assert_vec_eq(&actual, &expected);
}

#[test]
fn test_add_assign_zero() {
    let mut a = SVec16::<16>::new(
        std::array::from_fn(|i| (i + 1) as f32)
    );

    let original = a.clone();

    a += &SVec16::<16>::zero();

    assert_vec_eq(&a, &original);
}

#[test]
fn test_add_assign_self() {
    let mut a = SVec16::<16>::new(
        std::array::from_fn(|i| (i + 1) as f32)
    );

    let original = a.clone();

    a += &original;

    let expected = &original + &original;

    assert_vec_eq(&a, &expected);
}

// -----------------------------------------------------------------------------
// SubAssign
// -----------------------------------------------------------------------------

#[test]
fn test_sub_assign() {
    let a_values =
        std::array::from_fn(|i| (i * 3 + 10) as f32);

    let b_values =
        std::array::from_fn(|i| (i + 2) as f32);

    let mut actual = SVec16::<16>::new(a_values);

    let b = SVec16::<16>::new(b_values);

    actual -= &b;

    let expected_values =
        std::array::from_fn(|i| a_values[i] - b_values[i]);

    let expected = SVec16::<16>::new(expected_values);

    assert_vec_eq(&actual, &expected);
}

#[test]
fn test_sub_assign_zero() {
    let mut a = SVec16::<16>::new(
        std::array::from_fn(|i| (i + 1) as f32)
    );

    let original = a.clone();

    a -= &SVec16::<16>::zero();

    assert_vec_eq(&a, &original);
}

#[test]
fn test_sub_assign_self() {
    let mut a = SVec16::<16>::new(
        std::array::from_fn(|i| (i + 1) as f32)
    );

    let original = a.clone();

    a -= &original;

    assert_vec_eq(
        &a,
        &SVec16::<16>::zero(),
    );
}

// -----------------------------------------------------------------------------
// Algebraic consistency
// -----------------------------------------------------------------------------

#[test]
fn test_add_sub_consistency() {
    let a = SVec16::<16>::new(
        std::array::from_fn(|i| (i * 3 + 1) as f32)
    );

    let b = SVec16::<16>::new(
        std::array::from_fn(|i| (i * 2 + 5) as f32)
    );

    let c = &a + &b;
    let result = &c - &b;

    assert_vec_eq(&result, &a);
}

#[test]
fn test_add_assign_matches_add() {
    let a = SVec16::<16>::new(
        std::array::from_fn(|i| (i + 1) as f32)
    );

    let b = SVec16::<16>::new(
        std::array::from_fn(|i| (i * 2) as f32)
    );

    let expected = &a + &b;

    let mut actual = a.clone();
    actual += &b;

    assert_vec_eq(&actual, &expected);
}

#[test]
fn test_sub_assign_matches_sub() {
    let a = SVec16::<16>::new(
        std::array::from_fn(|i| (i * 3 + 10) as f32)
    );

    let b = SVec16::<16>::new(
        std::array::from_fn(|i| (i + 2) as f32)
    );

    let expected = &a - &b;

    let mut actual = a.clone();
    actual -= &b;

    assert_vec_eq(&actual, &expected);
}

// -----------------------------------------------------------------------------
// Outer product consistency
// -----------------------------------------------------------------------------

#[test]
fn test_outer_product_matches_elementwise_scaling() {
    let a_values =
        std::array::from_fn(|i| (i + 1) as f32);

    let b_values =
        std::array::from_fn(|i| (i * 2 + 1) as f32);

    let a = SVec16::<16>::new(a_values);
    let b = SRowVec16::<16>::new(b_values);

    let result = &a * &b;

    for i in 0..16 {
        for j in 0..16 {
            assert!(
                (result[i][j] - a_values[i] * b_values[j]).abs() < EPS,
                "wrong outer product at [{i}][{j}]"
            );
        }
    }
}

// -----------------------------------------------------------------------------
// SIMD boundary / remainder handling
// -----------------------------------------------------------------------------

#[test]
fn test_add_with_remainder() {
    const N: usize = 17;

    let a = SVec16::<N>::new(
        std::array::from_fn(|i| (i + 1) as f32)
    );

    let b = SVec16::<N>::new(
        std::array::from_fn(|i| (i * 3) as f32)
    );

    let result = &a + &b;

    let expected = SVec16::<N>::new(
        std::array::from_fn(|i| {
            (i + 1) as f32 + (i * 3) as f32
        })
    );

    assert_vec_eq(&result, &expected);
}

#[test]
fn test_sub_with_remainder() {
    const N: usize = 17;

    let a = SVec16::<N>::new(
        std::array::from_fn(|i| (i * 4 + 10) as f32)
    );

    let b = SVec16::<N>::new(
        std::array::from_fn(|i| (i + 2) as f32)
    );

    let result = &a - &b;

    let expected = SVec16::<N>::new(
        std::array::from_fn(|i| {
            (i * 4 + 10) as f32 - (i + 2) as f32
        })
    );

    assert_vec_eq(&result, &expected);
}

#[test]
fn test_mul_with_remainder() {
    const N: usize = 17;

    let a = SVec16::<N>::new(
        std::array::from_fn(|i| (i + 1) as f32)
    );

    let b = SVec16::<N>::new(
        std::array::from_fn(|i| (i * 2 + 1) as f32)
    );

    let result = &a * &b;

    let expected = SVec16::<N>::new(
        std::array::from_fn(|i| {
            (i + 1) as f32 * (i * 2 + 1) as f32
        })
    );

    assert_vec_eq(&result, &expected);
}

#[test]
fn test_add_assign_with_remainder() {
    const N: usize = 17;

    let a_values =
        std::array::from_fn(|i| (i + 1) as f32);

    let b_values =
        std::array::from_fn(|i| (i * 2) as f32);

    let mut actual = SVec16::<N>::new(a_values);

    actual += &SVec16::<N>::new(b_values);

    let expected = SVec16::<N>::new(
        std::array::from_fn(|i| {
            a_values[i] + b_values[i]
        })
    );

    assert_vec_eq(&actual, &expected);
}

#[test]
fn test_sub_assign_with_remainder() {
    const N: usize = 17;

    let a_values =
        std::array::from_fn(|i| (i * 3 + 10) as f32);

    let b_values =
        std::array::from_fn(|i| (i + 2) as f32);

    let mut actual = SVec16::<N>::new(a_values);

    actual -= &SVec16::<N>::new(b_values);

    let expected = SVec16::<N>::new(
        std::array::from_fn(|i| {
            a_values[i] - b_values[i]
        })
    );

    assert_vec_eq(&actual, &expected);
}

// -----------------------------------------------------------------------------
// Exact SIMD boundaries
// -----------------------------------------------------------------------------

#[test]
fn test_exactly_one_simd_block() {
    const N: usize = 16;

    let a = SVec16::<N>::new([2.0; N]);
    let b = SVec16::<N>::new([3.0; N]);

    let result = &a * &b;

    assert_vec_eq(
        &result,
        &SVec16::<N>::new([6.0; N]),
    );
}

#[test]
fn test_two_simd_blocks() {
    const N: usize = 32;

    let a = SVec16::<N>::new(
        std::array::from_fn(|i| (i + 1) as f32)
    );

    let b = SVec16::<N>::new([2.0; N]);

    let result = &a * &b;

    let expected = SVec16::<N>::new(
        std::array::from_fn(|i| ((i + 1) * 2) as f32)
    );

    assert_vec_eq(&result, &expected);
}

// -----------------------------------------------------------------------------
// Remainder sizes
// -----------------------------------------------------------------------------

#[test]
fn test_remainder_one_element() {
    const N: usize = 17;

    let a = SVec16::<N>::new(
        std::array::from_fn(|i| (i + 1) as f32)
    );

    let b = SVec16::<N>::new([2.0; N]);

    let result = &a + &b;

    assert_eq!(result[15], 18.0);
    assert_eq!(result[16], 19.0);
}

#[test]
fn test_remainder_fifteen_elements() {
    const N: usize = 31;

    let a = SVec16::<N>::new([1.0; N]);
    let b = SVec16::<N>::new([2.0; N]);

    let result = &a + &b;

    for i in 0..N {
        assert_eq!(result[i], 3.0);
    }
}

#[test]
fn test_remainder_one_element_subtraction() {
    const N: usize = 17;

    let a = SVec16::<N>::new([10.0; N]);
    let b = SVec16::<N>::new([3.0; N]);

    let result = &a - &b;

    assert_eq!(result[15], 7.0);
    assert_eq!(result[16], 7.0);
}

#[test]
fn test_remainder_one_element_multiplication() {
    const N: usize = 17;

    let a = SVec16::<N>::new([4.0; N]);
    let b = SVec16::<N>::new([5.0; N]);

    let result = &a * &b;

    assert_eq!(result[15], 20.0);
    assert_eq!(result[16], 20.0);
}

// -----------------------------------------------------------------------------
// Small vectors
// -----------------------------------------------------------------------------

#[test]
fn test_single_element_vector() {
    let a = SVec16::<1>::new([6.0]);
    let b = SVec16::<1>::new([7.0]);

    assert_vec_eq(
        &(&a + &b),
        &SVec16::<1>::new([13.0]),
    );

    assert_vec_eq(
        &(&a - &b),
        &SVec16::<1>::new([-1.0]),
    );

    assert_vec_eq(
        &(&a * &b),
        &SVec16::<1>::new([42.0]),
    );
}

#[test]
fn test_fifteen_element_vector() {
    const N: usize = 15;

    let a = SVec16::<N>::new(
        std::array::from_fn(|i| (i + 1) as f32)
    );

    let b = SVec16::<N>::new([2.0; N]);

    let result = &a * &b;

    let expected = SVec16::<N>::new(
        std::array::from_fn(|i| ((i + 1) * 2) as f32)
    );

    assert_vec_eq(&result, &expected);
}

#[test]
fn test_seventeen_element_vector() {
    const N: usize = 17;

    let a = SVec16::<N>::new(
        std::array::from_fn(|i| (i + 1) as f32)
    );

    let b = SVec16::<N>::new([2.0; N]);

    let result = &a * &b;

    let expected = SVec16::<N>::new(
        std::array::from_fn(|i| ((i + 1) * 2) as f32)
    );

    assert_vec_eq(&result, &expected);
}

// -----------------------------------------------------------------------------
// Negative values
// -----------------------------------------------------------------------------

#[test]
fn test_negative_values_add() {
    let a = SVec16::<16>::new([-2.0; 16]);
    let b = SVec16::<16>::new([5.0; 16]);

    let result = &a + &b;

    assert_vec_eq(
        &result,
        &SVec16::<16>::new([3.0; 16]),
    );
}

#[test]
fn test_negative_values_sub() {
    let a = SVec16::<16>::new([-2.0; 16]);
    let b = SVec16::<16>::new([5.0; 16]);

    let result = &a - &b;

    assert_vec_eq(
        &result,
        &SVec16::<16>::new([-7.0; 16]),
    );
}

#[test]
fn test_negative_values_mul() {
    let a = SVec16::<16>::new([-2.0; 16]);
    let b = SVec16::<16>::new([5.0; 16]);

    let result = &a * &b;

    assert_vec_eq(
        &result,
        &SVec16::<16>::new([-10.0; 16]),
    );
}

#[test]
fn test_mixed_signs() {
    let a = SVec16::<16>::new(
        std::array::from_fn(|i| {
            if i % 2 == 0 {
                i as f32
            } else {
                -(i as f32)
            }
        })
    );

    let b = SVec16::<16>::new([2.0; 16]);

    let result = &a * &b;

    let expected = SVec16::<16>::new(
        std::array::from_fn(|i| {
            let value = i as f32;

            if i % 2 == 0 {
                value * 2.0
            } else {
                -value * 2.0
            }
        })
    );

    assert_vec_eq(&result, &expected);
}

// -----------------------------------------------------------------------------
// SIMD mutation
// -----------------------------------------------------------------------------

#[test]
fn test_simd_iter_mut() {
    const N: usize = 32;

    let mut vec = SVec16::<N>::zero();

    for chunk in vec.simd_iter_mut() {
        *chunk += wide::f32x16::splat(5.0);
    }

    for value in &vec {
        assert_eq!(*value, 5.0);
    }
}

#[test]
fn test_simd_iter_mut_does_not_touch_remainder() {
    const N: usize = 17;

    let mut vec = SVec16::<N>::new([1.0; N]);

    for chunk in vec.simd_iter_mut() {
        *chunk *= wide::f32x16::splat(10.0);
    }

    for i in 0..16 {
        assert_eq!(vec[i], 10.0);
    }

    // The scalar remainder must not be included in simd_iter_mut().
    assert_eq!(vec[16], 1.0);
}

// -----------------------------------------------------------------------------
// Remainder API
// -----------------------------------------------------------------------------

#[test]
fn test_remainder_empty_for_exact_simd_size() {
    let vec = SVec16::<16>::new([1.0; 16]);

    assert!(vec.remainder().is_empty());
}

#[test]
fn test_remainder_empty_for_multiple_of_simd_width() {
    let vec = SVec16::<32>::new([1.0; 32]);

    assert!(vec.remainder().is_empty());
}

#[test]
fn test_remainder_length() {
    let vec = SVec16::<37>::new([1.0; 37]);

    assert_eq!(vec.remainder().len(), 5);
}

#[test]
fn test_remainder_values() {
    let vec = SVec16::<19>::new(
        std::array::from_fn(|i| (i + 1) as f32)
    );

    assert_eq!(
        vec.remainder(),
        &[17.0, 18.0, 19.0]
    );
}

// -----------------------------------------------------------------------------
// Non-square outer products
// -----------------------------------------------------------------------------

#[test]
fn test_outer_product_non_square() {
    const M: usize = 17;
    const N: usize = 19;

    let a_values =
        std::array::from_fn(|i| (i + 1) as f32);

    let b_values =
        std::array::from_fn(|i| (i + 1) as f32);

    let a = SVec16::<M>::new(a_values);
    let b = SRowVec16::<N>::new(b_values);

    let result = &a * &b;

    for i in 0..M {
        for j in 0..N {
            let expected = a_values[i] * b_values[j];

            assert!(
                (result[i][j] - expected).abs() < EPS,
                "wrong value at [{i}][{j}]: got {}, expected {}",
                result[i][j],
                expected
            );
        }
    }
}

#[test]
fn test_outer_product_with_remainder_columns() {
    const M: usize = 16;
    const N: usize = 17;

    let a = SVec16::<M>::new([2.0; M]);
    let b = SRowVec16::<N>::new([3.0; N]);

    let result = &a * &b;

    for i in 0..M {
        for j in 0..N {
            assert_eq!(result[i][j], 6.0);
        }
    }
}

#[test]
fn test_outer_product_with_remainder_rows() {
    const M: usize = 17;
    const N: usize = 16;

    let a = SVec16::<M>::new([2.0; M]);
    let b = SRowVec16::<N>::new([3.0; N]);

    let result = &a * &b;

    for i in 0..M {
        for j in 0..N {
            assert_eq!(result[i][j], 6.0);
        }
    }
}

#[test]
fn test_outer_product_with_remainder_rows_and_columns() {
    const M: usize = 17;
    const N: usize = 19;

    let a = SVec16::<M>::new([2.0; M]);
    let b = SRowVec16::<N>::new([3.0; N]);

    let result = &a * &b;

    for i in 0..M {
        for j in 0..N {
            assert_eq!(result[i][j], 6.0);
        }
    }
}

// -----------------------------------------------------------------------------
// Algebraic properties
// -----------------------------------------------------------------------------

#[test]
fn test_add_associativity() {
    let a = SVec16::<17>::new([1.0; 17]);
    let b = SVec16::<17>::new([2.0; 17]);
    let c = SVec16::<17>::new([3.0; 17]);

    let lhs = &(&a + &b) + &c;
    let rhs = &a + &(&b + &c);

    assert_vec_eq(&lhs, &rhs);
}

#[test]
fn test_additive_identity() {
    let a = SVec16::<17>::new(
        std::array::from_fn(|i| (i + 1) as f32)
    );

    let zero = SVec16::<17>::zero();

    let lhs = &a + &zero;
    let rhs = &zero + &a;

    assert_vec_eq(&lhs, &a);
    assert_vec_eq(&rhs, &a);
}

#[test]
fn test_subtraction_inverse() {
    let a = SVec16::<17>::new(
        std::array::from_fn(|i| (i + 1) as f32)
    );

    let b = SVec16::<17>::new(
        std::array::from_fn(|i| (i * 2) as f32)
    );

    let result = &(&a - &b) + &b;

    assert_vec_eq(&result, &a);
}

#[test]
fn test_elementwise_mul_distributive_over_add() {
    let a = SVec16::<17>::new(
        std::array::from_fn(|i| (i + 1) as f32)
    );

    let b = SVec16::<17>::new(
        std::array::from_fn(|i| (i * 2) as f32)
    );

    let c = SVec16::<17>::new(
        std::array::from_fn(|i| (i * 3 + 1) as f32)
    );

    let lhs = &a * &(&b + &c);
    let rhs = &(&a * &b) + &(&a * &c);

    assert_vec_eq(&lhs, &rhs);
}

// -----------------------------------------------------------------------------
// Larger vector
// -----------------------------------------------------------------------------

#[test]
fn test_large_vector() {
    const N: usize = 1001;

    let a = SVec16::<N>::new(
        std::array::from_fn(|i| (i % 17) as f32)
    );

    let b = SVec16::<N>::new(
        std::array::from_fn(|i| (i % 11) as f32)
    );

    let result = &a * &b;

    for i in 0..N {
        let expected =
            (i % 17) as f32 * (i % 11) as f32;

        assert!(
            (result[i] - expected).abs() < EPS,
            "wrong value at index {i}: got {}, expected {}",
            result[i],
            expected
        );
    }
}
