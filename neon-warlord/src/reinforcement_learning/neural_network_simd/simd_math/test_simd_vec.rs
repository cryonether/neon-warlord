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
