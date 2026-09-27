//! Tests for simd_row_vec

use super::simd_mat::SMat16;
use super::simd_row_vec::SRowVec16;
use super::simd_vec::SVec16;

const EPS: f32 = 1e-5;

fn assert_row_vec_eq<const N: usize>(
    a: &SRowVec16<N>,
    b: &SRowVec16<N>,
) {
    for (x, y) in a.iter().zip(b.iter()) {
        assert!(
            (x - y).abs() < EPS,
            "vector values differ: left={x}, right={y}"
        );
    }
}

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

// -----------------------------------------------------------------------------
// new()
// -----------------------------------------------------------------------------

#[test]
fn test_new() {
    let values = std::array::from_fn(|i| (i + 1) as f32);

    let vec = SRowVec16::<16>::new(values);

    for i in 0..16 {
        assert_eq!(vec[i], values[i]);
    }
}

// -----------------------------------------------------------------------------
// zero()
// -----------------------------------------------------------------------------

#[test]
fn test_zero() {
    let vec = SRowVec16::<16>::zero();

    for value in &vec {
        assert_eq!(*value, 0.0);
    }
}

// -----------------------------------------------------------------------------
// as_column_vec()
// -----------------------------------------------------------------------------

#[test]
fn test_as_column_vec() {
    let values = std::array::from_fn(|i| (i + 1) as f32);

    let row = SRowVec16::<16>::new(values);
    let column = row.as_column_vec();

    let expected = SVec16::<16>::new(values);

    assert_vec_eq(&column, &expected);
}

// -----------------------------------------------------------------------------
// Deref
// -----------------------------------------------------------------------------

#[test]
fn test_deref() {
    let mut vec = SRowVec16::<16>::zero();

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
    let mut vec = SRowVec16::<16>::zero();

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

    let vec = SRowVec16::<16>::new(values);

    let result: Vec<f32> = vec.into_iter().collect();

    assert_eq!(result.len(), 16);

    for i in 0..16 {
        assert_eq!(result[i], values[i]);
    }
}

#[test]
fn test_into_iterator_shared() {
    let values = std::array::from_fn(|i| (i + 1) as f32);

    let vec = SRowVec16::<16>::new(values);

    let mut count = 0;

    for (i, value) in (&vec).into_iter().enumerate() {
        assert_eq!(*value, values[i]);
        count += 1;
    }

    assert_eq!(count, 16);
}

#[test]
fn test_into_iterator_mut() {
    let mut vec = SRowVec16::<16>::zero();

    for (i, value) in (&mut vec).into_iter().enumerate() {
        *value = (i + 1) as f32;
    }

    for i in 0..16 {
        assert_eq!(vec[i], (i + 1) as f32);
    }
}

// -----------------------------------------------------------------------------
// Vector-matrix multiplication
// -----------------------------------------------------------------------------

#[test]
fn test_matrix_mul_identity() {
    let mut matrix = [[0.0f32; 16]; 16];

    for i in 0..16 {
        matrix[i][i] = 1.0;
    }

    let mat = SMat16::<16, 16>::new(matrix);

    let values = std::array::from_fn(|i| (i + 1) as f32);
    let row = SRowVec16::<16>::new(values);

    let result = &row * &mat;

    let expected = SRowVec16::<16>::new(values);

    assert_row_vec_eq(&result, &expected);
}

#[test]
fn test_matrix_mul_zero_matrix() {
    let mat = SMat16::<16, 16>::zero();

    let values = std::array::from_fn(|i| (i + 1) as f32);
    let row = SRowVec16::<16>::new(values);

    let result = &row * &mat;

    let expected = SRowVec16::<16>::zero();

    assert_row_vec_eq(&result, &expected);
}

#[test]
fn test_matrix_mul_known_values() {
    let matrix = std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            (i + j + 1) as f32
        })
    });

    let mat = SMat16::<16, 16>::new(matrix);

    let values = std::array::from_fn(|i| (i + 1) as f32);
    let row = SRowVec16::<16>::new(values);

    let result = &row * &mat;

    // Expected:
    //
    // result[j] = sum_i(row[i] * matrix[i][j])
    //
    let expected_values = std::array::from_fn(|j| {
        (0..16)
            .map(|i| values[i] * matrix[i][j])
            .sum::<f32>()
    });

    let expected = SRowVec16::<16>::new(expected_values);

    assert_row_vec_eq(&result, &expected);
}

#[test]
fn test_matrix_mul_single_nonzero() {
    let mut matrix = [[0.0f32; 16]; 16];

    // Only row 3 contributes.
    for j in 0..16 {
        matrix[3][j] = (j + 1) as f32;
    }

    let mat = SMat16::<16, 16>::new(matrix);

    let mut values = [0.0f32; 16];
    values[3] = 5.0;

    let row = SRowVec16::<16>::new(values);

    let result = &row * &mat;

    let expected_values =
        std::array::from_fn(|j| 5.0 * (j + 1) as f32);

    let expected = SRowVec16::<16>::new(expected_values);

    assert_row_vec_eq(&result, &expected);
}

#[test]
fn test_matrix_mul_single_nonzero_column() {
    let mut matrix = [[0.0f32; 16]; 16];

    // Only column 7 contributes.
    for i in 0..16 {
        matrix[i][7] = (i + 1) as f32;
    }

    let mat = SMat16::<16, 16>::new(matrix);

    let values = std::array::from_fn(|i| (i + 1) as f32);
    let row = SRowVec16::<16>::new(values);

    let result = &row * &mat;

    let expected_sum = (0..16)
        .map(|i| values[i] * matrix[i][7])
        .sum::<f32>();

    assert!(
        (result[7] - expected_sum).abs() < EPS,
        "expected {}, got {}",
        expected_sum,
        result[7]
    );

    for j in 0..16 {
        if j != 7 {
            assert_eq!(result[j], 0.0);
        }
    }
}

// -----------------------------------------------------------------------------
// Row-vector / column-vector conversion
// -----------------------------------------------------------------------------

#[test]
fn test_row_column_round_trip() {
    let values = std::array::from_fn(|i| (i * 3 + 1) as f32);

    let row = SRowVec16::<16>::new(values);
    let column = row.as_column_vec();

    let expected = SVec16::<16>::new(values);

    assert_vec_eq(&column, &expected);
}

// -----------------------------------------------------------------------------
// Algebraic consistency
// -----------------------------------------------------------------------------

#[test]
fn test_row_vector_mul_identity_is_unchanged() {
    let mut matrix = [[0.0f32; 16]; 16];

    for i in 0..16 {
        matrix[i][i] = 1.0;
    }

    let mat = SMat16::<16, 16>::new(matrix);

    let values = std::array::from_fn(|i| {
        (i * 7 + 3) as f32
    });

    let row = SRowVec16::<16>::new(values);

    let result = &row * &mat;

    assert_row_vec_eq(&result, &row);
}

#[test]
fn test_row_vector_mul_zero_is_zero() {
    let row = SRowVec16::<16>::new(
        std::array::from_fn(|i| (i + 1) as f32)
    );

    let zero = SMat16::<16, 16>::zero();

    let result = &row * &zero;

    assert_row_vec_eq(
        &result,
        &SRowVec16::<16>::zero(),
    );
}
