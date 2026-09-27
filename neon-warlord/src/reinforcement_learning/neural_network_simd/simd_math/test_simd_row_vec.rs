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

// -----------------------------------------------------------------------------
// SIMD boundary / remainder handling
// -----------------------------------------------------------------------------

#[test]
fn test_matrix_mul_with_remainder_columns() {
    const M: usize = 16;
    const N: usize = 17;

    let matrix = std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            (i + j + 1) as f32
        })
    });

    let mat = SMat16::<M, N>::new(matrix);

    let values =
        std::array::from_fn(|i| (i + 1) as f32);

    let row = SRowVec16::<M>::new(values);

    let result = &row * &mat;

    let expected_values = std::array::from_fn(|j| {
        (0..M)
            .map(|i| values[i] * matrix[i][j])
            .sum::<f32>()
    });

    let expected = SRowVec16::<N>::new(expected_values);

    assert_row_vec_eq(&result, &expected);
}

#[test]
fn test_matrix_mul_with_remainder_columns_large() {
    const M: usize = 17;
    const N: usize = 19;

    let matrix = std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            ((i + 1) * (j + 2)) as f32
        })
    });

    let mat = SMat16::<M, N>::new(matrix);

    let values =
        std::array::from_fn(|i| (i + 1) as f32);

    let row = SRowVec16::<M>::new(values);

    let result = &row * &mat;

    let expected_values = std::array::from_fn(|j| {
        (0..M)
            .map(|i| values[i] * matrix[i][j])
            .sum::<f32>()
    });

    let expected = SRowVec16::<N>::new(expected_values);

    assert_row_vec_eq(&result, &expected);
}

#[test]
fn test_matrix_mul_exactly_one_simd_block() {
    const M: usize = 16;
    const N: usize = 16;

    let matrix = std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            if i == j { 1.0 } else { 0.0 }
        })
    });

    let mat = SMat16::<M, N>::new(matrix);

    let row = SRowVec16::<M>::new(
        std::array::from_fn(|i| (i + 1) as f32)
    );

    let result = &row * &mat;

    assert_row_vec_eq(&result, &row);
}

#[test]
fn test_matrix_mul_two_simd_blocks() {
    const M: usize = 32;
    const N: usize = 32;

    let matrix = std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            if i == j { 1.0 } else { 0.0 }
        })
    });

    let mat = SMat16::<M, N>::new(matrix);

    let row = SRowVec16::<M>::new(
        std::array::from_fn(|i| (i + 1) as f32)
    );

    let result = &row * &mat;

    assert_row_vec_eq(&result, &row);
}

// -----------------------------------------------------------------------------
// Matrix dimensions around SIMD boundary
// -----------------------------------------------------------------------------

#[test]
fn test_matrix_mul_15_by_15() {
    const M: usize = 15;
    const N: usize = 15;

    let matrix = std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            (i + j + 1) as f32
        })
    });

    let mat = SMat16::<M, N>::new(matrix);

    let values =
        std::array::from_fn(|i| (i + 1) as f32);

    let row = SRowVec16::<M>::new(values);

    let result = &row * &mat;

    let expected_values = std::array::from_fn(|j| {
        (0..M)
            .map(|i| values[i] * matrix[i][j])
            .sum::<f32>()
    });

    let expected = SRowVec16::<N>::new(expected_values);

    assert_row_vec_eq(&result, &expected);
}

#[test]
fn test_matrix_mul_17_by_17() {
    const M: usize = 17;
    const N: usize = 17;

    let matrix = std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            (i + j + 1) as f32
        })
    });

    let mat = SMat16::<M, N>::new(matrix);

    let values =
        std::array::from_fn(|i| (i + 1) as f32);

    let row = SRowVec16::<M>::new(values);

    let result = &row * &mat;

    let expected_values = std::array::from_fn(|j| {
        (0..M)
            .map(|i| values[i] * matrix[i][j])
            .sum::<f32>()
    });

    let expected = SRowVec16::<N>::new(expected_values);

    assert_row_vec_eq(&result, &expected);
}

// -----------------------------------------------------------------------------
// Special matrix structures
// -----------------------------------------------------------------------------

#[test]
fn test_matrix_mul_diagonal() {
    const N: usize = 17;

    let diagonal: [f32; N] =
        std::array::from_fn(|i| (i + 2) as f32);

    let matrix = std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            if i == j {
                diagonal[i]
            } else {
                0.0
            }
        })
    });

    let mat = SMat16::<N, N>::new(matrix);

    let row = SRowVec16::<N>::new(
        std::array::from_fn(|i| (i + 1) as f32)
    );

    let result = &row * &mat;

    let expected = SRowVec16::<N>::new(
        std::array::from_fn(|i| {
            (i + 1) as f32 * diagonal[i]
        })
    );

    assert_row_vec_eq(&result, &expected);
}

#[test]
fn test_matrix_mul_constant_matrix() {
    const M: usize = 17;
    const N: usize = 19;

    let mat = SMat16::<M, N>::new(
        [[2.0; N]; M]
    );

    let row = SRowVec16::<M>::new(
        std::array::from_fn(|i| (i + 1) as f32)
    );

    let result = &row * &mat;

    // Every output element is:
    //
    // 2 * sum(i + 1), i = 0..16
    //
    // = 2 * 153
    // = 306
    let expected = SRowVec16::<N>::new(
        [306.0; N]
    );

    assert_row_vec_eq(&result, &expected);
}

#[test]
fn test_matrix_mul_single_nonzero_row_with_remainder() {
    const M: usize = 17;
    const N: usize = 19;

    let mut matrix = [[0.0f32; N]; M];

    for j in 0..N {
        matrix[16][j] = (j + 1) as f32;
    }

    let mat = SMat16::<M, N>::new(matrix);

    let mut values = [0.0f32; M];
    values[16] = 5.0;

    let row = SRowVec16::<M>::new(values);

    let result = &row * &mat;

    let expected = SRowVec16::<N>::new(
        std::array::from_fn(|j| {
            5.0 * (j + 1) as f32
        })
    );

    assert_row_vec_eq(&result, &expected);
}

#[test]
fn test_matrix_mul_single_nonzero_column_with_remainder() {
    const M: usize = 17;
    const N: usize = 19;

    let mut matrix = [[0.0f32; N]; M];

    for i in 0..M {
        matrix[i][18] = (i + 1) as f32;
    }

    let mat = SMat16::<M, N>::new(matrix);

    let row = SRowVec16::<M>::new(
        std::array::from_fn(|i| (i + 1) as f32)
    );

    let result = &row * &mat;

    let expected_sum = (0..M)
        .map(|i| {
            (i + 1) as f32 * (i + 1) as f32
        })
        .sum::<f32>();

    assert!(
        (result[18] - expected_sum).abs() < EPS,
        "expected {}, got {}",
        expected_sum,
        result[18]
    );

    for j in 0..18 {
        assert_eq!(result[j], 0.0);
    }
}

// -----------------------------------------------------------------------------
// Zero and one vectors
// -----------------------------------------------------------------------------

#[test]
fn test_zero_row_vector_times_any_matrix() {
    const M: usize = 17;
    const N: usize = 19;

    let row = SRowVec16::<M>::zero();

    let matrix = std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            (i * 3 + j + 1) as f32
        })
    });

    let mat = SMat16::<M, N>::new(matrix);

    let result = &row * &mat;

    assert_row_vec_eq(
        &result,
        &SRowVec16::<N>::zero(),
    );
}

#[test]
fn test_row_vector_times_identity_with_remainder() {
    const N: usize = 17;

    let matrix = std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            if i == j { 1.0 } else { 0.0 }
        })
    });

    let mat = SMat16::<N, N>::new(matrix);

    let row = SRowVec16::<N>::new(
        std::array::from_fn(|i| (i * 3 + 1) as f32)
    );

    let result = &row * &mat;

    assert_row_vec_eq(&result, &row);
}

// -----------------------------------------------------------------------------
// Negative values
// -----------------------------------------------------------------------------

#[test]
fn test_matrix_mul_negative_values() {
    const M: usize = 17;
    const N: usize = 19;

    let matrix = std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            if (i + j) % 2 == 0 {
                (i + j + 1) as f32
            } else {
                -((i + j + 1) as f32)
            }
        })
    });

    let mat = SMat16::<M, N>::new(matrix);

    let row = SRowVec16::<M>::new(
        std::array::from_fn(|i| {
            if i % 2 == 0 {
                (i + 1) as f32
            } else {
                -((i + 1) as f32)
            }
        })
    );

    let result = &row * &mat;

    let expected = SRowVec16::<N>::new(
        std::array::from_fn(|j| {
            (0..M)
                .map(|i| {
                    let x = if i % 2 == 0 {
                        (i + 1) as f32
                    } else {
                        -((i + 1) as f32)
                    };

                    let y = if (i + j) % 2 == 0 {
                        (i + j + 1) as f32
                    } else {
                        -((i + j + 1) as f32)
                    };

                    x * y
                })
                .sum::<f32>()
        })
    );

    assert_row_vec_eq(&result, &expected);
}

// -----------------------------------------------------------------------------
// Conversion / mutation
// -----------------------------------------------------------------------------

#[test]
fn test_column_row_round_trip_with_remainder() {
    const N: usize = 17;

    let values =
        std::array::from_fn(|i| (i * 3 + 1) as f32);

    let row = SRowVec16::<N>::new(values);
    let column = row.as_column_vec();
    let round_trip = column.as_row_vec();

    assert_row_vec_eq(&round_trip, &SRowVec16::<N>::new(values));
}

#[test]
fn test_row_vector_mutation_after_conversion() {
    const N: usize = 17;

    let values =
        std::array::from_fn(|i| (i + 1) as f32);

    let row = SRowVec16::<N>::new(values);

    let mut column = row.as_column_vec();

    for i in 0..N {
        column[i] *= 2.0;
    }

    let row = column.as_row_vec();

    let expected = SRowVec16::<N>::new(
        std::array::from_fn(|i| ((i + 1) * 2) as f32)
    );

    assert_row_vec_eq(&row, &expected);
}

// -----------------------------------------------------------------------------
// Iterator behaviour
// -----------------------------------------------------------------------------

#[test]
fn test_iterator_length_with_remainder() {
    const N: usize = 17;

    let row = SRowVec16::<N>::new([1.0; N]);

    assert_eq!(row.iter().count(), N);

    let simd_count = row.simd_iter().count();

    assert_eq!(simd_count, 1);
    assert_eq!(row.remainder().len(), 1);
}

#[test]
fn test_iterator_length_without_remainder() {
    const N: usize = 32;

    let row = SRowVec16::<N>::new([1.0; N]);

    assert_eq!(row.iter().count(), N);
    assert_eq!(row.simd_iter().count(), 2);
    assert!(row.remainder().is_empty());
}

#[test]
fn test_remainder_values() {
    const N: usize = 19;

    let row = SRowVec16::<N>::new(
        std::array::from_fn(|i| (i + 1) as f32)
    );

    assert_eq!(
        row.remainder(),
        &[17.0, 18.0, 19.0]
    );
}

// -----------------------------------------------------------------------------
// Algebraic consistency
// -----------------------------------------------------------------------------

#[test]
fn test_matrix_mul_distributivity() {
    const M: usize = 17;
    const N: usize = 19;

    let matrix = std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            (i + j + 1) as f32
        })
    });

    let mat = SMat16::<M, N>::new(matrix);

    let a = SRowVec16::<M>::new(
        std::array::from_fn(|i| (i + 1) as f32)
    );

    let b = SRowVec16::<M>::new(
        std::array::from_fn(|i| (i * 2) as f32)
    );

    // (a + b)A = aA + bA
    //
    // There is no row-vector Add implementation in the supplied
    // implementation, so calculate the sum explicitly.
    let summed = SRowVec16::<M>::new(
        std::array::from_fn(|i| a[i] + b[i])
    );

    let lhs = &summed * &mat;
    let a_result = &a * &mat;
    let b_result = &b * &mat;

    let rhs = SRowVec16::<N>::new(
        std::array::from_fn(|j| {
            a_result[j] + b_result[j]
        })
    );

    assert_row_vec_eq(&lhs, &rhs);
}

#[test]
fn test_matrix_mul_zero_plus_matrix() {
    const M: usize = 17;
    const N: usize = 19;

    let matrix = std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            (i * 2 + j + 1) as f32
        })
    });

    let mat = SMat16::<M, N>::new(matrix);

    let row = SRowVec16::<M>::new(
        std::array::from_fn(|i| (i + 1) as f32)
    );

    let zero = SRowVec16::<M>::zero();

    let lhs = &row * &mat;
    let rhs = &zero * &mat;

    assert_row_vec_eq(
        &rhs,
        &SRowVec16::<N>::zero(),
    );

    assert!(
        lhs.iter().any(|x| x.abs() > EPS),
        "non-zero row multiplied by non-zero matrix unexpectedly produced zero"
    );
}

// -----------------------------------------------------------------------------
// Large dimensions
// -----------------------------------------------------------------------------

#[test]
fn test_large_matrix_multiplication() {
    const M: usize = 33;
    const N: usize = 35;

    let matrix = std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            if i == j {
                1.0
            } else {
                0.0
            }
        })
    });

    let mat = SMat16::<M, N>::new(matrix);

    let row = SRowVec16::<M>::new(
        std::array::from_fn(|i| (i + 1) as f32)
    );

    let result = &row * &mat;

    // The first M columns correspond to the identity portion.
    for i in 0..M {
        assert!(
            (result[i] - row[i]).abs() < EPS,
            "wrong value at index {i}: got {}, expected {}",
            result[i],
            row[i]
        );
    }

    // Remaining columns contain zeroes.
    for i in M..N {
        assert_eq!(result[i], 0.0);
    }
}
