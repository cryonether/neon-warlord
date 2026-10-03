//! Tests for simd_mat

use super::simd_mat::*;
use super::simd_vec::SVec16;

const EPS: f32 = 1e-5;

fn assert_matrix_eq<const M: usize, const N: usize>(a: &SMat16<M, N>, b: &SMat16<M, N>) {
    for (row_a, row_b) in a.iter().zip(b.iter()) {
        for (x, y) in row_a.iter().zip(row_b.iter()) {
            assert!(
                (x - y).abs() < EPS,
                "matrix values differ: left={x}, right={y}"
            );
        }
    }
}

fn assert_vec_eq<const N: usize>(a: &SVec16<N>, b: &SVec16<N>) {
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
    let matrix = std::array::from_fn(|i| std::array::from_fn(|j| (i * 16 + j) as f32));

    let mat = SMat16::<16, 16>::new(matrix);

    for i in 0..16 {
        for j in 0..16 {
            assert_eq!(mat[i][j], matrix[i][j]);
        }
    }
}

// -----------------------------------------------------------------------------
// zero()
// -----------------------------------------------------------------------------

#[test]
fn test_zero() {
    let mat = SMat16::<16, 16>::zero();

    for row in &mat {
        for value in row.iter() {
            assert_eq!(*value, 0.0);
        }
    }
}

// -----------------------------------------------------------------------------
// Deref / DerefMut
// -----------------------------------------------------------------------------

#[test]
fn test_deref() {
    let mut mat = SMat16::<16, 16>::zero();

    mat[0][0] = 42.0;
    mat[15][15] = 99.0;

    assert_eq!(mat[0][0], 42.0);
    assert_eq!(mat[15][15], 99.0);
}

#[test]
fn test_deref_mut() {
    let mut mat = SMat16::<16, 16>::zero();

    for i in 0..16 {
        for j in 0..16 {
            mat[i][j] = (i + j) as f32;
        }
    }

    for i in 0..16 {
        for j in 0..16 {
            assert_eq!(mat[i][j], (i + j) as f32);
        }
    }
}

// -----------------------------------------------------------------------------
// IntoIterator
// -----------------------------------------------------------------------------

#[test]
fn test_into_iterator_owned() {
    let matrix = std::array::from_fn(|i| std::array::from_fn(|j| (i * 16 + j) as f32));

    let mat = SMat16::<16, 16>::new(matrix);

    let rows: Vec<_> = mat.into_iter().collect();

    assert_eq!(rows.len(), 16);

    for i in 0..16 {
        for j in 0..16 {
            assert_eq!(rows[i][j], matrix[i][j]);
        }
    }
}

#[test]
fn test_into_iterator_shared() {
    let matrix = std::array::from_fn(|i| std::array::from_fn(|j| (i * 16 + j) as f32));

    let mat = SMat16::<16, 16>::new(matrix);

    let mut count = 0;

    for (i, row) in (&mat).into_iter().enumerate() {
        for (j, value) in row.iter().enumerate() {
            assert_eq!(*value, matrix[i][j]);
            count += 1;
        }
    }

    assert_eq!(count, 16 * 16);
}

#[test]
fn test_into_iterator_mut() {
    let mut mat = SMat16::<16, 16>::zero();

    for (i, row) in (&mut mat).into_iter().enumerate() {
        for (j, value) in row.iter_mut().enumerate() {
            *value = (i * 16 + j) as f32;
        }
    }

    for i in 0..16 {
        for j in 0..16 {
            assert_eq!(mat[i][j], (i * 16 + j) as f32);
        }
    }
}

// -----------------------------------------------------------------------------
// Matrix-vector multiplication
// -----------------------------------------------------------------------------

#[test]
fn test_matrix_vector_mul_identity() {
    let mut matrix = [[0.0f32; 16]; 16];

    for i in 0..16 {
        matrix[i][i] = 1.0;
    }

    let mat = SMat16::<16, 16>::new(matrix);

    let mut values = [0.0f32; 16];
    for i in 0..16 {
        values[i] = (i + 1) as f32;
    }

    let vec = SVec16::new(values);

    let result = &mat * &vec;

    assert_vec_eq(&result, &vec);
}

#[test]
fn test_matrix_vector_mul_zero_matrix() {
    let mat = SMat16::<16, 16>::zero();

    let values = std::array::from_fn(|i| (i + 1) as f32);
    let vec = SVec16::new(values);

    let result = &mat * &vec;

    let expected = SVec16::zero();

    assert_vec_eq(&result, &expected);
}

#[test]
fn test_matrix_vector_mul_known_values() {
    let matrix = std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            // Matrix contains simple deterministic values.
            (i + j + 1) as f32
        })
    });

    let mat = SMat16::<16, 16>::new(matrix);

    let values = std::array::from_fn(|i| (i + 1) as f32);
    let vec = SVec16::new(values);

    let result = &mat * &vec;

    let expected_values =
        std::array::from_fn(|i| (0..16).map(|j| matrix[i][j] * values[j]).sum::<f32>());

    let expected = SVec16::new(expected_values);

    assert_vec_eq(&result, &expected);
}

#[test]
fn test_matrix_vector_mul_single_nonzero() {
    let mut matrix = [[0.0f32; 16]; 16];

    for i in 0..16 {
        matrix[i][3] = (i + 1) as f32;
    }

    let mat = SMat16::<16, 16>::new(matrix);

    let mut values = [0.0f32; 16];
    values[3] = 5.0;

    let vec = SVec16::new(values);

    let result = &mat * &vec;

    let expected_values = std::array::from_fn(|i| 5.0 * (i + 1) as f32);
    let expected = SVec16::new(expected_values);

    assert_vec_eq(&result, &expected);
}

// -----------------------------------------------------------------------------
// Addition
// -----------------------------------------------------------------------------

#[test]
fn test_add() {
    let a = SMat16::<16, 16>::new(std::array::from_fn(|i| {
        std::array::from_fn(|j| (i + j) as f32)
    }));

    let b = SMat16::<16, 16>::new(std::array::from_fn(|i| {
        std::array::from_fn(|j| (i * j) as f32)
    }));

    let result = &a + &b;

    let expected = SMat16::<16, 16>::new(std::array::from_fn(|i| {
        std::array::from_fn(|j| (i + j) as f32 + (i * j) as f32)
    }));

    assert_matrix_eq(&result, &expected);
}

#[test]
fn test_add_zero() {
    let a = SMat16::<16, 16>::new(std::array::from_fn(|i| {
        std::array::from_fn(|j| (i * 16 + j) as f32)
    }));

    let zero = SMat16::<16, 16>::zero();

    let result = &a + &zero;

    assert_matrix_eq(&result, &a);
}

#[test]
fn test_add_commutative() {
    let a = SMat16::<16, 16>::new(std::array::from_fn(|i| {
        std::array::from_fn(|j| (i + j) as f32)
    }));

    let b = SMat16::<16, 16>::new(std::array::from_fn(|i| {
        std::array::from_fn(|j| (i * j) as f32)
    }));

    let a_b = &a + &b;
    let b_a = &b + &a;

    assert_matrix_eq(&a_b, &b_a);
}

// -----------------------------------------------------------------------------
// Subtraction
// -----------------------------------------------------------------------------

#[test]
fn test_sub() {
    let a = SMat16::<16, 16>::new(std::array::from_fn(|i| {
        std::array::from_fn(|j| (i * 16 + j) as f32)
    }));

    let b = SMat16::<16, 16>::new(std::array::from_fn(|i| {
        std::array::from_fn(|j| (i + j) as f32)
    }));

    let result = &a - &b;

    let expected = SMat16::<16, 16>::new(std::array::from_fn(|i| {
        std::array::from_fn(|j| (i * 16 + j) as f32 - (i + j) as f32)
    }));

    assert_matrix_eq(&result, &expected);
}

#[test]
fn test_sub_zero() {
    let a = SMat16::<16, 16>::new(std::array::from_fn(|i| {
        std::array::from_fn(|j| (i * 16 + j) as f32)
    }));

    let zero = SMat16::<16, 16>::zero();

    let result = &a - &zero;

    assert_matrix_eq(&result, &a);
}

#[test]
fn test_sub_self() {
    let a = SMat16::<16, 16>::new(std::array::from_fn(|i| {
        std::array::from_fn(|j| (i * 16 + j) as f32)
    }));

    let result = &a - &a;
    let expected = SMat16::<16, 16>::zero();

    assert_matrix_eq(&result, &expected);
}

// -----------------------------------------------------------------------------
// SubAssign
// -----------------------------------------------------------------------------

#[test]
fn test_sub_assign() {
    let mut a = SMat16::<16, 16>::new(std::array::from_fn(|i| {
        std::array::from_fn(|j| (i * 16 + j) as f32)
    }));

    let b = SMat16::<16, 16>::new(std::array::from_fn(|i| {
        std::array::from_fn(|j| (i + j) as f32)
    }));

    let expected = &a - &b;

    a -= &b;

    assert_matrix_eq(&a, &expected);
}

#[test]
fn test_sub_assign_zero() {
    let mut a = SMat16::<16, 16>::new(std::array::from_fn(|i| {
        std::array::from_fn(|j| (i * 16 + j) as f32)
    }));

    let original = a.clone();

    a -= &SMat16::<16, 16>::zero();

    assert_matrix_eq(&a, &original);
}

#[test]
fn test_sub_assign_self() {
    let mut a = SMat16::<16, 16>::new(std::array::from_fn(|i| {
        std::array::from_fn(|j| (i * 16 + j) as f32)
    }));

    let copy = a.clone();

    a -= &copy;

    assert_matrix_eq(&a, &SMat16::<16, 16>::zero());
}

// -----------------------------------------------------------------------------
// Algebraic consistency
// -----------------------------------------------------------------------------

#[test]
fn test_add_sub_consistency() {
    let a = SMat16::<16, 16>::new(std::array::from_fn(|i| {
        std::array::from_fn(|j| (i * 2 + j) as f32)
    }));

    let b = SMat16::<16, 16>::new(std::array::from_fn(|i| {
        std::array::from_fn(|j| (i + j * 2) as f32)
    }));

    let c = &a + &b;
    let result = &c - &b;

    assert_matrix_eq(&result, &a);
}

#[test]
fn test_sub_assign_matches_sub() {
    let a = SMat16::<16, 16>::new(std::array::from_fn(|i| {
        std::array::from_fn(|j| (i * 3 + j) as f32)
    }));

    let b = SMat16::<16, 16>::new(std::array::from_fn(|i| {
        std::array::from_fn(|j| (i + j * 3) as f32)
    }));

    let expected = &a - &b;

    let mut actual = a.clone();
    actual -= &b;

    assert_matrix_eq(&actual, &expected);
}

// -----------------------------------------------------------------------------
// Non-SIMD / remainder dimensions
// -----------------------------------------------------------------------------

#[test]
fn test_matrix_vector_mul_remainder() {
    // N = 18 exercises one SIMD chunk + 2 scalar remainder elements.
    let matrix = std::array::from_fn(|i| std::array::from_fn(|j| (i + j + 1) as f32));

    let mat = SMat16::<18, 18>::new(matrix);

    let values = std::array::from_fn(|i| (i + 1) as f32);
    let vec = SVec16::<18>::new(values);

    let result = &mat * &vec;

    let expected_values =
        std::array::from_fn(|i| (0..18).map(|j| matrix[i][j] * values[j]).sum::<f32>());

    let expected = SVec16::<18>::new(expected_values);

    assert_vec_eq(&result, &expected);
}

#[test]
fn test_matrix_vector_mul_remainder_only() {
    // N < 16 means the entire operation uses the scalar remainder path.
    let matrix = std::array::from_fn(|i| std::array::from_fn(|j| (i + j + 1) as f32));

    let mat = SMat16::<7, 7>::new(matrix);

    let values = std::array::from_fn(|i| (i + 1) as f32);
    let vec = SVec16::<7>::new(values);

    let result = &mat * &vec;

    let expected_values =
        std::array::from_fn(|i| (0..7).map(|j| matrix[i][j] * values[j]).sum::<f32>());

    let expected = SVec16::<7>::new(expected_values);

    assert_vec_eq(&result, &expected);
}

#[test]
fn test_matrix_vector_mul_exact_simd_width() {
    // Exactly one SIMD vector.
    let matrix = std::array::from_fn(|i| std::array::from_fn(|j| (i * 2 + j + 1) as f32));

    let mat = SMat16::<16, 16>::new(matrix);

    let values = std::array::from_fn(|i| (i + 1) as f32);
    let vec = SVec16::<16>::new(values);

    let result = &mat * &vec;

    let expected_values =
        std::array::from_fn(|i| (0..16).map(|j| matrix[i][j] * values[j]).sum::<f32>());

    let expected = SVec16::<16>::new(expected_values);

    assert_vec_eq(&result, &expected);
}

#[test]
fn test_matrix_vector_mul_multiple_simd_chunks_and_remainder() {
    // 34 = 2 * 16 + 2 remainder elements.
    let matrix =
        std::array::from_fn(|i| std::array::from_fn(|j| ((i * 3 + j * 2 + 1) % 17) as f32));

    let mat = SMat16::<34, 34>::new(matrix);

    let values = std::array::from_fn(|i| ((i * 5 + 1) % 11) as f32);

    let vec = SVec16::<34>::new(values);

    let result = &mat * &vec;

    let expected_values =
        std::array::from_fn(|i| (0..34).map(|j| matrix[i][j] * values[j]).sum::<f32>());

    let expected = SVec16::<34>::new(expected_values);

    assert_vec_eq(&result, &expected);
}

// -----------------------------------------------------------------------------
// Addition with remainder
// -----------------------------------------------------------------------------

#[test]
fn test_add_with_remainder() {
    let a = SMat16::<3, 18>::new(std::array::from_fn(|i| {
        std::array::from_fn(|j| (i * 10 + j) as f32)
    }));

    let b = SMat16::<3, 18>::new(std::array::from_fn(|i| {
        std::array::from_fn(|j| (i + j * 2) as f32)
    }));

    let result = &a + &b;

    let expected = SMat16::<3, 18>::new(std::array::from_fn(|i| {
        std::array::from_fn(|j| (i * 10 + j) as f32 + (i + j * 2) as f32)
    }));

    assert_matrix_eq(&result, &expected);
}

#[test]
fn test_sub_with_remainder() {
    let a = SMat16::<3, 18>::new(std::array::from_fn(|i| {
        std::array::from_fn(|j| (i * 20 + j * 3) as f32)
    }));

    let b = SMat16::<3, 18>::new(std::array::from_fn(|i| {
        std::array::from_fn(|j| (i * 2 + j) as f32)
    }));

    let result = &a - &b;

    let expected = SMat16::<3, 18>::new(std::array::from_fn(|i| {
        std::array::from_fn(|j| (i * 20 + j * 3) as f32 - (i * 2 + j) as f32)
    }));

    assert_matrix_eq(&result, &expected);
}

// -----------------------------------------------------------------------------
// AddAssign / SubAssign with remainder
// -----------------------------------------------------------------------------

#[test]
fn test_sub_assign_with_remainder() {
    let mut a = SMat16::<3, 18>::new(std::array::from_fn(|i| {
        std::array::from_fn(|j| (i * 20 + j * 3) as f32)
    }));

    let b = SMat16::<3, 18>::new(std::array::from_fn(|i| {
        std::array::from_fn(|j| (i * 2 + j) as f32)
    }));

    let expected = &a - &b;

    a -= &b;

    assert_matrix_eq(&a, &expected);
}

// -----------------------------------------------------------------------------
// Degenerate dimensions
// -----------------------------------------------------------------------------

#[test]
fn test_single_row_matrix_vector_mul() {
    let matrix = [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]];

    let mat = SMat16::<1, 8>::new(matrix);

    let vec = SVec16::<8>::new([1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);

    let result = &mat * &vec;

    // 1 + 4 + 3 + 8 + 5 + 12 + 7 + 16 = 56
    assert!((result[0] - 56.0).abs() < EPS);
}

#[test]
fn test_single_column_matrix_vector_mul() {
    let matrix = std::array::from_fn(|i| [((i + 1) * 2) as f32]);

    let mat = SMat16::<8, 1>::new(matrix);

    let vec = SVec16::<1>::new([3.0]);

    let result = &mat * &vec;

    for i in 0..8 {
        let expected = ((i + 1) * 2) as f32 * 3.0;

        assert!(
            (result[i] - expected).abs() < EPS,
            "row {i}: expected {expected}, got {}",
            result[i]
        );
    }
}

// -----------------------------------------------------------------------------
// Special floating-point values
// -----------------------------------------------------------------------------

#[test]
fn test_matrix_vector_mul_negative_values() {
    let matrix = std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            if (i + j) % 2 == 0 {
                (i + j + 1) as f32
            } else {
                -((i + j + 1) as f32)
            }
        })
    });

    let values = std::array::from_fn(|i| {
        if i % 2 == 0 {
            (i + 1) as f32
        } else {
            -((i + 1) as f32)
        }
    });

    let mat = SMat16::<16, 16>::new(matrix);
    let vec = SVec16::<16>::new(values);

    let result = &mat * &vec;

    let expected_values =
        std::array::from_fn(|i| (0..16).map(|j| matrix[i][j] * values[j]).sum::<f32>());

    let expected = SVec16::<16>::new(expected_values);

    assert_vec_eq(&result, &expected);
}

// -----------------------------------------------------------------------------
// Identity for non-16 dimensions
// -----------------------------------------------------------------------------

#[test]
fn test_matrix_vector_mul_identity_with_remainder() {
    let mut matrix = [[0.0f32; 18]; 18];

    for i in 0..18 {
        matrix[i][i] = 1.0;
    }

    let mat = SMat16::<18, 18>::new(matrix);

    let values = std::array::from_fn(|i| (i * 3 + 1) as f32);

    let vec = SVec16::<18>::new(values);

    let result = &mat * &vec;

    assert_vec_eq(&result, &vec);
}

// -----------------------------------------------------------------------------
// Matrix arithmetic consistency with remainder
// -----------------------------------------------------------------------------

#[test]
fn test_add_sub_consistency_with_remainder() {
    let a = SMat16::<3, 18>::new(std::array::from_fn(|i| {
        std::array::from_fn(|j| (i * 3 + j) as f32)
    }));

    let b = SMat16::<3, 18>::new(std::array::from_fn(|i| {
        std::array::from_fn(|j| (i * 7 + j * 2) as f32)
    }));

    let c = &a + &b;
    let result = &c - &b;

    assert_matrix_eq(&result, &a);
}
