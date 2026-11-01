use {
    crate::{
        sparse_matrix::SparseMatrix,
        utils::{unzip_double_array, workload_size},
        FieldElement, R1CS,
    },
    ark_std::{One, Zero},
    std::array,
    tracing::instrument,
};

/// Compute the sum of a vector valued function over the boolean hypercube in
/// the leading variable.
pub fn sumcheck_fold_map_reduce<const N: usize, const M: usize>(
    mles: [&mut [FieldElement]; N],
    fold: Option<FieldElement>,
    map: impl Fn([(FieldElement, FieldElement); N]) -> [FieldElement; M] + Send + Sync + Copy,
) -> [FieldElement; M] {
    let size = mles[0].len();
    assert!(size.is_power_of_two());
    assert!(size >= 2);
    assert!(mles.iter().all(|mle| mle.len() == size));

    if let Some(fold) = fold {
        assert!(size >= 4);
        let slices = mles.map(|mle| {
            let (p0, tail) = mle.split_at_mut(size / 4);
            let (p1, tail) = tail.split_at_mut(size / 4);
            let (p2, p3) = tail.split_at_mut(size / 4);
            [p0, p1, p2, p3]
        });
        sumcheck_fold_map_reduce_inner::<N, M>(slices, fold, map)
    } else {
        let slices = mles.map(|mle| mle.split_at(size / 2));
        sumcheck_map_reduce_inner::<N, M>(slices, map)
    }
}

fn sumcheck_map_reduce_inner<const N: usize, const M: usize>(
    mles: [(&[FieldElement], &[FieldElement]); N],
    map: impl Fn([(FieldElement, FieldElement); N]) -> [FieldElement; M] + Send + Sync + Copy,
) -> [FieldElement; M] {
    let size = mles[0].0.len();
    if size * N * 2 > workload_size::<FieldElement>() {
        // Split slices
        let pairs = mles.map(|(p0, p1)| (p0.split_at(size / 2), p1.split_at(size / 2)));
        let left = pairs.map(|((l0, _), (l1, _))| (l0, l1));
        let right = pairs.map(|((_, r0), (_, r1))| (r0, r1));

        // Parallel recurse
        let (l, r) = rayon::join(
            || sumcheck_map_reduce_inner(left, map),
            || sumcheck_map_reduce_inner(right, map),
        );

        // Combine results
        array::from_fn(|i| l[i] + r[i])
    } else {
        let mut result = [FieldElement::zero(); M];
        for i in 0..size {
            let e = mles.map(|(p0, p1)| (p0[i], p1[i]));
            let local = map(e);
            result.iter_mut().zip(local).for_each(|(r, l)| *r += l);
        }
        result
    }
}

fn sumcheck_fold_map_reduce_inner<const N: usize, const M: usize>(
    mut mles: [[&mut [FieldElement]; 4]; N],
    fold: FieldElement,
    map: impl Fn([(FieldElement, FieldElement); N]) -> [FieldElement; M] + Send + Sync + Copy,
) -> [FieldElement; M] {
    let size = mles[0][0].len();
    if size * N * 4 > workload_size::<FieldElement>() {
        // Split slices
        let pairs = mles.map(|mles| mles.map(|p| p.split_at_mut(size / 2)));
        let (left, right) = unzip_double_array(pairs);

        // Parallel recurse
        let (l, r) = rayon::join(
            || sumcheck_fold_map_reduce_inner(left, fold, map),
            || sumcheck_fold_map_reduce_inner(right, fold, map),
        );

        // Combine results
        array::from_fn(|i| l[i] + r[i])
    } else {
        let mut result = [FieldElement::zero(); M];
        for i in 0..size {
            let e = array::from_fn(|j| {
                let mle = &mut mles[j];
                mle[0][i] += fold * (mle[2][i] - mle[0][i]);
                mle[1][i] += fold * (mle[3][i] - mle[1][i]);
                (mle[0][i], mle[1][i])
            });
            let local = map(e);
            result.iter_mut().zip(local).for_each(|(r, l)| *r += l);
        }
        result
    }
}

// TODO: Add unit tests for calculate_evaluations_over_boolean_hypercube_for_eq,
// eval_eq, calculate_eq, and the transposed matrix multiplication helpers.

/// List of evaluations for eq(r, x) over the boolean hypercube, truncated to
/// `num_entries` elements. When `num_entries < 2^r.len()`, avoids allocating
/// the full hypercube.
#[instrument(skip_all)]
pub fn calculate_evaluations_over_boolean_hypercube_for_eq(
    r: &[FieldElement],
    num_entries: usize,
) -> Vec<FieldElement> {
    let full_size = 1usize << r.len();
    assert!(
        num_entries <= full_size,
        "num_entries ({num_entries}) exceeds 2^{} = {full_size}",
        r.len()
    );
    let mut result = vec![FieldElement::zero(); num_entries];
    eval_eq(r, &mut result, FieldElement::one(), full_size);
    result
}

/// Evaluates the equality polynomial recursively. `subtree_size` tracks the
/// logical size of this recursion level so that truncated output buffers are
/// split correctly.
fn eval_eq(
    eval: &[FieldElement],
    out: &mut [FieldElement],
    scalar: FieldElement,
    subtree_size: usize,
) {
    debug_assert!(out.len() <= subtree_size);
    if let Some((&x, tail)) = eval.split_first() {
        let half = subtree_size / 2;
        let left_len = out.len().min(half);
        let right_len = out.len().saturating_sub(half);
        let (o0, o1) = out.split_at_mut(left_len);
        let s1 = scalar * x;
        let s0 = scalar - s1;
        if right_len == 0 {
            eval_eq(tail, o0, s0, half);
        } else if subtree_size > workload_size::<FieldElement>() {
            rayon::join(
                || eval_eq(tail, o0, s0, half),
                || eval_eq(tail, o1, s1, half),
            );
        } else {
            eval_eq(tail, o0, s0, half);
            eval_eq(tail, o1, s1, half);
        }
    } else {
        out[0] += scalar;
    }
}

/// Evaluates a cubic polynomial on a value
pub fn eval_cubic_poly(poly: [FieldElement; 4], point: FieldElement) -> FieldElement {
    poly[0] + point * (poly[1] + point * (poly[2] + point * poly[3]))
}

/// Given a path to JSON file with sparce matrices and a witness, calculates
/// matrix-vector multiplication and returns them
#[instrument(skip_all)]
pub fn calculate_witness_bounds(
    r1cs: &R1CS,
    witness: &[FieldElement],
) -> (Vec<FieldElement>, Vec<FieldElement>, Vec<FieldElement>) {
    let (a, b) = rayon::join(|| r1cs.a() * witness, || r1cs.b() * witness);

    let target_len = a.len().next_power_of_two();
    let mut c = Vec::with_capacity(target_len);
    c.extend(a.iter().zip(b.iter()).map(|(a, b)| *a * *b));
    c.resize(target_len, FieldElement::zero());

    let mut a = a;
    let mut b = b;
    a.resize(target_len, FieldElement::zero());
    b.resize(target_len, FieldElement::zero());
    (a, b, c)
}

/// Calculates eq(r, alpha)
pub fn calculate_eq(r: &[FieldElement], alpha: &[FieldElement]) -> FieldElement {
    r.iter()
        .zip(alpha.iter())
        .fold(FieldElement::from(1), |acc, (&r, &alpha)| {
            acc * (r * alpha + (FieldElement::from(1) - r) * (FieldElement::from(1) - alpha))
        })
}

/// Transpose all three R1CS matrices in parallel.
///
/// This depends only on the R1CS structure (from the verifier key), not on any
/// proof-specific data, so it can run concurrently with sumcheck verification.
#[instrument(skip_all)]
pub fn transpose_r1cs_matrices(r1cs: &R1CS) -> (SparseMatrix, SparseMatrix, SparseMatrix) {
    let ((at, bt), ct) = rayon::join(
        || rayon::join(|| r1cs.a.transpose(), || r1cs.b.transpose()),
        || r1cs.c.transpose(),
    );
    (at, bt, ct)
}

/// Multiply pre-transposed R1CS matrices by eq(alpha, ·) to compute the
/// external row.
#[instrument(skip_all)]
pub fn multiply_transposed_by_eq_alpha(
    at: &SparseMatrix,
    bt: &SparseMatrix,
    ct: &SparseMatrix,
    alpha: &[FieldElement],
    r1cs: &R1CS,
) -> [Vec<FieldElement>; 3] {
    let eq_alpha =
        calculate_evaluations_over_boolean_hypercube_for_eq(alpha, r1cs.num_constraints());
    let interner = &r1cs.interner;
    let ((a, b), c) = rayon::join(
        || {
            rayon::join(
                || at.hydrate(interner) * eq_alpha.as_slice(),
                || bt.hydrate(interner) * eq_alpha.as_slice(),
            )
        },
        || ct.hydrate(interner) * eq_alpha.as_slice(),
    );
    [a, b, c]
}

/// Calculates a random row of R1CS matrix extension. Made possible due to
/// sparseness.
///
/// Computes `eq(alpha, ·) * [A, B, C]` using transposed matrices for
/// parallel right-multiplication instead of sequential left-multiplication.
#[instrument(skip_all)]
pub fn calculate_external_row_of_r1cs_matrices(
    alpha: &[FieldElement],
    r1cs: &R1CS,
) -> [Vec<FieldElement>; 3] {
    let (at, bt, ct) = transpose_r1cs_matrices(r1cs);
    multiply_transposed_by_eq_alpha(&at, &bt, &ct, alpha, r1cs)
}

/// Computes the same external row as
/// [`calculate_external_row_of_r1cs_matrices`] without transposing the
/// matrices.
///
/// Each matrix is multiplied on the left by eq(alpha, ·), which walks it row by
/// row and adds every entry into its column's sum. Rows are visited in
/// ascending order, so each column's terms are multiplied and added in the
/// same order as in the transposed product, and the result is identical.
///
/// Suits a verifier, which computes this once per proof: the transposes cost
/// more than the product itself, and only the prover's row-parallel product
/// needs them.
#[instrument(skip_all)]
pub fn calculate_external_row_without_transpose(
    alpha: &[FieldElement],
    r1cs: &R1CS,
) -> [Vec<FieldElement>; 3] {
    let eq_alpha =
        calculate_evaluations_over_boolean_hypercube_for_eq(alpha, r1cs.num_constraints());
    let eq_alpha = eq_alpha.as_slice();
    let ((a, b), c) = rayon::join(
        || rayon::join(|| eq_alpha * r1cs.a(), || eq_alpha * r1cs.b()),
        || eq_alpha * r1cs.c(),
    );
    [a, b, c]
}

#[cfg(test)]
mod tests {
    use {super::*, ark_ff::PrimeField};

    /// SplitMix64, so the random instances are reproducible without a
    /// test-only RNG dependency.
    fn next_random(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// A uniformly random field element, or one of the edge values 0, 1 and -1
    /// (the largest element), each far more often than chance.
    fn random_field_element(state: &mut u64) -> FieldElement {
        match next_random(state) % 8 {
            0 => FieldElement::zero(),
            1 => FieldElement::one(),
            2 => -FieldElement::one(),
            _ => {
                let bytes: Vec<u8> = (0..4)
                    .flat_map(|_| next_random(state).to_le_bytes())
                    .collect();
                FieldElement::from_le_bytes_mod_order(&bytes)
            }
        }
    }

    /// A random R1CS whose linear combinations each have up to `max_terms`
    /// terms over `witnesses` witnesses.
    fn random_r1cs(state: &mut u64, constraints: usize, witnesses: usize, max_terms: u64) -> R1CS {
        let mut r1cs = R1CS::new();
        r1cs.add_witnesses(witnesses);
        let linear_combination = |state: &mut u64| {
            (0..next_random(state) % (max_terms + 1))
                .map(|_| {
                    (
                        random_field_element(state),
                        next_random(state) as usize % witnesses,
                    )
                })
                .collect::<Vec<_>>()
        };
        for _ in 0..constraints {
            let a = linear_combination(state);
            let b = linear_combination(state);
            let c = linear_combination(state);
            r1cs.add_constraint(&a, &b, &c);
        }
        r1cs
    }

    #[test]
    fn external_row_without_transpose_matches_transposed_product() {
        let mut state = 0xe7_u64;
        for (constraints, witnesses, max_terms) in [
            (1, 1, 1),
            (1, 9, 4),
            (7, 3, 6),
            (64, 64, 3),
            (100, 37, 12),
            (513, 300, 5),
            (1000, 1200, 20),
        ] {
            let r1cs = random_r1cs(&mut state, constraints, witnesses, max_terms);
            let num_variables = usize::BITS - (constraints - 1).leading_zeros();
            let alpha: Vec<FieldElement> = (0..num_variables.max(1))
                .map(|_| random_field_element(&mut state))
                .collect();
            assert_eq!(
                calculate_external_row_without_transpose(&alpha, &r1cs),
                calculate_external_row_of_r1cs_matrices(&alpha, &r1cs),
                "the external row without transposes differs from the transposed product for \
                 {constraints} constraints over {witnesses} witnesses"
            );
        }
    }
}
