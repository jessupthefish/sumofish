//! `_softmax_over_legal`. **NOT WIRED INTO THE SEARCH, and it cannot be.**
//!
//! # Verdict, decided by measurement 2026-07-29
//!
//! Moving the prior softmax into Rust was the highest-value remaining item on
//! the port's roadmap: it is the binding term in the 8.9x speedup, and removing
//! it would give ~10.8x. **It is not achievable at bit-identity, and the reason
//! is not fixable by care.**
//!
//! `numpy`'s **float32** `exp` is its own SIMD implementation and is not
//! correctly rounded. Measured against a correctly-rounded double `exp` rounded
//! to f32 -- which is what Rust's `f32::exp` gives, via glibc -- it **differs on
//! 39.9% of realistic inputs**, by up to 1 ULP of f32 (1.7e-7 relative).
//! Every f64 path agrees exactly (0.0% divergence over 42,201 values), which is
//! what made this look feasible at first: the initial check was run in f64, and
//! the engine runs f32, because `policy._logprobs` ends in `.float()`.
//!
//! Reproducing it would mean reimplementing numpy's SIMD exp polynomial, pinned
//! to a numpy version and a CPU feature set, with a bit-exactness test that any
//! numpy upgrade breaks. That is a fragile coupling bought for ~2% of engine wall
//! clock. Not worth it.
//!
//! # So why is this file still here
//!
//! Two reasons. The pairwise summation below IS verified bit-exact in both
//! precisions and is the harder half of the problem; it will be wanted the moment
//! the project accepts a deliberate **epoch boundary** for the softmax, at which
//! point bit-identity stops being required and only a re-measured result matters.
//! And the negative result deserves to be findable in code rather than only in a
//! transcript, so nobody spends the afternoon again.
//!
//! # What IS established, by measurement
//!
//! The first three hold. The fourth is the blocker above.
//!
//! 1. **`np.exp` on f64 is bit-identical to libm's `exp`.** 42,201 realistic
//!    values, every array length 1 to 500: zero mismatches. On **f32 it is not**;
//!    see the verdict above.
//! 2. **`np.exp` is length-independent**, so there is no SIMD-versus-scalar-tail
//!    hazard to worry about separately.
//! 3. **numpy's `sum` is pairwise, not sequential**, and a naive sequential sum
//!    disagreed with it on 41% of real positions by 1 ULP.
//! 4. **The pairwise algorithm below reproduces `np.sum` exactly**, verified over
//!    1,812 trials in f64 and 1,206 in f32 across lengths 1 to 1968. This half of
//!    the problem is solved and stays solved.
//!
//! # The dtype is f32, and that is the whole problem
//!
//! `policy._logprobs` ends in `.float()`, so the engine's logits are float32.
//! `_softmax_over_legal` indexes that array, so `scores` is a float32 array and
//! the whole softmax -- subtract, exp, sum, divide -- happens in single
//! precision. Only `.tolist()` at the end widens to Python floats.
//!
//! Computing this in f64 and rounding would NOT match. It would be more accurate
//! and it would be wrong, which is the same trap as every other "improvement" in
//! this port.

/// numpy's `PW_BLOCKSIZE`.
const PW_BLOCKSIZE: usize = 128;

/// numpy's `pairwise_sum_FLOAT`, structure for structure.
///
/// The eight accumulators are not a vectorisation trick chosen here: numpy
/// unrolls by eight specifically so that AVX can vectorise the loop *without
/// changing the summation order*, and the order is what has to be preserved.
pub fn pairwise_sum_f32(a: &[f32]) -> f32 {
    let n = a.len();

    if n < 8 {
        let mut res = 0.0f32;
        for &x in a {
            res += x;
        }
        res
    } else if n <= PW_BLOCKSIZE {
        let mut r = [
            a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7],
        ];
        // `for (i = 8; i < n - (n % 8); i += 8)`
        let bound = n - (n % 8);
        let mut i = 8;
        while i < bound {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        // Accumulated in this exact tree shape, not left to right.
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        // The non-multiple-of-8 remainder, sequentially.
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        // Halve, but keep the split on a multiple of the unroll factor.
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_sum_f32(&a[..n2]) + pairwise_sum_f32(&a[n2..])
    }
}

/// The f64 variant, kept because the identity harness can be driven in either
/// precision and a mismatch between them is a useful signal on its own.
pub fn pairwise_sum_f64(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut res = 0.0f64;
        for &x in a {
            res += x;
        }
        res
    } else if n <= PW_BLOCKSIZE {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let bound = n - (n % 8);
        let mut i = 8;
        while i < bound {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_sum_f64(&a[..n2]) + pairwise_sum_f64(&a[n2..])
    }
}

/// `mcts._softmax_over_legal(row, legal)`, given the action index per legal move.
///
/// ```text
/// scores = np.array([row[a] if a >= 0 else -1e9 for a in actions])
/// exp    = np.exp(scores - scores.max())
/// return (exp / exp.sum()).tolist()
/// ```
///
/// Returns f64 because that is what `.tolist()` produces and what the tree
/// stores; the arithmetic above it is all f32.
///
/// The `-1e9` branch cannot be reached by a legal move -- the action space is a
/// superset of every legal move in any position and `tests/verify_data.py`
/// checks that over real data -- but it is kept rather than asserted, because a
/// panic here would kill a rated game over something survivable.
pub fn softmax_over_legal(row: &[f32], actions: &[i32]) -> Vec<f64> {
    debug_assert!(!actions.is_empty(), "softmax over an empty move list");

    let mut scores: Vec<f32> = Vec::with_capacity(actions.len());
    for &a in actions {
        scores.push(if a >= 0 { row[a as usize] } else { -1e9f32 });
    }

    // numpy's `.max()` on a non-empty float array; NaN would propagate, and a
    // NaN logit means the network produced one, which is not this function's
    // problem to hide.
    let mut mx = scores[0];
    for &s in &scores[1..] {
        if s > mx {
            mx = s;
        }
    }

    let exp: Vec<f32> = scores.iter().map(|&s| (s - mx).exp()).collect();
    let total = pairwise_sum_f32(&exp);

    exp.iter().map(|&e| (e / total) as f64).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pairwise_matches_sequential_when_short() {
        // Under 8 elements numpy is sequential, so the two must agree exactly.
        let a = [0.5f32, 0.25, 0.125, 1.0, 2.0, 4.0, 8.0];
        let mut seq = 0.0f32;
        for &x in &a {
            seq += x;
        }
        assert_eq!(pairwise_sum_f32(&a), seq);
    }

    #[test]
    fn softmax_sums_to_about_one() {
        let mut row = vec![0.0f32; 1968];
        for (i, v) in row.iter_mut().enumerate() {
            *v = (i % 17) as f32 * 0.1;
        }
        let actions: Vec<i32> = (0..30).collect();
        let p = softmax_over_legal(&row, &actions);
        let s: f64 = p.iter().sum();
        assert!((s - 1.0).abs() < 1e-6, "priors summed to {}", s);
    }

    #[test]
    fn out_of_space_action_is_crushed_not_panicked() {
        let row = vec![1.0f32; 1968];
        let p = softmax_over_legal(&row, &[0, -1, 2]);
        assert!(p[1] < 1e-30, "sentinel prior was {}", p[1]);
        assert_eq!(p.len(), 3);
    }
}
