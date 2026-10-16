//! Prefix networks: fixed arrangements of pairwise combines that compute every
//! prefix `items[0]`, `combine(items[0], items[1])`, ... with a far shorter
//! longest chain of combines than a loop. Ported from handwired's
//! `prefix-networks.ts`.
//!
//! `combine(earlier, later)` has to be associative. That holds exactly for
//! selections, e.g. reading tables of integer states, and for sums of integers
//! below 2^24, while other FP32 sums hold it only up to rounding. It doesn't
//! have to be commutative. E.g. every row's bottom from the row heights:
//! `brent_kung_prefix(&heights, |&earlier, &later| circuit.round_affine(earlier + later))`.

/// Sklansky (1960): for blocks of 2, 4, 8, ... items, combine the last prefix of
/// the block's left half into every prefix of its right half. For k items, a
/// power of two, that's (k / 2) × log2(k) combines, and a longest chain of log2(k).
pub fn sklansky_prefix<T: Clone>(items: &[T], mut combine: impl FnMut(&T, &T) -> T) -> Vec<T> {
    let mut prefixes = items.to_vec();
    let mut half = 1;
    while half < prefixes.len() {
        for block_start in (0..prefixes.len()).step_by(2 * half) {
            let left_last = block_start + half - 1;
            let block_end = (block_start + 2 * half).min(prefixes.len());
            for index in left_last + 1..block_end {
                prefixes[index] = combine(&prefixes[left_last], &prefixes[index]);
            }
        }
        half *= 2;
    }
    prefixes
}

/// Brent-Kung (1982): an upward pass combines neighbors, then pairs of pairs,
/// and so on, which completes the prefixes of the first 2, 4, 8, ... items. A
/// downward pass then completes every other prefix by combining one completed
/// prefix with one smaller block. For k items, a power of two and at least 4,
/// that's 2k - 2 - log2(k) combines, and a longest chain of 2 × log2(k) - 2.
pub fn brent_kung_prefix<T: Clone>(items: &[T], mut combine: impl FnMut(&T, &T) -> T) -> Vec<T> {
    let mut prefixes = items.to_vec();
    let mut span = 1;
    while span < prefixes.len() {
        for index in (2 * span - 1..prefixes.len()).step_by(2 * span) {
            prefixes[index] = combine(&prefixes[index - span], &prefixes[index]);
        }
        span *= 2;
    }
    span /= 2;
    while span >= 1 {
        for index in (3 * span - 1..prefixes.len()).step_by(2 * span) {
            prefixes[index] = combine(&prefixes[index - span], &prefixes[index]);
        }
        span /= 2;
    }
    prefixes
}
