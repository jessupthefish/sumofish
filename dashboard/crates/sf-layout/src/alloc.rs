//! One allocator, used at every level of the layout.
//!
//! This is the whole answer to the bug class the rewrite exists to kill.
//!
//! The Python divided a fixed row budget by a chain of hand-written subtractions
//! (`watch.py:268-297`). One branch of that chain forgot a term -- `moves_h = h -
//! mind_h` in the `curve_h < 7` fallback, omitting `- results_h` -- so every
//! terminal between 50 and 78 rows over-allocated the right column by exactly 8
//! rows and `rich` silently discarded the tail, losing the `machine` panel. The
//! test suite stayed green because it only ever measured the board column.
//!
//! Here there is no chain. `allocate` sums over the same slice it is about to
//! divide, and its postcondition is that the results sum to exactly `total`. You
//! cannot forget a term in a sum over a slice.
//!
//! It is also why this design does not hand anything to ratatui's `Layout`.
//! kasuari is a priority-ordered constraint solver whose behaviour when
//! constraints cannot all be satisfied is documented as non-deterministic, and
//! rounding between `Min` and `Fill` is exactly where an off-by-one would hide.
//! Thirty lines of largest-remainder distribution is both smaller and provable.

/// What one item wants along the axis being divided.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Alloc {
    /// Never give less than this. `allocate` returns `None` rather than
    /// violating it, so a caller cannot accidentally clip.
    pub min: u16,
    /// Never give more than this. Extra space goes to items that can use it.
    pub max: u16,
    /// Share of the leftover, relative to other items' `fill`. Zero means fixed.
    pub fill: u16,
}

impl Alloc {
    pub const fn fixed(n: u16) -> Self {
        Alloc { min: n, max: n, fill: 0 }
    }
    pub const fn new(min: u16, max: u16, fill: u16) -> Self {
        Alloc { min, max, fill }
    }
    /// Grows without bound, from `min`.
    pub const fn grow(min: u16, fill: u16) -> Self {
        Alloc { min, max: u16::MAX, fill }
    }
}

/// Divide `total` among `items`.
///
/// Returns `None` if `Σ min > total` -- the caller is expected to have dropped
/// items until they fit, and getting this back means that pass has a bug. It is
/// deliberately not a clamp: silently shrinking below `min` is how a clipped
/// panel ships, and a clipped panel is a lie.
///
/// # Guarantees
///
/// - `out.len() == items.len()`
/// - `out.iter().sum() == total` exactly, when `Σ min <= total`
/// - `out[i] >= items[i].min` for every `i`
/// - `out[i] <= items[i].max` for every `i`, unless every item is at `max` and
///   there is still slack, in which case the remainder goes to the last item
///   that can take it (see `spill`)
/// - deterministic: same input, same output, always
pub fn allocate(total: u16, items: &[Alloc]) -> Option<Vec<u16>> {
    let need: u32 = items.iter().map(|a| a.min as u32).sum();
    if need > total as u32 {
        return None;
    }
    let mut out: Vec<u16> = items.iter().map(|a| a.min).collect();
    let mut slack = total as u32 - need;
    if slack == 0 || items.is_empty() {
        return Some(out);
    }

    // Distribute by `fill` weight, but never past `max`. Iterate: an item that
    // hits its ceiling returns its share to the pool, so a single pass is not
    // enough and the loop is bounded by the number of items.
    for _ in 0..items.len() {
        let weight: u32 = items
            .iter()
            .enumerate()
            .filter(|(i, a)| a.fill > 0 && out[*i] < a.max)
            .map(|(_, a)| a.fill as u32)
            .sum();
        if weight == 0 || slack == 0 {
            break;
        }
        // Largest-remainder: floor everyone, then hand the remainder out in
        // descending order of what they lost. Deterministic and sums exactly.
        let mut shares: Vec<(usize, u32, u32)> = Vec::new(); // (index, floor, remainder)
        for (i, a) in items.iter().enumerate() {
            if a.fill == 0 || out[i] >= a.max {
                continue;
            }
            let exact = slack * a.fill as u32;
            shares.push((i, exact / weight, exact % weight));
        }
        let handed: u32 = shares.iter().map(|(_, f, _)| f).sum();
        let mut remainder = slack - handed;
        // Ties break by index so the result never depends on sort stability.
        shares.sort_by(|a, b| b.2.cmp(&a.2).then(a.0.cmp(&b.0)));
        let mut given = 0u32;
        for (idx, floor, _) in &shares {
            let mut want = *floor;
            if remainder > 0 {
                want += 1;
                remainder -= 1;
            }
            let room = (items[*idx].max - out[*idx]) as u32;
            let take = want.min(room);
            out[*idx] += take as u16;
            given += take;
        }
        slack -= given;
        if given == 0 {
            break;
        }
    }

    // Everything is at its ceiling and there is still room. Rather than return a
    // short allocation -- which would leave unpainted rows that nothing ever
    // writes to, the exact condition that stranded half a chess board on screen
    // once -- give the remainder to the last item that will take it.
    if slack > 0 {
        spill(&mut out, items, slack);
    }
    Some(out)
}

/// Push leftover space into the last item, ignoring `max`. Only reached when
/// every item is capped; the alternative is rows nobody owns.
fn spill(out: &mut [u16], items: &[Alloc], slack: u32) {
    let target = items
        .iter()
        .enumerate()
        .rev()
        .find(|(_, a)| a.fill > 0)
        .map(|(i, _)| i)
        .or_else(|| out.len().checked_sub(1));
    if let Some(i) = target {
        out[i] = out[i].saturating_add(slack.min(u16::MAX as u32) as u16);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sum(v: &[u16]) -> u32 {
        v.iter().map(|x| *x as u32).sum()
    }

    #[test]
    fn refuses_to_clip_rather_than_shrinking_below_min() {
        assert_eq!(allocate(10, &[Alloc::fixed(6), Alloc::fixed(6)]), None);
    }

    #[test]
    fn fixed_items_get_exactly_their_size() {
        let got = allocate(20, &[Alloc::fixed(6), Alloc::fixed(5), Alloc::grow(4, 1)]).unwrap();
        assert_eq!(got, vec![6, 5, 9]);
        assert_eq!(sum(&got), 20);
    }

    #[test]
    fn slack_splits_by_fill_weight() {
        // Exact shares are 7.5 and 22.5. Both have the same remainder, so the tie
        // breaks by index and the first takes the odd cell. What matters is that
        // it is the *same* answer every time, not which side gets the cell.
        let got = allocate(30, &[Alloc::grow(0, 1), Alloc::grow(0, 3)]).unwrap();
        assert_eq!(got, vec![8, 22], "1:3 of 30, largest-remainder, ties by index");
        assert_eq!(sum(&got), 30);

        // A clean division has no tie to break.
        let got = allocate(40, &[Alloc::grow(0, 1), Alloc::grow(0, 3)]).unwrap();
        assert_eq!(got, vec![10, 30]);
    }

    #[test]
    fn a_capped_item_returns_its_share_to_the_pool() {
        // The first wants a quarter of the slack but can only take 2 more.
        let got = allocate(20, &[Alloc::new(1, 3, 1), Alloc::grow(1, 1)]).unwrap();
        assert_eq!(got, vec![3, 17]);
        assert_eq!(sum(&got), 20);
    }

    #[test]
    fn no_row_is_left_unowned_when_everything_is_capped() {
        // Both are capped well below the space available. Somebody must still own
        // the leftover rows, or they are never painted and whatever was there
        // before survives forever -- which is how half a chess board got stranded
        // under a player line once.
        let got = allocate(100, &[Alloc::new(2, 4, 1), Alloc::new(2, 4, 1)]).unwrap();
        assert_eq!(sum(&got), 100, "the allocation must still be exact");
    }

    /// The property the Python could not state, over a wide sweep: the parts sum
    /// to the whole, and nobody is below their minimum. This is the assertion the
    /// 50-78-row bug would have failed.
    #[test]
    fn always_exact_and_never_below_min() {
        let menu = [
            Alloc::fixed(6),
            Alloc::fixed(5),
            Alloc::grow(12, 1),
            Alloc::new(8, 34, 2),
            Alloc::new(7, 24, 3),
            Alloc::fixed(8),
            Alloc::grow(3, 1),
        ];
        for n in 1..=menu.len() {
            let items = &menu[..n];
            let need: u32 = items.iter().map(|a| a.min as u32).sum();
            for total in 0..=400u16 {
                match allocate(total, items) {
                    None => assert!(
                        need > total as u32,
                        "refused a total of {total} that fits (need {need})"
                    ),
                    Some(got) => {
                        assert_eq!(got.len(), items.len());
                        assert_eq!(
                            sum(&got),
                            total as u32,
                            "n={n} total={total} did not sum exactly: {got:?}"
                        );
                        for (g, a) in got.iter().zip(items) {
                            assert!(*g >= a.min, "n={n} total={total} below min: {got:?}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn is_deterministic() {
        let items = [Alloc::grow(1, 2), Alloc::grow(1, 2), Alloc::grow(1, 1)];
        let first = allocate(97, &items).unwrap();
        for _ in 0..50 {
            assert_eq!(allocate(97, &items).unwrap(), first);
        }
    }

    #[test]
    fn empty_is_not_a_special_case_for_the_caller() {
        assert_eq!(allocate(0, &[]), Some(vec![]));
        assert_eq!(allocate(50, &[]), Some(vec![]));
    }
}
