//! Oracle 1: perft against published node counts.
//!
//! These numbers are from the Chess Programming Wiki's standard perft suite and
//! are **not** derived from `python-chess`. That independence is the point: the
//! differential test in `../tests/differential.py` checks that this crate agrees
//! with the implementation it is replacing, which would happily pass if both
//! shared a misconception. Perft checks against the outside world instead.
//!
//! Each position targets a different class of bug:
//!
//! * **startpos** - the baseline; a wrong number here means something basic.
//! * **kiwipete** - castling both sides, pins, a dense middlegame.
//! * **position 3** - promotions and an en-passant discovered check; the
//!   classic case where a pinned pawn may not capture en passant.
//! * **position 4** - promotion under check, both colours, mirrored.
//! * **position 5** - a known trap for castling-rights bookkeeping.
//! * **position 6** - a quiet position where a wrong number means the ordinary
//!   path is wrong rather than an edge case.
//!
//! Depths are capped so the whole suite stays quick enough to run on every
//! change. The deep counts (startpos d6 = 119,060,324) are worth running by
//! hand once the generator is fast, and are listed here so nobody has to look
//! them up again.

use sumofish_core::board::Board;
use sumofish_core::movegen::perft;

const STARTPOS: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
const KIWIPETE: &str = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
const POS3: &str = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1";
const POS4: &str = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1";
const POS5: &str = "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8";
const POS6: &str = "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10";

/// `(fen, [(depth, expected_nodes), ...])`
fn suite() -> Vec<(&'static str, Vec<(u32, u64)>)> {
    vec![
        (
            STARTPOS,
            vec![(1, 20), (2, 400), (3, 8_902), (4, 197_281), (5, 4_865_609)],
        ),
        (
            KIWIPETE,
            vec![(1, 48), (2, 2_039), (3, 97_862), (4, 4_085_603)],
        ),
        (
            POS3,
            vec![(1, 14), (2, 191), (3, 2_812), (4, 43_238), (5, 674_624)],
        ),
        (POS4, vec![(1, 6), (2, 264), (3, 9_467), (4, 422_333)]),
        (POS5, vec![(1, 44), (2, 1_486), (3, 62_379), (4, 2_103_487)]),
        (POS6, vec![(1, 46), (2, 2_079), (3, 89_890), (4, 3_894_594)]),
    ]
}

#[test]
fn fen_roundtrips_exactly() {
    // This one passes today. FEN is written; move generation is not.
    for (fen, _) in suite() {
        let b = Board::from_fen(fen).expect("suite FEN parses");
        assert_eq!(b.to_fen(), fen, "FEN did not round-trip: {}", fen);
    }
}

#[test]
fn perft_matches_published_counts() {
    for (fen, expectations) in suite() {
        let b = Board::from_fen(fen).expect("suite FEN parses");
        for (depth, expected) in expectations {
            let got = perft(&b, depth);
            assert_eq!(
                got, expected,
                "perft({}) at depth {} gave {}, expected {}",
                fen, depth, got, expected
            );
        }
    }
}

/// The deep counts. Slow (about 13 seconds in release, 594M nodes), so they are
/// `#[ignore]`d and run explicitly:
///
///     cargo test --release --test perft -- --ignored --nocapture
///
/// They live here rather than in a shell history because they are the strongest
/// correctness evidence this crate has, and evidence that is not checked in is
/// evidence nobody will re-run.
#[test]
#[ignore]
fn perft_deep_matches_published_counts() {
    let deep: Vec<(&str, u32, u64)> = vec![
        (STARTPOS, 6, 119_060_324),
        (KIWIPETE, 5, 193_690_690),
        (POS3, 6, 11_030_083),
        (POS4, 5, 15_833_292),
        (POS5, 5, 89_941_194),
        (POS6, 5, 164_075_551),
    ];
    let mut total = 0u64;
    for (fen, depth, expected) in deep {
        let b = Board::from_fen(fen).expect("suite FEN parses");
        let got = perft(&b, depth);
        assert_eq!(got, expected, "deep perft({}) depth {}", fen, depth);
        total += got;
        println!("  ok depth {}: {:>12} nodes", depth, got);
    }
    println!("  {} nodes verified against published counts", total);
}
