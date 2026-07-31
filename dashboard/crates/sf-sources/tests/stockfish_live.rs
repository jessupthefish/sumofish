//! Drive the real Stockfish.
//!
//! The unit tests cover the parsing; this covers the part that can deadlock. Skipped
//! when the binary is absent, which is also the case the code has to survive: move
//! grading is the one source whose absence costs nothing.

use sf_sources::stockfish::{Stockfish, default_binary};
use std::path::Path;

const ROOT: &str = "/home/nomad/dev/active/sumofish";

#[tokio::test]
async fn a_real_engine_answers_and_agrees_with_itself() {
    let bin = default_binary(Path::new(ROOT));
    let Some(sf) = Stockfish::spawn(&bin).await.expect("spawn must not error") else {
        eprintln!("SKIP: no Stockfish at {}", bin.display());
        return;
    };

    // The start position is roughly level.
    let v = sf.analyse("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 10)
        .await
        .expect("analyse the start position");
    println!("start: cp {} depth {}", v.cp_white, v.depth);
    assert!(v.cp_white.abs() < 150.0, "the start position is not {} cp", v.cp_white);
    assert!(v.depth >= 10);

    // A rook up must read as clearly better for White. Note the sign convention:
    // this is side-to-move relative until `to_white` flips it, and White is to move
    // here, so it should already be positive.
    let v = sf.analyse("4k3/8/8/8/8/8/8/R3K3 w - - 0 1", 10).await.expect("analyse");
    assert!(v.cp_white > 200.0 || v.mate.is_some(), "a rook up read as {v:?}");

    // Repeated calls must not deadlock -- this is the failure `tokio::process`
    // documents when stdin and stdout are driven from one future.
    for i in 0..8 {
        let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        sf.analyse(fen, 8).await.unwrap_or_else(|e| panic!("call {i} failed: {e}"));
    }
}

/// A mate must be reported as a mate, not as a very large centipawn score: the
/// distinction is what lets the curve saturate honestly.
#[tokio::test]
async fn a_forced_mate_is_reported_as_one() {
    let bin = default_binary(Path::new(ROOT));
    let Some(sf) = Stockfish::spawn(&bin).await.expect("spawn") else {
        eprintln!("SKIP: no Stockfish");
        return;
    };
    // Back-rank mate in one for White.
    let v = sf.analyse("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", 12).await.expect("analyse");
    println!("mate position: {v:?}");
    assert!(v.mate.is_some(), "expected a mate score, got {v:?}");
    assert_eq!(v.wp_white(), 1.0);
}

/// A missing binary is not an error. Grading is optional and the dashboard must
/// start without it.
#[tokio::test]
async fn an_absent_binary_is_not_an_error() {
    let r = Stockfish::spawn(Path::new("/definitely/not/here/stockfish")).await;
    assert!(matches!(r, Ok(None)), "a missing engine must be None, not Err");
}
