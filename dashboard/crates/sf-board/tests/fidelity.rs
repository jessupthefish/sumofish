//! Is the ported board renderer faithful to `chess.svg.board()`?
//!
//! Not by reading it. This execs the original and diffs against it, which is
//! CLAUDE.md's rule for porting a reference implementation: "exec the original and
//! diff against it rather than reading it carefully and reimplementing. Catches
//! what careful reading misses, and it is less work." It did: the first draft of
//! the port guessed the coordinate placement, drew 16 glyphs where python-chess
//! draws 32, and put the check gradient inside the square loop. All three were
//! caught here rather than by inspection.
//!
//! **The diff is over pixels, not SVG bytes.** Both strings go through the same
//! rasteriser and the resulting images must match exactly. Byte-identical SVG
//! would additionally pin `ElementTree`'s attribute insertion order and float
//! formatting, which is a means rather than the end, and python-chess also emits a
//! `<desc><pre>` of the ASCII board that a rasteriser ignores. Pixels are the
//! thing that can actually be wrong in a way anybody would notice.
//!
//! Skipped, loudly, when the venv is absent, so a fresh clone does not fail a test
//! for a reason unrelated to the code.

use sf_board::raster::Renderer;
use sf_board::svg::{Colors, Request, board_svg};
use shakmaty::{Board, CastlingMode, Chess, EnPassantMode, Position, fen::Fen};
use std::process::Command;

const PYTHON: &str = "/home/nomad/dev/active/sumofish/.venv/bin/python";
/// Small enough to keep 30 subprocess spawns quick, and still a multiple of 26 so
/// the comparison is not dominated by antialiasing on off-grid square edges.
const PX: u32 = 390;

fn available() -> bool {
    std::path::Path::new(PYTHON).exists()
}

/// python-chess's own SVG for the same inputs.
fn reference_svg(fen: &str, flip: bool, last: Option<&str>, check: Option<&str>) -> String {
    let script = r##"
import sys, chess, chess.svg
fen, flip, last, check, px = sys.argv[1], sys.argv[2] == "1", sys.argv[3], sys.argv[4], int(sys.argv[5])
b = chess.Board(fen)
sys.stdout.write(chess.svg.board(
    b,
    orientation=chess.BLACK if flip else chess.WHITE,
    lastmove=chess.Move.from_uci(last) if last else None,
    check=chess.parse_square(check) if check else None,
    size=px, coordinates=True,
    colors={"square light": "#f0d9b5", "square dark": "#b58863",
            "square light lastmove": "#cdd26a", "square dark lastmove": "#aaa23a",
            "margin": "#2b2724", "coord": "#e5e0d5"},
))
"##;
    let out = Command::new(PYTHON)
        .args([
            "-c",
            script,
            fen,
            if flip { "1" } else { "0" },
            last.unwrap_or(""),
            check.unwrap_or(""),
        ])
        .arg(PX.to_string())
        .output()
        .expect("run python-chess");
    assert!(out.status.success(), "python-chess failed: {}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8(out.stdout).expect("utf8")
}

fn ours_svg(fen: &str, flip: bool, last: Option<&str>, check: Option<&str>) -> String {
    let board = placement(fen);
    let last_sq = last.map(|u| {
        (
            sf_board::square_from_name(&u[0..2]).expect("from square"),
            sf_board::square_from_name(&u[2..4]).expect("to square"),
        )
    });
    let req = Request {
        board_fen: board.board_fen().to_string(),
        flip,
        last: last_sq,
        check: check.and_then(sf_board::square_from_name),
        px: PX,
    };
    board_svg(&board, &req, &Colors::default())
}

/// The placement field only. Not via `Chess`, which validates: an empty board is
/// `EMPTY_BOARD | MISSING_KING`, and the picture draws boards that are not legal
/// positions perfectly happily.
fn placement(fen: &str) -> Board {
    fen.split(' ').next().unwrap().parse::<Board>().expect("placement fen")
}

/// Render both and report how many pixels differ.
fn compare(fen: &str, flip: bool, last: Option<&str>, check: Option<&str>) -> (usize, u8) {
    let r = Renderer::new();
    let want = r.render_svg(&reference_svg(fen, flip, last, check), PX).expect("render reference");
    let got = r.render_svg(&ours_svg(fen, flip, last, check), PX).expect("render ours");
    assert_eq!(want.rgba.len(), got.rgba.len());
    let mut differing = 0;
    let mut worst = 0u8;
    for (a, b) in want.rgba.chunks_exact(4).zip(got.rgba.chunks_exact(4)) {
        let d = (0..4).map(|i| a[i].abs_diff(b[i])).max().unwrap();
        worst = worst.max(d);
        if d > 0 {
            differing += 1;
        }
    }
    (differing, worst)
}

/// The cases that exercise every branch: an empty board, all twelve glyphs, a
/// highlight on each square colour, a king in check, and both orientations.
///
/// Same rasteriser on both sides, so the requirement is **exact**: any difference
/// at all is a geometry or paint error, not antialiasing.
#[test]
fn pixels_match_python_chess_on_representative_positions() {
    if !available() {
        eprintln!("SKIP: {PYTHON} absent, cannot diff against the reference");
        return;
    }
    let cases: &[(&str, bool, Option<&str>, Option<&str>)] = &[
        ("8/8/8/8/8/8/8/8 w - - 0 1", false, None, None),
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", false, None, None),
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", true, None, None),
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", false, Some("e2e4"), None),
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", false, Some("d2d4"), None),
        (
            "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
            false,
            Some("d8h4"),
            Some("e1"),
        ),
        (
            "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
            true,
            Some("d8h4"),
            Some("e1"),
        ),
        ("r5k1/1b1rp2p/p4pp1/1p6/1P2P3/P1N2P2/2P3PP/1R1R2K1 b - - 2 24", false, Some("b5b4"), None),
        ("r5k1/1b1rp2p/p4pp1/1p6/1P2P3/P1N2P2/2P3PP/1R1R2K1 b - - 2 24", true, Some("b5b4"), None),
        ("QQQQQQQQ/8/8/8/8/8/8/qqqqqqqq w - - 0 1", false, None, None),
    ];
    for (fen, flip, last, check) in cases {
        let (differing, worst) = compare(fen, *flip, *last, *check);
        assert_eq!(
            differing, 0,
            "{differing} pixels differ (worst channel delta {worst}) for \
             fen={fen} flip={flip} last={last:?} check={check:?}"
        );
    }
}

/// A long deterministic walk out of the opening, so the `<defs>` set, the square
/// fills and the coordinate orientation vary in combinations no single position
/// reaches. This is the pass that would catch a `<defs>` ordering bug.
#[test]
fn pixels_match_python_chess_over_a_long_walk() {
    if !available() {
        eprintln!("SKIP: {PYTHON} absent");
        return;
    }
    // Always take the Nth legal move with N cycling: reproducible without a seed,
    // and it wanders through captures, checks and promotions.
    let mut pos = Chess::default();
    let mut n = 0usize;
    let mut checked = 0;
    for step in 0..120 {
        let moves = pos.legal_moves();
        if moves.is_empty() {
            pos = Chess::default();
            continue;
        }
        n = (n + 7) % moves.len();
        let mv = moves[n];
        let from = mv.from().expect("a real move has a source");
        let to = mv.to();
        pos = pos.clone().play(mv).expect("legal");

        // Every fifth position gets a subprocess; the walk's variety is the point,
        // not the count.
        if step % 5 != 0 {
            continue;
        }
        let fen = Fen::from_position(&pos, EnPassantMode::Legal).to_string();
        let uci = format!("{from}{to}");
        let check = pos
            .is_check()
            .then(|| pos.board().king_of(pos.turn()).map(|s| s.to_string()))
            .flatten();
        let (differing, worst) = compare(&fen, step % 2 == 0, Some(&uci), check.as_deref());
        assert_eq!(
            differing, 0,
            "step {step}: {differing} pixels differ (worst {worst}) at fen={fen} last={uci} check={check:?}"
        );
        checked += 1;
    }
    assert!(checked >= 20, "the walk only checked {checked} positions");
}

/// Structural checks that do not need python-chess, so they still run on a machine
/// without the venv.
#[test]
fn the_svg_has_the_right_shape() {
    let svg = ours_svg("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", false, None, None);

    // 64 squares, once each.
    assert_eq!(svg.matches(r#"class="square"#).count(), 64);
    // Coordinates on all four sides: 8 files x 2 + 8 ranks x 2.
    assert_eq!(svg.matches("<g transform=\"translate(").count(), 32, "32 coordinate glyphs");
    // Every piece present is defined exactly once.
    let defs = svg
        .split_once("<defs>")
        .and_then(|(_, r)| r.split_once("</defs>"))
        .map(|(d, _)| d)
        .expect("a defs block");
    for name in [
        "white-pawn", "white-knight", "white-bishop", "white-rook", "white-queen", "white-king",
        "black-pawn", "black-knight", "black-bishop", "black-rook", "black-queen", "black-king",
    ] {
        assert_eq!(defs.matches(&format!(r#"id="{name}""#)).count(), 1, "{name} in <defs>");
    }
    // 32 pieces on the board at the start.
    assert_eq!(svg.matches("<use xlink:href=").count(), 32);

    // An empty board defines nothing and places nothing.
    let empty = ours_svg("8/8/8/8/8/8/8/8 w - - 0 1", false, None, None);
    assert!(empty.contains("<defs></defs>"));
    assert_eq!(empty.matches("<use xlink:href=").count(), 0);
    assert_eq!(empty.matches(r#"class="square"#).count(), 64);

    // The check gradient appears only when there is a check.
    assert!(!svg.contains("check_gradient"));
    let checked = ours_svg(
        "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
        false,
        Some("d8h4"),
        Some("e1"),
    );
    assert_eq!(checked.matches("check_gradient").count(), 2, "one def, one reference");
    assert_eq!(checked.matches(r#"class="check""#).count(), 1, "the mark is emitted once");
}
