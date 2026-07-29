//! `xtask board` — render one position and write a PNG.
//!
//! The pixel diff in `sf-board` proves the port is equivalent to python-chess.
//! Only looking proves the whole pipeline produces something that reads as a chess
//! board. Per the Lab Notes, a screenshot judges how it *looks* and never whether
//! the numbers add up -- this is the "how it looks" half.

use anyhow::{Context, Result};
use sf_board::{Colors, Renderer, Request, square_from_name};
use shakmaty::Board;

pub fn run(args: &[String]) -> Result<()> {
    let get = |k: &str| args.iter().find_map(|a| a.strip_prefix(&format!("--{k}=")));
    let fen = get("fen").unwrap_or("r5k1/1b1rp2p/p4pp1/1p6/1P2P3/P1N2P2/2P3PP/1R1R2K1");
    let px: u32 = get("px").and_then(|s| s.parse().ok()).unwrap_or(390);
    let out = get("out").unwrap_or("docs/board.png");
    let flip = args.iter().any(|a| a == "--flip");

    let board: Board = fen.split(' ').next().unwrap().parse().context("placement fen")?;
    let req = Request {
        board_fen: board.board_fen().to_string(),
        flip,
        last: get("last").map(|u| {
            (
                square_from_name(&u[0..2]).expect("from square"),
                square_from_name(&u[2..4]).expect("to square"),
            )
        }),
        check: get("check").and_then(square_from_name),
        px: sf_layout_snap(px),
    };
    let raster = Renderer::new().render(&board, &req, &Colors::default())?;

    let file = std::fs::File::create(out).with_context(|| format!("create {out}"))?;
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), raster.px, raster.px);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header()?.write_image_data(&raster.rgba)?;

    // The seam count only means anything on a board with no highlight: it counts
    // pixels that are neither square colour, and a lastmove square is neither.
    // Re-render clean to get an honest number rather than reporting the highlight
    // as if it were a rasterisation artefact.
    let clean = Renderer::new().render(
        &board,
        &Request { last: None, check: None, ..req.clone() },
        &Colors::default(),
    )?;
    let seams = sf_board::seam_count(&clean, [0xf0, 0xd9, 0xb5], [0xb5, 0x88, 0x63]);
    println!(
        "wrote {out}  {}px (snapped from {px})  seams={seams} {}",
        raster.px,
        if seams == 0 { "(on-grid)" } else { "(OFF-GRID: the board has a hairline it should not)" }
    );
    Ok(())
}

/// Duplicated from `sf_layout::snap26` rather than depending on the layout crate
/// from xtask: one line is cheaper than the dependency edge.
fn sf_layout_snap(px: u32) -> u32 {
    px.saturating_sub(px % 26).max(26)
}
