//! Development tasks for the SumoFish dashboard.
//!
//! `probe` answers the M0 falsification questions from docs/plan.md. Nothing in
//! the real design is committed until it reports, because two of the seven can
//! invalidate a settled decision and finding out costs an afternoon.

mod probe_board;
mod probe_svg;
mod probe_sys;
mod probe_term;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or("help");
    match cmd {
        // F6: does resvg match rsvg-convert for cburnett, and does the
        // multiple-of-26 seam rule hold in the new rasteriser.
        "probe-svg" => probe_svg::run(&args[1..]),
        // F7: nvml per-process VRAM under the systemd sandbox; zbus on the user bus.
        "probe-sys" => probe_sys::run(),
        // F1/F2/F3: kitty graphics in Konsole. Needs a real tty; run it IN a
        // Konsole window, not through a pipe.
        "probe-term" => probe_term::run(&args[1..]),
        // Render one board to a PNG, so a human can look at the pipeline's output.
        // A pixel diff proves equivalence to python-chess; only an eye proves the
        // whole thing looks like a chess board.
        "board" => probe_board::run(&args[1..]),
        _ => {
            eprintln!("usage: xtask <probe-svg|probe-sys|probe-term|board>");
            std::process::exit(2);
        }
    }
}
