//! Does the results reader actually find this machine's finished games?
//!
//! The panel was hidden on a machine with four of our PGNs sitting on disk, which is
//! either correct (nothing to show) or a silent parse failure. Those look identical
//! from the outside, which is the whole reason to check.

#[test]
fn the_real_pgn_directory_yields_our_games() {
    let dir = std::path::Path::new("/home/nomad/chess-gpu/logs/games");
    if !dir.exists() {
        eprintln!("SKIP: no PGN directory");
        return;
    }
    let games = sf_sources::files::read_results(dir, "SumoFish", None).expect("read");
    println!("{} games parsed", games.len());
    for g in games.iter().take(6) {
        println!(
            "  {:<7} vs {:<18} {:>5} {}  {}",
            g.result,
            g.opponent,
            g.rating_delta.map(|d| format!("{d:+}")).unwrap_or_default(),
            g.reason,
            g.at.map(|t| t.to_string()).unwrap_or_default()
        );
    }
    let on_disk = std::fs::read_dir(dir)
        .unwrap()
        .flatten()
        .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("pgn"))
        .count();
    assert!(on_disk > 0, "the directory has PGNs");
    assert!(
        !games.is_empty(),
        "{on_disk} PGN files on disk and the reader found no games -- a silent parse \
         failure looks exactly like having nothing to show"
    );
}
