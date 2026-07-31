//! Does the results reader actually find this machine's finished games?
//!
//! The panel was hidden on a machine with four of our PGNs sitting on disk, which is
//! either correct (nothing to show) or a silent parse failure. Those look identical
//! from the outside, which is the whole reason to check.

#[test]
fn the_real_pgn_directory_yields_our_games() {
    let dir = std::path::Path::new("/home/nomad/dev/active/sumofish/logs/games");
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

/// The version reader, against the real registry. Takes the LAST line, which is the
/// current version -- the Python read only its timestamp, as a cutoff, and its label.
#[test]
fn the_real_versions_file_yields_the_current_version() {
    let p = std::path::Path::new("/home/nomad/dev/active/sumofish/VERSIONS.jsonl");
    if !p.exists() {
        eprintln!("SKIP: no VERSIONS.jsonl");
        return;
    }
    let v = sf_sources::files::read_version(p).expect("read the registry");
    println!("{} — {}", v.version, v.title);
    println!("  checkpoint {:?} step {:?}", v.checkpoint_sha, v.step);
    assert!(!v.version.is_empty());
    assert!(!v.title.is_empty(), "a version with no title says nothing on screen");
    // The one field that answers "is the bot playing what this version shipped".
    assert!(v.checkpoint_sha.is_some(), "a version must pin its checkpoint");
    // And it must be the LAST entry, not the first.
    let lines = std::fs::read_to_string(p).unwrap();
    let last = lines.lines().filter(|l| !l.trim().is_empty()).next_back().unwrap();
    assert!(
        last.contains(&format!("\"{}\"", v.version)),
        "read {} but the registry's last line is {last}",
        v.version
    );
}
