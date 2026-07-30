//! The event loop, and the one task that owns the state.
//!
//! ```text
//!   sources ──mpsc::Sender<Update>──▶ ┌──────────────────────────┐
//!   encoder ──mpsc::Sender<Painted>──▶│ owns AppState. no locks. │
//!   crossterm EventStream ───────────▶│ select! { .. }           │
//!   render interval ─────────────────▶└──────────────────────────┘
//!                          watch::Sender<bool> ──▶ every task, for shutdown
//! ```
//!
//! No `Arc<RwLock<AppState>>`. The render needs a consistent snapshot for a whole
//! frame; with a lock you either hold it across the render, blocking every writer,
//! or you render a torn state.

use crate::config::{Bot, Config};
use crate::render;
use crate::Args;
use anyhow::{Context, Result};
use crossterm::event::{Event, EventStream, KeyCode, KeyEventKind, KeyModifiers};
use futures_util::StreamExt;
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;
use sf_board::{Colors, Renderer, Request};
use sf_layout::Tier;
use sf_model::state::{BotConfig, BotId, BotState, Overlay};
use sf_model::{AppState, Update, apply};
use sf_term::{Caps, Graphics, ImageKey, Painter};
use std::io::Write;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, watch};

/// A finished picture, coming back from the encoder thread.
struct Painted {
    key: ImageKey,
    raster: sf_board::Raster,
}

pub async fn run(cfg: Config, args: Args) -> Result<()> {
    let mut state = seed_state(&cfg);

    let (updates_tx, mut updates_rx) = mpsc::channel::<Update>(512);
    let (paint_tx, mut paint_rx) = mpsc::channel::<Painted>(4);
    let (want_tx, want_rx) = mpsc::channel::<(ImageKey, Request)>(4);
    let (shutdown_tx, shutdown_rx) = watch::channel(false);

    spawn_sources(&cfg, &args, updates_tx.clone(), shutdown_rx.clone());
    spawn_machine(&cfg, updates_tx.clone(), shutdown_rx.clone());
    spawn_files(&cfg, updates_tx.clone(), shutdown_rx.clone());
    if !args.offline {
        spawn_lichess(&cfg, updates_tx.clone(), shutdown_rx.clone());
    } else {
        tracing::info!("offline: no lichess connection will be opened");
    }
    spawn_encoder(want_rx, paint_tx);

    // --- terminal ---
    let mut terminal = enter()?;
    // Probe AFTER the alternate screen and BEFORE the event stream starts, or the
    // capability replies are consumed as keystrokes and the terminal looks as
    // though it supports nothing.
    let forced = args
        .board
        .as_deref()
        .or(Some(cfg.dash.board.as_str()))
        .filter(|s| *s != "auto")
        .and_then(Graphics::parse);
    let caps = sf_term::probe(forced);

    let tier = args.tier.as_deref().or(Some(cfg.dash.tier.as_str())).and_then(Tier::parse);
    let result = event_loop(
        &mut terminal,
        &mut state,
        caps,
        tier,
        cfg.dash.fps,
        &mut updates_rx,
        &mut paint_rx,
        want_tx,
    )
    .await;

    let _ = shutdown_tx.send(true);
    leave(&mut terminal);
    result
}

#[allow(clippy::too_many_arguments)]
async fn event_loop(
    terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    state: &mut AppState,
    caps: Caps,
    tier: Option<Tier>,
    fps: u32,
    updates: &mut mpsc::Receiver<Update>,
    paints: &mut mpsc::Receiver<Painted>,
    want: mpsc::Sender<(ImageKey, Request)>,
) -> Result<()> {
    let mut events = EventStream::new();
    let mut ticker = tokio::time::interval(Duration::from_secs_f64(1.0 / fps as f64));
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

    let mut painter = Painter::default();
    let mut ready: Option<Painted> = None;
    let mut frame: u64 = 0;
    let mut dirty = true;
    let mut last_size = terminal.size()?;

    loop {
        tokio::select! {
            // Batch whatever has arrived rather than redrawing per update: the
            // telemetry feed is ~6.7Hz and a burst of ten records is one frame's
            // worth of news.
            Some(u) = updates.recv() => {
                let now = Instant::now();
                let mut applied = apply(state, u, now);
                while let Ok(next) = updates.try_recv() {
                    applied = applied.merge(apply(state, next, now));
                }
                dirty |= applied.state_changed;
            }
            Some(p) = paints.recv() => {
                ready = Some(p);
                dirty = true;
            }
            maybe = events.next() => {
                match maybe {
                    Some(Ok(Event::Key(k))) if k.kind == KeyEventKind::Press => {
                        match (k.code, k.modifiers) {
                            (KeyCode::Char('q'), _) | (KeyCode::Char('c'), KeyModifiers::CONTROL) => break,
                            (KeyCode::Char('r'), _) => {
                                // Forget the picture and redraw from nothing.
                                painter.forget();
                                let mut out = std::io::stdout();
                                let _ = sf_term::clear(&mut out, &caps);
                                terminal.clear()?;
                                dirty = true;
                            }
                            (KeyCode::Char('?'), _) => {
                                state.ui.overlay = match state.ui.overlay {
                                    Overlay::Layout => Overlay::None,
                                    _ => Overlay::Layout,
                                };
                                dirty = true;
                            }
                            (KeyCode::Tab, _) => {
                                cycle_focus(state);
                                painter.forget();
                                dirty = true;
                            }
                            // Direct selection, so you do not have to Tab past a bot
                            // to reach the one you want. The numbers are on screen.
                            (KeyCode::Char(c @ '1'..='9'), _) => {
                                let n = c as usize - '1' as usize;
                                if let Some(id) = state.bots.keys().nth(n).cloned() {
                                    if state.focus.as_ref() != Some(&id) {
                                        state.focus = Some(id);
                                        painter.forget();
                                        dirty = true;
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                    Some(Ok(Event::Resize(w, h))) => {
                        // The layout rebuilds instantly and a render takes time, so
                        // the picture must go now rather than be painted at the old
                        // size into the new layout.
                        painter.forget();
                        ready = None;
                        let mut out = std::io::stdout();
                        let _ = sf_term::clear(&mut out, &caps);
                        last_size = ratatui::layout::Size { width: w, height: h };
                        dirty = true;
                    }
                    Some(Err(e)) => { tracing::warn!(error = %e, "terminal event error"); }
                    None => break,
                    _ => {}
                }
            }
            _ = ticker.tick() => {
                if !dirty { continue; }
                dirty = false;
                frame += 1;

                let size = terminal.size()?;
                if size != last_size {
                    painter.forget();
                    ready = None;
                    last_size = size;
                }

                let now = Instant::now();
                let wall = jiff::Timestamp::now();
                let mut request = None;
                let mut solved = None;
                terminal.draw(|f| {
                    let area = f.area();
                    let (s, req) = render::draw(
                        f.buffer_mut(), area, state, now, wall, frame, caps.cell, tier,
                    );
                    request = req;
                    solved = Some(s);
                })?;

                // The picture, after the text, and only ever the one drawn for THIS
                // layout. `Painter` enforces both rules.
                if let (Some(req), Some(s)) = (request.as_ref(), solved.as_ref()) {
                    if let Some(pic) = s.picture {
                        let key = ImageKey {
                            board_fen: req.fen.split(' ').next().unwrap_or_default().to_string(),
                            flip: req.flip,
                            last: req.last.clone(),
                            check: req.check.clone(),
                            px: pic.px,
                            // 1-based screen cell, from the solve. Never a magic
                            // absolute row.
                            at: (pic.cells.y + 1, pic.cells.x + 1),
                        };
                        match ready.as_ref() {
                            Some(p) if p.key == key => {
                                // Flush ratatui's pending frame BEFORE the image
                                // bytes, or it lands on top of the picture.
                                let mut out = std::io::stdout();
                                let _ = out.flush();
                                let _ = painter.paint(&mut out, &caps, &key, &p.raster);
                            }
                            _ => {
                                // Ask for it. Newest wins; a stale render is dropped.
                                let _ = want.try_send((key, board_request(req, pic.px)));
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

fn board_request(req: &sf_layout::PictureRequest, px: u32) -> Request {
    Request {
        board_fen: req.fen.split(' ').next().unwrap_or_default().to_string(),
        flip: req.flip,
        last: req.last.as_deref().and_then(|u| {
            Some((sf_board::square_from_name(u.get(0..2)?)?, sf_board::square_from_name(u.get(2..4)?)?))
        }),
        check: req.check.as_deref().and_then(sf_board::square_from_name),
        px,
    }
}

/// The encoder runs on a blocking thread and always renders the **newest** request,
/// dropping anything that went stale while it worked. That bounds board lag at one
/// render however fast the game is -- the Python's first version encoded on the
/// render loop, and since the bot moved faster than the 400ms encode, every move
/// queued behind the last and the board fell further behind for the whole game.
fn spawn_encoder(mut want: mpsc::Receiver<(ImageKey, Request)>, out: mpsc::Sender<Painted>) {
    std::thread::spawn(move || {
        let renderer = Renderer::new();
        while let Some((mut key, mut req)) = want.blocking_recv() {
            // Drain to the newest before doing any work.
            while let Ok((k, r)) = want.try_recv() {
                key = k;
                req = r;
            }
            let Ok(board) = req.board_fen.parse::<shakmaty::Board>() else {
                tracing::debug!(fen = %req.board_fen, "unparseable placement for the picture");
                continue;
            };
            match renderer.render(&board, &req, &Colors::default()) {
                Ok(raster) => {
                    if out.blocking_send(Painted { key, raster }).is_err() {
                        break;
                    }
                }
                Err(e) => tracing::warn!(error = %e, "board render failed"),
            }
        }
    });
}

fn spawn_sources(
    cfg: &Config,
    args: &Args,
    tx: mpsc::Sender<Update>,
    shutdown: watch::Receiver<bool>,
) {
    for bot in &cfg.bots {
        if !bot.watch {
            continue;
        }
        let path = args.replay.clone().unwrap_or_else(|| bot.telemetry.clone());
        let id = BotId(bot.id.clone());
        // Attribution normally comes from `/api/account/playing`: game -> bot, then
        // pid -> game. With no lichess there is nothing to ask, so a replay or an
        // offline run would show an empty board forever. Adopt every game seen
        // instead -- safe precisely because there is only one bot's worth of
        // telemetry to misattribute, and never enabled when lichess is live, where
        // guessing an owner is the mistake that put two games on one curve.
        let adopt = args.offline || args.replay.is_some();
        let tx = tx.clone();
        let mut shutdown = shutdown.clone();
        tokio::spawn(async move {
            let mut src = sf_sources::TelemetrySource::new(path);
            if adopt {
                src = src.adopt_all(id.clone());
            }
            // Until `/api/account/playing` lands, nothing owns any game, so records
            // are counted as unattributed rather than guessed at. The lichess
            // source calls `own()` as soon as it knows.
            let mut ticker = tokio::time::interval(sf_sources::tailer::POLL);
            loop {
                tokio::select! {
                    _ = shutdown.changed() => break,
                    _ = ticker.tick() => {
                        match src.poll(Instant::now()).await {
                            Ok(updates) => {
                                for u in updates {
                                    if tx.send(u).await.is_err() { return; }
                                }
                            }
                            Err(e) => tracing::warn!(bot = %id, error = %e, "telemetry poll failed"),
                        }
                    }
                }
            }
        });
    }
}

/// Everything the project writes to disk. All of it is a `read_to_string` and a
/// parse, so one slow task covers the lot -- and unlike the lichess sources it costs
/// nothing to poll, which is why this runs even in `--offline`.
fn spawn_files(cfg: &Config, tx: mpsc::Sender<Update>, shutdown: watch::Receiver<bool>) {
    for bot in &cfg.bots {
        let src = sf_sources::FileSource {
            bot: BotId(bot.id.clone()),
            user: bot.user.clone(),
            // Global things, attached to each bot's poll for simplicity; the
            // updates they produce are global and idempotent.
            lab_state: cfg.machine.lab_state.clone(),
            matches_dir: cfg.machine.matches_dir.clone(),
            train_active: cfg.machine.train_active.clone(),
            versions: bot.versions.clone(),
            pgn_dir: bot.pgn_dir.clone(),
            rating_log: bot.rating_log.clone(),
            watchdog: dirs_state().map(|d| d.join("chess-gpu/watchdog.json")),
        };
        let tx = tx.clone();
        let mut shutdown = shutdown.clone();
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(Duration::from_secs(5));
            loop {
                tokio::select! {
                    _ = shutdown.changed() => break,
                    _ = ticker.tick() => {
                        // Parsing a few hundred KB of PGN and JSONL is milliseconds,
                        // but it is still blocking work and the UI task must never
                        // wait on it.
                        let s = std::sync::Arc::new(src_clone(&src));
                        let updates = match tokio::task::spawn_blocking(move || s.poll()).await {
                            Ok(u) => u,
                            Err(_) => break,
                        };
                        for u in updates {
                            if tx.send(u).await.is_err() { return; }
                        }
                    }
                }
            }
        });
    }
}

fn src_clone(s: &sf_sources::FileSource) -> sf_sources::FileSource {
    sf_sources::FileSource {
        bot: s.bot.clone(),
        user: s.user.clone(),
        lab_state: s.lab_state.clone(),
        matches_dir: s.matches_dir.clone(),
        train_active: s.train_active.clone(),
        versions: s.versions.clone(),
        pgn_dir: s.pgn_dir.clone(),
        rating_log: s.rating_log.clone(),
        watchdog: s.watchdog.clone(),
    }
}

fn dirs_state() -> Option<std::path::PathBuf> {
    std::env::var("XDG_STATE_HOME")
        .ok()
        .map(std::path::PathBuf::from)
        .or_else(|| {
            std::env::var("HOME").ok().map(|h| std::path::PathBuf::from(h).join(".local/state"))
        })
}

/// The GPU and the units. Global, not per-bot: there is one machine.
fn spawn_machine(cfg: &Config, tx: mpsc::Sender<Update>, shutdown: watch::Receiver<bool>) {
    let names = cfg.machine.units.clone();
    let mut shutdown_gpu = shutdown.clone();
    let tx_gpu = tx.clone();
    tokio::spawn(async move {
        // NVML is a blocking FFI call of a few milliseconds. Off the UI task, on its
        // own cadence.
        let gpu = tokio::task::spawn_blocking(sf_sources::GpuSource::new).await.ok();
        let Some(gpu) = gpu else { return };
        let gpu = Arc::new(gpu);
        let mut ticker = tokio::time::interval(Duration::from_secs(2));
        loop {
            tokio::select! {
                _ = shutdown_gpu.changed() => break,
                _ = ticker.tick() => {
                    let g = gpu.clone();
                    if let Ok(update) = tokio::task::spawn_blocking(move || g.poll()).await {
                        if tx_gpu.send(update).await.is_err() { return; }
                    }
                }
            }
        }
    });

    let mut shutdown_units = shutdown;
    tokio::spawn(async move {
        let units = sf_sources::UnitsSource::new(names).await;
        let mut ticker = tokio::time::interval(Duration::from_secs(6));
        loop {
            tokio::select! {
                _ = shutdown_units.changed() => break,
                _ = ticker.tick() => {
                    if tx.send(units.poll().await).await.is_err() { return; }
                }
            }
        }
    });
}

/// One governor for the whole process, and one polling task per watched bot. The
/// governor is what makes "add more bots" affordable: the endpoint budgets are
/// shared, so the effective cadence degrades rather than the IP getting throttled.
fn spawn_lichess(cfg: &Config, tx: mpsc::Sender<Update>, shutdown: watch::Receiver<bool>) {
    let fetcher = match sf_sources::HttpFetcher::new() {
        Ok(f) => Arc::new(f),
        Err(e) => {
            tracing::error!(error = %e, "no HTTP client; lichess sources disabled");
            return;
        }
    };
    let (call_tx, call_rx) = mpsc::channel(64);
    let (status_tx, mut status_rx) = tokio::sync::watch::channel(Default::default());
    tokio::spawn(
        sf_sources::Governor::new(cfg.api.min_gap(), fetcher)
            .reporting(status_tx)
            .run(call_rx),
    );
    let api = sf_sources::Api::new(call_tx);

    // The budget, on screen. The whole multi-bot story is that the effective
    // interval degrades as bots are added, so it has to be visible rather than
    // inferred from a bot that went quiet.
    {
        let tx = tx.clone();
        tokio::spawn(async move {
            while status_rx.changed().await.is_ok() {
                let s = status_rx.borrow_and_update().clone();
                if tx.send(Update::Api(s.into())).await.is_err() {
                    return;
                }
            }
        });
    }

    for bot in &cfg.bots {
        if !bot.watch {
            continue;
        }
        let id = BotId(bot.id.clone());
        // Read once at startup, held in memory, and never logged or displayed.
        let token = bot
            .token_file
            .as_deref()
            .and_then(|f| crate::config::read_token(f, &bot.token_var));
        if token.is_none() {
            tracing::warn!(bot = %id, "no token; only public endpoints will work");
        }
        let mut src = sf_sources::LichessSource::new(id.clone(), token, api.clone());
        let tx = tx.clone();
        let mut shutdown = shutdown.clone();
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(Duration::from_millis(500));
            loop {
                tokio::select! {
                    _ = shutdown.changed() => break,
                    _ = ticker.tick() => {
                        for u in src.tick(Instant::now()).await {
                            if tx.send(u).await.is_err() { return; }
                        }
                    }
                }
            }
        });
    }
}

fn cycle_focus(state: &mut AppState) {
    let ids: Vec<BotId> = state.bots.keys().cloned().collect();
    if ids.len() < 2 {
        return;
    }
    let cur = state.focus.as_ref().and_then(|f| ids.iter().position(|i| i == f)).unwrap_or(0);
    state.focus = Some(ids[(cur + 1) % ids.len()].clone());
}

/// Build the initial state from config. Shared with `--snapshot`, so a snapshot and
/// a live run start from the same place.
pub fn seed_state(cfg: &Config) -> AppState {
    let mut state = AppState::default();
    for bot in &cfg.bots {
        let c = Arc::new(to_bot_config(bot));
        let id = c.id.clone();
        state.bots.insert(id.clone(), BotState::new(c));
        if state.focus.is_none() {
            state.focus = Some(id);
        }
    }
    state
}

fn to_bot_config(b: &Bot) -> BotConfig {
    BotConfig {
        id: BotId(b.id.clone()),
        user: b.user.clone(),
        label: b.label.clone().unwrap_or_else(|| b.id.clone()),
        unit: b.unit.clone(),
        telemetry: b.telemetry.clone(),
        pgn_dir: b.pgn_dir.clone(),
        versions: b.versions.clone(),
        rating_log: b.rating_log.clone(),
        watch: b.watch,
    }
}

// ---------------------------------------------------------------- terminal

fn enter() -> Result<Terminal<CrosstermBackend<std::io::Stdout>>> {
    install_panic_hook();
    crossterm::terminal::enable_raw_mode().context("enable raw mode")?;
    let mut out = std::io::stdout();
    crossterm::execute!(
        out,
        crossterm::terminal::EnterAlternateScreen,
        crossterm::cursor::Hide
    )?;
    Ok(Terminal::new(CrosstermBackend::new(out))?)
}

fn leave(terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>) {
    restore();
    let _ = terminal.show_cursor();
}

fn restore() {
    let mut out = std::io::stdout();
    // Delete any graphics before leaving, or the picture survives on the shell.
    let _ = write!(out, "\x1b_Ga=d,d=A,q=2\x1b\\");
    let _ = crossterm::execute!(
        out,
        crossterm::terminal::LeaveAlternateScreen,
        crossterm::cursor::Show
    );
    let _ = crossterm::terminal::disable_raw_mode();
    let _ = out.flush();
}

/// Without this, a panic in any task leaves the user in raw mode with an invisible
/// cursor and graphics garbage on screen, and the backtrace is unreadable because
/// newlines do not return the carriage.
fn install_panic_hook() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        restore();
        tracing::error!(%info, "panic");
        previous(info);
    }));
}
