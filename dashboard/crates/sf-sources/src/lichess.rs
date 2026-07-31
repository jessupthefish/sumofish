//! The lichess client, hand-rolled.
//!
//! `licheszter` is the maintained crate and it was rejected for two specific
//! reasons found by reading its source: it configures **no timeouts at all**, so a
//! stalled TCP connection hangs forever; and it drops the keepalive blank lines,
//! which are the only liveness signal a long-lived stream has. We need three
//! endpoints and a line reader, which is forty lines.
//!
//! Two client configurations, because the timeout that is right for a poll is wrong
//! for a stream:
//!
//! - Polls get a total `timeout`.
//! - Streams get `read_timeout` and never a total one. `timeout` is a deadline for
//!   the whole response, so setting it on a stream kills a healthy connection on
//!   schedule. `read_timeout` "resets after a successful read", which is what
//!   detects a stalled socket without punishing a working one. The public move
//!   stream sends a blank line every 50 seconds, so 90 is the floor.

use crate::governor::{Api, Endpoint, FetchError};
use anyhow::Result;
use serde::Deserialize;
use sf_model::state::*;
// The trait `Chess::play` comes from -- needed for `pos.play(mv)` in
// `game_export_updates`, even though every other shakmaty type here is used
// through a fully-qualified path.
use shakmaty::Position as _;
use sf_model::{BotId, GameId, Update};
use std::time::{Duration, Instant};

/// lichess asks for a descriptive agent. Being identifiable is the polite half of
/// staying inside a budget nobody publishes.
pub const USER_AGENT: &str = concat!("sumofish-dash/", env!("CARGO_PKG_VERSION"));

// ---------------------------------------------------------------- wire types

#[derive(Debug, Deserialize)]
pub struct AccountJson {
    pub username: String,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub perfs: indexmap::IndexMap<String, PerfJson>,
    #[serde(default)]
    pub count: Option<CountJson>,
}

#[derive(Debug, Deserialize)]
pub struct PerfJson {
    #[serde(default)]
    pub rating: i32,
    #[serde(default)]
    pub games: u32,
    /// Rating deviation. The Python dropped it, and a provisional 2400 and a settled
    /// 2400 are different claims about the same number.
    #[serde(default)]
    pub rd: i32,
    #[serde(default)]
    pub prov: bool,
    #[serde(default)]
    pub prog: i32,
}

#[derive(Debug, Deserialize)]
pub struct CountJson {
    #[serde(default)]
    pub all: u32,
}

#[derive(Debug, Deserialize)]
pub struct PlayingJson {
    #[serde(default)]
    #[serde(rename = "nowPlaying")]
    pub now_playing: Vec<NowPlaying>,
}

#[derive(Debug, Deserialize)]
pub struct NowPlaying {
    #[serde(rename = "gameId")]
    pub game_id: String,
    pub color: String,
    #[serde(default)]
    pub fen: String,
    #[serde(default, rename = "lastMove")]
    pub last_move: Option<String>,
    #[serde(default, rename = "isMyTurn")]
    pub is_my_turn: bool,
    #[serde(default, rename = "secondsLeft")]
    pub seconds_left: Option<u32>,
    #[serde(default)]
    pub opponent: OpponentJson,
    #[serde(default)]
    pub speed: String,
    #[serde(default)]
    pub rated: bool,
}

#[derive(Debug, Default, Deserialize)]
pub struct OpponentJson {
    #[serde(default)]
    pub username: String,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub rating: Option<i32>,
    #[serde(default)]
    pub provisional: bool,
    #[serde(default)]
    pub ai: Option<u32>,
}

/// `/game/export/{id}?moves=true&clocks=true&evals=true`, JSON response.
/// Delayed by three moves on an ongoing game, by lichess's own policy, same as
/// the public game stream -- see `Endpoint::GameExport`'s doc comment.
#[derive(Debug, Default, Deserialize)]
pub struct GameExportJson {
    /// Space-separated SAN, e.g. "e4 e5 Nf3 Nc6". Never UCI -- shakmaty replays
    /// it to get both the UCI form and the position each move was played from,
    /// the same way `sf_model::position::parse_uci` needs a position to resolve
    /// an ambiguous UCI move.
    #[serde(default)]
    pub moves: String,
    /// Centiseconds remaining for the side that just moved, one entry per ply.
    #[serde(default)]
    pub clocks: Option<Vec<i64>>,
    /// One entry per ply when the game has been analysed; absent entirely for
    /// most in-progress bot games, which is not an error, just "not analysed
    /// yet" -- `game_export_updates` distinguishes the two rather than reading
    /// a missing array as every move being flagged.
    #[serde(default)]
    pub analysis: Option<Vec<MoveAnalysisJson>>,
    /// Only present for a non-standard start; absent means the standard
    /// starting position, same convention `initialFen` uses lichess-side.
    #[serde(default, rename = "initialFen")]
    pub initial_fen: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct MoveAnalysisJson {
    #[serde(default)]
    pub judgment: Option<JudgmentJson>,
}

#[derive(Debug, Deserialize)]
pub struct JudgmentJson {
    pub name: String,
}

// ---------------------------------------------------------------- conversion

pub fn account_update(bot: &BotId, json: &str) -> Result<Update> {
    let a: AccountJson = serde_json::from_str(json)?;
    Ok(Update::Account {
        bot: bot.clone(),
        account: Account {
            username: a.username,
            title: a.title,
            perfs: a
                .perfs
                .into_iter()
                .map(|(k, p)| {
                    (
                        k,
                        Perf {
                            rating: p.rating,
                            games: p.games,
                            rd: p.rd,
                            provisional: p.prov,
                            prog: p.prog,
                        },
                    )
                })
                .collect(),
            games_total: a.count.map(|c| c.all).unwrap_or(0),
        },
    })
}

/// Turn `/api/account/playing` into a `Playing` update **and** the position
/// observations it implicitly carries.
///
/// Free position data: the response has `fen` and `lastMove` for every game, and we
/// poll it anyway. Whether it is subject to lichess's three-move delay is F5 and is
/// unmeasured, so it ranks below the engine's telemetry and above the public
/// stream -- which the resolver's `Source` ordering encodes.
/// Returns the games alongside the updates -- not just inside
/// `Update::Playing` -- so `LichessSource::tick` can remember which games to
/// poll `/game/export/{id}` for without re-parsing its own output.
pub fn playing_updates(bot: &BotId, json: &str, now: Instant) -> Result<(Vec<Update>, Vec<PlayingGame>)> {
    let p: PlayingJson = serde_json::from_str(json)?;
    let mut out = Vec::new();
    let mut games = Vec::new();
    for g in p.now_playing {
        let id = GameId(g.game_id.clone());
        let colour = if g.color == "black" { shakmaty::Color::Black } else { shakmaty::Color::White };
        // The position, if it parses. A bad FEN here is not fatal: telemetry is the
        // primary source and this is a bonus.
        if let Ok(pos) = sf_model::position::parse_fen(&g.fen) {
            let last = g
                .last_move
                .as_deref()
                .and_then(|u| sf_model::position::parse_uci(&pos, u).ok());
            out.push(Update::Position {
                bot: bot.clone(),
                obs: sf_model::Obs::Playing { game: id.clone(), pos, last, at: now },
            });
        }
        if let Some(secs) = g.seconds_left {
            let d = Duration::from_secs(secs as u64);
            let (white, black) = match colour {
                shakmaty::Color::White => (Some(d), None),
                shakmaty::Color::Black => (None, Some(d)),
            };
            out.push(Update::Clocks {
                bot: bot.clone(),
                game: id.clone(),
                clocks: Clocks {
                    white,
                    black,
                    as_of: Some(now),
                    source: Some(sf_model::PosSource::Playing),
                    ticking: g.is_my_turn.then_some(colour),
                },
            });
        }
        let (username, title) = split_bot_title(g.opponent.username, g.opponent.title);
        games.push(PlayingGame {
            id,
            our_colour: colour,
            fen: g.fen,
            last_move: g.last_move,
            is_my_turn: g.is_my_turn,
            seconds_left: g.seconds_left,
            opponent: Opponent {
                username,
                title,
                rating: g.opponent.rating,
                provisional: g.opponent.provisional,
                ai_level: g.opponent.ai,
            },
            speed: g.speed,
            rated: g.rated,
        });
    }
    // The Playing update goes LAST, so the games it creates already have their
    // positions applied rather than being created empty and filled a frame later.
    let games_out = games.clone();
    out.push(Update::Playing { bot: bot.clone(), games });
    Ok((out, games_out))
}

/// `/api/account/playing`'s `opponent.username` has no `title` field in its
/// own OpenAPI schema at all -- confirmed against the spec, not assumed --
/// yet a BOT-titled opponent's `username` arrives as e.g. `"BOT rudim-bot"`,
/// undocumented but measured live. Split it back into a clean name and a
/// derived title so the two carry the same information every other endpoint
/// gives as separate fields, rather than a name every display site would
/// otherwise have to remember to de-prefix itself.
fn split_bot_title(username: String, title: Option<String>) -> (String, Option<String>) {
    match username.strip_prefix("BOT ") {
        Some(rest) => (rest.to_string(), Some("BOT".to_string())),
        None => (username, title),
    }
}

/// Turn `/game/export/{id}` into an `Update::Moves`. `our_colour` comes from
/// `PlayingGame` (already known from `/api/account/playing`) rather than from
/// `players.white`/`players.black` here, so this function does not need the
/// bot's own username at all.
///
/// Delayed by three moves on an ongoing game -- lichess's own anti-cheat
/// policy on this endpoint, not a bug here. `mind`/`curve` stay authoritative
/// for "what is happening right now"; this is authoritative for "what
/// actually got played," a few plies behind. Both true at once, see
/// `sf_model::position::Source` for how they're ranked against each other.
pub fn game_export_updates(bot: &BotId, game: &GameId, json: &str) -> Result<Update> {
    let g: GameExportJson = serde_json::from_str(json)?;

    let start = match g.initial_fen.as_deref() {
        Some(fen) => sf_model::position::parse_fen(fen).map_err(anyhow::Error::msg)?,
        None => shakmaty::Chess::default(),
    };

    let mut pos = start;
    let mut moves = Vec::new();
    for (ply, token) in g.moves.split_whitespace().enumerate() {
        let san: shakmaty::san::San = match token.parse() {
            Ok(s) => s,
            // A bad token this far in means the reply is truncated or the game
            // is a variant this parser does not support -- stop rather than
            // desync every ply after it from its real position.
            Err(_) => break,
        };
        let mv = match san.to_move(&pos) {
            Ok(m) => m,
            Err(_) => break,
        };
        let uci = mv.to_uci(shakmaty::CastlingMode::Standard).to_string();
        let san_text = san.to_string();

        let clock = g
            .clocks
            .as_ref()
            .and_then(|c| c.get(ply))
            .map(|cs| Duration::from_millis((*cs).max(0) as u64 * 10));

        // A move with no `judgment` in an analysed game is Best or Good, which
        // `Grade::mark` already renders identically (see its own doc comment) --
        // so collapsing that distinction here costs nothing on screen and keeps
        // this function from inventing a rule `Grade` does not have. An
        // unanalysed game (`analysis` entirely absent) leaves every grade
        // `None`: no data is a different claim from "this move was fine."
        let grade = g.analysis.as_ref().and_then(|a| a.get(ply)).map(|m| match &m.judgment {
            Some(j) => match j.name.as_str() {
                "Blunder" => Grade::Blunder,
                "Mistake" => Grade::Mistake,
                "Inaccuracy" => Grade::Inaccuracy,
                _ => Grade::Good,
            },
            None => Grade::Good,
        });

        moves.push(MoveRec {
            ply: ply as sf_model::Ply,
            san: san_text,
            uci,
            origin: MoveOrigin::Unknown,
            authority: sf_model::PosSource::Stream,
            clock,
            grade,
        });

        pos = match pos.play(mv) {
            Ok(p) => p,
            Err(_) => break,
        };
    }

    Ok(Update::Moves { bot: bot.clone(), game: game.clone(), moves })
}

/// The opponent's most recently known remaining time, read off what
/// `game_export_updates` already parsed rather than the response body a
/// second time. `MoveRec.clock` is "the time left for whoever just moved,
/// after moving" (see `GameExportJson::clocks`'s doc comment), so the
/// opponent's own most recent entry is exactly their last-known clock.
///
/// This is the only source the opponent's clock has at all: unlike our own
/// side, `/api/account/playing`'s `secondsLeft` is documented as the account
/// owner's own remaining time and never the opponent's. It carries the same
/// three-move delay as everything else this endpoint feeds -- not a bug, see
/// `game_export_updates`'s own doc comment -- and it costs nothing extra: no
/// new connection, no new endpoint, no new budget, because the export poll
/// that produces it already runs every 16s for the move list and grading.
pub fn opponent_clock(moves: &[MoveRec], our_colour: shakmaty::Color) -> Option<Duration> {
    let opponent = !our_colour;
    moves.iter().rev().find(|m| mover(m.ply) == opponent).and_then(|m| m.clock)
}

fn mover(ply: sf_model::Ply) -> shakmaty::Color {
    if ply % 2 == 0 { shakmaty::Color::White } else { shakmaty::Color::Black }
}

// ---------------------------------------------------------------- the source

/// Polls one bot's lichess endpoints through the governor.
pub struct LichessSource {
    pub bot: BotId,
    pub token: Option<String>,
    api: Api,
    account_due: Instant,
    playing_due: Instant,
    /// Games last seen from `/api/account/playing`, kept only so the export
    /// poll below knows which game IDs exist -- not a second copy of game
    /// state, `AppState` owns that once `Update::Playing` lands.
    games: Vec<PlayingGame>,
    moves_due: Instant,
}

impl LichessSource {
    pub fn new(bot: BotId, token: Option<String>, api: Api) -> Self {
        let now = Instant::now();
        LichessSource {
            bot,
            token,
            api,
            account_due: now,
            playing_due: now,
            games: Vec::new(),
            moves_due: now,
        }
    }

    /// One pass. The governor decides whether the call actually happens, so the
    /// intervals here are *wants*, not guarantees -- with two bots the effective
    /// cadence halves and the `api` panel says so.
    pub async fn tick(&mut self, now: Instant) -> Vec<Update> {
        let mut out = Vec::new();

        if now >= self.playing_due {
            self.playing_due = now + Duration::from_secs(2);
            match self
                .api
                .get(Endpoint::AccountPlaying, "/api/account/playing", self.token.clone())
                .await
            {
                Ok(body) => match playing_updates(&self.bot, &body, now) {
                    Ok((us, games)) => {
                        self.games = games;
                        out.extend(us);
                    }
                    Err(e) => out.push(Update::PlayingFailed {
                        bot: self.bot.clone(),
                        why: format!("parse: {e}"),
                    }),
                },
                Err(FetchError::NoBudget) | Err(FetchError::PenaltyBox(_)) => {
                    // Not a failure of the bot; a failure to have budget. Do not
                    // mark the field errored, or a busy governor looks like a dead
                    // connection.
                }
                Err(e) => {
                    out.push(Update::PlayingFailed { bot: self.bot.clone(), why: e.to_string() })
                }
            }
        }

        if now >= self.account_due {
            self.account_due = now + Duration::from_secs(45);
            match self.api.get(Endpoint::Account, "/api/account", self.token.clone()).await {
                Ok(body) => match account_update(&self.bot, &body) {
                    Ok(u) => out.push(u),
                    Err(e) => out.push(Update::AccountFailed {
                        bot: self.bot.clone(),
                        why: format!("parse: {e}"),
                    }),
                },
                Err(FetchError::NoBudget) | Err(FetchError::PenaltyBox(_)) => {}
                Err(e) => {
                    out.push(Update::AccountFailed { bot: self.bot.clone(), why: e.to_string() })
                }
            }
        }

        // The move list, for `moves`. One export per currently-playing game, no
        // per-game due-tracking: `self.games` is rarely more than one or two
        // long, and the governor's own budget (10/min on `GameExport`) is what
        // actually paces this when there is more than one, exactly like
        // `AccountPlaying` degrading gracefully with a second bot -- see the
        // `api` panel for the same "wanted vs effective" story applied here.
        //
        // 16s, not 8: unlike `AccountPlaying`'s 2s (measured -- 6/6 calls
        // answered, no 429, see that endpoint's doc comment), lichess does
        // not publish GameExport's real limit at all ("varied and ever
        // changing", per their own API guidance), and our 10/min is an
        // unverified guess. A 429 here costs a full 60s in the penalty box
        // (`Endpoint::GameExport::penalty()`) -- which is exactly the
        // "moves lag by almost a minute" symptom reported 2026-07-30, traced
        // to `penalty_until` being silently dropped before it ever reached
        // the screen (fixed separately) and rediscovered as a real risk
        // while looking into it: 8s against an unconfirmed budget was
        // needlessly aggressive for data a human is glancing at, not racing.
        if now >= self.moves_due {
            self.moves_due = now + Duration::from_secs(16);
            for g in self.games.clone() {
                let path = format!(
                    "/game/export/{}?moves=true&clocks=true&evals=true&opening=false&tags=false&division=false&pgnInJson=false",
                    g.id.0
                );
                match self.api.get(Endpoint::GameExport, path, self.token.clone()).await {
                    Ok(body) => match game_export_updates(&self.bot, &g.id, &body) {
                        Ok(u) => {
                            // The opponent's clock rides along on the same poll --
                            // see `opponent_clock`'s doc comment for why this is
                            // its only source at all.
                            if let Update::Moves { moves, .. } = &u {
                                if let Some(remaining) = opponent_clock(moves, g.our_colour) {
                                    out.push(Update::OpponentClock {
                                        bot: self.bot.clone(),
                                        game: g.id.clone(),
                                        remaining,
                                    });
                                }
                            }
                            out.push(u);
                        }
                        Err(e) => {
                            tracing::debug!(game = %g.id.0, error = %e, "unparseable game export");
                        }
                    },
                    Err(FetchError::NoBudget) | Err(FetchError::PenaltyBox(_)) => {}
                    Err(e) => {
                        tracing::debug!(game = %g.id.0, error = %e, "game export failed");
                    }
                }
            }
        }

        out
    }
}

/// The real fetcher.
pub struct HttpFetcher {
    poll: reqwest::Client,
}

impl HttpFetcher {
    pub fn new() -> Result<Self> {
        let poll = reqwest::Client::builder()
            .user_agent(USER_AGENT)
            .timeout(Duration::from_secs(10))
            .build()?;
        Ok(HttpFetcher { poll })
    }
}

impl crate::governor::Fetcher for HttpFetcher {
    fn get(
        &self,
        _endpoint: Endpoint,
        path: String,
        token: Option<String>,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<String, FetchError>> + Send>> {
        let client = self.poll.clone();
        Box::pin(async move {
            let mut req = client.get(format!("https://lichess.org{path}"));
            if let Some(t) = token {
                req = req.bearer_auth(t);
            }
            // `/game/export/{id}` serves PGN unless asked for JSON, and the Python
            // spent a while feeding PGN to `json.load` and calling the result a
            // network error.
            req = req.header("Accept", "application/json");
            let resp = req.send().await.map_err(|e| FetchError::Http(e.to_string()))?;
            if resp.status().as_u16() == 429 {
                return Err(FetchError::Throttled);
            }
            if !resp.status().is_success() {
                return Err(FetchError::Http(format!("HTTP {}", resp.status())));
            }
            resp.text().await.map_err(|e| FetchError::Http(e.to_string()))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ACCOUNT: &str = r#"{"username":"SumoFish","title":"BOT","perfs":{"rapid":{"rating":2379,"games":44,"rd":65,"prov":false,"prog":12},"bullet":{"rating":1950,"games":38,"rd":69,"prov":false}},"count":{"all":131}}"#;

    #[test]
    fn account_keeps_the_rating_deviation_the_python_dropped() {
        let u = account_update(&BotId("a".into()), ACCOUNT).unwrap();
        let Update::Account { account, .. } = u else { panic!() };
        assert_eq!(account.username, "SumoFish");
        assert_eq!(account.games_total, 131);
        let rapid = &account.perfs["rapid"];
        assert_eq!(rapid.rating, 2379);
        assert_eq!(rapid.rd, 65, "a provisional 2400 and a settled 2400 are different claims");
        assert!(!rapid.provisional);
        // Insertion order, so the header reads the same way every time.
        assert_eq!(account.perfs.keys().collect::<Vec<_>>(), vec!["rapid", "bullet"]);
    }

    const PLAYING: &str = r#"{"nowPlaying":[{"gameId":"abcd1234","color":"black","fen":"rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1","lastMove":"e2e4","isMyTurn":true,"secondsLeft":880,"opponent":{"username":"Foe","rating":2100,"provisional":true},"speed":"rapid","rated":true}]}"#;

    #[test]
    fn playing_yields_a_position_a_clock_and_the_list() {
        let now = Instant::now();
        let (ups, _) = playing_updates(&BotId("a".into()), PLAYING, now).unwrap();
        assert!(ups.iter().any(|u| matches!(
            u,
            Update::Position { obs: sf_model::Obs::Playing { .. }, .. }
        )));
        assert!(ups.iter().any(|u| matches!(u, Update::Clocks { .. })));
        // The list LAST, so the games already have their positions.
        assert!(matches!(ups.last(), Some(Update::Playing { .. })));
    }

    #[test]
    fn a_provisional_opponent_is_flagged() {
        let (ups, _) = playing_updates(&BotId("a".into()), PLAYING, Instant::now()).unwrap();
        let Some(Update::Playing { games, .. }) = ups.last() else { panic!() };
        assert!(games[0].opponent.provisional, "a provisional rating is noise, and must say so");
        assert_eq!(games[0].our_colour, shakmaty::Color::Black);
    }

    /// Measured live, 2026-07-31: `/api/account/playing` sends a BOT
    /// opponent's name as `"BOT rudim-bot"` in `username` with no separate
    /// `title` field at all (confirmed against the endpoint's own OpenAPI
    /// schema, which has no title property on `opponent`). Every display
    /// site should see a clean name plus a real title, not a name it has to
    /// remember to strip itself.
    #[test]
    fn a_bot_titled_opponent_has_the_title_split_out_of_the_name() {
        let bot_opponent = PLAYING.replace(r#""username":"Foe""#, r#""username":"BOT rudim-bot""#);
        let (ups, _) = playing_updates(&BotId("a".into()), &bot_opponent, Instant::now()).unwrap();
        let Some(Update::Playing { games, .. }) = ups.last() else { panic!() };
        assert_eq!(games[0].opponent.username, "rudim-bot", "the name must not carry the title as a prefix");
        assert_eq!(games[0].opponent.title.as_deref(), Some("BOT"));
    }

    /// A human opponent's name never starts with "BOT ", so this must be a
    /// no-op for the overwhelming common case rather than something that
    /// could ever mangle a real name that happens to start similarly.
    #[test]
    fn a_human_opponents_name_is_untouched() {
        let (ups, _) = playing_updates(&BotId("a".into()), PLAYING, Instant::now()).unwrap();
        let Some(Update::Playing { games, .. }) = ups.last() else { panic!() };
        assert_eq!(games[0].opponent.username, "Foe");
        assert_eq!(games[0].opponent.title, None);
    }

    #[test]
    fn the_clock_belongs_to_our_colour_and_ticks_only_on_our_turn() {
        let (ups, _) = playing_updates(&BotId("a".into()), PLAYING, Instant::now()).unwrap();
        let clocks = ups
            .iter()
            .find_map(|u| match u {
                Update::Clocks { clocks, .. } => Some(clocks),
                _ => None,
            })
            .unwrap();
        assert_eq!(clocks.black, Some(Duration::from_secs(880)));
        assert_eq!(clocks.white, None);
        assert_eq!(clocks.ticking, Some(shakmaty::Color::Black));
        assert_eq!(clocks.source, Some(sf_model::PosSource::Playing));
    }

    #[test]
    fn an_empty_list_still_produces_the_update_that_clears_the_board() {
        let (ups, _) = playing_updates(&BotId("a".into()), r#"{"nowPlaying":[]}"#, Instant::now())
            .unwrap();
        assert_eq!(ups.len(), 1);
        assert!(matches!(&ups[0], Update::Playing { games, .. } if games.is_empty()));
    }

    #[test]
    fn a_game_with_an_unparseable_fen_still_yields_its_entry() {
        let bad = PLAYING.replace(
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            "not a fen",
        );
        let (ups, _) = playing_updates(&BotId("a".into()), &bad, Instant::now()).unwrap();
        assert!(!ups.iter().any(|u| matches!(u, Update::Position { .. })));
        assert!(matches!(ups.last(), Some(Update::Playing { .. })), "the game itself survives");
    }

    #[test]
    fn unknown_fields_do_not_break_the_parse() {
        let future = PLAYING.replace(r#""rated":true"#, r#""rated":true,"newThing":{"a":1}"#);
        assert!(playing_updates(&BotId("a".into()), &future, Instant::now()).is_ok());
    }

    #[test]
    fn the_user_agent_identifies_us() {
        assert!(USER_AGENT.starts_with("sumofish-dash/"));
    }

    // ---------------------------------------------------------- game export

    const EXPORT: &str = r#"{
        "id": "abcd1234",
        "moves": "e4 e5 Nf3 Nc6 Bb5",
        "clocks": [30000, 29800, 30100, 29600, 30050],
        "analysis": [
            {},
            {},
            {},
            {"eval": 320},
            {"eval": -50, "judgment": {"name": "Mistake", "comment": "Mistake. Bc4 was best."}}
        ]
    }"#;

    #[test]
    fn moves_replay_into_correct_uci_with_the_right_ply_count() {
        let u = game_export_updates(&BotId("a".into()), &GameId("abcd1234".into()), EXPORT)
            .unwrap();
        let Update::Moves { moves, .. } = u else { panic!("expected Update::Moves") };
        assert_eq!(moves.len(), 5, "all five plies must replay, none dropped");
        let uci: Vec<&str> = moves.iter().map(|m| m.uci.as_str()).collect();
        // Ruy Lopez: e2e4 e7e5 g1f3 b8c6 f1b5. Getting these right end to end is
        // the actual point of replaying through shakmaty rather than trusting
        // the SAN string as display text -- a wrong UCI here is a wrong move
        // fed back into anything that trusts `MoveRec.uci`.
        assert_eq!(uci, vec!["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"]);
    }

    // ------------------------------------------------------ opponent clock

    /// The only source the opponent's clock has at all: ply parity picks out
    /// their moves from `game_export_updates`' own output, and the most
    /// recent one is their most recently known remaining time.
    #[test]
    fn opponent_clock_reads_their_own_most_recent_move_not_ours() {
        let u = game_export_updates(&BotId("a".into()), &GameId("abcd1234".into()), EXPORT)
            .unwrap();
        let Update::Moves { moves, .. } = u else { panic!() };

        // We're White: the opponent is Black, whose plies are 1 (e5) and 3
        // (Nc6). The most recent is ply 3, clock 29600cs = 296s.
        assert_eq!(
            opponent_clock(&moves, shakmaty::Color::White),
            Some(Duration::from_secs(296))
        );

        // We're Black: the opponent is White, whose plies are 0 (e4), 2
        // (Nf3), 4 (Bb5). The most recent is ply 4, clock 30050cs = 300.5s.
        assert_eq!(
            opponent_clock(&moves, shakmaty::Color::Black),
            Some(Duration::from_millis(300_500))
        );
    }

    /// Before the opponent has made a single move yet, there is nothing to
    /// report -- not a zero, not a stale guess, `None`.
    #[test]
    fn opponent_clock_is_none_before_their_first_move() {
        const ONE_MOVE: &str = r#"{"id": "x", "moves": "e4", "clocks": [30000]}"#;
        let u = game_export_updates(&BotId("a".into()), &GameId("x".into()), ONE_MOVE).unwrap();
        let Update::Moves { moves, .. } = u else { panic!() };
        // We're White and just moved; Black (the opponent) has not moved yet.
        assert_eq!(opponent_clock(&moves, shakmaty::Color::White), None);
    }

    /// No `clocks` array at all (an unanalysed or very fresh export) must not
    /// invent a value.
    #[test]
    fn opponent_clock_is_none_with_no_clocks_array() {
        const NO_CLOCKS: &str = r#"{"id": "x", "moves": "e4 e5"}"#;
        let u = game_export_updates(&BotId("a".into()), &GameId("x".into()), NO_CLOCKS).unwrap();
        let Update::Moves { moves, .. } = u else { panic!() };
        assert_eq!(opponent_clock(&moves, shakmaty::Color::White), None);
    }

    #[test]
    fn clocks_convert_centiseconds_to_a_duration() {
        let u = game_export_updates(&BotId("a".into()), &GameId("abcd1234".into()), EXPORT)
            .unwrap();
        let Update::Moves { moves, .. } = u else { panic!() };
        assert_eq!(moves[0].clock, Some(Duration::from_millis(300_000)));
    }

    #[test]
    fn an_analysed_move_with_no_judgment_grades_good_not_none() {
        let u = game_export_updates(&BotId("a".into()), &GameId("abcd1234".into()), EXPORT)
            .unwrap();
        let Update::Moves { moves, .. } = u else { panic!() };
        // Ply 3 (Nc6) has an eval but no judgment: analysed and fine, which is a
        // different claim from "never analysed" -- see the doc comment on the
        // `grade` computation in `game_export_updates`.
        assert_eq!(moves[3].grade, Some(Grade::Good));
    }

    #[test]
    fn a_flagged_move_carries_its_judgment_through() {
        let u = game_export_updates(&BotId("a".into()), &GameId("abcd1234".into()), EXPORT)
            .unwrap();
        let Update::Moves { moves, .. } = u else { panic!() };
        assert_eq!(moves[4].grade, Some(Grade::Mistake));
    }

    #[test]
    fn no_analysis_array_at_all_leaves_every_grade_none() {
        const NO_ANALYSIS: &str = r#"{
            "id": "abcd1234",
            "moves": "e4 e5 Nf3 Nc6 Bb5",
            "clocks": [30000, 29800, 30100, 29600, 30050]
        }"#;
        let u = game_export_updates(&BotId("a".into()), &GameId("abcd1234".into()), NO_ANALYSIS)
            .unwrap();
        let Update::Moves { moves, .. } = u else { panic!() };
        assert!(
            moves.iter().all(|m| m.grade.is_none()),
            "an unanalysed game must not invent grades: {:?}",
            moves.iter().map(|m| m.grade).collect::<Vec<_>>()
        );
    }

    #[test]
    fn a_truncated_or_illegal_move_stops_the_replay_without_panicking() {
        let bad = EXPORT.replace("Nc6 Bb5", "Nc6 Zz9");
        let u = game_export_updates(&BotId("a".into()), &GameId("abcd1234".into()), &bad)
            .unwrap();
        let Update::Moves { moves, .. } = u else { panic!() };
        assert_eq!(moves.len(), 4, "stops at the bad token, keeps everything before it");
    }
}
