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
pub fn playing_updates(bot: &BotId, json: &str, now: Instant) -> Result<Vec<Update>> {
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
        games.push(PlayingGame {
            id,
            our_colour: colour,
            fen: g.fen,
            last_move: g.last_move,
            is_my_turn: g.is_my_turn,
            seconds_left: g.seconds_left,
            opponent: Opponent {
                username: g.opponent.username,
                title: g.opponent.title,
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
    out.push(Update::Playing { bot: bot.clone(), games });
    Ok(out)
}

// ---------------------------------------------------------------- the source

/// Polls one bot's lichess endpoints through the governor.
pub struct LichessSource {
    pub bot: BotId,
    pub token: Option<String>,
    api: Api,
    account_due: Instant,
    playing_due: Instant,
}

impl LichessSource {
    pub fn new(bot: BotId, token: Option<String>, api: Api) -> Self {
        let now = Instant::now();
        LichessSource { bot, token, api, account_due: now, playing_due: now }
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
                    Ok(us) => out.extend(us),
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
        let ups = playing_updates(&BotId("a".into()), PLAYING, now).unwrap();
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
        let ups = playing_updates(&BotId("a".into()), PLAYING, Instant::now()).unwrap();
        let Some(Update::Playing { games, .. }) = ups.last() else { panic!() };
        assert!(games[0].opponent.provisional, "a provisional rating is noise, and must say so");
        assert_eq!(games[0].our_colour, shakmaty::Color::Black);
    }

    #[test]
    fn the_clock_belongs_to_our_colour_and_ticks_only_on_our_turn() {
        let ups = playing_updates(&BotId("a".into()), PLAYING, Instant::now()).unwrap();
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
        let ups = playing_updates(&BotId("a".into()), r#"{"nowPlaying":[]}"#, Instant::now())
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
        let ups = playing_updates(&BotId("a".into()), &bad, Instant::now()).unwrap();
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
}
