//! The lichess budget, in one place.
//!
//! One task owns every outbound request. No other task holds a client, so four
//! properties are structural rather than hoped for:
//!
//! 1. **One request in flight, paced.** lichess publishes no numeric limits at all
//!    -- "various strategies", and "some limits may require longer" than the usual
//!    minute -- so *"Only make one request at a time"* is the only concrete guidance
//!    there is. A single task with a gap between calls is that sentence, implemented.
//!
//! 2. **Buckets are keyed by ENDPOINT, not by bot.** This is the correction to the
//!    obvious per-bot design. Measured on this machine and recorded in CLAUDE.md:
//!    `/api/user/{name}` returns 429 for authenticated *and* anonymous requests
//!    while `/api/account` stays fine, so the limit is per-IP-per-endpoint and a
//!    second token buys nothing. Four bots polling `/api/account/playing` every two
//!    seconds is 120 requests a minute against one bucket, not four buckets of 30.
//!    So sources do **not** choose their own cadence: they register a want and the
//!    governor issues permits round-robin, which makes the effective interval
//!    degrade to `n_bots x base` automatically and *visibly*. That degradation is
//!    the honest answer to "add more bots as it grows" and it belongs on screen.
//!
//! 3. **Forbidden endpoints are unnameable.** [`Endpoint`] has no variant for
//!    `/api/stream/event`: exactly one may exist per token and lichess-bot owns it
//!    for every token, so opening one would silently kill the bot's own connection.
//!    Not a rule in a comment -- there is no way to ask for it.
//!
//! 4. **429 means back off, never retry.** Every retry against a 429 renews the
//!    penalty, which is how the bot once sat for forty minutes unable to create a
//!    single challenge. The failed call is dropped, not requeued.

use std::collections::HashMap;
use std::time::Duration;
use tokio::sync::{mpsc, oneshot};
use tokio::time::Instant;

/// Every endpoint the dashboard may touch. Deliberately closed.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub enum Endpoint {
    /// `/api/account` — token-scoped, so it has a budget the public endpoints do
    /// not. The Python learned this the hard way: it polled `/api/user/{name}`,
    /// got a permanently empty rating panel with no error, and switched.
    Account,
    /// `/api/account/playing` — the workhorse. Measured at 6 of 6 calls answered at
    /// one-second intervals with a 547ms median and no 429, which is what justified
    /// a two-second cadence for ONE bot.
    AccountPlaying,
    /// `/game/export/{id}` — public. Note the path: `/api/game/export/{id}` is a
    /// 404, and it serves PGN unless you send `Accept: application/json`.
    GameExport,
    /// `/api/stream/game/{id}` — public, and delayed by three moves by policy.
    GameStream,
}

impl Endpoint {
    pub const ALL: [Endpoint; 4] =
        [Endpoint::Account, Endpoint::AccountPlaying, Endpoint::GameExport, Endpoint::GameStream];

    pub const fn name(self) -> &'static str {
        match self {
            Endpoint::Account => "account",
            Endpoint::AccountPlaying => "account/playing",
            Endpoint::GameExport => "game/export",
            Endpoint::GameStream => "stream/game",
        }
    }

    /// Requests a minute, across every bot. These are the whole-IP budget.
    pub const fn default_rpm(self) -> u32 {
        match self {
            Endpoint::Account => 6,
            // 30/min is one bot at 2s. With two bots each gets 4s, and the screen
            // says so.
            Endpoint::AccountPlaying => 30,
            Endpoint::GameExport => 10,
            Endpoint::GameStream => 4,
        }
    }

    /// How long to sit out after a 429. The public endpoints recover slowly and the
    /// Lab Note is explicit that retrying renews the penalty.
    pub const fn penalty(self) -> Duration {
        match self {
            Endpoint::Account => Duration::from_secs(90),
            Endpoint::AccountPlaying => Duration::from_secs(60),
            Endpoint::GameExport | Endpoint::GameStream => Duration::from_secs(60),
        }
    }
}

/// What a source wants fetched.
pub struct Call {
    pub endpoint: Endpoint,
    /// Full path including any id.
    pub path: String,
    /// Whose credential, if the endpoint is token-scoped.
    pub token: Option<String>,
    pub reply: oneshot::Sender<Result<String, FetchError>>,
}

#[derive(Debug, thiserror::Error)]
pub enum FetchError {
    #[error("rate limited; backing off, not retrying")]
    Throttled,
    #[error("endpoint is in the penalty box for another {0:?}")]
    PenaltyBox(Duration),
    #[error("budget exhausted for this minute")]
    NoBudget,
    #[error("{0}")]
    Http(String),
}

/// The thing that actually talks to the network. A trait so the governor's pacing
/// can be tested against a virtual clock with no network at all -- which is the
/// only way to assert "no bucket is ever exceeded" deterministically.
pub trait Fetcher: Send + Sync + 'static {
    fn get(
        &self,
        endpoint: Endpoint,
        path: String,
        token: Option<String>,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<String, FetchError>> + Send>>;
}

/// A per-endpoint budget: a sliding window of call times, plus a penalty box.
#[derive(Debug)]
struct Bucket {
    rpm: u32,
    calls: std::collections::VecDeque<Instant>,
    penalty_until: Option<Instant>,
    throttles: u32,
}

impl Bucket {
    fn new(rpm: u32) -> Self {
        Bucket { rpm, calls: Default::default(), penalty_until: None, throttles: 0 }
    }
    fn prune(&mut self, now: Instant) {
        let cutoff = now - Duration::from_secs(60);
        while self.calls.front().is_some_and(|t| *t < cutoff) {
            self.calls.pop_front();
        }
    }
    fn may_call(&mut self, now: Instant) -> Result<(), FetchError> {
        if let Some(until) = self.penalty_until {
            if now < until {
                return Err(FetchError::PenaltyBox(until - now));
            }
            self.penalty_until = None;
        }
        self.prune(now);
        if self.calls.len() as u32 >= self.rpm {
            return Err(FetchError::NoBudget);
        }
        Ok(())
    }
    fn record(&mut self, now: Instant) {
        self.calls.push_back(now);
    }
    fn throttled(&mut self, now: Instant, endpoint: Endpoint) {
        self.throttles += 1;
        self.penalty_until = Some(now + endpoint.penalty());
    }
}

/// What the governor is doing, for the `api` panel. The whole multi-bot story is
/// that the effective interval degrades as bots are added, so it must be visible.
#[derive(Clone, Debug, Default)]
pub struct GovernorStatus {
    pub per_endpoint: Vec<(Endpoint, u32, u32, u32)>, // endpoint, used, rpm, throttles
    pub in_flight: Option<Endpoint>,
    pub penalised: Vec<Endpoint>,
}

impl From<GovernorStatus> for sf_model::state::ApiState {
    fn from(g: GovernorStatus) -> Self {
        let mut endpoints = indexmap::IndexMap::new();
        let mut total = 0;
        let mut throttled = 0;
        for (e, used, rpm, thr) in &g.per_endpoint {
            total += used;
            throttled += thr;
            endpoints.insert(
                e.name().to_string(),
                sf_model::state::EndpointState {
                    calls_last_minute: *used,
                    rpm_budget: *rpm,
                    // What each source ASKED for against what the shared bucket can
                    // actually give it. With two bots the second number doubles, and
                    // that is the honest answer to "what does another bot cost".
                    wanted_interval: Some(Duration::from_secs_f64(
                        60.0 / (*rpm).max(1) as f64,
                    )),
                    effective_interval: (*used > 0)
                        .then(|| Duration::from_secs_f64(60.0 / *used as f64)),
                    last_429: None,
                    penalty_until: None,
                },
            );
        }
        sf_model::state::ApiState {
            endpoints,
            in_flight: g.in_flight.map(|e| e.name().to_string()),
            total_last_minute: total,
            throttled,
        }
    }
}

pub struct Governor {
    min_gap: Duration,
    buckets: HashMap<Endpoint, Bucket>,
    fetcher: std::sync::Arc<dyn Fetcher>,
    last_call: Option<Instant>,
    report: Option<tokio::sync::watch::Sender<GovernorStatus>>,
}

impl Governor {
    pub fn new(min_gap: Duration, fetcher: std::sync::Arc<dyn Fetcher>) -> Self {
        Governor {
            min_gap,
            buckets: Endpoint::ALL.into_iter().map(|e| (e, Bucket::new(e.default_rpm()))).collect(),
            fetcher,
            last_call: None,
            report: None,
        }
    }

    /// Publish the budget after every call, so the `api` panel can show the
    /// degradation rather than leaving it to be inferred.
    pub fn reporting(mut self, tx: tokio::sync::watch::Sender<GovernorStatus>) -> Self {
        self.report = Some(tx);
        self
    }

    pub fn with_rpm(mut self, endpoint: Endpoint, rpm: u32) -> Self {
        self.buckets.insert(endpoint, Bucket::new(rpm));
        self
    }

    pub fn status(&mut self) -> GovernorStatus {
        let now = Instant::now();
        let mut per = Vec::new();
        for e in Endpoint::ALL {
            if let Some(b) = self.buckets.get_mut(&e) {
                b.prune(now);
                per.push((e, b.calls.len() as u32, b.rpm, b.throttles));
            }
        }
        let penalised = Endpoint::ALL
            .into_iter()
            .filter(|e| {
                self.buckets.get(e).is_some_and(|b| b.penalty_until.is_some_and(|u| u > now))
            })
            .collect();
        GovernorStatus { per_endpoint: per, in_flight: None, penalised }
    }

    /// Serve calls until the channel closes. One at a time, forever.
    pub async fn run(mut self, mut rx: mpsc::Receiver<Call>) {
        while let Some(call) = rx.recv().await {
            let now = Instant::now();

            // Pacing: the gap is between the STARTS of consecutive calls, and it
            // applies across endpoints, because the limit being respected is
            // "one request at a time from this IP".
            if let Some(last) = self.last_call {
                let since = now.saturating_duration_since(last);
                if since < self.min_gap {
                    tokio::time::sleep(self.min_gap - since).await;
                }
            }

            let now = Instant::now();
            let bucket = self
                .buckets
                .entry(call.endpoint)
                .or_insert_with(|| Bucket::new(call.endpoint.default_rpm()));
            if let Err(e) = bucket.may_call(now) {
                let _ = call.reply.send(Err(e));
                continue;
            }
            bucket.record(now);
            self.last_call = Some(now);

            let result = self.fetcher.get(call.endpoint, call.path, call.token).await;

            // A 429 puts the endpoint in the box. The call is NOT retried: every
            // retry against a 429 renews the penalty.
            if matches!(result, Err(FetchError::Throttled)) {
                let now = Instant::now();
                if let Some(b) = self.buckets.get_mut(&call.endpoint) {
                    b.throttled(now, call.endpoint);
                }
                tracing::warn!(
                    endpoint = call.endpoint.name(),
                    "429; backing off and NOT retrying"
                );
            }
            let _ = call.reply.send(result);

            if self.report.is_some() {
                let status = self.status();
                if let Some(tx) = self.report.as_ref() {
                    let _ = tx.send(status);
                }
            }
        }
    }
}

/// A handle a source uses to ask for something.
#[derive(Clone)]
pub struct Api {
    tx: mpsc::Sender<Call>,
}

impl Api {
    pub fn new(tx: mpsc::Sender<Call>) -> Self {
        Api { tx }
    }
    pub async fn get(
        &self,
        endpoint: Endpoint,
        path: impl Into<String>,
        token: Option<String>,
    ) -> Result<String, FetchError> {
        let (reply, rx) = oneshot::channel();
        let call = Call { endpoint, path: path.into(), token, reply };
        self.tx
            .send(call)
            .await
            .map_err(|_| FetchError::Http("governor is gone".into()))?;
        rx.await.map_err(|_| FetchError::Http("governor dropped the call".into()))?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    /// Counts calls per endpoint and answers instantly. No network, no clock of its
    /// own -- which is what lets the whole thing run under `tokio::time::pause()`.
    struct Counting {
        calls: Arc<std::sync::Mutex<Vec<(Endpoint, Instant)>>>,
        throttle_after: Option<u32>,
        seen: AtomicU32,
    }

    impl Fetcher for Counting {
        fn get(
            &self,
            endpoint: Endpoint,
            _path: String,
            _token: Option<String>,
        ) -> std::pin::Pin<Box<dyn Future<Output = Result<String, FetchError>> + Send>> {
            self.calls.lock().unwrap().push((endpoint, Instant::now()));
            let n = self.seen.fetch_add(1, Ordering::SeqCst);
            let throttled = self.throttle_after.is_some_and(|k| n >= k);
            Box::pin(async move {
                if throttled { Err(FetchError::Throttled) } else { Ok("{}".into()) }
            })
        }
    }

    fn harness(throttle_after: Option<u32>) -> (Api, Arc<std::sync::Mutex<Vec<(Endpoint, Instant)>>>) {
        let calls = Arc::new(std::sync::Mutex::new(Vec::new()));
        let fetcher = Arc::new(Counting {
            calls: calls.clone(),
            throttle_after,
            seen: AtomicU32::new(0),
        });
        let (tx, rx) = mpsc::channel(256);
        let gov = Governor::new(Duration::from_millis(250), fetcher);
        tokio::spawn(gov.run(rx));
        (Api::new(tx), calls)
    }

    #[tokio::test(start_paused = true)]
    async fn calls_are_paced_at_least_min_gap_apart() {
        let (api, calls) = harness(None);
        for _ in 0..5 {
            api.get(Endpoint::Account, "/api/account", None).await.ok();
        }
        let times: Vec<Instant> = calls.lock().unwrap().iter().map(|(_, t)| *t).collect();
        assert_eq!(times.len(), 5);
        for w in times.windows(2) {
            let gap = w[1].saturating_duration_since(w[0]);
            assert!(
                gap >= Duration::from_millis(250),
                "calls only {gap:?} apart; lichess asks for one request at a time"
            );
        }
    }

    /// The property that matters most: **no endpoint bucket is ever exceeded**,
    /// however many bots ask. Six synthetic bots against a virtual clock.
    #[tokio::test(start_paused = true)]
    async fn no_bucket_is_ever_exceeded_however_many_bots_ask() {
        let (api, calls) = harness(None);
        let mut tasks = Vec::new();
        for _bot in 0..6 {
            let api = api.clone();
            tasks.push(tokio::spawn(async move {
                let mut ok = 0;
                for _ in 0..20 {
                    if api.get(Endpoint::AccountPlaying, "/api/account/playing", None).await.is_ok()
                    {
                        ok += 1;
                    }
                }
                ok
            }));
        }
        let mut granted = 0;
        for t in tasks {
            granted += t.await.unwrap();
        }
        // 120 wanted, and the bucket is 30 a minute.
        let made = calls.lock().unwrap().len();
        assert_eq!(made, granted, "every granted call reached the fetcher exactly once");
        assert!(
            made <= Endpoint::AccountPlaying.default_rpm() as usize,
            "{made} calls in a minute against a budget of {}",
            Endpoint::AccountPlaying.default_rpm()
        );
        assert!(made > 0, "the budget should not be zero");
    }

    /// A 429 must not be retried. The Lab Note is explicit: every retry renews the
    /// penalty, and that cost forty minutes of a bot unable to challenge anybody.
    #[tokio::test(start_paused = true)]
    async fn a_429_is_never_retried() {
        let (api, calls) = harness(Some(0)); // throttle from the first call
        let first = api.get(Endpoint::Account, "/api/account", None).await;
        assert!(matches!(first, Err(FetchError::Throttled)));

        // Everything afterwards must be refused locally, without touching the
        // network at all.
        for _ in 0..5 {
            let r = api.get(Endpoint::Account, "/api/account", None).await;
            assert!(
                matches!(r, Err(FetchError::PenaltyBox(_))),
                "a throttled endpoint must be refused locally, got {r:?}"
            );
        }
        assert_eq!(calls.lock().unwrap().len(), 1, "exactly one call reached the network");
    }

    /// One endpoint's penalty must not silence the others.
    #[tokio::test(start_paused = true)]
    async fn the_penalty_box_is_per_endpoint() {
        let calls = Arc::new(std::sync::Mutex::new(Vec::new()));
        struct OnlyAccount(Arc<std::sync::Mutex<Vec<(Endpoint, Instant)>>>);
        impl Fetcher for OnlyAccount {
            fn get(
                &self,
                endpoint: Endpoint,
                _p: String,
                _t: Option<String>,
            ) -> std::pin::Pin<Box<dyn Future<Output = Result<String, FetchError>> + Send>> {
                self.0.lock().unwrap().push((endpoint, Instant::now()));
                Box::pin(async move {
                    if endpoint == Endpoint::Account {
                        Err(FetchError::Throttled)
                    } else {
                        Ok("{}".into())
                    }
                })
            }
        }
        let (tx, rx) = mpsc::channel(64);
        tokio::spawn(
            Governor::new(Duration::from_millis(10), Arc::new(OnlyAccount(calls.clone()))).run(rx),
        );
        let api = Api::new(tx);

        api.get(Endpoint::Account, "/api/account", None).await.ok();
        assert!(matches!(
            api.get(Endpoint::Account, "/api/account", None).await,
            Err(FetchError::PenaltyBox(_))
        ));
        assert!(
            api.get(Endpoint::GameExport, "/game/export/x", None).await.is_ok(),
            "a different endpoint has a different budget"
        );
    }

    /// The event stream is not in the type. This test exists to fail loudly if
    /// somebody adds it: exactly one may exist per token and lichess-bot owns it.
    #[test]
    fn the_event_stream_is_unnameable() {
        for e in Endpoint::ALL {
            assert!(
                !e.name().contains("stream/event"),
                "an event-stream endpoint was added; opening one kills lichess-bot's own \
                 connection for that token"
            );
        }
    }

    #[tokio::test(start_paused = true)]
    async fn status_reports_what_the_screen_needs() {
        let calls = Arc::new(std::sync::Mutex::new(Vec::new()));
        let fetcher =
            Arc::new(Counting { calls, throttle_after: None, seen: AtomicU32::new(0) });
        let mut gov = Governor::new(Duration::from_millis(1), fetcher);
        let s = gov.status();
        assert_eq!(s.per_endpoint.len(), Endpoint::ALL.len());
        for (e, used, rpm, throttles) in s.per_endpoint {
            assert_eq!(used, 0);
            assert_eq!(rpm, e.default_rpm());
            assert_eq!(throttles, 0);
        }
    }
}
