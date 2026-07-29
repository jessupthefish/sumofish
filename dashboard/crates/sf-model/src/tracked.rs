//! Staleness in the data model, not in the styling.
//!
//! `Tracked<T>` replaces the Python `Field` and closes its one loophole.
//! `State.get()` there returned the bare value and discarded how old it was, so
//! the docstring's promise -- "you cannot read a value without being handed how
//! old it is" -- was aspirational. Here there is **no accessor that returns the
//! value alone**: `get` hands back `(&T, Track)` or nothing.
//!
//! The failure this exists to prevent is the original loop's
//! `profile = api(...) or profile`, where a dead connection rendered
//! pixel-identical to a live one.

use std::time::{Duration, Instant};

pub use sf_track::Track;

/// Re-exported so `sf-model` does not depend on `sf-theme` (which owns the ink
/// for each track) while still naming the same three states.
mod sf_track {
    /// Freshness of a value. Mirrors `sf_theme::Track`; the conversion is in
    /// sf-layout, which sees both.
    #[derive(Clone, Copy, PartialEq, Eq, Debug, Default, PartialOrd, Ord)]
    pub enum Track {
        #[default]
        Live,
        Coast,
        Lost,
    }
}

/// A value plus its provenance. `interval` is how often the source *intends* to
/// refresh, which is what makes "late" meaningful without a second constant.
#[derive(Clone, Debug)]
pub struct Tracked<T> {
    name: &'static str,
    interval: Duration,
    value: Option<T>,
    updated: Option<Instant>,
    error: Option<String>,
    /// How many times a value has landed. Zero means never, which is `Lost` and
    /// not `Live`, because a field that has never been fed is not fresh.
    fills: u64,
}

impl<T> Tracked<T> {
    pub fn new(name: &'static str, interval: Duration) -> Self {
        Self { name, interval, value: None, updated: None, error: None, fills: 0 }
    }

    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn interval(&self) -> Duration {
        self.interval
    }

    /// The only way to read the value, and it always comes with the track.
    pub fn get(&self, now: Instant) -> Option<(&T, Track)> {
        self.value.as_ref().map(|v| (v, self.track(now)))
    }

    /// For the rare caller that wants the track without borrowing the value.
    pub fn track(&self, now: Instant) -> Track {
        let Some(updated) = self.updated else { return Track::Lost };
        if self.fills == 0 {
            return Track::Lost;
        }
        let age = now.saturating_duration_since(updated);
        if age > self.interval * 6 {
            Track::Lost
        } else if self.error.is_some() && age > self.interval * 2 {
            // An error on top of being late is worse than being late.
            Track::Lost
        } else if age > self.interval * 2 {
            Track::Coast
        } else {
            Track::Live
        }
    }

    pub fn age(&self, now: Instant) -> Option<Duration> {
        self.updated.map(|u| now.saturating_duration_since(u))
    }

    pub fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }

    /// A value landed. Clears any error, because arriving is the definition of
    /// recovered.
    pub fn set(&mut self, value: T, now: Instant) {
        self.value = Some(value);
        self.updated = Some(now);
        self.error = None;
        self.fills += 1;
    }

    /// A fetch failed. Records the reason and **leaves the value and its
    /// timestamp alone**, so the field coasts and then goes lost on its own
    /// schedule rather than blanking the instant one request fails.
    pub fn fail(&mut self, why: impl Into<String>) {
        self.error = Some(why.into());
    }

    /// Mutate in place, for sources that accumulate rather than replace (a tail,
    /// a curve). Stamps the update the same way `set` does.
    pub fn update_with(&mut self, now: Instant, f: impl FnOnce(&mut T)) -> bool {
        match self.value.as_mut() {
            Some(v) => {
                f(v);
                self.updated = Some(now);
                self.error = None;
                self.fills += 1;
                true
            }
            None => false,
        }
    }

    pub fn or_insert_with(&mut self, now: Instant, make: impl FnOnce() -> T) -> &mut T {
        if self.value.is_none() {
            self.value = Some(make());
            self.updated = Some(now);
            self.fills += 1;
        }
        self.value.as_mut().expect("just inserted")
    }
}

/// Human-readable age, matching the Python `ago()` so the screen reads the same:
/// under 90s in seconds, under 90 minutes in minutes, then hours.
pub fn ago(d: Option<Duration>) -> String {
    match d {
        None => "never".into(),
        Some(d) => {
            let s = d.as_secs();
            if s < 90 {
                format!("{s}s")
            } else if s < 5400 {
                format!("{}m", s / 60)
            } else {
                format!("{}h", s / 3600)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(base: Instant, secs: u64) -> Instant {
        base + Duration::from_secs(secs)
    }

    #[test]
    fn never_fed_is_lost_not_live() {
        let t: Tracked<u32> = Tracked::new("x", Duration::from_secs(5));
        let now = Instant::now();
        assert_eq!(t.track(now), Track::Lost);
        assert!(t.get(now).is_none());
    }

    #[test]
    fn grades_by_multiples_of_the_intended_interval() {
        let base = Instant::now();
        let mut t = Tracked::new("x", Duration::from_secs(5));
        t.set(1u32, base);
        assert_eq!(t.track(at(base, 0)), Track::Live);
        assert_eq!(t.track(at(base, 9)), Track::Live); // < 2x
        assert_eq!(t.track(at(base, 11)), Track::Coast); // > 2x
        assert_eq!(t.track(at(base, 29)), Track::Coast); // < 6x
        assert_eq!(t.track(at(base, 31)), Track::Lost); // > 6x
    }

    #[test]
    fn a_failure_coasts_rather_than_blanking() {
        let base = Instant::now();
        let mut t = Tracked::new("x", Duration::from_secs(5));
        t.set(7u32, base);
        t.fail("timeout");
        // The value survives, which is the whole point: one failed request must
        // not empty the panel.
        let (v, track) = t.get(at(base, 1)).expect("value survives a failure");
        assert_eq!(*v, 7);
        assert_eq!(track, Track::Live);
        // But an error plus lateness is lost sooner than lateness alone.
        assert_eq!(t.track(at(base, 11)), Track::Lost);
    }

    #[test]
    fn ago_reads_like_the_python() {
        assert_eq!(ago(None), "never");
        assert_eq!(ago(Some(Duration::from_secs(3))), "3s");
        assert_eq!(ago(Some(Duration::from_secs(89))), "89s");
        assert_eq!(ago(Some(Duration::from_secs(90))), "1m");
        assert_eq!(ago(Some(Duration::from_secs(5399))), "89m");
        assert_eq!(ago(Some(Duration::from_secs(5400))), "1h");
    }
}
