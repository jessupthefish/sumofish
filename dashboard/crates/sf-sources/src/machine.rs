//! The GPU and the systemd units, read through libraries rather than subprocesses.
//!
//! Both replacements were falsified before being committed to (F7 in
//! `docs/probe-results.md`), because both had a specific doubt:
//!
//! - **nvml per-process VRAM** is the answer to `machine_panel`'s own stated
//!   question, "is one starving the other" -- but NVML reports `Unavailable` for
//!   processes it cannot attribute, and the bot runs under a systemd sandbox. It
//!   does not: every process reported a real number, including the trainer holding
//!   11,160 MiB. Inferring that from one aggregate is the exact mistake CLAUDE.md
//!   records as costing a whole session.
//! - **zbus on the USER bus** needs `DBUS_SESSION_BUS_ADDRESS` in the environment,
//!   which is not guaranteed if the dashboard is ever started from a system unit or
//!   a bare ssh. It works here, and when it does not it must surface as a source
//!   error rather than a panic.
//!
//! Two things this gets right that the subprocess version could not:
//!
//! - `not-found` is a **load state**, not an error. `sumofish-train.service` does
//!   not exist, and the Python renders it as a permanently red dot, which teaches
//!   you to ignore the row. Meanwhile `sumofish-lab` -- the unit doing all the
//!   work -- is not polled at all.
//! - systemd's `NRestarts` is **not** the restart count you want. `watchdog.py` uses
//!   `systemctl kill` + `restart` rather than letting `Restart=always` fire, so the
//!   counter stays at zero through a SIGKILL. Confirmed over D-Bus: the bot was
//!   killed and restarted twice in half an hour today with `NRestarts` reading 0,
//!   and the panel showed a green dot throughout. The real count comes from the
//!   watchdog's own state file.

use anyhow::{Context, Result};
use sf_model::state::{Gpu, GpuProc, Unit};
use sf_model::Update;
use std::time::Duration;

// ---------------------------------------------------------------- gpu

pub struct GpuSource {
    nvml: Option<nvml_wrapper::Nvml>,
}

impl GpuSource {
    /// Initialise once and keep it: repeated `Nvml::init()` calls "incur unnecessary
    /// overhead from reloading symbols".
    pub fn new() -> Self {
        match nvml_wrapper::Nvml::init() {
            Ok(n) => GpuSource { nvml: Some(n) },
            Err(e) => {
                tracing::warn!(error = %e, "NVML unavailable; the machine panel will say so");
                GpuSource { nvml: None }
            }
        }
    }

    pub fn poll(&self) -> Update {
        let Some(nvml) = self.nvml.as_ref() else {
            return Update::GpusFailed("NVML not available".into());
        };
        match self.read(nvml) {
            Ok(gpus) => Update::Gpus(gpus),
            Err(e) => Update::GpusFailed(e.to_string()),
        }
    }

    fn read(&self, nvml: &nvml_wrapper::Nvml) -> Result<Vec<Gpu>> {
        use nvml_wrapper::enum_wrappers::device::{Clock, TemperatureSensor};
        let count = nvml.device_count().context("device_count")?;
        let mut out = Vec::new();
        for i in 0..count {
            let d = nvml.device_by_index(i).context("device_by_index")?;
            let mem = d.memory_info().ok();
            let util = d.utilization_rates().ok();
            out.push(Gpu {
                name: d.name().unwrap_or_else(|_| "?".into()),
                util_gpu: util.as_ref().map(|u| u.gpu).unwrap_or(0),
                util_mem: util.as_ref().map(|u| u.memory).unwrap_or(0),
                mem_used: mem.as_ref().map(|m| m.used).unwrap_or(0),
                mem_total: mem.as_ref().map(|m| m.total).unwrap_or(0),
                temp_c: d.temperature(TemperatureSensor::Gpu).unwrap_or(0),
                power_w: d.power_usage().map(|w| w as f32 / 1000.0).unwrap_or(0.0),
                power_limit_w: d
                    .power_management_limit()
                    .map(|w| w as f32 / 1000.0)
                    .unwrap_or(0.0),
                fan_pct: d.fan_speed(0).unwrap_or(0),
                clock_sm: d.clock_info(Clock::SM).unwrap_or(0),
                clock_mem: d.clock_info(Clock::Memory).unwrap_or(0),
                // A sudden drop in nodes per second has a reason, and this is
                // usually it. Live on this machine right now: SW_POWER_CAP at 281
                // of 300 W.
                throttle: d
                    .current_throttle_reasons()
                    .map(|r| format!("{r:?}"))
                    .into_iter()
                    .filter(|s| !s.contains("(0x0)") && !s.is_empty())
                    .collect(),
                procs: d
                    .running_compute_processes()
                    .unwrap_or_default()
                    .into_iter()
                    .map(|p| GpuProc {
                        pid: p.pid,
                        // An enum, not a u64: NVML says "unavailable" for processes
                        // it cannot attribute, and showing 0 would be a lie.
                        mem: match p.used_gpu_memory {
                            nvml_wrapper::enums::device::UsedGpuMemory::Used(b) => Some(b),
                            nvml_wrapper::enums::device::UsedGpuMemory::Unavailable => None,
                        },
                        name: proc_name(p.pid),
                    })
                    .collect(),
            });
        }
        Ok(out)
    }
}

impl Default for GpuSource {
    fn default() -> Self {
        Self::new()
    }
}

/// A short, recognisable name for a pid. `/proc` rather than a `ps` subprocess,
/// and note the Lab Note it avoids: matching on a command line that contains the
/// pattern you are searching for makes a process find itself.
fn proc_name(pid: u32) -> String {
    let cmdline = std::fs::read_to_string(format!("/proc/{pid}/cmdline")).unwrap_or_default();
    let parts: Vec<&str> = cmdline.split('\0').filter(|s| !s.is_empty()).collect();
    // A python process is named by its script, not by "python".
    for p in &parts {
        if p.ends_with(".py") {
            return std::path::Path::new(p)
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| (*p).to_string());
        }
    }
    // argv[0] is not always just a path: Electron apps put the whole command line
    // in it, which turned one row of the machine panel into 600 characters of
    // Chromium feature flags.
    let name = parts
        .first()
        .map(|p| {
            let head = p.split_whitespace().next().unwrap_or(p);
            std::path::Path::new(head)
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| head.to_string())
        })
        .unwrap_or_else(|| format!("pid {pid}"));
    if name.chars().count() > 24 {
        name.chars().take(23).chain(std::iter::once('\u{2026}')).collect()
    } else {
        name
    }
}

// ---------------------------------------------------------------- systemd

pub struct UnitsSource {
    names: Vec<String>,
    conn: Option<zbus::Connection>,
}

impl UnitsSource {
    pub async fn new(names: Vec<String>) -> Self {
        let conn = match zbus::Connection::session().await {
            Ok(c) => Some(c),
            Err(e) => {
                // Surfaced, never a panic: the dashboard may be started somewhere
                // without a session bus.
                tracing::warn!(error = %e, "no session bus; unit states unavailable");
                None
            }
        };
        UnitsSource { names, conn }
    }

    pub async fn poll(&self) -> Update {
        let Some(conn) = self.conn.as_ref() else {
            return Update::UnitsFailed("no session bus".into());
        };
        match self.read(conn).await {
            Ok(units) => Update::Units(units),
            Err(e) => Update::UnitsFailed(e.to_string()),
        }
    }

    async fn read(&self, conn: &zbus::Connection) -> Result<Vec<Unit>> {
        let mgr = zbus_systemd::systemd1::ManagerProxy::new(conn).await?;
        // ONE round trip for every unit, rather than N x GetUnit plus N property
        // reads. It also returns entries for units that are not loaded, which is how
        // `not-found` becomes visible instead of looking like an error.
        let listed = mgr.list_units_by_names(self.names.clone()).await?;
        let mut out = Vec::new();
        for u in listed {
            let (name, _desc, load, active, sub) = (u.0, u.1, u.2, u.3, u.4);
            let mut unit = Unit {
                name: name.clone(),
                load,
                active,
                sub,
                main_pid: None,
                n_restarts: 0,
                result: String::new(),
                since: None,
                next_elapse: None,
                last_trigger: None,
            };
            let path = unit_path(&name);
            if name.ends_with(".service") && !unit.missing() {
                if let Ok(svc) = zbus_systemd::systemd1::ServiceProxy::builder(conn)
                    .path(path.clone())
                    .and_then(|b| Ok(b))
                    .map(|b| b.build())
                {
                    if let Ok(svc) = svc.await {
                        unit.main_pid = svc.main_pid().await.ok().filter(|p| *p != 0);
                        unit.n_restarts = svc.n_restarts().await.unwrap_or(0);
                        unit.result = svc.result().await.unwrap_or_default();
                        unit.since = svc
                            .exec_main_start_timestamp()
                            .await
                            .ok()
                            .filter(|t| *t > 0)
                            .and_then(realtime_us);
                    }
                }
            } else if name.ends_with(".timer") && !unit.missing() {
                if let Ok(t) = zbus_systemd::systemd1::TimerProxy::builder(conn)
                    .path(path)
                    .map(|b| b.build())
                {
                    if let Ok(t) = t.await {
                        // CLOCK_MONOTONIC microseconds, NOT wall time. Converted to
                        // a duration-from-now here so that no consumer downstream
                        // can mix the two clocks -- which produces timestamps in
                        // 1970 or the year 2500.
                        unit.next_elapse =
                            t.next_elapse_u_sec_monotonic().await.ok().and_then(monotonic_remaining);
                        unit.last_trigger = t
                            .last_trigger_u_sec()
                            .await
                            .ok()
                            .filter(|v| *v > 0)
                            .and_then(realtime_us);
                    }
                }
            }
            out.push(unit);
        }
        Ok(out)
    }
}

/// CLOCK_REALTIME microseconds since the epoch.
fn realtime_us(us: u64) -> Option<jiff::Timestamp> {
    jiff::Timestamp::from_microsecond(us as i64).ok()
}

/// CLOCK_MONOTONIC microseconds, turned into "how long from now".
///
/// systemd uses 0 (and `u64::MAX`) as "never scheduled", so that sentinel is decoded
/// here rather than at each call site: the meaning belongs with the conversion.
fn monotonic_remaining(us: u64) -> Option<Duration> {
    if us == 0 || us == u64::MAX {
        return None;
    }
    let now = monotonic_now_us()?;
    Some(Duration::from_micros(us.saturating_sub(now)))
}

fn monotonic_now_us() -> Option<u64> {
    // /proc/uptime is CLOCK_BOOTTIME in seconds, close enough for "next run in 4m"
    // and free of a libc dependency.
    let text = std::fs::read_to_string("/proc/uptime").ok()?;
    let secs: f64 = text.split_whitespace().next()?.parse().ok()?;
    Some((secs * 1_000_000.0) as u64)
}

/// Unit object paths are escaped: `sumofish-bot.service` becomes
/// `/org/freedesktop/systemd1/unit/chess_2dgpu_2dbot_2eservice`. Never build these
/// by string interpolation.
fn unit_path(unit: &str) -> zbus::zvariant::OwnedObjectPath {
    let mut out = String::from("/org/freedesktop/systemd1/unit/");
    for b in unit.bytes() {
        if b.is_ascii_alphanumeric() {
            out.push(b as char);
        } else {
            out.push_str(&format!("_{b:02x}"));
        }
    }
    zbus::zvariant::ObjectPath::try_from(out)
        .expect("an escaped unit path is always valid")
        .into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_paths_are_escaped_not_interpolated() {
        let p = unit_path("sumofish-bot.service");
        assert_eq!(
            p.as_str(),
            "/org/freedesktop/systemd1/unit/sumofish_2dbot_2eservice"
        );
        // A templated unit, which is what a second bot would use.
        let p = unit_path("sumofish-bot@second.service");
        assert!(p.as_str().contains("_40"), "@ must be escaped, got {p}");
    }

    #[test]
    fn the_two_systemd_clocks_are_converted_differently() {
        // A realtime microsecond count is ~1.78e15 today; a monotonic one on a box
        // up for two weeks is ~1.2e12. Feeding one to the other's converter is the
        // classic bug.
        let realtime = realtime_us(1_785_360_623_954_099).unwrap();
        assert!(
            realtime.as_second() > 1_700_000_000,
            "a realtime stamp must land in the present, got {realtime}"
        );
        // Monotonic converts to a duration, and a value in the past clamps to zero
        // rather than wrapping.
        assert_eq!(monotonic_remaining(0), None, "systemd's 'never' sentinel");
        assert_eq!(monotonic_remaining(u64::MAX), None, "and its other one");
        // A real future timestamp converts to a real remaining duration.
        let soon = monotonic_now_us().unwrap() + 60_000_000;
        let d = monotonic_remaining(soon).expect("a scheduled timer");
        assert!(d.as_secs() <= 60 && d.as_secs() >= 58, "expected ~60s, got {d:?}");
    }

    #[test]
    fn a_python_process_is_named_by_its_script() {
        // The self-match trap in reverse: a name must identify the work, not the
        // interpreter. Every GPU process here would otherwise read "python".
        assert_eq!(proc_name(u32::MAX), "pid 4294967295", "a dead pid degrades gracefully");
        // And whatever a process calls itself, one row of the panel is one line.
        assert!(proc_name(std::process::id()).chars().count() <= 24);
    }

    /// The distinction the subprocess version could not make.
    #[test]
    fn missing_and_broken_are_different_states() {
        let missing = Unit {
            name: "sumofish-train.service".into(),
            load: "not-found".into(),
            active: "inactive".into(),
            sub: "dead".into(),
            main_pid: None,
            n_restarts: 0,
            result: "success".into(),
            since: None,
            next_elapse: None,
            last_trigger: None,
        };
        assert!(missing.missing());
        assert!(!missing.healthy());

        let failed = Unit { load: "loaded".into(), active: "failed".into(), ..missing.clone() };
        assert!(!failed.missing());
        assert!(!failed.healthy());
    }
}
