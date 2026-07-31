//! F7 — can nvml and zbus replace the `nvidia-smi` and `systemctl show`
//! subprocesses?
//!
//! Two specific doubts, not general ones:
//!
//! - **nvml per-process VRAM.** `running_compute_processes()` is the answer to
//!   `machine_panel`'s own stated question, "is one starving the other". But the
//!   bot runs under a systemd sandbox with `ProtectHome=read-only` and friends,
//!   and NVML reports `UsedGpuMemory::Unavailable` for processes it cannot
//!   attribute. If that is what we get, the panel keeps guessing from one
//!   aggregate — which is exactly the mistake that cost a whole session.
//! - **zbus on the USER bus.** Every unit here is `--user`. `Connection::session()`
//!   needs `DBUS_SESSION_BUS_ADDRESS`/`XDG_RUNTIME_DIR` in the environment, which
//!   is not guaranteed if the dashboard is ever launched from a system unit or a
//!   bare ssh. Also: unit object paths are escaped, so they must be built with
//!   the helper and never by string interpolation.

use anyhow::{Context, Result};
use std::fmt::Write as _;

/// The units that actually matter. Note what the Python polls today:
/// ("sumofish-bot", "sumofish-train") — and `sumofish-train.service` does not
/// exist, so it renders a permanently red dot, while `sumofish-lab` (the thing
/// doing all the work) is absent entirely.
const UNITS: &[&str] = &[
    "sumofish-bot.service",
    "sumofish-lab.service",
    "sumofish-train.service",
    "sumofish-watchdog.timer",
    "sumofish-train-watchdog.timer",
    "sumofish-rating.timer",
];

pub fn run() -> Result<()> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let report = rt.block_on(probe())?;
    println!("{report}");
    std::fs::create_dir_all("docs").ok();
    std::fs::write("docs/probe-f7.md", &report).context("write docs/probe-f7.md")?;
    println!("wrote docs/probe-f7.md");
    Ok(())
}

async fn probe() -> Result<String> {
    let mut r = String::from("## F7 — nvml and zbus instead of subprocesses\n\n");

    nvml_section(&mut r);
    systemd_section(&mut r).await;

    Ok(r)
}

fn nvml_section(r: &mut String) {
    let _ = writeln!(r, "### nvml-wrapper\n");
    // Loads `libnvidia-ml.so.1` dynamically (the SONAME that ships with the
    // runtime driver, not the devel symlink — the crate's own doc comment
    // saying ".so" is stale).
    let nvml = match nvml_wrapper::Nvml::init() {
        Ok(n) => n,
        Err(e) => {
            let _ = writeln!(r, "**FAIL** `Nvml::init()`: {e}\n\nKeep the `nvidia-smi` subprocess.\n");
            return;
        }
    };
    let _ = writeln!(
        r,
        "- driver: `{}`\n- NVML: `{}`\n- devices: {}",
        nvml.sys_driver_version().unwrap_or_else(|e| e.to_string()),
        nvml.sys_nvml_version().unwrap_or_else(|e| e.to_string()),
        nvml.device_count().map(|c| c.to_string()).unwrap_or_else(|e| e.to_string()),
    );

    let dev = match nvml.device_by_index(0) {
        Ok(d) => d,
        Err(e) => {
            let _ = writeln!(r, "\n**FAIL** `device_by_index(0)`: {e}\n");
            return;
        }
    };

    let _ = writeln!(r, "\n| field | value | today |\n|---|---|---|");
    macro_rules! row {
        ($label:expr, $today:expr, $e:expr) => {
            match $e {
                Ok(v) => { let _ = writeln!(r, "| {} | {} | {} |", $label, v, $today); }
                Err(e) => { let _ = writeln!(r, "| {} | **ERR** {} | {} |", $label, e, $today); }
            }
        };
    }
    row!("name", "polled", dev.name());
    row!("gpu util %", "polled", dev.utilization_rates().map(|u| format!("{}", u.gpu)));
    row!("mem util %", "NEW", dev.utilization_rates().map(|u| format!("{}", u.memory)));
    row!(
        "memory",
        "polled",
        dev.memory_info().map(|m| format!("{} / {} MiB", m.used / 1048576, m.total / 1048576))
    );
    row!("temp C", "polled", dev.temperature(nvml_wrapper::enum_wrappers::device::TemperatureSensor::Gpu));
    row!("power W", "polled", dev.power_usage().map(|w| format!("{:.1}", w as f64 / 1000.0)));
    row!("power limit W", "NEW", dev.power_management_limit().map(|w| format!("{:.1}", w as f64 / 1000.0)));
    row!("fan %", "NEW", dev.fan_speed(0));
    row!("SM clock MHz", "NEW", dev.clock_info(nvml_wrapper::enum_wrappers::device::Clock::SM));
    row!("mem clock MHz", "NEW", dev.clock_info(nvml_wrapper::enum_wrappers::device::Clock::Memory));
    row!("pstate", "NEW", dev.performance_state().map(|p| format!("{p:?}")));
    row!(
        "throttle reasons",
        "NEW",
        dev.current_throttle_reasons().map(|v| format!("{v:?}"))
    );

    // The one that matters most.
    let _ = writeln!(r, "\n**Per-process VRAM** (the R9 question):\n");
    match dev.running_compute_processes() {
        Ok(procs) if procs.is_empty() => {
            let _ = writeln!(r, "No compute processes running right now — rerun while the bot or trainer is up.");
        }
        Ok(procs) => {
            let _ = writeln!(r, "| pid | used VRAM | cmdline |\n|---|---|---|");
            let mut any_unavailable = false;
            for p in &procs {
                use nvml_wrapper::enums::device::UsedGpuMemory::*;
                let used = match p.used_gpu_memory {
                    Used(b) => format!("{} MiB", b / 1048576),
                    Unavailable => {
                        any_unavailable = true;
                        "**Unavailable**".into()
                    }
                };
                let _ = writeln!(r, "| {} | {} | `{}` |", p.pid, used, cmdline(p.pid));
            }
            let _ = writeln!(
                r,
                "\nverdict: **{}**",
                if any_unavailable {
                    "PARTIAL — some processes are unattributable; show what we can and mark the rest"
                } else {
                    "PASS — per-process VRAM is readable; `machine` panel can answer the starving question directly"
                }
            );
        }
        Err(e) => {
            let _ = writeln!(
                r,
                "**FAIL** `running_compute_processes()`: {e}\n\nFall back to `nvidia-smi --query-compute-apps`."
            );
        }
    }
    let _ = writeln!(r);
}

async fn systemd_section(r: &mut String) {
    let _ = writeln!(r, "### zbus + zbus_systemd on the USER bus\n");
    let _ = writeln!(
        r,
        "- `XDG_RUNTIME_DIR={}`\n- `DBUS_SESSION_BUS_ADDRESS={}`\n",
        std::env::var("XDG_RUNTIME_DIR").unwrap_or_else(|_| "(unset)".into()),
        if std::env::var("DBUS_SESSION_BUS_ADDRESS").is_ok() { "(set)" } else { "(unset)" },
    );

    let conn = match zbus::Connection::session().await {
        Ok(c) => c,
        Err(e) => {
            let _ = writeln!(
                r,
                "**FAIL** `Connection::session()`: {e}\n\nThis must be surfaced as a source error, never a panic. Keep `systemctl --user show`.\n"
            );
            return;
        }
    };
    let mgr = match zbus_systemd::systemd1::ManagerProxy::new(&conn).await {
        Ok(m) => m,
        Err(e) => {
            let _ = writeln!(r, "**FAIL** ManagerProxy: {e}\n");
            return;
        }
    };

    // One round trip for every unit, rather than N x GetUnit + N x property read.
    // `ListUnitsByNames` also returns entries for units that are not loaded,
    // which is how `not-found` becomes visible instead of looking like an error.
    let names: Vec<String> = UNITS.iter().map(|s| s.to_string()).collect();
    match mgr.list_units_by_names(names).await {
        Ok(units) => {
            let _ = writeln!(
                r,
                "`ListUnitsByNames` returned {} entries in one round trip.\n",
                units.len()
            );
            let _ = writeln!(r, "| unit | load | active | sub | description |\n|---|---|---|---|---|");
            for u in &units {
                let _ = writeln!(
                    r,
                    "| `{}` | {} | **{}** | {} | {} |",
                    u.0, u.2, u.3, u.4, u.1
                );
            }
        }
        Err(e) => {
            let _ = writeln!(r, "**FAIL** `list_units_by_names`: {e}\n");
            return;
        }
    }

    // Service-specific properties, on the escaped object path. NRestarts is the
    // field `machine_panel` reads today, and it stays 0 when the watchdog uses
    // `systemctl kill` + `restart` rather than letting Restart=always fire —
    // which is why the panel showed a green dot through two SIGKILLs.
    let _ = writeln!(r, "\n**Service properties** (note NRestarts vs the watchdog's own count):\n");
    let _ = writeln!(r, "| unit | MainPID | NRestarts | Result | started |\n|---|---|---|---|---|");
    for unit in UNITS.iter().filter(|u| u.ends_with(".service")) {
        match service_props(&conn, unit).await {
            Ok((pid, n, res, ts)) => {
                let _ = writeln!(r, "| `{unit}` | {pid} | {n} | {res} | {ts} |");
            }
            Err(e) => {
                let _ = writeln!(r, "| `{unit}` | — | — | — | ERR: {e} |");
            }
        }
    }

    let _ = writeln!(
        r,
        "\n**Timers.** `NextElapseUSecMonotonic` is CLOCK_MONOTONIC microseconds, not wall time;\n\
         mixing it with `SystemTime::now()` is the classic bug. `ExecMainStartTimestamp` is\n\
         CLOCK_REALTIME microseconds. Different conversions.\n"
    );
    let _ = writeln!(r, "| timer | next elapse (monotonic us) | last trigger |\n|---|---|---|");
    for unit in UNITS.iter().filter(|u| u.ends_with(".timer")) {
        match timer_props(&conn, unit).await {
            Ok((next, last)) => {
                let _ = writeln!(r, "| `{unit}` | {next} | {last} |");
            }
            Err(e) => {
                let _ = writeln!(r, "| `{unit}` | — | ERR: {e} |");
            }
        }
    }
    let _ = writeln!(r);
}

async fn service_props(
    conn: &zbus::Connection,
    unit: &str,
) -> Result<(u32, u32, String, u64)> {
    let path = unit_path(unit);
    let svc = zbus_systemd::systemd1::ServiceProxy::builder(conn)
        .path(path)?
        .build()
        .await?;
    Ok((
        svc.main_pid().await.unwrap_or(0),
        svc.n_restarts().await.unwrap_or(0),
        svc.result().await.unwrap_or_else(|_| "?".into()),
        svc.exec_main_start_timestamp().await.unwrap_or(0),
    ))
}

async fn timer_props(conn: &zbus::Connection, unit: &str) -> Result<(u64, u64)> {
    let path = unit_path(unit);
    let t = zbus_systemd::systemd1::TimerProxy::builder(conn)
        .path(path)?
        .build()
        .await?;
    Ok((
        t.next_elapse_u_sec_monotonic().await.unwrap_or(0),
        t.last_trigger_u_sec().await.unwrap_or(0),
    ))
}

/// Unit object paths are escaped: `sumofish-bot.service` becomes
/// `/org/freedesktop/systemd1/unit/chess_2dgpu_2dbot_2eservice`. Never build
/// these by interpolation.
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
        .expect("escaped unit path is always valid")
        .into()
}

fn cmdline(pid: u32) -> String {
    std::fs::read_to_string(format!("/proc/{pid}/cmdline"))
        .map(|s| {
            let s = s.replace('\0', " ").trim().to_string();
            if s.len() > 60 { format!("{}…", &s[..60]) } else { s }
        })
        .unwrap_or_else(|_| "(gone)".into())
}
