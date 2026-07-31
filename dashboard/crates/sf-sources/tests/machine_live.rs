//! M5's verification: the library readings must agree with the subprocesses they
//! replace.
//!
//! Not a mock. `nvidia-smi` and `systemctl --user show` are the incumbents, they
//! work today, and the only question worth asking is whether the library says the
//! same thing. Skipped when the hardware or the bus is absent.

use sf_model::Update;
use std::process::Command;

fn sh(cmd: &str) -> Option<String> {
    let out = Command::new("sh").args(["-c", cmd]).output().ok()?;
    out.status.success().then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// nvml against `nvidia-smi`, on the numbers that are stable enough to compare.
#[test]
fn nvml_agrees_with_nvidia_smi() {
    let Some(reference) =
        sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits")
    else {
        eprintln!("SKIP: nvidia-smi unavailable");
        return;
    };
    let src = sf_sources::GpuSource::new();
    let Update::Gpus(gpus) = src.poll() else {
        eprintln!("SKIP: NVML unavailable");
        return;
    };
    assert!(!gpus.is_empty(), "nvidia-smi answered but NVML reported no devices");

    let line = reference.lines().next().unwrap();
    let mut parts = line.split(',').map(str::trim);
    let want_name = parts.next().unwrap();
    let want_total: u64 = parts.next().unwrap().parse().unwrap();

    let g = &gpus[0];
    assert_eq!(g.name, want_name);
    let got_total = g.mem_total / 1_048_576;
    assert!(
        got_total.abs_diff(want_total) <= 2,
        "total VRAM disagrees: nvml {got_total} MiB vs nvidia-smi {want_total} MiB"
    );

    // Utilisation and power move between the two readings, so only sanity-check.
    assert!(g.util_gpu <= 100);
    assert!(g.temp_c > 0 && g.temp_c < 130, "implausible temperature {}", g.temp_c);
    assert!(g.mem_used <= g.mem_total);

    println!(
        "{} {}% gpu {}% mem {}/{} MiB {}C {:.0}/{:.0}W fan {}% sm {}MHz throttle {:?}",
        g.name,
        g.util_gpu,
        g.util_mem,
        g.mem_used / 1_048_576,
        got_total,
        g.temp_c,
        g.power_w,
        g.power_limit_w,
        g.fan_pct,
        g.clock_sm,
        g.throttle
    );
    for p in &g.procs {
        println!(
            "  pid {:>7} {:>6} {}",
            p.pid,
            p.mem.map(|m| format!("{}M", m / 1_048_576)).unwrap_or_else(|| "n/a".into()),
            p.name
        );
    }
}

/// The per-process split is the answer to `machine_panel`'s own question, and it is
/// the reading most likely to be blocked by the sandbox.
#[test]
fn per_process_vram_is_attributable() {
    let Some(reference) = sh(
        "nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits",
    ) else {
        eprintln!("SKIP: nvidia-smi unavailable");
        return;
    };
    if reference.is_empty() {
        eprintln!("SKIP: no compute processes running right now");
        return;
    }
    let src = sf_sources::GpuSource::new();
    let Update::Gpus(gpus) = src.poll() else {
        eprintln!("SKIP: NVML unavailable");
        return;
    };
    let procs = &gpus[0].procs;
    assert!(!procs.is_empty(), "nvidia-smi lists compute processes but NVML reported none");
    // Every one must be attributable, or the panel is back to inferring from one
    // aggregate -- the mistake CLAUDE.md records as costing a session.
    let unattributable = procs.iter().filter(|p| p.mem.is_none()).count();
    assert_eq!(
        unattributable, 0,
        "{unattributable} of {} processes report Unavailable VRAM",
        procs.len()
    );
    // And the pids must match what nvidia-smi sees.
    let want: Vec<u32> = reference
        .lines()
        .filter_map(|l| l.split(',').next()?.trim().parse().ok())
        .collect();
    for pid in want {
        assert!(procs.iter().any(|p| p.pid == pid), "NVML missed pid {pid}");
    }
}

/// zbus against `systemctl --user show`, including the two states the subprocess
/// version conflated.
#[tokio::test]
async fn zbus_agrees_with_systemctl() {
    let names: Vec<String> = ["sumofish-bot.service", "sumofish-lab.service", "sumofish-train.service"]
        .into_iter()
        .map(String::from)
        .collect();
    if sh("systemctl --user --version").is_none() {
        eprintln!("SKIP: no user systemd");
        return;
    }
    let src = sf_sources::UnitsSource::new(names.clone()).await;
    let Update::Units(units) = src.poll().await else {
        eprintln!("SKIP: no session bus");
        return;
    };
    assert_eq!(units.len(), names.len(), "one row per requested unit, present or not");

    for u in &units {
        let want_load =
            sh(&format!("systemctl --user show -p LoadState --value {}", u.name)).unwrap_or_default();
        let want_active = sh(&format!("systemctl --user show -p ActiveState --value {}", u.name))
            .unwrap_or_default();
        assert_eq!(u.load, want_load, "LoadState for {}", u.name);
        assert_eq!(u.active, want_active, "ActiveState for {}", u.name);
        println!(
            "{:<32} {:<10} {:<10} {:<8} pid={:?} nrestarts={}",
            u.name, u.load, u.active, u.sub, u.main_pid, u.n_restarts
        );
    }

    // The distinction the Python could not make: a unit that does not exist is not
    // a unit that is broken.
    if let Some(train) = units.iter().find(|u| u.name == "sumofish-train.service") {
        if train.missing() {
            println!(
                "sumofish-train.service is NOT-FOUND -- a distinct state, not a red dot, \
                 and its watchdog timer is still armed and pointed at it"
            );
        }
    }
}

/// The timer conversion, against the real bus. `NextElapseUSecMonotonic` is a
/// different clock from `ExecMainStartTimestamp` and mixing them lands in 1970.
#[tokio::test]
async fn timer_next_elapse_is_a_plausible_duration() {
    let names = vec![
        "sumofish-watchdog.timer".to_string(),
        "sumofish-rating.timer".to_string(),
    ];
    let src = sf_sources::UnitsSource::new(names).await;
    let Update::Units(units) = src.poll().await else {
        eprintln!("SKIP: no session bus");
        return;
    };
    for u in &units {
        if u.missing() {
            continue;
        }
        if let Some(next) = u.next_elapse {
            // A rating sample is every 15 minutes and the watchdog every minute, so
            // anything beyond an hour means the clocks got crossed.
            assert!(
                next.as_secs() < 3600,
                "{} next elapse is {next:?} -- the monotonic clock was probably read as wall time",
                u.name
            );
            println!("{} fires in {}s", u.name, next.as_secs());
        }
        if let Some(t) = u.last_trigger {
            assert!(
                t.as_second() > 1_600_000_000,
                "{} last trigger {t} is before this project existed",
                u.name
            );
        }
    }
}
