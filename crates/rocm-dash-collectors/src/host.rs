// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Host system metrics via sysinfo. Backs the daemon's per-tick `SystemMetrics`.

use std::collections::{HashMap, HashSet};
use std::ffi::{OsStr, OsString};
use std::time::{Duration, Instant};

use rocm_dash_core::metrics::SystemMetrics;
use sysinfo::{DiskRefreshKind, Disks, MemoryRefreshKind, Networks, RefreshKind, System};

#[derive(Clone, Copy, Default)]
struct IoCounters {
    read: u64,
    written: u64,
}

pub struct HostCollector {
    sys: System,
    disks: Disks,
    networks: Networks,
    previous_disks: Option<HashMap<OsString, IoCounters>>,
    previous_networks: Option<HashMap<String, IoCounters>>,
    last_tick: Option<Instant>,
    cpu_model: String,
}

impl Default for HostCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl HostCollector {
    pub fn new() -> Self {
        let refresh = RefreshKind::nothing()
            .with_cpu(sysinfo::CpuRefreshKind::everything())
            .with_memory(MemoryRefreshKind::everything());
        let mut sys = System::new_with_specifics(refresh);
        // First refresh primes CPU deltas; the very first read is meaningless.
        sys.refresh_cpu_all();
        sys.refresh_memory();
        let cpu_model = detect_cpu_model(&sys);
        let disk_refresh = DiskRefreshKind::nothing().with_io_usage();
        Self {
            sys,
            disks: Disks::new_with_refreshed_list_specifics(disk_refresh),
            networks: Networks::new_with_refreshed_list(),
            previous_disks: None,
            previous_networks: None,
            last_tick: None,
            cpu_model,
        }
    }

    pub fn tick(&mut self) -> SystemMetrics {
        self.sys.refresh_cpu_all();
        self.sys.refresh_memory();
        self.disks
            .refresh_specifics(true, DiskRefreshKind::nothing().with_io_usage());
        self.networks.refresh(true);

        let now = Instant::now();
        let elapsed = self.last_tick.map(|last| now.duration_since(last));
        self.last_tick = Some(now);

        let disks = disk_snapshot(&self.disks);
        let networks = network_snapshot(&self.networks);
        let disk_delta = self
            .previous_disks
            .as_ref()
            .map_or_else(IoCounters::default, |previous| {
                counter_delta(previous, &disks)
            });
        let network_delta = self
            .previous_networks
            .as_ref()
            .map_or_else(IoCounters::default, |previous| {
                counter_delta(previous, &networks)
            });
        self.previous_disks = Some(disks);
        self.previous_networks = Some(networks);

        let cpu_overall_pct = self.sys.global_cpu_usage();
        let cpu_per_core_pct: Vec<f32> = self
            .sys
            .cpus()
            .iter()
            .map(sysinfo::Cpu::cpu_usage)
            .collect();

        // sysinfo reports memory in bytes.
        let memory_used_mb = self.sys.used_memory() / 1024 / 1024;
        let memory_total_mb = self.sys.total_memory() / 1024 / 1024;
        let swap_used_mb = self.sys.used_swap() / 1024 / 1024;
        let swap_total_mb = self.sys.total_swap() / 1024 / 1024;

        SystemMetrics {
            cpu_model: self.cpu_model.clone(),
            cpu_overall_pct,
            cpu_per_core_pct,
            memory_used_mb,
            memory_total_mb,
            swap_used_mb,
            swap_total_mb,
            disk_read_bps: elapsed.map_or(0, |elapsed| rate_bps(disk_delta.read, elapsed)),
            disk_write_bps: elapsed.map_or(0, |elapsed| rate_bps(disk_delta.written, elapsed)),
            net_rx_bps: elapsed.map_or(0, |elapsed| rate_bps(network_delta.read, elapsed)),
            net_tx_bps: elapsed.map_or(0, |elapsed| rate_bps(network_delta.written, elapsed)),
        }
    }
}

fn disk_snapshot(disks: &Disks) -> HashMap<OsString, IoCounters> {
    deduplicated_disk_snapshot(disks.iter().map(|disk| {
        let usage = disk.usage();
        (
            disk.name(),
            disk.file_system(),
            IoCounters {
                read: usage.total_read_bytes,
                written: usage.total_written_bytes,
            },
        )
    }))
}

fn deduplicated_disk_snapshot<'a>(
    disks: impl IntoIterator<Item = (&'a OsStr, &'a OsStr, IoCounters)>,
) -> HashMap<OsString, IoCounters> {
    let mut seen = HashSet::new();
    disks
        .into_iter()
        .filter(|(name, file_system, _)| include_disk(name, file_system))
        .filter_map(|(name, _, counters)| {
            let key = canonical_device_name(name);
            seen.insert(key.clone()).then_some((key, counters))
        })
        .collect()
}

fn network_snapshot(networks: &Networks) -> HashMap<String, IoCounters> {
    networks
        .iter()
        .filter(|(name, data)| {
            include_network(
                name,
                data.ip_networks()
                    .iter()
                    .any(|network| network.addr.is_loopback()),
            )
        })
        .map(|(name, data)| {
            (
                name.clone(),
                IoCounters {
                    read: data.total_received(),
                    written: data.total_transmitted(),
                },
            )
        })
        .collect()
}

fn counter_delta<K: Eq + std::hash::Hash>(
    previous: &HashMap<K, IoCounters>,
    current: &HashMap<K, IoCounters>,
) -> IoCounters {
    current
        .iter()
        .fold(IoCounters::default(), |mut total, (key, current)| {
            if let Some(previous) = previous.get(key) {
                total.read = total
                    .read
                    .saturating_add(current.read.saturating_sub(previous.read));
                total.written = total
                    .written
                    .saturating_add(current.written.saturating_sub(previous.written));
            }
            total
        })
}

fn canonical_device_name(name: &OsStr) -> OsString {
    std::fs::canonicalize(name)
        .map_or_else(|_| name.to_os_string(), std::path::PathBuf::into_os_string)
}

fn include_disk(name: &OsStr, file_system: &OsStr) -> bool {
    let fs = file_system.to_string_lossy();
    !name.is_empty()
        && !matches!(
            fs.as_ref(),
            "rootfs"
                | "sysfs"
                | "proc"
                | "devtmpfs"
                | "devpts"
                | "tmpfs"
                | "overlay"
                | "cgroup"
                | "cgroup2"
                | "pstore"
                | "squashfs"
                | "rpc_pipefs"
                | "hugetlbfs"
                | "mqueue"
        )
}

fn include_network(name: &str, has_loopback_address: bool) -> bool {
    name != "lo" && !has_loopback_address
}

fn rate_bps(delta_bytes: u64, elapsed: Duration) -> u64 {
    let elapsed_nanos = elapsed.as_nanos();
    if elapsed_nanos == 0 {
        return 0;
    }

    let bytes_per_second = u128::from(delta_bytes).saturating_mul(1_000_000_000) / elapsed_nanos;
    bytes_per_second.min(u128::from(u64::MAX)) as u64
}

fn detect_cpu_model(sys: &System) -> String {
    sys.cpus()
        .first()
        .map(sysinfo::Cpu::brand)
        .map(str::trim)
        .filter(|name| !name.is_empty() && !name.eq_ignore_ascii_case("unknown cpu"))
        .map(str::to_string)
        .or_else(|| {
            std::fs::read_to_string("/proc/cpuinfo")
                .ok()
                .and_then(|cpuinfo| proc_cpu_model(&cpuinfo).map(str::to_string))
        })
        .unwrap_or_else(|| "Unknown CPU".to_string())
}

fn proc_cpu_model(cpuinfo: &str) -> Option<&str> {
    cpuinfo.lines().find_map(|line| {
        let (key, value) = line.split_once(':')?;
        (key.trim() == "model name")
            .then(|| value.trim())
            .filter(|value| !value.is_empty())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tick_produces_metrics_with_some_cpus() {
        let mut c = HostCollector::new();
        let m = c.tick();
        assert!(!m.cpu_per_core_pct.is_empty(), "expected at least one CPU");
        assert_ne!(m.cpu_model, "Unknown CPU", "expected a CPU model name");
        assert!(m.memory_total_mb > 0, "expected nonzero total memory");
        assert_eq!(m.disk_read_bps, 0);
        assert_eq!(m.disk_write_bps, 0);
        assert_eq!(m.net_rx_bps, 0);
        assert_eq!(m.net_tx_bps, 0);
    }

    #[test]
    fn parses_linux_cpu_model_name() {
        assert_eq!(
            proc_cpu_model("processor: 0\nmodel name: Intel Core i9-14900KF\n"),
            Some("Intel Core i9-14900KF")
        );
    }

    #[test]
    fn rate_uses_measured_elapsed_time() {
        assert_eq!(rate_bps(4_096, Duration::from_millis(500)), 8_192);
    }

    #[test]
    fn rate_handles_zero_and_near_zero_elapsed_time() {
        assert_eq!(rate_bps(4_096, Duration::ZERO), 0);
        assert_eq!(rate_bps(1, Duration::from_nanos(1)), 1_000_000_000);
    }

    #[test]
    fn missing_baseline_produces_zero_delta() {
        let current = HashMap::from([(
            "sda".to_string(),
            IoCounters {
                read: 1_024,
                written: 2_048,
            },
        )]);

        let delta = Option::<&HashMap<String, IoCounters>>::None
            .map_or_else(IoCounters::default, |previous| {
                counter_delta(previous, &current)
            });

        assert_eq!(delta.read, 0);
        assert_eq!(delta.written, 0);
    }

    #[test]
    fn counter_reset_does_not_create_a_spike() {
        let previous = HashMap::from([(
            "sda",
            IoCounters {
                read: u64::MAX - 10,
                written: 50_000,
            },
        )]);
        let current = HashMap::from([(
            "sda",
            IoCounters {
                read: 5,
                written: 100,
            },
        )]);

        let delta = counter_delta(&previous, &current);

        assert_eq!(delta.read, 0);
        assert_eq!(delta.written, 0);
    }

    #[test]
    fn loopback_interfaces_are_excluded() {
        assert!(!include_network("lo", false));
        assert!(!include_network("loopback-alias", true));
        assert!(include_network("eth0", false));
    }

    #[test]
    fn pseudo_disks_are_excluded() {
        assert!(!include_disk(OsStr::new("overlay"), OsStr::new("overlay")));
        assert!(!include_disk(OsStr::new("tmpfs"), OsStr::new("tmpfs")));
        assert!(include_disk(
            OsStr::new("/dev/nvme0n1p1"),
            OsStr::new("ext4")
        ));
    }

    #[cfg(unix)]
    #[test]
    fn duplicate_device_mounts_have_the_same_deduplication_key() {
        use std::os::unix::fs::symlink;

        let dir = tempfile::tempdir().expect("create temp directory");
        let device = dir.path().join("device");
        let alias = dir.path().join("alias");
        std::fs::write(&device, []).expect("create stand-in device");
        symlink(&device, &alias).expect("create device alias");

        let snapshot = deduplicated_disk_snapshot([
            (
                device.as_os_str(),
                OsStr::new("ext4"),
                IoCounters {
                    read: 1_000,
                    written: 2_000,
                },
            ),
            (
                alias.as_os_str(),
                OsStr::new("ext4"),
                IoCounters {
                    read: 1_000,
                    written: 2_000,
                },
            ),
        ]);

        assert_eq!(snapshot.len(), 1);
        let counters = snapshot.values().next().expect("one physical device");
        assert_eq!(counters.read, 1_000);
        assert_eq!(counters.written, 2_000);
    }

    #[cfg(target_os = "linux")]
    #[test]
    #[ignore = "CI temporary directories may be tmpfs; run manually on a block-backed TMPDIR"]
    fn writing_and_syncing_a_file_produces_disk_write_activity() {
        use std::io::Write;

        let mut collector = HostCollector::new();
        let first = collector.tick();
        assert_eq!(first.disk_write_bps, 0);

        std::thread::sleep(Duration::from_millis(100));
        let mut file = tempfile::NamedTempFile::new().expect("create temporary file");
        file.write_all(&vec![0x5a; 8 * 1024 * 1024])
            .expect("write temporary file");
        file.as_file().sync_all().expect("sync temporary file");
        let second = collector.tick();

        assert!(
            second.disk_write_bps > 0,
            "expected block-device write activity after writing and syncing 8 MiB"
        );
    }
}
