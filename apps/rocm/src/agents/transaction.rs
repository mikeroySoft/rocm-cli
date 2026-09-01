// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

use anyhow::{Context, Result, bail};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static UNIQUE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone)]
pub(super) struct Snapshot {
    pub(super) raw: Option<Vec<u8>>,
    pub(super) mode: Option<u32>,
    pub(super) symlink: bool,
}

#[derive(Debug, Clone)]
pub(super) struct Rollback {
    path: PathBuf,
    before: Snapshot,
    after: Option<Vec<u8>>,
}

impl Rollback {
    pub(super) const fn new(path: PathBuf, before: Snapshot, after: Option<Vec<u8>>) -> Self {
        Self {
            path,
            before,
            after,
        }
    }

    fn ensure_restorable(&self) -> Result<()> {
        reject_symlink(&self.path)?;
        if read_optional(&self.path)? != self.after {
            bail!(
                "refusing to roll back stale configuration {}; it changed after setup",
                self.path.display()
            );
        }
        Ok(())
    }

    fn restore(&self) -> Result<()> {
        self.ensure_restorable()?;
        restore_snapshot(&self.path, &self.before)
    }
}

pub(super) fn restore_all(rollbacks: &[Rollback]) -> Result<()> {
    for rollback in rollbacks {
        rollback.ensure_restorable()?;
    }
    let failures = rollbacks
        .iter()
        .rev()
        .filter_map(|rollback| rollback.restore().err().map(|error| format!("{error:#}")))
        .collect::<Vec<_>>();
    if failures.is_empty() {
        Ok(())
    } else {
        bail!("failed to roll back configuration: {}", failures.join("; "))
    }
}

pub(super) fn restore_failed_write(path: &Path, before: &Snapshot, intended: &[u8]) -> Result<()> {
    reject_symlink(path)?;
    let actual = read_optional(path)?;
    if actual == before.raw {
        return Ok(());
    }
    if actual.as_deref() != Some(intended) {
        bail!(
            "refusing to restore stale configuration {}; it changed during setup",
            path.display()
        );
    }
    restore_snapshot(path, before)
}

pub(super) fn snapshot(path: &Path) -> Result<Snapshot> {
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => Some(metadata),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => {
            return Err(error).with_context(|| format!("failed to inspect {}", path.display()));
        }
    };
    let symlink = metadata.as_ref().is_some_and(fs::Metadata::is_symlink);
    let raw = read_optional(path)?;
    #[cfg(unix)]
    let mode = {
        use std::os::unix::fs::PermissionsExt;
        metadata
            .as_ref()
            .filter(|metadata| !metadata.is_symlink())
            .map(|metadata| metadata.permissions().mode())
    };
    #[cfg(not(unix))]
    let mode = None;
    Ok(Snapshot { raw, mode, symlink })
}

pub(super) fn ensure_fresh(path: &Path, before: &Snapshot) -> Result<()> {
    reject_symlink(path)?;
    if read_optional(path)? != before.raw {
        bail!(
            "refusing stale setup plan for {}; the configuration changed after planning",
            path.display()
        );
    }
    Ok(())
}

pub(super) fn read_optional(path: &Path) -> Result<Option<Vec<u8>>> {
    match fs::read(path) {
        Ok(bytes) => Ok(Some(bytes)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error).with_context(|| format!("failed to read {}", path.display())),
    }
}

pub(super) fn reject_symlink(path: &Path) -> Result<()> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.is_symlink() => bail!(
            "refusing to write symlinked configuration {}; point the harness directly at a regular file",
            path.display()
        ),
        Ok(_) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error).with_context(|| format!("failed to inspect {}", path.display())),
    }
}

pub(super) fn restore_if_changed(path: &Path, before: &Snapshot) -> Result<()> {
    if read_optional(path)? != before.raw {
        restore_snapshot(path, before)?;
    }
    Ok(())
}

fn restore_snapshot(path: &Path, before: &Snapshot) -> Result<()> {
    if let Some(bytes) = before.raw.as_deref() {
        atomic_write(path, bytes, before.mode)
    } else {
        reject_symlink(path)?;
        match fs::remove_file(path) {
            Ok(()) => sync_parent(path),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error)
                .with_context(|| format!("failed to remove {} during rollback", path.display())),
        }
    }
}

pub(super) fn atomic_write(path: &Path, bytes: &[u8], mode: Option<u32>) -> Result<()> {
    reject_symlink(path)?;
    let parent = path.parent().ok_or_else(|| {
        anyhow::anyhow!(
            "configuration path {} has no parent directory",
            path.display()
        )
    })?;
    fs::create_dir_all(parent).with_context(|| {
        format!(
            "failed to create configuration directory {}",
            parent.display()
        )
    })?;
    let (temporary, mut file) = unique_temp(path)?;
    let result = (|| -> Result<()> {
        file.write_all(bytes).with_context(|| {
            format!(
                "failed to write temporary configuration {}",
                temporary.display()
            )
        })?;
        set_mode(&temporary, mode)?;
        file.sync_all().with_context(|| {
            format!(
                "failed to sync temporary configuration {}",
                temporary.display()
            )
        })?;
        drop(file);
        publish(&temporary, path)?;
        sync_parent(path)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

fn unique_temp(path: &Path) -> Result<(PathBuf, File)> {
    let parent = path.parent().expect("checked by atomic_write");
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("config");
    for _ in 0..32 {
        let candidate = parent.join(format!("{}.tmp", unique_token(&format!(".{name}.rocm"))));
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(file) => return Ok((candidate, file)),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => {
                return Err(error).with_context(|| {
                    format!(
                        "failed to create temporary configuration in {}",
                        parent.display()
                    )
                });
            }
        }
    }
    bail!(
        "failed to allocate a unique temporary configuration beside {}",
        path.display()
    )
}

pub(super) fn unique_token(prefix: &str) -> String {
    let sequence = UNIQUE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{prefix}-{:x}-{nanos:x}-{sequence:x}", std::process::id())
}

#[cfg(unix)]
fn set_mode(path: &Path, mode: Option<u32>) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(mode.unwrap_or(0o600)))
        .with_context(|| format!("failed to set permissions on {}", path.display()))
}

#[cfg(not(unix))]
fn set_mode(_path: &Path, _mode: Option<u32>) -> Result<()> {
    Ok(())
}

#[cfg(not(windows))]
fn publish(temporary: &Path, path: &Path) -> Result<()> {
    fs::rename(temporary, path)
        .with_context(|| format!("failed to atomically replace {}", path.display()))
}

#[cfg(windows)]
#[allow(unsafe_code)] // MoveFileExW is the Windows atomic-replacement primitive
fn publish(temporary: &Path, path: &Path) -> Result<()> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::{
        MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH, MoveFileExW,
    };
    let source: Vec<u16> = temporary.as_os_str().encode_wide().chain(Some(0)).collect();
    let destination: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
    let result = unsafe {
        MoveFileExW(
            source.as_ptr(),
            destination.as_ptr(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
        )
    };
    if result == 0 {
        Err(std::io::Error::last_os_error())
            .with_context(|| format!("failed to atomically replace {}", path.display()))
    } else {
        Ok(())
    }
}

#[cfg(unix)]
fn sync_parent(path: &Path) -> Result<()> {
    let parent = path.parent().ok_or_else(|| {
        anyhow::anyhow!(
            "configuration path {} has no parent directory",
            path.display()
        )
    })?;
    File::open(parent)
        .and_then(|directory| directory.sync_all())
        .with_context(|| {
            format!(
                "failed to sync configuration directory {}",
                parent.display()
            )
        })
}

#[cfg(not(unix))]
fn sync_parent(_path: &Path) -> Result<()> {
    Ok(())
}
