// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

//! Hardware instrument cluster for Observe.
//!
//! A wide GPU column mirrors GPUFLO's paired activity/VRAM instruments. A
//! narrower CPU column mirrors btop's overall meter and labeled per-core
//! histories. Memory, swap, and combined disk/network I/O share one footer.

use ratatui::Frame;
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use rocm_dash_core::metrics::{GpuMetrics, GpuSystemInfo, Snapshot};

use crate::app::AppState;
use crate::ui::format;
use crate::ui::gradient::GradientGauge;
use crate::ui::panel::{self, BoxRole};
use crate::ui::sparkline::BrailleSparkline;
use crate::ui::theme::Theme;
use crate::ui::widgets::{
    POWER_CRIT_W, gpu_stats_line, instances_on_gpu, node_efficiency, power_style,
    temperature_style, trunc,
};

/// Rows outside the GPU column in [`draw`] (the shared host footer).
const ROWS_ABOVE_GPUS: u16 = 3;
/// Height of the GPU-section summary header (partition + efficiency).
const GPU_HEADER_H: u16 = 1;
/// Minimum slot height that still leaves one visible row in each paired instrument.
const FULL_PANEL_MIN_H: u16 = 8;

pub fn draw(f: &mut Frame, area: Rect, state: &AppState, theme: &Theme) {
    let Some(snap) = state.latest.as_ref() else {
        let inner = panel::bento(f, area, Some("Hardware"), BoxRole::Neutral, false, theme);
        let p = Paragraph::new(Line::from(Span::styled(
            "waiting for first snapshot…",
            Style::default().fg(theme.muted),
        )));
        f.render_widget(p, inner);
        return;
    };

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(0), Constraint::Length(3)])
        .split(area);
    let instruments = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Ratio(3, 5), Constraint::Ratio(2, 5)])
        .split(rows[0]);
    let host = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Ratio(1, 4),
            Constraint::Ratio(1, 4),
            Constraint::Ratio(2, 4),
        ])
        .split(rows[1]);

    draw_gpus(f, instruments[0], state, snap, theme);
    let cpu_rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Min(0)])
        .split(instruments[1]);
    f.render_widget(Paragraph::new(cpu_section_header(snap, theme)), cpu_rows[0]);
    draw_cpu(f, cpu_rows[1], state, snap, theme);
    draw_gauge_block(
        f,
        host[0],
        "Memory",
        snap.host.memory_used_mb,
        snap.host.memory_total_mb,
        BoxRole::Success,
        theme,
    );
    draw_gauge_block(
        f,
        host[1],
        "Swap",
        snap.host.swap_used_mb,
        snap.host.swap_total_mb,
        BoxRole::Secondary,
        theme,
    );
    draw_io_row(f, host[2], snap, theme);
}

/// Disk and network throughput share one host-I/O box.
fn draw_io_row(f: &mut Frame, area: Rect, snap: &Snapshot, theme: &Theme) {
    let inner = panel::bento(f, area, Some("Disk + Net"), BoxRole::Primary, false, theme);
    if inner.height == 0 {
        return;
    }
    let line = Line::from(vec![
        Span::styled("read ", Style::default().fg(theme.muted)),
        Span::styled(
            format::bps(snap.host.disk_read_bps as f64),
            Style::default().fg(theme.accent),
        ),
        Span::styled("  write ", Style::default().fg(theme.muted)),
        Span::styled(
            format::bps(snap.host.disk_write_bps as f64),
            Style::default().fg(theme.accent),
        ),
        Span::styled("   rx ", Style::default().fg(theme.muted)),
        Span::styled(
            format::bps(snap.host.net_rx_bps as f64),
            Style::default().fg(theme.accent_2),
        ),
        Span::styled("  tx ", Style::default().fg(theme.muted)),
        Span::styled(
            format::bps(snap.host.net_tx_bps as f64),
            Style::default().fg(theme.accent_2),
        ),
    ]);
    f.render_widget(Paragraph::new(line), inner);
}

fn cpu_section_header(snap: &Snapshot, theme: &Theme) -> Line<'static> {
    let model = snap.host.cpu_model.trim();
    let model = if model.is_empty() {
        "Unknown CPU".to_string()
    } else {
        trunc(model, 22)
    };
    Line::from(vec![
        Span::styled(
            "CPU",
            Style::default()
                .fg(theme.accent)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(format!(" · {model}"), Style::default().fg(theme.fg)),
        Span::styled(
            format!(
                " · {}c · {}",
                snap.host.cpu_per_core_pct.len(),
                format::pct(snap.host.cpu_overall_pct)
            ),
            Style::default().fg(theme.muted),
        ),
    ])
}

fn draw_cpu(f: &mut Frame, area: Rect, state: &AppState, snap: &Snapshot, theme: &Theme) {
    let cores = &snap.host.cpu_per_core_pct;
    let title = "CPU cores";
    let inner = panel::bento(f, area, Some(title), BoxRole::Secondary, false, theme);
    if inner.height == 0 {
        return;
    }

    let ratio = f64::from(snap.host.cpu_overall_pct.clamp(0.0, 100.0)) / 100.0;
    let meter_label = format!("CPU {}", format::pct(snap.host.cpu_overall_pct));
    let meter = GradientGauge::new(ratio)
        .stops(theme.ok, theme.warn, theme.err)
        .track_bg(theme.surface_2)
        .label(&meter_label)
        .label_fg(theme.fg);
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Min(0)])
        .split(inner);
    f.render_widget(meter, rows[0]);
    draw_cpu_cores(f, rows[1], cores, &state.history, theme);
}

fn draw_cpu_cores(
    f: &mut Frame,
    area: Rect,
    cores: &[f32],
    history: &std::collections::VecDeque<Snapshot>,
    theme: &Theme,
) {
    const MIN_CELL_WIDTH: usize = 10;

    if area.width == 0 || area.height == 0 || cores.is_empty() {
        return;
    }

    let rows = area.height as usize;
    let columns = cores
        .len()
        .div_ceil(rows)
        .min((area.width as usize / MIN_CELL_WIDTH).max(1));
    let grid_rows = rows - usize::from(rows * columns < cores.len());
    let visible = cores.len().min(grid_rows * columns);
    let cell_width = area.width / columns as u16;
    let label_width = format!("C{} ", cores.len().saturating_sub(1)).len() as u16;
    let mut samples = Vec::with_capacity(history.len());

    for (index, value) in cores.iter().take(visible).enumerate() {
        let column = index / grid_rows;
        let row = index % grid_rows;
        let x = area.x + column as u16 * cell_width;
        let width = if column + 1 == columns {
            area.right().saturating_sub(x)
        } else {
            cell_width
        };
        let cell = Rect::new(x, area.y + row as u16, width, 1);
        let style = cpu_core_style(*value, theme);
        if width < label_width + 4 {
            f.render_widget(
                Paragraph::new(format!("C{index} {:.0}%", value.clamp(0.0, 100.0))).style(style),
                cell,
            );
            continue;
        }

        let graph_width = width - label_width - 4;
        f.render_widget(
            Paragraph::new(format!("C{index} ")).style(Style::default().fg(theme.fg)),
            Rect::new(x, cell.y, label_width, 1),
        );
        samples.clear();
        samples.extend(history.iter().filter_map(|snapshot| {
            snapshot
                .host
                .cpu_per_core_pct
                .get(index)
                .map(|sample| sample.clamp(0.0, 100.0) as u64)
        }));
        f.render_widget(
            BrailleSparkline::new(&samples).max(100).style(style),
            Rect::new(x + label_width, cell.y, graph_width, 1),
        );
        f.render_widget(
            Paragraph::new(format!("{:.0}%", value.clamp(0.0, 100.0)))
                .alignment(Alignment::Right)
                .style(style),
            Rect::new(x + width - 4, cell.y, 4, 1),
        );
    }

    if visible < cores.len() {
        let hidden = cores.len() - visible;
        f.render_widget(
            Paragraph::new(format!("+{hidden} cores")).style(Style::default().fg(theme.muted)),
            Rect::new(
                area.x,
                area.y + area.height.saturating_sub(1),
                area.width,
                1,
            ),
        );
    }
}

fn cpu_core_style(value: f32, theme: &Theme) -> Style {
    Style::default().fg(if value >= 90.0 {
        theme.err
    } else if value >= 70.0 {
        theme.warn
    } else {
        theme.ok
    })
}

fn draw_gauge_block(
    f: &mut Frame,
    area: Rect,
    label: &str,
    used_mb: u64,
    total_mb: u64,
    role: BoxRole,
    theme: &Theme,
) {
    let (ratio, title) = if total_mb > 0 {
        let r = (used_mb as f64 / total_mb as f64).clamp(0.0, 1.0);
        (
            r,
            format!("{label} · {}", format::mib_pair(used_mb, total_mb)),
        )
    } else {
        // No total reported: render flat gauge with just the used value.
        (0.0, format!("{label} · {}", format::mib(used_mb)))
    };
    let inner = panel::bento(f, area, Some(&title), role, false, theme);
    if inner.height == 0 {
        return;
    }
    let label = format::pct((ratio * 100.0) as f32);
    let gauge = GradientGauge::new(ratio)
        .stops(theme.ok, theme.warn, theme.err)
        .track_bg(theme.surface_2)
        .label(&label)
        .label_fg(theme.fg);
    f.render_widget(gauge, inner);
}

fn draw_gpus(f: &mut Frame, area: Rect, state: &AppState, snap: &Snapshot, theme: &Theme) {
    if snap.gpus.is_empty() {
        // Keep the instrument labels visible in the unavailable state so the
        // layout remains readable in plain output and on hosts without GPUs.
        let mut lines = vec![Line::from(vec![
            Span::styled("GPU activity · —", Style::default().fg(theme.muted)),
            Span::styled("   VRAM occupancy · —", Style::default().fg(theme.muted)),
        ])];
        let role = if snap.warnings.is_empty() {
            lines.push(Line::from(Span::styled(
                "no GPUs reported",
                Style::default().fg(theme.muted),
            )));
            BoxRole::Neutral
        } else {
            lines.extend(
                snap.warnings
                    .iter()
                    .map(|w| Line::from(Span::styled(w.clone(), Style::default().fg(theme.warn)))),
            );
            BoxRole::Warning
        };
        let inner = panel::bento(f, area, Some("GPUs"), role, false, theme);
        f.render_widget(Paragraph::new(lines), inner);
        return;
    }

    let n = snap.gpus.len();
    let sel = state.gpu_sel.min(n - 1);

    // Reserve one row for the section header (partition + efficiency) when there
    // is room; otherwise hand the whole area to the panels.
    let body = if area.height >= 2 {
        let split = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(GPU_HEADER_H), Constraint::Min(0)])
            .split(area);
        f.render_widget(Paragraph::new(gpu_section_header(snap, theme)), split[0]);
        split[1]
    } else {
        area
    };

    let (full, visible) = plan_gpu_rows(body.height, n);
    if full {
        draw_gpus_full(f, body, state, snap, theme, sel);
    } else {
        draw_gpus_compact(f, body, snap, theme, sel, state.gpu_scroll, visible);
    }
}

/// One-line GPU identity and serving-efficiency summary.
fn gpu_section_header(snap: &Snapshot, theme: &Theme) -> Line<'static> {
    let n = snap.gpus.len();
    let mut spans = vec![Span::styled(
        format!("{n} GPU"),
        Style::default()
            .fg(theme.accent)
            .add_modifier(Modifier::BOLD),
    )];
    if let Some(si) = snap.gpu_system_info.as_ref() {
        let model = si.gpu_model.trim();
        if !model.is_empty() {
            spans.push(Span::styled(
                format!(" · {}", trunc(model, 24)),
                Style::default().fg(theme.fg),
            ));
        }
        if si.physical_gpu_count != n as u32 || si.logical_gpu_count != n as u32 {
            spans.push(Span::styled(
                format!(" · {}p/{}l", si.physical_gpu_count, si.logical_gpu_count),
                Style::default().fg(theme.muted),
            ));
        }
        if si.vram_per_logical_gpu_mb > 0 {
            spans.push(Span::styled(
                format!(" · {}/l", format::mib(si.vram_per_logical_gpu_mb)),
                Style::default().fg(theme.muted),
            ));
        }
    }
    let total_power: f64 = snap.gpus.iter().map(|g| f64::from(g.power_w)).sum();
    spans.push(Span::styled(
        format!(" · {:.1}kW", total_power / 1000.0),
        Style::default().fg(theme.fg),
    ));
    let eff = node_efficiency(snap);
    spans.push(Span::styled(
        format!(" · eff {}", format::tokens_per_watt(eff)),
        Style::default().fg(if eff.is_some() { theme.ok } else { theme.muted }),
    ));
    Line::from(spans)
}

/// Decide how the GPU section renders for a given section height and GPU count.
/// Returns `(full_panels, visible_rows)`:
/// - `full_panels == true`  → every GPU gets a bordered panel (`visible == n`).
/// - `full_panels == false` → compact one-line rows; `visible_rows` is how many
///   GPU rows are shown (one row is reserved for the overflow indicator when
///   the list is longer than the section).
fn plan_gpu_rows(section_h: u16, n: usize) -> (bool, usize) {
    if n == 0 {
        return (true, 0);
    }
    if (n as u16).saturating_mul(FULL_PANEL_MIN_H) <= section_h {
        return (true, n);
    }
    let cap = section_h as usize;
    if n <= cap {
        (false, n) // every GPU fits as a compact row — no scroll needed
    } else {
        (false, cap.saturating_sub(1).max(1)) // reserve one row for the indicator
    }
}

/// Visible compact-row count for the Hardware Observe sub-panel given the full body height.
///
/// Used by `AppState` to keep `gpu_scroll` in sync on navigation. Accounts for
/// both the rows above the GPU section and the section's own header row.
pub fn gpu_visible_count(body_h: u16, n: usize) -> usize {
    let section = body_h.saturating_sub(ROWS_ABOVE_GPUS);
    let panels = if section >= 2 {
        section - GPU_HEADER_H
    } else {
        section
    };
    plan_gpu_rows(panels, n).1
}

/// Advance/clamp a scroll offset so index `sel` lies within
/// `[scroll, scroll + visible)`.
pub const fn scroll_to_show(sel: usize, scroll: usize, visible: usize) -> usize {
    if visible == 0 {
        0
    } else if sel < scroll {
        sel
    } else if sel >= scroll + visible {
        sel + 1 - visible
    } else {
        scroll
    }
}

/// A short inline utilization bar like `▕███░░░▏`.
fn util_bar(pct: f32, cells: usize) -> String {
    let filled = (((pct.clamp(0.0, 100.0) / 100.0) * cells as f32).round() as usize).min(cells);
    let mut s = String::with_capacity(cells + 2);
    s.push('▕');
    for _ in 0..filled {
        s.push('█');
    }
    for _ in 0..cells - filled {
        s.push('░');
    }
    s.push('▏');
    s
}

/// One dense line summarizing a GPU: selection marker, id, util bar, util%,
/// temperature and power (threshold-colored), and — when the width allows —
/// vram. Pure; used by the compact GPU layout.
pub fn gpu_compact_line(
    g: &GpuMetrics,
    theme: &Theme,
    width: u16,
    selected: bool,
) -> Line<'static> {
    let marker = if selected { "▸" } else { " " };
    let id = trunc(&g.device_id, 7);
    let id_style = if selected {
        Style::default()
            .fg(theme.accent)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(theme.fg)
    };
    let mut spans = vec![
        Span::styled(format!("{marker}{id:<7} "), id_style),
        Span::styled(
            format!("{} ", util_bar(g.gpu_utilization_pct, 6)),
            Style::default().fg(theme.accent),
        ),
        Span::styled(
            format!("{:>3.0}% ", g.gpu_utilization_pct),
            Style::default().fg(theme.fg),
        ),
        Span::styled(
            format!("{:>3.0}°C ", g.temperature_c),
            temperature_style(g.temperature_c, theme),
        ),
        Span::styled(
            format!("{:>4.0} W", g.power_w),
            power_style(g.power_w, theme),
        ),
    ];
    let base_w: usize = spans.iter().map(|s| s.content.chars().count()).sum();
    let used_g = g.vram_used_mb as f64 / 1024.0;
    let tot_g = g.vram_total_mb as f64 / 1024.0;
    let vram = format!("  {used_g:.0}/{tot_g:.0} GB");
    if base_w + vram.chars().count() <= width as usize {
        spans.push(Span::styled(vram, Style::default().fg(theme.muted)));
    }
    Line::from(spans)
}

/// Full-panel GPU rendering: one bordered block per GPU with paired
/// GPUFLO-style activity and VRAM occupancy instruments.
fn draw_gpus_full(
    f: &mut Frame,
    area: Rect,
    state: &AppState,
    snap: &Snapshot,
    theme: &Theme,
    sel: usize,
) {
    let n = snap.gpus.len() as u16;
    let per_gpu = (area.height / n).max(1);
    let constraints: Vec<Constraint> = (0..n).map(|_| Constraint::Length(per_gpu)).collect();
    let slots = Layout::default()
        .direction(Direction::Vertical)
        .constraints(constraints)
        .split(area);
    let sysinfo = snap.gpu_system_info.as_ref();

    for (i, g) in snap.gpus.iter().enumerate() {
        let slot = slots[i];
        if slot.height == 0 {
            continue;
        }
        let role = if i == sel {
            BoxRole::Primary
        } else {
            BoxRole::Secondary
        };
        let title = format!("GPU {}", g.device_id);
        let inner = panel::bento(f, slot, Some(&title), role, false, theme);
        if inner.height == 0 {
            continue;
        }

        let stats_line = gpu_stats_line(g, theme);
        let info_line = gpu_info_line(g.clock_mhz, sysinfo, theme);
        if inner.height == 1 {
            f.render_widget(Paragraph::new(stats_line), inner);
            continue;
        }
        if inner.height == 2 {
            f.render_widget(Paragraph::new(vec![stats_line, info_line]), inner);
            continue;
        }

        let serving = serving_line(
            &instances_on_gpu(&g.device_id, &snap.instances),
            inner.width,
            theme,
        );
        let want_serving = serving.is_some() && inner.height >= 6;
        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints(if want_serving {
                vec![
                    Constraint::Length(1),
                    Constraint::Length(1),
                    Constraint::Length(1),
                    Constraint::Min(1),
                ]
            } else {
                vec![
                    Constraint::Length(1),
                    Constraint::Length(1),
                    Constraint::Min(1),
                ]
            })
            .split(inner);
        f.render_widget(Paragraph::new(stats_line), rows[0]);
        f.render_widget(Paragraph::new(info_line), rows[1]);
        let graphs = if want_serving {
            f.render_widget(Paragraph::new(serving.unwrap()), rows[2]);
            rows[3]
        } else {
            rows[2]
        };
        let graph_rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Ratio(1, 2), Constraint::Ratio(1, 2)])
            .split(graphs);

        let activity_history: Vec<u64> = state
            .history
            .iter()
            .filter_map(|s| {
                s.gpus
                    .get(i)
                    .map(|gpu| gpu.gpu_utilization_pct.clamp(0.0, 100.0) as u64)
            })
            .collect();
        let vram_history: Vec<u64> = state
            .history
            .iter()
            .filter_map(|s| {
                let gpu = s.gpus.get(i)?;
                (gpu.vram_total_mb > 0)
                    .then_some((100 * gpu.vram_used_mb / gpu.vram_total_mb).min(100))
            })
            .collect();
        draw_gpu_metric(
            f,
            graph_rows[0],
            &format!("GPU activity · {:.1}%", g.gpu_utilization_pct),
            &activity_history,
            BoxRole::Primary,
            theme,
        );
        let vram_title = if g.vram_total_mb > 0 {
            format!(
                "VRAM occupancy · {:.1}%",
                100.0 * g.vram_used_mb as f64 / g.vram_total_mb as f64
            )
        } else {
            "VRAM occupancy · —".to_string()
        };
        draw_gpu_metric(
            f,
            graph_rows[1],
            &vram_title,
            &vram_history,
            BoxRole::Secondary,
            theme,
        );
    }
}

fn draw_gpu_metric(
    f: &mut Frame,
    area: Rect,
    title: &str,
    history: &[u64],
    role: BoxRole,
    theme: &Theme,
) {
    let inner = panel::bento(f, area, Some(title), role, false, theme);
    if inner.height == 0 {
        return;
    }
    f.render_widget(
        BrailleSparkline::new(history)
            .max(100)
            .style(Style::default().fg(theme.accent))
            .gradient(theme.ok, theme.warn, theme.err),
        inner,
    );
}

/// "serving: model[, model]" line for a GPU's instances, or `None` when the
/// GPU has no instances. Truncated to fit `width`.
fn serving_line(
    insts: &[&rocm_dash_core::metrics::Instance],
    width: u16,
    theme: &Theme,
) -> Option<Line<'static>> {
    if insts.is_empty() {
        return None;
    }
    let models = insts
        .iter()
        .map(|i| i.model_name.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    let budget = (width as usize).saturating_sub("serving: ".len());
    let models = trunc(&models, budget.max(1));
    Some(Line::from(vec![
        Span::styled("serving: ".to_string(), Style::default().fg(theme.muted)),
        Span::styled(models, Style::default().fg(theme.fg)),
    ]))
}

/// Compact GPU rendering: one dense line per GPU within a scrolled window,
/// with an overflow indicator when GPUs are hidden above or below.
fn draw_gpus_compact(
    f: &mut Frame,
    area: Rect,
    snap: &Snapshot,
    theme: &Theme,
    sel: usize,
    scroll: usize,
    visible: usize,
) {
    let n = snap.gpus.len();
    let eff_scroll = scroll_to_show(sel, scroll, visible).min(n.saturating_sub(visible));
    let end = (eff_scroll + visible).min(n);
    let hidden_above = eff_scroll;
    let hidden_below = n - end;

    let mut y = area.y;
    let bottom = area.y + area.height;
    for i in eff_scroll..end {
        if y >= bottom {
            break;
        }
        let g = &snap.gpus[i];
        let line = gpu_compact_line(g, theme, area.width, i == sel);
        f.render_widget(Paragraph::new(line), Rect::new(area.x, y, area.width, 1));
        y += 1;
    }
    if (hidden_above > 0 || hidden_below > 0) && y < bottom {
        let txt = format!(" ↑ {hidden_above}   ↓ {hidden_below} more");
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(
                txt,
                Style::default().fg(theme.muted),
            ))),
            Rect::new(area.x, y, area.width, 1),
        );
    }
}

fn gpu_info_line<'a>(
    clock_mhz: Option<f32>,
    sysinfo: Option<&'a GpuSystemInfo>,
    theme: &Theme,
) -> Line<'a> {
    let (compute, memory, rocm, driver) = match sysinfo {
        Some(si) => (
            format!("{:?}", si.compute_partition_mode).to_uppercase(),
            format!("{:?}", si.memory_partition_mode).to_uppercase(),
            si.rocm_version.as_deref().unwrap_or("?").to_string(),
            si.driver_version.as_deref().unwrap_or("?").to_string(),
        ),
        None => ("?".into(), "?".into(), "?".into(), "?".into()),
    };
    let clk = match clock_mhz {
        Some(v) => format::mhz(v.round() as u64),
        None => "-".to_string(),
    };
    Line::from(vec![Span::styled(
        format!("partition: {compute}/{memory}  ·  ROCm {rocm}  ·  driver {driver}  ·  clk {clk}"),
        Style::default().fg(theme.muted),
    )])
}

// ---- detail modal -----------------------------------------------------------

/// Heatmap redline for temperature (°C): a full bar means junction-redline-hot.
const HEATMAP_TEMP_MAX_C: f64 = 100.0;

/// Largest of a fixed `floor` (the semantic redline) and the observed maximum
/// in `data`. Keeps a heatmap row normalized to a meaningful limit while still
/// growing if telemetry exceeds that limit.
fn semantic_max(floor: f64, data: &[f64]) -> f64 {
    data.iter().copied().fold(floor, f64::max)
}

/// The detail-modal "now" line, with temperature and power threshold-colored
/// via [`temperature_style`] / [`power_style`]. Pure.
fn detail_now_line(g: &GpuMetrics, theme: &Theme) -> Line<'static> {
    let clk = match g.clock_mhz {
        Some(v) => format::mhz(v.round() as u64),
        None => "-".into(),
    };
    Line::from(vec![
        Span::styled(format!("{:<12} ", "now"), Style::default().fg(theme.muted)),
        Span::styled(
            format!("util {} · ", format::pct(g.gpu_utilization_pct)),
            Style::default().fg(theme.fg),
        ),
        Span::styled(
            format::celsius(g.temperature_c),
            temperature_style(g.temperature_c, theme),
        ),
        Span::styled(" · ".to_string(), Style::default().fg(theme.fg)),
        Span::styled(format::watts(g.power_w), power_style(g.power_w, theme)),
        Span::styled(format!(" · clk {clk}"), Style::default().fg(theme.fg)),
    ])
}

/// Full-screen detail for the currently-selected GPU.
///
/// Pulls per-tick samples out of `state.history` to build a metric × time heatmap (util, temp,
/// power, vram%) alongside a summary header and a footer hint.
pub fn draw_detail(f: &mut Frame, area: Rect, state: &AppState, theme: &Theme) {
    use crate::ui::heatmap::{Heatmap, HeatmapRow};
    use crate::ui::modal::{centered_rect, draw_popup_frame};

    let popup = centered_rect(85, 85, 140, 32, area);
    let Some(snap) = state.latest.as_ref() else {
        let inner = draw_popup_frame(f, popup, "GPU detail", theme);
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(
                "no snapshot yet",
                Style::default().fg(theme.muted),
            ))),
            inner,
        );
        return;
    };
    if snap.gpus.is_empty() {
        let inner = draw_popup_frame(f, popup, "GPU detail", theme);
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(
                "no GPUs reported",
                Style::default().fg(theme.muted),
            ))),
            inner,
        );
        return;
    }
    let i = state.gpu_sel.min(snap.gpus.len() - 1);
    let g = &snap.gpus[i];

    let title = format!(" GPU {} · detail ", g.device_id);
    let inner = draw_popup_frame(f, popup, &title, theme);
    if inner.height == 0 {
        return;
    }

    // Vertical layout: 4 summary lines + heatmap (Min) + 1 footer hint.
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1), // gap
            Constraint::Min(4),
            Constraint::Length(1),
        ])
        .split(inner);

    let sysinfo = snap.gpu_system_info.as_ref();
    let model = sysinfo.map_or("?", |si| si.gpu_model.as_str());
    let rocm = sysinfo
        .and_then(|si| si.rocm_version.as_deref())
        .unwrap_or("?");
    let driver = sysinfo
        .and_then(|si| si.driver_version.as_deref())
        .unwrap_or("?");
    let partitions = match sysinfo {
        Some(si) => format!(
            "{:?} / {:?}",
            si.compute_partition_mode, si.memory_partition_mode
        )
        .to_uppercase(),
        None => "? / ?".into(),
    };
    // Summary lines.
    let kv = |k: &'static str, v: String, tone: Style| -> Line<'static> {
        Line::from(vec![
            Span::styled(format!("{k:<12} "), Style::default().fg(theme.muted)),
            Span::styled(v, tone),
        ])
    };
    f.render_widget(
        Paragraph::new(vec![
            kv(
                "device",
                format!("{} · {model}", g.device_id),
                Style::default()
                    .fg(theme.accent)
                    .add_modifier(Modifier::BOLD),
            ),
            kv(
                "vram",
                format::mib_pair(g.vram_used_mb, g.vram_total_mb),
                Style::default().fg(theme.fg),
            ),
            detail_now_line(g, theme),
            kv(
                "platform",
                format!("partition {partitions} · ROCm {rocm} · driver {driver}"),
                Style::default().fg(theme.muted),
            ),
        ]),
        Rect::new(inner.x, inner.y, inner.width, 4),
    );

    // Heatmap rows derived from state.history.
    let history = &state.history;
    let util: Vec<f64> = history
        .iter()
        .filter_map(|s| s.gpus.get(i).map(|gpu| f64::from(gpu.gpu_utilization_pct)))
        .collect();
    let temp: Vec<f64> = history
        .iter()
        .filter_map(|s| s.gpus.get(i).map(|gpu| f64::from(gpu.temperature_c)))
        .collect();
    let power: Vec<f64> = history
        .iter()
        .filter_map(|s| s.gpus.get(i).map(|gpu| f64::from(gpu.power_w)))
        .collect();
    let vram: Vec<f64> = history
        .iter()
        .filter_map(|s| {
            s.gpus.get(i).map(|gpu| {
                if gpu.vram_total_mb > 0 {
                    100.0 * gpu.vram_used_mb as f64 / gpu.vram_total_mb as f64
                } else {
                    0.0
                }
            })
        })
        .collect();

    // Normalize temp/power to fixed semantic redlines so a full bar means
    // "near the limit", not "near the largest value seen this session". The
    // row still grows if telemetry ever exceeds the redline.
    let max_temp = semantic_max(HEATMAP_TEMP_MAX_C, &temp);
    let max_power = semantic_max(f64::from(POWER_CRIT_W), &power);
    let rows_vec = vec![
        HeatmapRow::new("util %", util, 100.0).stops(theme.ok, theme.warn, theme.err),
        HeatmapRow::new("temp °C", temp, max_temp).stops(theme.ok, theme.warn, theme.err),
        HeatmapRow::new("power W", power, max_power).stops(theme.ok, theme.warn, theme.err),
        HeatmapRow::new("vram %", vram, 100.0).stops(theme.ok, theme.warn, theme.err),
    ];
    let heat = Heatmap::new(&rows_vec)
        .stops(theme.ok, theme.warn, theme.err)
        .track_bg(theme.surface_2)
        .label_style(Style::default().fg(theme.muted))
        .label_width(10);
    f.render_widget(heat, rows[5]);

    // Footer hint.
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(
            " each row = a metric over the last N ticks · newest on the right · Esc close",
            Style::default().fg(theme.muted),
        ))),
        rows[6],
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::{ActiveTab, AppState};
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use rocm_dash_core::metrics::{GpuSystemInfo, SystemMetrics};
    use rocm_dash_core::partition::{ComputePartitionMode, MemoryPartitionMode};
    use std::collections::VecDeque;

    /// Render `draw` to a TestBackend and return the buffer as a flat string.
    fn render_to_string(state: &AppState, cols: u16, rows: u16) -> String {
        let backend = TestBackend::new(cols, rows);
        let mut term = Terminal::new(backend).unwrap();
        term.draw(|f| draw(f, f.area(), state, &state.theme))
            .unwrap();
        let buf = term.backend().buffer().clone();
        buf.content()
            .iter()
            .map(ratatui::buffer::Cell::symbol)
            .collect()
    }

    fn render_cpu_cores_to_lines(cores: &[f32], cols: u16, rows: u16) -> Vec<String> {
        let backend_cols = cols.max(1);
        let backend = TestBackend::new(backend_cols, rows.max(1));
        let mut term = Terminal::new(backend).unwrap();
        let theme = Theme::default_dark();
        let history = VecDeque::default();
        term.draw(|f| {
            draw_cpu_cores(f, Rect::new(0, 0, cols, rows), cores, &history, &theme);
        })
        .unwrap();
        term.backend()
            .buffer()
            .content()
            .chunks(backend_cols as usize)
            .map(|line| line.iter().map(ratatui::buffer::Cell::symbol).collect())
            .collect()
    }

    fn state_with_snapshot(snap: Snapshot) -> AppState {
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.active_tab = ActiveTab::Observe;
        s.latest = Some(snap);
        s
    }

    #[test]
    fn draw_renders_disk_and_net_io() {
        let snap = Snapshot {
            host: SystemMetrics {
                disk_read_bps: 1_200_000,
                disk_write_bps: 512,
                net_rx_bps: 2_500_000,
                net_tx_bps: 4_096,
                ..Default::default()
            },
            ..Default::default()
        };
        let out = render_to_string(&state_with_snapshot(snap), 120, 30);
        assert!(
            out.contains("Disk + Net"),
            "missing combined Disk + Net label: {out:?}"
        );
        assert!(out.contains("rx"), "missing rx label");
        assert!(out.contains("tx"), "missing tx label");
        assert!(out.contains("read"), "missing read label");
        assert!(out.contains("write"), "missing write label");
        assert!(out.contains("/s"), "missing rate suffix");
        // a scaled value should be present
        assert!(out.contains("M/s"), "missing M/s scaled rate: {out:?}");
    }

    #[test]
    fn draw_does_not_panic_when_squeezed() {
        let snap = Snapshot {
            host: SystemMetrics {
                disk_read_bps: 1_000,
                net_rx_bps: 2_000,
                ..Default::default()
            },
            ..Default::default()
        };
        let s = state_with_snapshot(snap);
        // Heights where the I/O row gets squeezed to 0–1 inner rows.
        for h in [1u16, 2, 3, 14, 16] {
            let _ = render_to_string(&s, 80, h);
        }
    }

    #[test]
    fn cpu_core_overflow_reserves_its_own_row() {
        let lines = render_cpu_cores_to_lines(&[1.0; 7], 20, 3);

        assert!(lines[0].contains("C0") && lines[0].contains("C2"));
        assert!(lines[1].contains("C1") && lines[1].contains("C3"));
        assert_eq!(lines[2].trim_end(), "+3 cores");
    }

    #[test]
    fn cpu_core_overflow_handles_narrow_and_zero_sized_areas() {
        let lines = render_cpu_cores_to_lines(&[1.0; 3], 9, 2);
        assert!(lines[0].contains("C0"));
        assert_eq!(lines[1].trim_end(), "+2 cores");

        let _ = render_cpu_cores_to_lines(&[1.0], 0, 1);
        let _ = render_cpu_cores_to_lines(&[1.0], 1, 0);
    }

    fn mk_gpu(id: &str, util: f32, temp: f32, power: f32) -> GpuMetrics {
        GpuMetrics {
            device_id: id.into(),
            vram_used_mb: 142 * 1024,
            vram_total_mb: 192 * 1024,
            gpu_utilization_pct: util,
            temperature_c: temp,
            power_w: power,
            clock_mhz: Some(1850.0),
        }
    }

    fn snap_with_gpus(n: usize) -> Snapshot {
        Snapshot {
            gpus: (0..n)
                .map(|i| mk_gpu(&format!("gpu-{i}"), 50.0 + i as f32, 65.0, 400.0))
                .collect(),
            ..Default::default()
        }
    }

    #[test]
    fn plan_full_panels_when_roomy() {
        // 2 GPUs, plenty of height → full panels for all.
        assert_eq!(plan_gpu_rows(40, 2), (true, 2));
    }

    #[test]
    fn plan_compact_then_scroll_when_tight() {
        // 8 GPUs, section height 8 → too tight for full panels, all fit compact.
        assert_eq!(plan_gpu_rows(8, 8), (false, 8));
        // 8 GPUs, section height 4 → compact + scroll, one row reserved for indicator.
        assert_eq!(plan_gpu_rows(4, 8), (false, 3));
    }

    #[test]
    fn scroll_to_show_tracks_selection() {
        // selection past the bottom of the window advances the scroll
        assert_eq!(scroll_to_show(5, 0, 3), 3);
        // selection above the window pulls the scroll up
        assert_eq!(scroll_to_show(1, 4, 3), 1);
        // selection already visible leaves scroll unchanged
        assert_eq!(scroll_to_show(2, 1, 3), 1);
        // degenerate visible
        assert_eq!(scroll_to_show(9, 4, 0), 0);
    }

    // ponytail: P3 folds Hardware into Observe; the main selection model now
    // drives the instances list rather than per-GPU scroll, so the former
    // `appstate_navigation_advances_gpu_scroll` test is removed.
    // `scroll_to_show` / `gpu_visible_count` stay unit-tested above and are
    // reused by Observe's GPU panel.

    #[test]
    fn gpu_compact_line_colors_and_truncates() {
        let theme = Theme::default_dark();
        // hot + high power → err color on both temp and power spans
        let hot = mk_gpu("gpu-2", 95.0, 88.0, 740.0);
        let line = gpu_compact_line(&hot, &theme, 120, true);
        let temp_span = line
            .spans
            .iter()
            .find(|s| s.content.contains("°C"))
            .unwrap();
        let pow_span = line.spans.iter().find(|s| s.content.contains('W')).unwrap();
        assert_eq!(temp_span.style.fg, Some(theme.err));
        assert_eq!(pow_span.style.fg, Some(theme.err));
        // wide → vram segment included
        assert!(line.spans.iter().any(|s| s.content.contains("GB")));
        // narrow → vram segment dropped (truncation by omission)
        let narrow = gpu_compact_line(&hot, &theme, 24, false);
        assert!(!narrow.spans.iter().any(|s| s.content.contains("GB")));
    }

    #[test]
    fn draw_full_panels_render_all_gpus() {
        let s = state_with_snapshot(snap_with_gpus(2));
        // tall terminal → full panels (each with a "GPU N" titled border)
        let out = render_to_string(&s, 100, 40);
        assert!(out.contains("GPU gpu-0"), "missing full panel 0: {out:?}");
        assert!(out.contains("GPU gpu-1"), "missing full panel 1");
        // full panels carry the info line (ROCm/partition), compact rows do not
        assert!(out.contains("partition"), "full panel info line missing");
    }

    #[test]
    fn draw_compact_rows_when_tight() {
        let s = state_with_snapshot(snap_with_gpus(6));
        // short terminal → compact one-line rows for each GPU, no panic
        let out = render_to_string(&s, 100, 24);
        // compact rows show the util bar glyphs and °C/W on one line
        assert!(out.contains('▕'), "missing util bar: {out:?}");
        assert!(out.contains("°C"));
    }

    #[test]
    fn draw_overflow_indicator_when_clipped() {
        let mut s = state_with_snapshot(snap_with_gpus(12));
        s.gpu_sel = 11;
        s.gpu_scroll = 5;
        // Short height forces a compact window with hidden rows above.
        let out = render_to_string(&s, 100, 12);
        assert!(out.contains("more"), "missing overflow affordance: {out:?}");
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn semantic_max_uses_floor_then_grows() {
        // all temps below the redline → max is the semantic floor
        assert_eq!(semantic_max(HEATMAP_TEMP_MAX_C, &[60.0, 78.0, 95.0]), 100.0);
        // an observed value above the floor wins
        assert_eq!(semantic_max(HEATMAP_TEMP_MAX_C, &[60.0, 110.0]), 110.0);
        // power: below critical → floor (POWER_CRIT_W)
        assert_eq!(
            semantic_max(f64::from(POWER_CRIT_W), &[400.0, 690.0]),
            700.0
        );
        // power: above critical → observed
        assert_eq!(
            semantic_max(f64::from(POWER_CRIT_W), &[400.0, 760.0]),
            760.0
        );
        // empty data → floor
        assert_eq!(semantic_max(f64::from(POWER_CRIT_W), &[]), 700.0);
    }

    #[test]
    fn detail_now_line_threshold_colors_power_and_temp() {
        let theme = Theme::default_dark();
        let hot = mk_gpu("gpu-0", 99.0, 88.0, 740.0);
        let line = detail_now_line(&hot, &theme);
        let temp_span = line
            .spans
            .iter()
            .find(|s| s.content.contains("°C"))
            .unwrap();
        let pow_span = line
            .spans
            .iter()
            .find(|s| s.content.contains(" W"))
            .unwrap();
        assert_eq!(
            temp_span.style.fg,
            Some(theme.err),
            "hot temp not err-colored"
        );
        assert_eq!(
            pow_span.style.fg,
            Some(theme.err),
            ">700W power not err-colored"
        );
        // a cool, low-power GPU is ok-colored
        let cool = mk_gpu("gpu-1", 10.0, 45.0, 300.0);
        let line2 = detail_now_line(&cool, &theme);
        let p2 = line2
            .spans
            .iter()
            .find(|s| s.content.contains(" W"))
            .unwrap();
        assert_eq!(p2.style.fg, Some(theme.ok));
    }

    #[test]
    fn draw_detail_renders_without_panic() {
        let mut s = state_with_snapshot(snap_with_gpus(4));
        // seed a little history so the heatmap has data
        for _ in 0..5 {
            s.history.push_back(snap_with_gpus(4));
        }
        s.modal = crate::app::Modal::Detail;
        let backend = TestBackend::new(140, 32);
        let mut term = Terminal::new(backend).unwrap();
        term.draw(|f| draw_detail(f, f.area(), &s, &s.theme))
            .unwrap();
        let buf = term.backend().buffer().clone();
        let out: String = buf
            .content()
            .iter()
            .map(ratatui::buffer::Cell::symbol)
            .collect();
        assert!(out.contains("detail"), "missing detail title: {out:?}");
        assert!(out.contains("power W"), "missing power heatmap row");
    }

    fn mk_instance(
        name: &str,
        gpu_ids: &[&str],
        gen_tps: Option<f64>,
    ) -> rocm_dash_core::metrics::Instance {
        rocm_dash_core::metrics::Instance {
            container_name: name.into(),
            model_name: name.into(),
            gpu_ids: gpu_ids
                .iter()
                .map(std::string::ToString::to_string)
                .collect(),
            gen_tps,
            ..Default::default()
        }
    }

    fn sysinfo_partitioned() -> GpuSystemInfo {
        GpuSystemInfo {
            rocm_version: Some("7.13.0".into()),
            driver_version: Some("6.10.5".into()),
            gpu_model: "MI355X".into(),
            physical_gpu_count: 4,
            logical_gpu_count: 8,
            vram_per_logical_gpu_mb: 24 * 1024,
            ..Default::default()
        }
    }

    #[test]
    fn serving_line_lists_models_or_none() {
        let theme = Theme::default_dark();
        assert!(serving_line(&[], 80, &theme).is_none());
        let a = mk_instance("llama-70b", &["3"], Some(100.0));
        let b = mk_instance("qwen-coder", &["3"], Some(50.0));
        let line = serving_line(&[&a, &b], 80, &theme).unwrap();
        let text: String = line.spans.iter().map(|s| s.content.as_ref()).collect();
        assert!(text.contains("serving:"));
        assert!(text.contains("llama-70b"));
        assert!(text.contains("qwen-coder"));
    }

    #[test]
    fn gpu_section_header_shows_partition_and_efficiency() {
        let theme = Theme::default_dark();
        let snap = Snapshot {
            gpus: vec![mk_gpu("gpu-0", 50.0, 65.0, 500.0)],
            gpu_system_info: Some(sysinfo_partitioned()),
            instances: vec![mk_instance("a", &["0"], Some(250.0))],
            ..Default::default()
        };
        let line = gpu_section_header(&snap, &theme);
        let text: String = line.spans.iter().map(|s| s.content.as_ref()).collect();
        assert!(text.contains("MI355X"), "missing GPU model: {text}");
        assert!(text.contains("4p/8l"), "missing partition counts: {text}");
        assert!(text.contains("/l"), "missing per-logical VRAM");
        assert!(text.contains("kW"), "missing total power");
        // 250 tok / 500 W = 0.50 tok/W
        assert!(text.contains("0.50 tok/W"), "missing efficiency: {text}");
    }

    #[test]
    fn gpu_section_header_efficiency_dash_without_traffic() {
        let theme = Theme::default_dark();
        let snap = Snapshot {
            gpus: vec![mk_gpu("gpu-0", 5.0, 45.0, 300.0)],
            gpu_system_info: Some(sysinfo_partitioned()),
            instances: vec![mk_instance("a", &["0"], None)], // no gen_tps
            ..Default::default()
        };
        let text: String = gpu_section_header(&snap, &theme)
            .spans
            .iter()
            .map(|s| s.content.as_ref())
            .collect();
        assert!(text.contains("eff -"), "expected dash efficiency: {text}");
    }

    #[test]
    fn draw_serving_line_appears_for_right_gpu_end_to_end() {
        // device_id "gpu-3" ↔ instance gpu_ids ["3"] normalization, rendered.
        let mut snap = snap_with_gpus(4);
        snap.gpu_system_info = Some(sysinfo_partitioned());
        snap.instances = vec![mk_instance("llama-70b", &["3"], Some(120.0))];
        let s = state_with_snapshot(snap);
        let out = render_to_string(&s, 110, 48); // tall → full panels w/ serving
        assert!(
            out.contains("serving:"),
            "no serving line rendered: {out:?}"
        );
        assert!(out.contains("llama-70b"), "model not shown on its GPU");
        // Header surfaces the GPU identity, partitioning, and efficiency end-to-end.
        assert!(out.contains("MI355X"), "header GPU model missing");
        assert!(out.contains("4p/8l"), "header partition counts missing");
        assert!(out.contains("tok/W"), "header efficiency missing");
    }

    #[test]
    fn draw_handles_degraded_and_empty_states() {
        // 1) No snapshot at all → "waiting…" placeholder, no panic.
        let mut s = AppState::new("t".into(), "default-dark".into());
        s.active_tab = ActiveTab::Observe;
        let out = render_to_string(&s, 100, 30);
        assert!(out.contains("waiting"), "no waiting placeholder: {out:?}");
        // 2) Snapshot with no GPUs → honest unavailable instruments.
        let no_gpus = state_with_snapshot(Snapshot::default());
        let out = render_to_string(&no_gpus, 100, 30);
        assert!(out.contains("no GPUs reported"), "missing no-GPU notice");
        assert!(out.contains("GPU activity · —"), "missing GPU placeholder");
        assert!(
            out.contains("VRAM occupancy · —"),
            "missing VRAM placeholder"
        );
        // 3) GPUs but no sysinfo and no instances and zero power → header still
        //    renders with "eff -" and "?"-free crash-free panels.
        let mut snap = snap_with_gpus(2);
        for g in &mut snap.gpus {
            g.power_w = 0.0;
        }
        let s3 = state_with_snapshot(snap);
        let out = render_to_string(&s3, 100, 30);
        assert!(
            out.contains("eff -"),
            "expected dash efficiency w/o power/traffic"
        );
        assert!(
            !out.contains("serving:"),
            "no serving line without instances"
        );

        // 4) Tiny terminals must not panic across a range of heights/widths.
        let s4 = state_with_snapshot(snap_with_gpus(8));
        for h in [1u16, 2, 3, 5, 8, 12, 18] {
            for w in [10u16, 24, 40, 80] {
                let _ = render_to_string(&s4, w, h);
            }
        }
    }

    #[test]
    fn info_line_renders_with_sysinfo() {
        let si = GpuSystemInfo {
            rocm_version: Some("6.2.0".into()),
            driver_version: Some("6.10.5".into()),
            compute_partition_mode: ComputePartitionMode::Spx,
            memory_partition_mode: MemoryPartitionMode::Nps1,
            ..Default::default()
        };
        let theme = Theme::default_dark();
        let line = gpu_info_line(Some(1850.0), Some(&si), &theme);
        let rendered: String = line.spans.iter().map(|s| s.content.as_ref()).collect();
        assert!(rendered.contains("partition: SPX/NPS1"));
        assert!(rendered.contains("ROCm 6.2.0"));
        assert!(rendered.contains("driver 6.10.5"));
        assert!(rendered.contains("clk 1.85 GHz"));
    }

    #[test]
    fn info_line_uses_placeholders_without_sysinfo() {
        let theme = Theme::default_dark();
        let line = gpu_info_line(None, None, &theme);
        let rendered: String = line.spans.iter().map(|s| s.content.as_ref()).collect();
        assert!(rendered.contains("partition: ?/?"));
        assert!(rendered.contains("ROCm ?"));
        assert!(rendered.contains("driver ?"));
        assert!(rendered.contains("clk -"));
    }
}
