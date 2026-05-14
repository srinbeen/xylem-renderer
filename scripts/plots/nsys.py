"""Nsys per-pipeline aggregate plots (consume Series.nsys_dir)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .loader import Series, load_nsys_csv
from .style import PIPELINE_COLORS, LINESTYLE_CYCLE, save_fig, legend_below


def _nsys_series(series_list: list[Series]) -> list[Series]:
    """Filter to series that actually have an nsys_results/ folder."""
    return [s for s in series_list if s.nsys_dir is not None]


def _series_colors(series_list: list[Series]) -> list[str]:
    return [PIPELINE_COLORS.get(s.pipeline, "#444444") for s in series_list]


def _annotate_bars(ax, xs, heights, *, fmt: str = "{:.1f}"):
    if len(heights) <= 8:
        for x, h in zip(xs, heights):
            if np.isnan(h):
                continue
            ax.text(x, h, fmt.format(h), ha="center", va="bottom", fontsize=7)


def plot_sm_throughput(series_list: list[Series], out_dir: Path, *,
                       fmt: str = "pdf+png") -> list[Path]:
    series_list = _nsys_series(series_list)
    if not series_list:
        return []
    values = []
    labels = []
    for s in series_list:
        df = load_nsys_csv(s, "05_per_pipeline_metrics.csv")
        if df is None:
            values.append(float("nan"))
        else:
            row = df[df["metricName"] == "SM Throughput [Throughput %]"]
            values.append(float(row["avgVal"].iloc[0]) if not row.empty else float("nan"))
        labels.append(s.label)

    fig, ax = plt.subplots()
    xs = np.arange(len(labels))
    ax.bar(xs, values, color=_series_colors(series_list),
           edgecolor="black", linewidth=0.4)
    _annotate_bars(ax, xs, values)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("SM Throughput (%)")
    ax.set_title("Average SM Throughput per pipeline")
    written = save_fig(fig, out_dir, "sm_throughput", fmt=fmt)
    plt.close(fig)
    return written


def plot_dram_bandwidth(series_list: list[Series], out_dir: Path, *,
                        fmt: str = "pdf+png") -> list[Path]:
    series_list = _nsys_series(series_list)
    if not series_list:
        return []
    reads: list[float] = []
    writes: list[float] = []
    labels = [s.label for s in series_list]
    for s in series_list:
        df = load_nsys_csv(s, "06_per_pipeline_memory_bandwidth.csv")
        r = w = float("nan")
        if df is not None:
            rr = df[df["metric"] == "GPU Memory Read Bandwidth [Throughput %]"]
            ww = df[df["metric"] == "GPU Memory Write Bandwidth [Throughput %]"]
            if not rr.empty:
                r = float(rr["avgGBps"].iloc[0])
            if not ww.empty:
                w = float(ww["avgGBps"].iloc[0])
        reads.append(r)
        writes.append(w)

    # Uniform colors across pipelines: Read vs Write is the only encoded
    # distinction. Pipeline identity is carried by the x-axis label.
    read_color  = "#4C72B0"   # blue (matches cpu_gpu_bar palette)
    write_color = "#7FB3D5"   # lighter blue
    fig, ax = plt.subplots()
    xs = np.arange(len(labels))
    width = 0.38
    reads_arr = np.array(reads)
    writes_arr = np.array(writes)
    ax.bar(xs - width / 2, reads_arr, width=width, color=read_color,
           edgecolor="black", linewidth=0.4, label="Read")
    ax.bar(xs + width / 2, writes_arr, width=width, color=write_color,
           edgecolor="black", linewidth=0.4, label="Write")

    for x, v in zip(xs - width / 2, reads_arr):
        if np.isfinite(v) and v > 0:
            ax.text(x, v, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
    for x, v in zip(xs + width / 2, writes_arr):
        if np.isfinite(v) and v > 0:
            ax.text(x, v, f"{v:.1f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("DRAM bandwidth (GB/s)")
    ax.set_title("Average DRAM read + write bandwidth")
    ax.grid(True, axis="y", alpha=0.25)
    legend_below(ax, anchor_y=-0.30)
    written = save_fig(fig, out_dir, "dram_bandwidth", fmt=fmt)
    plt.close(fig)
    return written


def plot_cache_hit_rates(series_list: list[Series], out_dir: Path, *,
                         fmt: str = "pdf+png") -> list[Path]:
    series_list = _nsys_series(series_list)
    if not series_list:
        return []
    categories = ("L1 Hit Rate [Ratio %]",
                  "L2 Hit Rate [Ratio %]",
                  "L2 Hit Rate from L1 [Ratio %]")
    short = ("L1", "L2", "L2 from L1")
    # values[i][j] = series i, category j
    values: list[list[float]] = []
    labels = [s.label for s in series_list]
    for s in series_list:
        df = load_nsys_csv(s, "08_per_pipeline_cache_hit_rates.csv")
        row: list[float] = []
        for cat in categories:
            if df is None:
                row.append(float("nan"))
            else:
                m = df[df["metric"] == cat]
                row.append(float(m["avgPct"].iloc[0]) if not m.empty else float("nan"))
        values.append(row)

    n_series = len(series_list)
    width = 0.8 / n_series
    xs = np.arange(len(categories))
    fig, ax = plt.subplots()
    for i, s in enumerate(series_list):
        offset = (i - (n_series - 1) / 2.0) * width
        ax.bar(xs + offset, values[i], width=width,
               color=PIPELINE_COLORS.get(s.pipeline, "#444444"),
               edgecolor="black", linewidth=0.3, label=s.label)
    ax.set_xticks(xs)
    ax.set_xticklabels(short)
    ax.set_ylabel("Hit rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Cache hit rates per pipeline")
    legend_below(ax, anchor_y=-0.28)
    written = save_fig(fig, out_dir, "cache_hit_rates", fmt=fmt)
    plt.close(fig)
    return written


def plot_warp_occupancy(series_list: list[Series], out_dir: Path, *,
                        fmt: str = "pdf+png") -> list[Path]:
    series_list = _nsys_series(series_list)
    if not series_list:
        return []
    # Five categories that conceptually partition SM cycles. We do NOT
    # normalize to 100% - the nsys metrics already are throughput %, so a
    # stack that exceeds 100% just means active SM slots overlap with warp
    # classes (which they do; this stack is a visualisation aid, not an
    # accounting partition).
    stacks = (
        ("Vertex/Tess/Geometry Warps [Throughput %]", "VTG",         "#4C72B0"),
        ("Pixel Warps [Throughput %]",                "Pixel",       "#DD8452"),
        ("Compute Warps [Throughput %]",              "Compute",     "#55A467"),
        ("Unallocated Warps in Active SMs [Throughput %]", "Unallocated", "#C44E52"),
        ("Idle SM Unused Warp Slots [Throughput %]",  "Idle slots",  "#8172B2"),
    )
    labels = [s.label for s in series_list]
    values: dict[str, list[float]] = {short: [] for _, short, _ in stacks}
    for s in series_list:
        df = load_nsys_csv(s, "12_per_pipeline_warp_occupancy.csv")
        for metric, short, _ in stacks:
            if df is None:
                values[short].append(float("nan"))
            else:
                m = df[df["metric"] == metric]
                values[short].append(float(m["avgVal"].iloc[0]) if not m.empty else float("nan"))

    fig, ax = plt.subplots()
    xs = np.arange(len(labels))
    bottoms = np.zeros(len(labels))
    for _, short, color in stacks:
        heights = np.array(values[short], dtype=float)
        heights_clean = np.nan_to_num(heights, nan=0.0)
        ax.bar(xs, heights_clean, bottom=bottoms, color=color, label=short,
               edgecolor="black", linewidth=0.3)
        bottoms = bottoms + heights_clean
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Throughput (%)")
    ax.set_title("Warp class occupancy per pipeline")
    legend_below(ax, anchor_y=-0.28)
    written = save_fig(fig, out_dir, "warp_occupancy", fmt=fmt)
    plt.close(fig)
    return written


# Geometry types we surface; each is timed twice in the renderer with
# Main_* (color pass) and Shadow_* (cascade pass) PIX markers. The other
# PIX markers (DepthPrepass, HiZ, SDSM, *Dispatch, Sky) are intentionally
# omitted here -- they overlap with plot_stage_grouped_bar at the
# parent-stage level (depthPrepassMs / hizMs / sdsmMs / cullMs / skyMs).
GEOMETRY_KINDS = ("Trunk", "Leaves", "Terrain", "Impostors")


def plot_pix_stages(series_list: list[Series], out_dir: Path, *,
                    fmt: str = "pdf+png") -> list[Path]:
    series_list = _nsys_series(series_list)
    if not series_list:
        return []
    # Per kind, per series, totalMs for Main_* and Shadow_* markers.
    main_ms: dict[str, list[float]] = {k: [] for k in GEOMETRY_KINDS}
    shadow_ms: dict[str, list[float]] = {k: [] for k in GEOMETRY_KINDS}
    for s in series_list:
        df = load_nsys_csv(s, "15_per_pipeline_pix_stages.csv")
        for kind in GEOMETRY_KINDS:
            def lookup(marker: str) -> float:
                if df is None:
                    return 0.0
                m = df[df["stage"] == marker]
                return float(m["totalMs"].iloc[0]) if not m.empty else 0.0
            main_ms[kind].append(lookup(f"Main_{kind}"))
            shadow_ms[kind].append(lookup(f"Shadow_{kind}"))
    # Drop kinds that are zero across every series.
    active = [k for k in GEOMETRY_KINDS
              if any(main_ms[k][i] + shadow_ms[k][i] > 0 for i in range(len(series_list)))]
    if not active:
        return []
    n_series = len(series_list)
    width = 0.8 / n_series
    xs = np.arange(len(active))

    # Fixed two-tone palette per pipeline: deep = main pass, light = shadow pass.
    pipeline_main_color = {
        "Traditional": "#4C72B0", "Compute": "#55A467", "MeshShader": "#DD8452",
    }
    pipeline_shadow_color = {
        "Traditional": "#A2B9D5", "Compute": "#A9D2B2", "MeshShader": "#EEC0A4",
    }

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    for i, s in enumerate(series_list):
        offset = (i - (n_series - 1) / 2.0) * width
        m_heights = np.array([main_ms[k][i] for k in active])
        s_heights = np.array([shadow_ms[k][i] for k in active])
        c_main = pipeline_main_color.get(s.pipeline, "#444444")
        c_shadow = pipeline_shadow_color.get(s.pipeline, "#999999")
        ax.bar(xs + offset, m_heights, width=width, color=c_main,
               edgecolor="black", linewidth=0.3,
               label=f"{s.label} (main)")
        ax.bar(xs + offset, s_heights, width=width, bottom=m_heights, color=c_shadow,
               edgecolor="black", linewidth=0.3,
               label=f"{s.label} (shadow)")
        for x, total in zip(xs + offset, m_heights + s_heights):
            if total > 0:
                ax.text(x, total, f"{total:.0f}", ha="center", va="bottom", fontsize=6)
    ax.set_xticks(xs)
    ax.set_xticklabels(active)
    ax.set_ylabel("Total GPU time inside pipeline window (ms)")
    ax.set_title("Per-geometry GPU time: main pass vs shadow pass")
    ax.grid(True, axis="y", alpha=0.25)
    legend_below(ax, anchor_y=-0.35, ncol=n_series)
    written = save_fig(fig, out_dir, "pix_stages", fmt=fmt)
    plt.close(fig)
    return written
