"""Nsys per-pipeline aggregate plots (consume Series.nsys_dir)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .loader import Series, load_nsys_csv
from .style import PIPELINE_COLORS, LINESTYLE_CYCLE, save_fig


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

    fig, ax = plt.subplots()
    xs = np.arange(len(labels))
    colors = _series_colors(series_list)
    ax.bar(xs, reads, color=colors, edgecolor="black", linewidth=0.4,
           label="Read", alpha=0.55, hatch="//")
    ax.bar(xs, writes, bottom=reads, color=colors, edgecolor="black",
           linewidth=0.4, label="Write")
    _annotate_bars(ax, xs, [r + w for r, w in zip(reads, writes)])
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("DRAM bandwidth (GB/s)")
    ax.set_title("Average DRAM read + write bandwidth")
    ax.legend(loc="best")
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
    ax.legend(loc="best")
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
    ax.legend(loc="upper right", fontsize=7)
    written = save_fig(fig, out_dir, "warp_occupancy", fmt=fmt)
    plt.close(fig)
    return written


def plot_zcull_rejection(series_list: list[Series], out_dir: Path, *,
                         fmt: str = "pdf+png") -> list[Path]:
    series_list = _nsys_series(series_list)
    if not series_list:
        return []
    values: list[float] = []
    labels = [s.label for s in series_list]
    for s in series_list:
        df = load_nsys_csv(s, "11_per_pipeline_zcull.csv")
        if df is None or df.empty:
            values.append(float("nan"))
        else:
            v = df["rejectionPct"].iloc[0]
            values.append(float(v) if v is not None else float("nan"))

    fig, ax = plt.subplots()
    xs = np.arange(len(labels))
    ax.bar(xs, values, color=_series_colors(series_list),
           edgecolor="black", linewidth=0.4)
    _annotate_bars(ax, xs, values, fmt="{:.1f}%")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("ZCULL rejection (%)")
    ax.set_title("Hardware ZCULL sample rejection rate")
    ax.set_ylim(0, 100)
    written = save_fig(fig, out_dir, "zcull_rejection", fmt=fmt)
    plt.close(fig)
    return written


# PIX markers we actually want to surface in the report. These are the
# *leaf-level* markers (no double-counting via parent markers). 'Cull' and
# 'Main' are parents in the renderer's tree; we list their children instead.
PIX_REPORTED = (
    "DepthPrepass", "BuildMipChain", "SDSM",
    "RegionDispatch", "MainDispatch", "ShadowDispatch",
    "Trunk", "Leaves", "Terrain", "Impostors", "Sky",
)


def plot_pix_stages(series_list: list[Series], out_dir: Path, *,
                    fmt: str = "pdf+png") -> list[Path]:
    series_list = _nsys_series(series_list)
    if not series_list:
        return []
    labels = [s.label for s in series_list]
    # For each PIX stage, collect totalMs per series. Missing stage -> 0.
    per_stage: dict[str, list[float]] = {st: [] for st in PIX_REPORTED}
    for s in series_list:
        df = load_nsys_csv(s, "15_per_pipeline_pix_stages.csv")
        for st in PIX_REPORTED:
            v = 0.0
            if df is not None:
                m = df[df["stage"] == st]
                if not m.empty:
                    v = float(m["totalMs"].iloc[0])
            per_stage[st].append(v)
    # Drop stages that are zero across every series so the x-axis stays tight.
    active = [st for st in PIX_REPORTED if any(v > 0 for v in per_stage[st])]
    if not active:
        return []
    n_series = len(series_list)
    width = 0.8 / n_series
    xs = np.arange(len(active))

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    for i, s in enumerate(series_list):
        offset = (i - (n_series - 1) / 2.0) * width
        heights = [per_stage[st][i] for st in active]
        ax.bar(xs + offset, heights, width=width,
               color=PIPELINE_COLORS.get(s.pipeline, "#444444"),
               edgecolor="black", linewidth=0.3, label=s.label)
    ax.set_xticks(xs)
    ax.set_xticklabels(active, rotation=30, ha="right")
    ax.set_ylabel("Total GPU time inside pipeline window (ms)")
    ax.set_title("Per-PIX-marker GPU time per pipeline")
    ax.legend(loc="best")
    written = save_fig(fig, out_dir, "pix_stages", fmt=fmt)
    plt.close(fig)
    return written
