"""Per-frame plots (consume Series.frames)."""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .loader import STAGE_COLS, Series
from .style import (LINESTYLE_CYCLE, PIPELINE_COLORS, STAGE_COLORS, save_fig,
                    variant_color, legend_below)


def _safe_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", s)


def _series_style(series_list: list[Series]) -> dict[int, tuple[str, str]]:
    """Returns {index: (color, linestyle)} so the same pipeline stays in the
    same hue family; variants of the same pipeline are darkened shades of the
    base color so they read as distinct on FPS / CDF / hist / stage plots."""
    pipeline_count: dict[str, int] = {}
    out: dict[int, tuple[str, str]] = {}
    for i, s in enumerate(series_list):
        idx = pipeline_count.get(s.pipeline, 0)
        pipeline_count[s.pipeline] = idx + 1
        base = PIPELINE_COLORS.get(s.pipeline, "#444444")
        color = variant_color(base, idx)
        # Solid lines everywhere; color carries the variant signal now.
        out[i] = (color, "-")
    return out


def plot_fps_over_time(series_list: list[Series], out_dir: Path, *,
                       fmt: str = "pdf+png", smooth: bool = False) -> list[Path]:
    if not series_list:
        return []
    styles = _series_style(series_list)
    fig, ax = plt.subplots()
    for i, s in enumerate(series_list):
        x = s.frames["simTimeMs"] / 1000.0
        y = s.frames["fps"]
        if smooth:
            y = y.ewm(alpha=0.1, adjust=False).mean()
        color, ls = styles[i]
        ax.plot(x, y, label=s.label, color=color, linestyle=ls, linewidth=1.0)
    ax.set_xlabel("Simulation time (s)")
    ax.set_ylabel("FPS (frames/s)")
    ax.set_title("Frame rate over time")
    legend_below(ax, anchor_y=-0.22)
    written = save_fig(fig, out_dir, "fps_over_time", fmt=fmt)
    plt.close(fig)
    return written


def plot_frame_time_cdf(series_list: list[Series], out_dir: Path, *,
                        fmt: str = "pdf+png") -> list[Path]:
    if not series_list:
        return []
    styles = _series_style(series_list)
    fig, ax = plt.subplots()

    for i, s in enumerate(series_list):
        ms = s.frames["gpuMs"].dropna().to_numpy()
        if ms.size == 0:
            continue
        ms_sorted = np.sort(ms)
        cdf = np.arange(1, ms_sorted.size + 1) / ms_sorted.size
        color, ls = styles[i]
        ax.plot(ms_sorted, cdf, label=s.label, color=color, linestyle=ls, linewidth=1.4)

    ax.set_xlabel("GPU frame time (ms)")
    ax.set_ylabel("Fraction of Frames")
    ax.set_title("Frame-time CDF")
    ax.set_ylim(0, 1.02)
    legend_below(ax, anchor_y=-0.22)
    written = save_fig(fig, out_dir, "frame_time_cdf", fmt=fmt)
    plt.close(fig)
    return written


def plot_frame_time_hist(series_list: list[Series], out_dir: Path, *,
                         fmt: str = "pdf+png", bins: int = 50) -> list[Path]:
    if not series_list:
        return []
    styles = _series_style(series_list)

    series_ms: list[np.ndarray] = [s.frames["gpuMs"].dropna().to_numpy()
                                   for s in series_list]
    flat = np.concatenate([m for m in series_ms if m.size]) if series_ms else np.array([])
    if flat.size == 0:
        return []
    # Shared bin edges across all panels so bar widths are directly comparable.
    edges = np.linspace(flat.min(), flat.max(), bins + 1)

    # One stacked panel per series, shared x. Shared y so the eye can compare
    # bin heights between panels at a glance.
    n = len(series_list)
    fig, axes = plt.subplots(n, 1, sharex=True, sharey=True,
                             figsize=(5.5, max(1.6, 1.1 * n) + 0.6))
    if n == 1:
        axes = [axes]

    for ax, s, ms, i in zip(axes, series_list, series_ms, range(n)):
        color, _ = styles[i]
        ax.set_ylabel(s.label, fontsize=7)
        if ms.size == 0:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center", color="0.6")
            continue
        ax.hist(ms, bins=edges, color=color, edgecolor=color, linewidth=0.5)
        mean_ms = ms.mean()
        ax.axvline(mean_ms, color="black", linewidth=0.8, linestyle=":",
                   label="mean")
        ax.tick_params(axis="both", labelsize=7)

    axes[-1].set_xlabel("GPU frame time (ms)", fontsize=8)
    axes[0].set_title("Frame-time distribution", fontsize=9)
    fig.align_ylabels(axes)
    fig.tight_layout()
    # Single-row legend below the bottom subplot (just the "mean" line).
    legend_below(axes[-1], anchor_y=-0.55)
    written = save_fig(fig, out_dir, "frame_time_hist", fmt=fmt)
    plt.close(fig)
    return written


def plot_cpu_gpu_bar(series_list: list[Series], out_dir: Path, *,
                     fmt: str = "pdf+png") -> list[Path]:
    if not series_list:
        return []
    labels = [s.label for s in series_list]
    cpu_means = np.array([s.frames["cpuMs"].dropna().mean() for s in series_list])
    gpu_means = np.array([s.frames["gpuMs"].dropna().mean() for s in series_list])

    # Uniform color across pipelines: CPU vs GPU is the only encoded distinction.
    # Pipeline identity is carried by the x-axis label.
    cpu_color = "#4C72B0"   # blue (matches Traditional/legend hue)
    gpu_color = "#7FB3D5"   # lighter blue, distinct from CPU
    fig, ax = plt.subplots()
    xs = np.arange(len(labels))
    width = 0.38
    ax.bar(xs - width / 2, cpu_means, width=width, color=cpu_color,
           edgecolor="black", linewidth=0.4, label="CPU")
    ax.bar(xs + width / 2, gpu_means, width=width, color=gpu_color,
           edgecolor="black", linewidth=0.4, label="GPU")

    # Per-bar value labels (always useful with log scale).
    for x, v in zip(xs - width / 2, cpu_means):
        if np.isfinite(v) and v > 0:
            ax.text(x, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    for x, v in zip(xs + width / 2, gpu_means):
        if np.isfinite(v) and v > 0:
            ax.text(x, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Mean time per frame (ms, log)")
    ax.set_title("CPU vs GPU mean frame time")
    ax.grid(True, which="both", axis="y", alpha=0.25)
    legend_below(ax, anchor_y=-0.30)
    written = save_fig(fig, out_dir, "cpu_gpu_bar", fmt=fmt)
    plt.close(fig)
    return written


def plot_stage_stacked_area(series_list: list[Series], out_dir: Path, *,
                            fmt: str = "pdf+png") -> list[Path]:
    written: list[Path] = []

    # First pass: compute a uniform y-limit across all series so framerate
    # differences between pipelines are visible at a glance (a faster pipeline
    # produces a visibly shorter stack in the same coordinate space).
    global_max = 0.0
    for s in series_list:
        df = s.frames
        active = [c for c in STAGE_COLS if df[c].notna().any()]
        if not active:
            continue
        totals = np.sum([df[c].fillna(0).to_numpy() for c in active], axis=0)
        if totals.size:
            # Use 99th percentile of the stack height (not max) so a single
            # outlier frame doesn't squash everyone else.
            global_max = max(global_max, float(np.percentile(totals, 99)))
    if global_max <= 0:
        global_max = 1.0
    y_top = global_max * 1.08  # small headroom

    for s in series_list:
        df = s.frames
        active = [c for c in STAGE_COLS if df[c].notna().any()]
        if not active:
            continue
        x = df["simTimeMs"] / 1000.0
        ys = [df[c].fillna(0).to_numpy() for c in active]
        colors = [STAGE_COLORS[c] for c in active]

        fig, ax = plt.subplots()
        ax.stackplot(x, *ys, labels=active, colors=colors, alpha=0.85,
                     linewidth=0)
        ax.set_xlabel("Simulation time (s)")
        ax.set_ylabel("Stage GPU time (ms)")
        ax.set_ylim(0, y_top)
        ax.set_title(f"Per-stage GPU time - {s.label}")
        legend_below(ax, anchor_y=-0.24)
        written.extend(save_fig(
            fig, out_dir,
            f"stage_stacked_area__{_safe_filename(s.label)}",
            fmt=fmt))
        plt.close(fig)
    return written


def plot_stage_grouped_bar(series_list: list[Series], out_dir: Path, *,
                           fmt: str = "pdf+png") -> list[Path]:
    if not series_list:
        return []
    # Stages on x-axis; one bar per series within each stage group. Stages
    # absent from EVERY series are dropped from the x-axis.
    active = [c for c in STAGE_COLS
              if any(s.frames[c].notna().any() for s in series_list)]
    if not active:
        return []
    n_series = len(series_list)
    width = 0.8 / n_series
    xs = np.arange(len(active))
    styles = _series_style(series_list)

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for i, s in enumerate(series_list):
        means = [s.frames[c].dropna().mean() if s.frames[c].notna().any() else 0.0
                 for c in active]
        offset = (i - (n_series - 1) / 2.0) * width
        color, _ = styles[i]
        ax.bar(xs + offset, means, width=width,
               color=color, edgecolor="black", linewidth=0.4,
               label=s.label)

    ax.set_xticks(xs)
    ax.set_xticklabels([c.replace("Ms", "") for c in active], rotation=20)
    ax.set_ylabel("Mean stage time (ms)")
    ax.set_title("Per-stage mean GPU time")
    legend_below(ax, anchor_y=-0.28)

    written = save_fig(fig, out_dir, "stage_grouped_bar", fmt=fmt)
    plt.close(fig)
    return written


def _summarize_ms(ms: np.ndarray) -> dict[str, float]:
    """Canonical frame-time summary stats (all in ms)."""
    if ms.size == 0:
        return {"n": 0, "min": np.nan, "q1": np.nan, "median": np.nan,
                "mean": np.nan, "q3": np.nan, "p99": np.nan, "max": np.nan,
                "std": np.nan}
    q1, med, q3 = np.percentile(ms, [25, 50, 75])
    return {
        "n": int(ms.size),
        "min": float(ms.min()),
        "q1": float(q1),
        "median": float(med),
        "mean": float(ms.mean()),
        "q3": float(q3),
        "p99": float(np.percentile(ms, 99)),
        "max": float(ms.max()),
        "std": float(ms.std(ddof=1)) if ms.size > 1 else 0.0,
    }


_STAT_KEYS: tuple[str, ...] = ("min", "q1", "median", "mean", "q3", "p99",
                               "max", "std")
_STAT_HEADERS: tuple[str, ...] = ("Min", "Q1", "Median", "Mean", "Q3", "p99",
                                  "Max", "Std")


_TEX_ESCAPES = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%",
                "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
                "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def _tex_escape(s: str) -> str:
    return "".join(_TEX_ESCAPES.get(ch, ch) for ch in s)


def _booktabs_tex(header: list[str], rows: list[list[str]], *,
                  label_colors: list[str] | None = None,
                  caption: str = "", tex_label: str = "",
                  row_rules: bool = True,
                  footer_row: list[str] | None = None) -> str:
    """Return a self-contained LaTeX `tabular` snippet using `booktabs` rules.

    The first column is left-aligned; every other column is right-aligned for
    numeric stats. If `label_colors` is provided, the first cell of each data
    row is wrapped in `\\textcolor[HTML]{...}{\\textbf{...}}` so each row label
    keeps the colour used by the matching plot series.

    Requires `\\usepackage{booktabs}` (and `\\usepackage{xcolor}` if colours
    are used) in the consuming document.
    """
    col_spec = "l" + "r" * (len(header) - 1)
    lines: list[str] = [
        r"% Requires: \usepackage{booktabs}"
        + (r"  \usepackage{xcolor}" if label_colors else ""),
        r"\begin{table}[h]",
        r"\centering",
    ]
    if caption:
        lines.append(rf"\caption{{{_tex_escape(caption)}}}")
    if tex_label:
        lines.append(rf"\label{{{tex_label}}}")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    lines.append(" & ".join(rf"\textbf{{{_tex_escape(h)}}}" for h in header)
                 + r" \\")
    lines.append(r"\midrule")
    for i, row in enumerate(rows):
        first = _tex_escape(row[0])
        if label_colors and i < len(label_colors) and label_colors[i]:
            hexcol = label_colors[i].lstrip("#").upper()
            first = rf"\textcolor[HTML]{{{hexcol}}}{{\textbf{{{first}}}}}"
        else:
            first = rf"\textbf{{{first}}}"
        cells = [first] + [_tex_escape(c) for c in row[1:]]
        lines.append(" & ".join(cells) + r" \\")
        # Thin separator between consecutive data rows (booktabs \midrule).
        # `\toprule` and `\bottomrule` are heavier on their own; this keeps
        # the iconic thick-top, thin-between, thick-bottom appearance.
        if row_rules and i < len(rows) - 1:
            lines.append(r"\midrule")
    lines.append(r"\bottomrule")
    if footer_row:
        # Extra row below \bottomrule with cells lined up to the table's
        # columns (e.g. a per-column sample-count summary). Cells here are
        # passed through verbatim so the caller can embed `\textbf{...}` etc.
        padded = list(footer_row) + [""] * max(0, len(header) - len(footer_row))
        padded = padded[:len(header)]
        lines.append(r"\addlinespace[2pt]")
        lines.append(" & ".join(padded) + r" \\")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def _write_tex(out_dir: Path, basename: str, content: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{basename}.tex"
    p.write_text(content, encoding="utf-8")
    return p


def plot_frame_stats_table(series_list: list[Series], out_dir: Path, *,
                           fmt: str = "pdf+png") -> list[Path]:
    """Emit a booktabs LaTeX table: one row per series, columns = frame-time
    summary stats. `fmt` is accepted for interface uniformity but ignored —
    output is always a `.tex` source file."""
    del fmt  # always .tex
    if not series_list:
        return []
    styles = _series_style(series_list)
    n_values = [_summarize_ms(s.frames["gpuMs"].dropna().to_numpy())["n"]
                for s in series_list]
    common_n = n_values[0] if len(set(n_values)) == 1 else None

    header = (["Pipeline", *_STAT_HEADERS] if common_n is not None
              else ["Pipeline", "n", *_STAT_HEADERS])
    rows: list[list[str]] = []
    row_colors: list[str] = []
    for i, s in enumerate(series_list):
        ms = s.frames["gpuMs"].dropna().to_numpy()
        stats = _summarize_ms(ms)
        row = [s.label]
        if common_n is None:
            row.append(f"{stats['n']:,}")
        for k in _STAT_KEYS:
            v = stats[k]
            row.append("---" if not np.isfinite(v) else f"{v:.2f}")
        rows.append(row)
        row_colors.append(styles[i][0])

    footer: list[str] | None = None
    if common_n is not None and "Min" in header:
        footer = [""] * len(header)
        footer[0] = r"\textbf{Frames}"
        footer[header.index("Min")] = f"{common_n:,}"
    tex = _booktabs_tex(header, rows, label_colors=row_colors,
                        caption="Frame-time summary (gpuMs, ms).",
                        tex_label="tab:frame-stats", footer_row=footer)
    return [_write_tex(out_dir, "frame_stats_table", tex)]


def plot_stage_stats_table(series_list: list[Series], out_dir: Path, *,
                           fmt: str = "pdf+png") -> list[Path]:
    """Emit one booktabs LaTeX table per series; rows are the stages that
    pipeline actually runs."""
    del fmt
    written: list[Path] = []
    for s in series_list:
        df = s.frames
        active = [c for c in STAGE_COLS if df[c].notna().any()]
        if not active:
            continue
        frame_ms = df["gpuMs"].dropna().to_numpy()
        frame_n = _summarize_ms(frame_ms)["n"]
        n_values = [_summarize_ms(df[c].dropna().to_numpy())["n"]
                    for c in active] + [frame_n]
        common_n = n_values[0] if len(set(n_values)) == 1 else None
        header = (["Stage", *_STAT_HEADERS] if common_n is not None
                  else ["Stage", "n", *_STAT_HEADERS])

        rows: list[list[str]] = []
        row_colors: list[str] = []
        for c in active:
            ms = df[c].dropna().to_numpy()
            stats = _summarize_ms(ms)
            row = [c.replace("Ms", "")]
            if common_n is None:
                row.append(f"{stats['n']:,}")
            for k in _STAT_KEYS:
                v = stats[k]
                row.append("---" if not np.isfinite(v) else f"{v:.3f}")
            rows.append(row)
            row_colors.append(STAGE_COLORS[c])

        # Whole-frame summary row (gpuMs) for cross-reference with the
        # stage rows above.
        frame_stats = _summarize_ms(frame_ms)
        frame_row = ["Frame"]
        if common_n is None:
            frame_row.append(f"{frame_stats['n']:,}")
        for k in _STAT_KEYS:
            v = frame_stats[k]
            frame_row.append("---" if not np.isfinite(v) else f"{v:.3f}")
        rows.append(frame_row)
        row_colors.append("")

        slug = _safe_filename(s.label)
        footer: list[str] | None = None
        if common_n is not None and "Min" in header:
            footer = [""] * len(header)
            footer[0] = r"\textbf{Frames}"
            footer[header.index("Min")] = f"{common_n:,}"
        tex = _booktabs_tex(
            header, rows, label_colors=row_colors,
            caption=f"Per-stage GPU time (ms) --- {s.label}.",
            tex_label=f"tab:stage-stats-{slug}", footer_row=footer)
        written.append(_write_tex(out_dir, f"stage_stats_table__{slug}", tex))
    return written


from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers '3d' projection)


def _resolve_color_metric(df, name: str):
    """Return the per-frame array to color the camera path by.

    Accepts any DataFrame column name; falls back to 'fps' if name is invalid
    (with a warning)."""
    if name in df.columns:
        return df[name].to_numpy()
    print(f"WARNING: color_by='{name}' not found; falling back to 'fps'")
    return df["fps"].to_numpy()


def _global_color_range(series_list: list[Series], color_by: str
                        ) -> tuple[float, float]:
    """Compute a single (vmin, vmax) for `color_by` across all series so the
    per-series camera-path plots are visually comparable."""
    vmin, vmax = np.inf, -np.inf
    for s in series_list:
        c = _resolve_color_metric(s.frames, color_by)
        c = c[np.isfinite(c)]
        if c.size == 0:
            continue
        vmin = min(vmin, float(c.min()))
        vmax = max(vmax, float(c.max()))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        # Degenerate: fall back to a tiny window so Normalize doesn't blow up.
        return (0.0, 1.0) if not np.isfinite(vmin) else (vmin, vmin + 1.0)
    return vmin, vmax


def plot_camera_path_3d(series_list: list[Series], out_dir: Path, *,
                        fmt: str = "pdf+png",
                        color_by: str = "fps",
                        arrow_stride: int = 30) -> list[Path]:
    written: list[Path] = []
    vmin, vmax = _global_color_range(series_list, color_by)
    norm = plt.Normalize(vmin, vmax)
    for s in series_list:
        df = s.frames
        x = df["camPosX"].to_numpy()
        y = df["camPosY"].to_numpy()
        z = df["camPosZ"].to_numpy()
        c = _resolve_color_metric(df, color_by)

        fig = plt.figure(figsize=(6.0, 5.0))
        ax = fig.add_subplot(111, projection="3d")

        # Polyline coloured by metric: render as small per-segment line collection
        # so the colour varies smoothly along the path. Shared `norm` across
        # all series means the same colour means the same value in every plot.
        pts = np.column_stack([x, y, z])
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
        lc = Line3DCollection(segs, cmap="viridis", norm=norm, linewidth=1.4)
        lc.set_array(c[:-1])
        ax.add_collection3d(lc)
        fig.colorbar(lc, ax=ax, label=f"{color_by} (shared scale)",
                     shrink=0.7, pad=0.1)

        # Direction arrows subsampled - bigger than the polyline so they read.
        idx = np.arange(0, len(df), max(1, arrow_stride))
        ax.quiver(
            x[idx], y[idx], z[idx],
            df["camDirX"].to_numpy()[idx],
            df["camDirY"].to_numpy()[idx],
            df["camDirZ"].to_numpy()[idx],
            length=max((x.max() - x.min()), 1.0) * 0.07,
            color="black", linewidth=1.1, alpha=0.85,
            arrow_length_ratio=0.4,
        )

        # Pad limits so the colorbar doesn't clip the data.
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())
        ax.set_zlim(z.min(), z.max())
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"Camera path coloured by {color_by} - {s.label}")
        written.extend(save_fig(
            fig, out_dir,
            f"camera_path_3d__{_safe_filename(s.label)}",
            fmt=fmt))
        plt.close(fig)
    return written


from matplotlib.collections import LineCollection


def plot_camera_path_2d(series_list: list[Series], out_dir: Path, *,
                        fmt: str = "pdf+png",
                        color_by: str = "fps",
                        arrow_stride: int = 30) -> list[Path]:
    written: list[Path] = []
    vmin, vmax = _global_color_range(series_list, color_by)
    norm = plt.Normalize(vmin, vmax)
    for s in series_list:
        df = s.frames
        x = df["camPosX"].to_numpy()
        z = df["camPosZ"].to_numpy()
        c = _resolve_color_metric(df, color_by)

        fig, ax = plt.subplots()
        pts = np.column_stack([x, z])
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap="viridis", norm=norm, linewidth=1.4)
        lc.set_array(c[:-1])
        ax.add_collection(lc)
        fig.colorbar(lc, ax=ax, label=f"{color_by} (shared scale)", shrink=0.85)

        idx = np.arange(0, len(df), max(1, arrow_stride))
        margin = max((x.max() - x.min()), (z.max() - z.min()), 1.0) * 0.06
        arrow_len = margin * 0.45
        # Normalize the (x,z) projection so the arrow length is constant.
        # camDir is a 3D unit vector; its XZ projection shrinks as the camera
        # pitches up/down, which made arrows look smaller at those points.
        dx = df["camDirX"].to_numpy()[idx]
        dz = df["camDirZ"].to_numpy()[idx]
        mag = np.hypot(dx, dz)
        mag = np.where(mag < 1e-6, 1.0, mag)
        ax.quiver(
            x[idx], z[idx], dx / mag, dz / mag,
            scale_units="xy", scale=1.0 / arrow_len,
            color="black", width=0.004, headwidth=4.0, headlength=4.5,
            alpha=0.85,
        )
        ax.set_xlim(x.min() - margin, x.max() + margin)
        ax.set_ylim(z.min() - margin, z.max() + margin)
        ax.set_aspect("equal")
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.set_title(f"Camera path (top-down) - {s.label}")
        written.extend(save_fig(
            fig, out_dir,
            f"camera_path_2d__{_safe_filename(s.label)}",
            fmt=fmt))
        plt.close(fig)
    return written
