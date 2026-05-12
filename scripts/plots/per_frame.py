"""Per-frame plots (consume Series.frames)."""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .loader import STAGE_COLS, Series
from .style import LINESTYLE_CYCLE, PIPELINE_COLORS, STAGE_COLORS, save_fig


def _safe_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", s)


def _series_style(series_list: list[Series]) -> dict[int, tuple[str, str]]:
    """Returns {index: (color, linestyle)} so the same pipeline keeps the same hue
    and only the linestyle varies across runs of that pipeline."""
    pipeline_count: dict[str, int] = {}
    out: dict[int, tuple[str, str]] = {}
    for i, s in enumerate(series_list):
        idx = pipeline_count.get(s.pipeline, 0)
        pipeline_count[s.pipeline] = idx + 1
        color = PIPELINE_COLORS.get(s.pipeline, "#444444")
        linestyle = LINESTYLE_CYCLE[idx % len(LINESTYLE_CYCLE)]
        out[i] = (color, linestyle)
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
    ax.legend(loc="best")
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
        ax.plot(ms_sorted, cdf, label=s.label, color=color, linestyle=ls, linewidth=1.2)
        mean_ms = ms.mean()
        p99 = np.percentile(ms, 99)
        ax.axvline(mean_ms, color=color, alpha=0.25, linestyle=":")
        ax.axvline(p99, color=color, alpha=0.5, linestyle="--")
        ax.text(p99, 0.5, f"p99={p99:.1f}", color=color, fontsize=7,
                rotation=90, va="center", ha="right")
    ax.set_xlabel("GPU frame time (ms)")
    ax.set_ylabel("Cumulative fraction of frames")
    ax.set_title("Frame-time CDF (1% low = p99)")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="lower right")
    written = save_fig(fig, out_dir, "frame_time_cdf", fmt=fmt)
    plt.close(fig)
    return written


def plot_frame_time_hist(series_list: list[Series], out_dir: Path, *,
                         fmt: str = "pdf+png", bins: int = 50) -> list[Path]:
    if not series_list:
        return []
    styles = _series_style(series_list)
    all_ms = []
    for s in series_list:
        all_ms.append(s.frames["gpuMs"].dropna().to_numpy())
    flat = np.concatenate(all_ms) if all_ms else np.array([])
    if flat.size == 0:
        return []
    edges = np.linspace(flat.min(), flat.max(), bins + 1)

    fig, ax = plt.subplots()
    for i, s in enumerate(series_list):
        ms = s.frames["gpuMs"].dropna().to_numpy()
        if ms.size == 0:
            continue
        color, _ = styles[i]
        ax.hist(ms, bins=edges, alpha=0.45, color=color, label=s.label,
                edgecolor=color, linewidth=0.6)
    ax.set_xlabel("GPU frame time (ms)")
    ax.set_ylabel("Frame count")
    ax.set_title("Frame-time distribution")
    ax.legend(loc="best")
    written = save_fig(fig, out_dir, "frame_time_hist", fmt=fmt)
    plt.close(fig)
    return written


def plot_cpu_gpu_bar(series_list: list[Series], out_dir: Path, *,
                     fmt: str = "pdf+png") -> list[Path]:
    if not series_list:
        return []
    styles = _series_style(series_list)
    labels = [s.label for s in series_list]
    cpu_means = [s.frames["cpuMs"].dropna().mean() for s in series_list]
    gpu_means = [s.frames["gpuMs"].dropna().mean() for s in series_list]
    colors = [styles[i][0] for i in range(len(series_list))]

    fig, ax = plt.subplots()
    xs = np.arange(len(labels))
    # CPU on bottom (hatch pattern to distinguish from GPU which uses solid).
    ax.bar(xs, cpu_means, color=colors, alpha=0.45, hatch="//",
           edgecolor=colors, linewidth=0.8, label="CPU")
    ax.bar(xs, gpu_means, bottom=cpu_means, color=colors,
           edgecolor="black", linewidth=0.4, label="GPU")
    # Value labels when there are few enough bars.
    if len(labels) <= 8:
        for x, c, g in zip(xs, cpu_means, gpu_means):
            ax.text(x, c + g + 0.2, f"{c + g:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Mean CPU + GPU frame time")
    ax.legend(loc="best")
    written = save_fig(fig, out_dir, "cpu_gpu_bar", fmt=fmt)
    plt.close(fig)
    return written


def plot_stage_stacked_area(series_list: list[Series], out_dir: Path, *,
                            fmt: str = "pdf+png") -> list[Path]:
    written: list[Path] = []
    for s in series_list:
        df = s.frames
        # Drop stages that are all-NaN (this pipeline doesn't run them).
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
        ax.set_title(f"Per-stage GPU time - {s.label}")
        ax.legend(loc="upper left", fontsize=7)
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
    # Group stages on x-axis; bars within each group are series. Stages with
    # no data in ANY series are dropped from the x-axis.
    active = [c for c in STAGE_COLS
              if any(s.frames[c].notna().any() for s in series_list)]
    if not active:
        return []
    n_series = len(series_list)
    width = 0.8 / n_series
    xs = np.arange(len(active))
    styles = _series_style(series_list)

    fig, ax = plt.subplots()
    for i, s in enumerate(series_list):
        means = [s.frames[c].dropna().mean() if s.frames[c].notna().any() else 0.0
                 for c in active]
        offset = (i - (n_series - 1) / 2.0) * width
        color, _ = styles[i]
        ax.bar(xs + offset, means, width=width, color=color, label=s.label,
               edgecolor="black", linewidth=0.3)
    ax.set_xticks(xs)
    ax.set_xticklabels([c.replace("Ms", "") for c in active], rotation=20)
    ax.set_ylabel("Mean stage time (ms)")
    ax.set_title("Per-stage mean GPU time")
    ax.legend(loc="best")
    written = save_fig(fig, out_dir, "stage_grouped_bar", fmt=fmt)
    plt.close(fig)
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


def plot_camera_path_3d(series_list: list[Series], out_dir: Path, *,
                        fmt: str = "pdf+png",
                        color_by: str = "fps",
                        arrow_stride: int = 30) -> list[Path]:
    written: list[Path] = []
    for s in series_list:
        df = s.frames
        x = df["camPosX"].to_numpy()
        y = df["camPosY"].to_numpy()
        z = df["camPosZ"].to_numpy()
        c = _resolve_color_metric(df, color_by)

        fig = plt.figure(figsize=(6.0, 5.0))
        ax = fig.add_subplot(111, projection="3d")

        # Polyline coloured by metric: render as small per-segment line collection
        # so the colour varies smoothly along the path.
        pts = np.column_stack([x, y, z])
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
        norm = plt.Normalize(np.nanmin(c), np.nanmax(c))
        lc = Line3DCollection(segs, cmap="viridis", norm=norm, linewidth=1.4)
        lc.set_array(c[:-1])
        ax.add_collection3d(lc)
        fig.colorbar(lc, ax=ax, label=color_by, shrink=0.7, pad=0.1)

        # Direction arrows subsampled.
        idx = np.arange(0, len(df), max(1, arrow_stride))
        ax.quiver(
            x[idx], y[idx], z[idx],
            df["camDirX"].to_numpy()[idx],
            df["camDirY"].to_numpy()[idx],
            df["camDirZ"].to_numpy()[idx],
            length=max((x.max() - x.min()), 1.0) * 0.03,
            color="black", linewidth=0.6, alpha=0.7,
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
    for s in series_list:
        df = s.frames
        x = df["camPosX"].to_numpy()
        z = df["camPosZ"].to_numpy()
        c = _resolve_color_metric(df, color_by)

        fig, ax = plt.subplots()
        pts = np.column_stack([x, z])
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        norm = plt.Normalize(np.nanmin(c), np.nanmax(c))
        lc = LineCollection(segs, cmap="viridis", norm=norm, linewidth=1.4)
        lc.set_array(c[:-1])
        ax.add_collection(lc)
        fig.colorbar(lc, ax=ax, label=color_by, shrink=0.85)

        idx = np.arange(0, len(df), max(1, arrow_stride))
        scale = max((x.max() - x.min()), (z.max() - z.min()), 1.0) * 0.03
        ax.quiver(
            x[idx], z[idx],
            df["camDirX"].to_numpy()[idx],
            df["camDirZ"].to_numpy()[idx],
            scale_units="xy", scale=1.0 / scale,
            color="black", width=0.003, alpha=0.7,
        )
        ax.set_xlim(x.min() - scale, x.max() + scale)
        ax.set_ylim(z.min() - scale, z.max() + scale)
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
