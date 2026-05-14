"""Shared matplotlib styling for benchmark plots.

Call `apply()` once at program start to install LaTeX-friendly rcParams.
PIPELINE_COLORS and STAGE_COLORS are the canonical color maps every plot
should use, so the same pipeline keeps the same color across all figures.
"""
from __future__ import annotations

import matplotlib as mpl


PIPELINE_COLORS: dict[str, str] = {
    "Traditional": "#4C72B0",  # blue
    "Compute":     "#DD8452",  # orange
    "MeshShader":  "#55A467",  # green
}


def variant_color(base_hex: str, variant_idx: int) -> str:
    """Return a shade of `base_hex` for the N-th run of the same pipeline.

    Variant 0 returns the base unchanged; subsequent variants are progressively
    darkened in HLS space so the eye still groups them as 'a kind of orange'
    while reading them as distinct series.
    """
    if variant_idx <= 0:
        return base_hex
    import colorsys
    import matplotlib.colors as mcolors
    r, g, b = mcolors.to_rgb(base_hex)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    new_l = max(0.12, l - 0.20 * variant_idx)
    # Bump saturation slightly so the darker shades don't read as muddy.
    new_s = min(1.0, s + 0.05 * variant_idx)
    r2, g2, b2 = colorsys.hls_to_rgb(h, new_l, new_s)
    return mcolors.to_hex((r2, g2, b2))

# Order matches the per-frame CSV column order: depthPrepass, hizMs, sdsmMs,
# cullMs, shadowMs, skyMs, sceneMs. Picked from a colour-blind-safe sequence.
STAGE_COLORS: dict[str, str] = {
    "depthPrepassMs": "#4C72B0",
    "hizMs":          "#DD8452",
    "sdsmMs":         "#55A467",
    "cullMs":         "#C44E52",
    "shadowMs":       "#8172B2",
    "skyMs":          "#937860",
    "sceneMs":        "#DA8BC3",
}

# Linestyles cycled through when multiple runs share the same pipeline.
LINESTYLE_CYCLE: tuple[str, ...] = ("-", "--", ":", "-.")


def apply() -> None:
    mpl.rcParams.update({
        "font.family":      "serif",
        "font.size":        9,
        "axes.titlesize":   10,
        "axes.labelsize":   9,
        "legend.fontsize":  8,
        "figure.figsize":   (5.5, 3.4),
        "axes.grid":        True,
        "grid.alpha":       0.25,
        "savefig.bbox":     "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype":     42,   # TrueType, required by many publishers
    })


def legend_below(ax, *, anchor_y: float = -0.22, ncol: int | None = None,
                 **overrides) -> None:
    """Place a single-row legend underneath an axes' horizontal axis.

    `anchor_y` is negative (axes-fraction below the bottom). Push it further
    down (e.g. -0.30) when the axis has rotated tick labels or a label of its
    own that needs clearance. `ncol` defaults to the number of handles, which
    is what produces the requested single horizontal row.
    """
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    n = ncol if ncol is not None else len(handles)
    params: dict = dict(loc="upper center", bbox_to_anchor=(0.5, anchor_y),
                        ncol=n, fontsize=7, framealpha=0.85, borderpad=0.3,
                        columnspacing=1.2, handlelength=1.6, frameon=False)
    params.update(overrides)
    ax.legend(handles, labels, **params)


def save_fig(fig, out_dir, basename: str, *, fmt: str = "pdf+png", dpi: int = 200) -> list:
    """Save `fig` to `out_dir/<basename>.{pdf,png}` per `fmt`.

    fmt is one of "pdf", "png", or "pdf+png". Returns the paths written.
    """
    from pathlib import Path
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list = []
    if "pdf" in fmt:
        p = out_dir / f"{basename}.pdf"
        fig.savefig(p)
        written.append(p)
    if "png" in fmt:
        p = out_dir / f"{basename}.png"
        fig.savefig(p, dpi=dpi)
        written.append(p)
    return written
