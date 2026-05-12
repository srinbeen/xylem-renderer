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
