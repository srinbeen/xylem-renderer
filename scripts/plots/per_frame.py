"""Per-frame plots (consume Series.frames)."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .loader import STAGE_COLS, Series
from .style import PIPELINE_COLORS, STAGE_COLORS, variant_color


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


def _tail_mean(sorted_ms: np.ndarray, frac: float) -> float:
    """Mean of the slowest `frac` fraction of frame times.

    `sorted_ms` must already be ascending. The slowest frames are the tail
    end of that array. `k = max(1, round(n * frac))` so very short captures
    still produce a finite value (degenerates to `max` for tiny n)."""
    n = sorted_ms.size
    if n == 0:
        return float("nan")
    k = max(1, int(round(n * frac)))
    return float(sorted_ms[-k:].mean())


def _summarize_ms(ms: np.ndarray) -> dict[str, float]:
    """Canonical frame-time summary stats (all in ms).

    `low_1pct` / `low_0p1pct` are the mean frame time of the slowest 1% /
    0.1% of frames -- the same tail averages graphics benchmarks report
    as ``1% Low FPS`` / ``0.1% Low FPS``, just kept in ms so they stack
    cleanly next to the other frame-time stats. Distinct from `p99`,
    which is a threshold (not a tail average)."""
    if ms.size == 0:
        return {"n": 0, "min": np.nan, "q1": np.nan, "median": np.nan,
                "mean": np.nan, "q3": np.nan, "p99": np.nan,
                "low_1pct": np.nan, "low_0p1pct": np.nan,
                "max": np.nan, "std": np.nan}
    sorted_ms = np.sort(ms)
    q1, med, q3 = np.percentile(ms, [25, 50, 75])
    return {
        "n": int(ms.size),
        "min": float(sorted_ms[0]),
        "q1": float(q1),
        "median": float(med),
        "mean": float(ms.mean()),
        "q3": float(q3),
        "p99": float(np.percentile(ms, 99)),
        "low_1pct": _tail_mean(sorted_ms, 0.01),
        "low_0p1pct": _tail_mean(sorted_ms, 0.001),
        "max": float(sorted_ms[-1]),
        "std": float(ms.std(ddof=1)) if ms.size > 1 else 0.0,
    }


_STAT_KEYS: tuple[str, ...] = ("median", "mean", "low_1pct")
_STAT_HEADERS: tuple[str, ...] = ("Median", "Mean", "Worst 1%")


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
    if common_n is not None and "Median" in header:
        footer = [""] * len(header)
        footer[0] = r"\textbf{Frames}"
        footer[header.index("Median")] = f"{common_n:,}"
    tex = _booktabs_tex(header, rows, label_colors=row_colors,
                        caption="Frame-time summary (gpuMs, ms). "
                                "Worst 1% = mean frame time of the slowest "
                                "1% of frames (the frame-time tail average "
                                "behind the ``1% Low FPS'' benchmarking "
                                "metric).",
                        tex_label="tab:frame-stats", footer_row=footer)
    chart_dir = out_dir / "stats_table"
    written = [_write_tex(chart_dir, "frame_stats_table", tex)]
    written.append(_write_tex(chart_dir, "frame_stats_table.standalone",
                              _make_standalone_table_wrapper("frame_stats_table")))
    return written


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
        if common_n is not None and "Median" in header:
            footer = [""] * len(header)
            footer[0] = r"\textbf{Frames}"
            footer[header.index("Median")] = f"{common_n:,}"
        tex = _booktabs_tex(
            header, rows, label_colors=row_colors,
            caption=f"Per-stage GPU time (ms) --- {s.label}.",
            tex_label=f"tab:stage-stats-{slug}", footer_row=footer)
        # One folder per pipeline under stats_table/, mirroring the layout
        # used by stage_stacked_area, camera_path_3d, etc.
        chart_dir = out_dir / "stats_table" / slug
        written.append(_write_tex(chart_dir, "stage_stats_table", tex))
        written.append(_write_tex(chart_dir, "stage_stats_table.standalone",
                                  _make_standalone_table_wrapper("stage_stats_table")))
    return written



def _resolve_color_metric(df, name: str):
    """Return the per-frame array to color the camera path by.

    Accepts any DataFrame column name; falls back to 'fps' if name is invalid
    (with a warning)."""
    if name in df.columns:
        return df[name].to_numpy()
    print(f"WARNING: color_by='{name}' not found; falling back to 'fps'")
    return df["fps"].to_numpy()


# Per-metric default colormap for camera-path plots. The semantic intent is
# encoded here so callers don't have to pass `--cmap` for the common cases:
#   fps         -> viridis  (sequential, "more is better": dark = slow frames,
#                            bright = fast frames; matches matplotlib default)
#   visibleFrac -> RdYlGn_r (diverging, "less is better": green = low fraction
#                            of scene drawn = cull paying off, red = high
#                            fraction = lots reaching the rasteriser)
# Any color_by name not listed here falls back to viridis.
_COLOR_BY_CMAP: dict[str, str] = {
    "fps":         "viridis",
    "visibleFrac": "RdYlGn_r",
}


def _cmap_for(color_by: str) -> str:
    return _COLOR_BY_CMAP.get(color_by, "viridis")


# Explicit pgfplots `\pgfplotsset{colormap={...}}` blocks for the colormaps
# referenced by _COLOR_BY_CMAP. pgfplots ships a built-in `viridis` since
# v1.15 but defining it explicitly keeps the .standalone.tex compileable
# against older preamble setups.
#
# viridis: 8-stop approximation of matplotlib's perceptually-uniform default
# (dark purple -> teal -> yellow-green -> yellow).
# RdYlGn_r: 9-stop ColorBrewer Red-Yellow-Green, reversed so green = low.
_PGFPLOTS_COLORMAP_DEFS: dict[str, str] = {
    "viridis": (
        "\\pgfplotsset{colormap={viridis}{\n"
        "    rgb255=(68,1,84) rgb255=(72,40,120) rgb255=(62,73,137)\n"
        "    rgb255=(49,104,142) rgb255=(38,130,142) rgb255=(31,158,137)\n"
        "    rgb255=(53,183,121) rgb255=(109,205,89) rgb255=(180,222,44)\n"
        "    rgb255=(253,231,37)}}"
    ),
    "RdYlGn_r": (
        "\\pgfplotsset{colormap={RdYlGn_r}{\n"
        "    rgb255=(26,152,80) rgb255=(102,189,99) rgb255=(166,217,106)\n"
        "    rgb255=(217,239,139) rgb255=(255,255,191) rgb255=(254,224,139)\n"
        "    rgb255=(253,174,97) rgb255=(244,109,67) rgb255=(215,48,39)}}"
    ),
}


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


def _write_dat(path: Path, header: list[str], rows: np.ndarray) -> None:
    """Write a pgfplots-readable whitespace-separated table file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(" ".join(header) + "\n")
        for row in rows:
            fh.write(" ".join(f"{v:.6g}" for v in row) + "\n")


def _camera_path_3d_tex(*, base: str, label: str, color_by: str,
                        x_min: float, x_max: float,
                        y_min: float, y_max: float,
                        z_min: float, z_max: float,
                        meta_min: float, meta_max: float,
                        xtick_str: str, ytick_str: str,
                        ztick_str: str) -> str:
    """Render the `\\input{}`-able .tex fragment for one camera-path series.

    Output is a bare `\\begin{tikzpicture}...\\end{tikzpicture}` block --
    callers must wrap it in a document with pgfplots loaded (>= 1.16) before
    compiling. The companion `*.standalone.tex` wrapper generated alongside
    by `plot_camera_path_3d_tex` does exactly that for previewing."""
    cmap_name = _cmap_for(color_by)
    cmap_def = _PGFPLOTS_COLORMAP_DEFS.get(cmap_name,
                                           _PGFPLOTS_COLORMAP_DEFS["viridis"])
    safe_label = _tex_escape(label)
    safe_color_by = _tex_escape(color_by)
    # Axis swap: pgfplots-X=worldX, pgfplots-Y=worldZ, pgfplots-Z=worldY so the
    # renderer's Y-up world reads naturally as "vertical" in the rendered plot.
    return (
        f"% Auto-generated by scripts/plots/per_frame.py - do not hand-edit.\n"
        f"% Insert with \\input{{{base}}} from a document whose preamble loads\n"
        f"% pgfplots (>= 1.16). Companion data files (same directory):\n"
        f"%   {base}.dat            - polyline samples: worldX worldY worldZ meta\n"
        f"%   {base}_sightlines.dat - sight-line segments (empty-line=jump format)\n"
        f"%   {base}_tips.dat       - sight-line tip markers\n"
        f"\\begin{{tikzpicture}}\n"
        f"{cmap_def}\n"
        f"  \\begin{{axis}}[\n"
        f"    view={{60}}{{25}},\n"
        f"    width=13cm, height=10.5cm,\n"
        f"    scale only axis=true,\n"
        f"    xlabel={{$X$}}, ylabel={{$Z$}}, zlabel={{$Y$}},\n"
        f"    xmin={x_min:.6g}, xmax={x_max:.6g},\n"
        f"    ymin={z_min:.6g}, ymax={z_max:.6g},\n"
        f"    zmin={y_min:.6g}, zmax={y_max:.6g},\n"
        f"    point meta min={meta_min:.6g},\n"
        f"    point meta max={meta_max:.6g},\n"
        f"    colormap name={cmap_name},\n"
        f"    colorbar,\n"
        f"    % Pin the colorbar so its horizontal centre is at axis-x=1.06;\n"
        f"    % the legend anchors at the same x below the bar for centering.\n"
        f"    colorbar style={{\n"
        f"      at={{(1.04, 0.0)}}, anchor=south west,\n"
        f"      width=0.35cm,\n"
        f"      height=0.95*\\pgfkeysvalueof{{/pgfplots/parent axis height}},\n"
        f"      yshift=0.025*\\pgfkeysvalueof{{/pgfplots/parent axis height}},\n"
        f"      ylabel={{{safe_color_by}}}, font=\\scriptsize,\n"
        f"    }},\n"
        f"    title={{Camera Path --- {safe_label}}},\n"
        f"    grid=major,\n"
        f"    % Legend sits directly under the colorbar, centred on its width.\n"
        f"    % `legend image code` draws an explicit line+dot symbol matching\n"
        f"    % the in-plot camDir glyph (just bigger so it reads in the box).\n"
        f"    legend style={{at={{(1.06, -0.02)}}, anchor=north,\n"
        f"                   draw=none, fill=none, font=\\scriptsize}},\n"
        f"    legend cell align=left,\n"
        f"    % Shrink tick numbers + place them at round values so the 3D\n"
        f"    % corner labels read cleanly and don't run into each other.\n"
        f"    tick label style={{font=\\scriptsize}},\n"
        f"    every axis label/.append style={{font=\\small}},\n"
        f"    xtick={{{xtick_str}}},\n"
        f"    ytick={{{ytick_str}}},\n"
        f"    ztick={{{ztick_str}}},\n"
        f"    minor tick num=0,\n"
        f"  ]\n"
        f"    % Colored polyline (mesh) - segment colour interpolates the metric.\n"
        f"    % `forget plot` keeps it out of the legend so the line+dot image\n"
        f"    % below attaches to its own \\addlegendentry.\n"
        f"    \\addplot3+[\n"
        f"      mesh, mark=none, line width=1pt,\n"
        f"      point meta=\\thisrow{{meta}},\n"
        f"      forget plot,\n"
        f"    ] table[x=worldX, y=worldZ, z=worldY, meta=meta]\n"
        f"      {{{base}.dat}};\n"
        f"\n"
        f"    % Sight lines: thin segments from each subsampled polyline sample\n"
        f"    % extending forward along camDir. `empty line=jump` translates blank\n"
        f"    % rows in the .dat into pgfplots line breaks so each consecutive pair\n"
        f"    % of rows is its own segment.\n"
        f"    \\addplot3[\n"
        f"      no marks, line width=0.45pt, black, opacity=0.75,\n"
        f"      empty line=jump, forget plot,\n"
        f"    ] table[x=worldX, y=worldZ, z=worldY]\n"
        f"      {{{base}_sightlines.dat}};\n"
        f"\n"
        f"    % Tip dots: small filled circle at each sight-line tip so the glyph\n"
        f"    % direction stays readable when foreshortened at oblique view angles.\n"
        f"    \\addplot3[\n"
        f"      only marks, mark=*, mark size=0.7pt,\n"
        f"      mark options={{fill=black, draw=black, fill opacity=0.85, draw opacity=0.85}},\n"
        f"      forget plot,\n"
        f"    ] table[x=worldX, y=worldZ, z=worldY]\n"
        f"      {{{base}_tips.dat}};\n"
        f"\n"
        f"    % Legend entry: dummy image with custom line+dot symbol matching\n"
        f"    % the sight-line / tip styling above (full opacity + slightly\n"
        f"    % bigger so it reads in the legend box). \\addlegendimage draws\n"
        f"    % only into the legend, not into the plot itself.\n"
        f"    \\addlegendimage{{\n"
        f"      legend image code/.code={{%\n"
        f"        \\draw[black, line width=0.6pt]\n"
        f"          (0cm,0cm) -- (0.5cm,0cm);\n"
        f"        \\filldraw[black] (0.5cm,0cm) circle (1.2pt);\n"
        f"      }},\n"
        f"    }}\n"
        f"    \\addlegendentry{{Camera direction}}\n"
        f"  \\end{{axis}}\n"
        f"\\end{{tikzpicture}}\n"
    )


def _make_standalone_wrapper(fragment_basename: str,
                             extra_preamble: str = "") -> str:
    """Return a tiny `\\documentclass{standalone}` wrapper that `\\input{}`s
    the fragment. Used for one-shot pdflatex previews. `extra_preamble`
    is appended verbatim after the base pgfplots load -- used by fragments
    that need extra libraries (e.g. groupplots for the hist subplots)."""
    extra = (extra_preamble.rstrip() + "\n") if extra_preamble else ""
    return (
        f"% Auto-generated wrapper for previewing {fragment_basename}.tex.\n"
        f"% Delete after compiling; the fragment itself is the canonical source.\n"
        f"\\documentclass[tikz,border=2pt]{{standalone}}\n"
        f"\\usepackage{{pgfplots}}\n"
        f"\\pgfplotsset{{compat=1.16}}\n"
        f"{extra}"
        f"\\begin{{document}}\n"
        f"\\input{{{fragment_basename}}}\n"
        f"\\end{{document}}\n"
    )


def _make_standalone_table_wrapper(fragment_basename: str) -> str:
    """Standalone wrapper for booktabs tables (`\\begin{table}` + `\\caption`).

    `standalone` typesets the document body with no float context, so the
    fragment's `\\begin{table}[h]` is reduced to a no-op env and both
    `\\caption` and `\\label` are dropped -- the standalone PDF is a bare
    tabular preview matching plots/orchard's layout, and the caption /
    cross-ref text lives in the `.tex` fragment for paper inclusion.
    """
    return (
        f"% Auto-generated wrapper for previewing {fragment_basename}.tex.\n"
        f"% Delete after compiling; the fragment itself is the canonical source.\n"
        f"\\documentclass[border=10pt,varwidth=true]{{standalone}}\n"
        f"\\usepackage{{booktabs}}\n"
        f"\\usepackage{{xcolor}}\n"
        f"% Strip float decoration + caption/label so the fragment renders as\n"
        f"% a bare booktabs tabular in the standalone preview.\n"
        f"\\renewenvironment{{table}}[1][]{{}}{{}}\n"
        f"\\renewcommand{{\\caption}}[1]{{}}\n"
        f"\\renewcommand{{\\label}}[1]{{}}\n"
        f"\\begin{{document}}\n"
        f"\\input{{{fragment_basename}}}\n"
        f"\\end{{document}}\n"
    )


def _nice_ticks(lo: float, hi: float, target_n: int = 4,
                edge_trim_frac: float = 0.15) -> list[float]:
    """Return tick positions within [lo, hi] at "nice" round numbers.

    Snaps the tick step to the nearest of {1, 2, 5} * 10^k so labels read as
    multiples of 10 / 50 / 100 / etc. rather than awkward `range/3` divides.
    Target is `target_n` ticks; actual count may differ by 1-2 depending on
    where the round multiples land inside [lo, hi].

    `edge_trim_frac` drops any tick that lands within `frac * step` of the
    axis edge. At a 3D plot's corner, the X axis's last tick and the Y
    axis's first tick share screen space; if both labels land in the
    corner, they overlap. Trimming edge-hugging ticks (e.g. a `1000` tick at
    `xmax=1001.6`) avoids that collision."""
    span = hi - lo
    if span <= 0 or not np.isfinite(span):
        return [lo]
    raw_step = span / max(1, target_n)
    mag = 10.0 ** np.floor(np.log10(raw_step))
    norm = raw_step / mag
    if norm < 1.5:
        nice = 1.0
    elif norm < 3.0:
        nice = 2.0
    elif norm < 7.0:
        nice = 5.0
    else:
        nice = 10.0
    step = nice * mag
    first_k = int(np.ceil(lo / step))
    last_k = int(np.floor(hi / step))
    ticks = [(first_k + i) * step for i in range(max(0, last_k - first_k + 1))]
    trim = step * edge_trim_frac
    return [t for t in ticks if (t - lo) >= trim and (hi - t) >= trim]


def _format_ticks(ticks: list[float]) -> str:
    """Format a tick list for pgfplots' `xtick={...}` option."""
    return ", ".join(f"{t:.6g}" for t in ticks)


def _write_segments_dat(path: Path,
                        starts: np.ndarray, tips: np.ndarray) -> None:
    """Write line segments as pgfplots-readable rows separated by empty lines
    (which `empty line=jump` interprets as line breaks). Each pair of
    consecutive non-empty rows is one segment from `starts[i]` to `tips[i]`.
    Shared by frustum-edge and sight-line emitters."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("worldX worldY worldZ\n")
        for s, t in zip(starts, tips):
            fh.write(f"{s[0]:.6g} {s[1]:.6g} {s[2]:.6g}\n")
            fh.write(f"{t[0]:.6g} {t[1]:.6g} {t[2]:.6g}\n")
            fh.write("\n")  # blank row -> pgfplots line break




def plot_camera_path_3d_tex(series_list: list[Series], out_dir: Path, *,
                            fmt: str = "pdf+png",
                            color_by: str = "fps",
                            arrow_stride: int = 30) -> list[Path]:
    """pgfplots/tikz equivalent of `plot_camera_path_3d`.

    For each series, emits five files into `out_dir/camera_path_3d/<Pipeline>/`:
      - `camera_path_3d.tex`              (\\input{}-able fragment)
      - `camera_path_3d.standalone.tex`   (wrapper for pdflatex preview)
      - `camera_path_3d.dat`              (polyline samples)
      - `camera_path_3d_sightlines.dat`   (one segment per sample)
      - `camera_path_3d_tips.dat`         (tip-dot positions)

    The .tex is a bare `\\begin{tikzpicture}...\\end{tikzpicture}` block, so
    it can be `\\input{}`-ed into a host document that already loads pgfplots.
    The .standalone.tex wrapper provides a minimal preamble for one-shot
    previewing with pdflatex.

    camDir is rendered as a thin "sight line" extending forward from each
    subsampled polyline point in the camera-look direction, terminated by a
    small filled circle at the tip. Both elements are 3D-native (line +
    point), so direction lines up with the projected polyline at any view
    angle -- no flat-arrowhead / 3D-polyline mismatch.

    `fmt` is accepted for interface uniformity but ignored - output is always
    .tex + .dat. Shares `vmin/vmax` across all series via `_global_color_range`
    so the colormap is comparable across emitted plots."""
    del fmt  # always .tex + .dat
    written: list[Path] = []
    if not series_list:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    vmin, vmax = _global_color_range(series_list, color_by)
    for s in series_list:
        df = s.frames
        x = df["camPosX"].to_numpy()
        y = df["camPosY"].to_numpy()
        z = df["camPosZ"].to_numpy()
        c = _resolve_color_metric(df, color_by).astype(float, copy=True)
        # Non-finite metric -> clamp to vmin so the colormap shows the
        # "low" end instead of an unrendered NaN segment.
        c = np.where(np.isfinite(c), c, vmin)

        # Sight-line length: 20% of the smallest axis extent. Long enough to
        # extend clearly off the polyline as a directional indicator.
        x_range = max(x.max() - x.min(), 1.0)
        y_range = max(y.max() - y.min(), 1.0)
        z_range = max(z.max() - z.min(), 1.0)
        sightline_len = 0.20 * min(x_range, y_range, z_range)

        # Arc-length-based subsampling: at slow / waypoint stretches the
        # camera dwells for many frames, so a fixed frame stride drops a pile
        # of glyphs on top of each other. We instead distribute glyphs evenly
        # along the *path*: ~len(df)/arrow_stride total glyphs, spaced by
        # cumulative distance. Falls back to frame stride if the path is too
        # short for arc-length spacing to make sense.
        seg = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2)
        arc = np.concatenate([[0.0], np.cumsum(seg)])
        total_len = float(arc[-1])
        target_count = max(2, len(df) // max(1, arrow_stride))
        if total_len > sightline_len * 1.5:
            targets = np.linspace(0.0, total_len, target_count + 2)[1:-1]
            idx = np.unique(np.clip(np.searchsorted(arc, targets),
                                    0, len(df) - 1))
        else:
            idx = np.arange(0, len(df), max(1, arrow_stride))

        dx = df["camDirX"].to_numpy()[idx]
        dy = df["camDirY"].to_numpy()[idx]
        dz = df["camDirZ"].to_numpy()[idx]
        d_mag = np.sqrt(dx * dx + dy * dy + dz * dz)
        d_mag = np.where(d_mag < 1e-8, 1.0, d_mag)
        dir_unit = np.column_stack([dx / d_mag, dy / d_mag, dz / d_mag])

        # Each series gets its own subfolder named by its label, so the
        # fragment + dat files + rendered PDF + standalone wrapper stay
        # grouped. Filenames inside are simple (basename "camera_path_3d")
        # since the label is in the folder name. Pass `--labels` to
        # `plot_benchmarks.py` to get clean folder + title names (otherwise
        # the timestamp ends up in both).
        pipeline_dir = (out_dir / "camera_path_3d"
                        / _safe_filename(s.label))
        pipeline_dir.mkdir(parents=True, exist_ok=True)
        base = "camera_path_3d"

        path_rows = np.column_stack([x, y, z, c])
        _write_dat(pipeline_dir / f"{base}.dat",
                   ["worldX", "worldY", "worldZ", "meta"], path_rows)
        written.append(pipeline_dir / f"{base}.dat")

        # Sight lines + tip dots. Each segment runs from a polyline sample
        # straight out along camDir for `sightline_len` world units; pgfplots
        # draws line + dot as 3D primitives that share the polyline's
        # projection, so the apparent direction always matches the projected
        # camDir at any view angle.
        starts = np.column_stack([x[idx], y[idx], z[idx]])
        tips = starts + dir_unit * sightline_len
        _write_segments_dat(pipeline_dir / f"{base}_sightlines.dat",
                            starts, tips)
        written.append(pipeline_dir / f"{base}_sightlines.dat")
        _write_dat(pipeline_dir / f"{base}_tips.dat",
                   ["worldX", "worldY", "worldZ"], tips)
        written.append(pipeline_dir / f"{base}_tips.dat")

        # Axis bounds need to cover polyline AND sight-line tips, otherwise
        # tips extending past the camPos range get clipped (= missing dots).
        # A small padding margin keeps glyphs from sitting right on the walls.
        all_x = np.concatenate([x, tips[:, 0]])
        all_y = np.concatenate([y, tips[:, 1]])
        all_z = np.concatenate([z, tips[:, 2]])

        def _padded(lo: float, hi: float, pad_frac: float = 0.02
                    ) -> tuple[float, float]:
            pad = max(hi - lo, 1.0) * pad_frac
            return lo - pad, hi + pad

        x_lo, x_hi = _padded(float(all_x.min()), float(all_x.max()))
        y_lo, y_hi = _padded(float(all_y.min()), float(all_y.max()))
        z_lo, z_hi = _padded(float(all_z.min()), float(all_z.max()))

        # Snap ticks to round values on each axis. pgfplots-X = worldX,
        # pgfplots-Y = worldZ, pgfplots-Z = worldY (Y-up swap).
        xtick_str = _format_ticks(_nice_ticks(x_lo, x_hi))
        ytick_str = _format_ticks(_nice_ticks(z_lo, z_hi))
        ztick_str = _format_ticks(_nice_ticks(y_lo, y_hi))

        tex = _camera_path_3d_tex(
            base=base, label=s.label, color_by=color_by,
            x_min=x_lo, x_max=x_hi,
            y_min=y_lo, y_max=y_hi,
            z_min=z_lo, z_max=z_hi,
            meta_min=float(vmin), meta_max=float(vmax),
            xtick_str=xtick_str, ytick_str=ytick_str, ztick_str=ztick_str,
        )
        written.append(_write_tex(pipeline_dir, base, tex))
        # Companion standalone wrapper so the fragment can be previewed via
        # `pdflatex camera_path_3d.standalone.tex` without touching the
        # fragment itself.
        wrapper_tex = _make_standalone_wrapper(base)
        written.append(_write_tex(pipeline_dir, f"{base}.standalone",
                                  wrapper_tex))
    return written


# ---------------------------------------------------------------------------
# pgfplots / TikZ variant of plot_cpu_gpu_bar. Single chart aggregating all
# series side-by-side; mirrors the camera_path_3d output layout (fragment +
# .dat + standalone wrapper + PDF inside its own subfolder).
# ---------------------------------------------------------------------------

def _cpu_gpu_bar_tex(*, base: str, labels: list[str],
                    y_min: float, y_max: float) -> str:
    """Return the `\\input{}`-able .tex fragment for the CPU vs GPU bar."""
    n = len(labels)
    # Numeric x positions (1..n) + explicit xticklabels so labels can contain
    # spaces / hyphens without breaking pgfplots' symbolic-coord parser.
    # Each label is brace-wrapped so commas-in-labels would survive too.
    xticks = ", ".join(str(i) for i in range(1, n + 1))
    xticklabels = ", ".join("{" + _tex_escape(lbl) + "}" for lbl in labels)
    # Scale chart width with series count so per-group horizontal spacing
    # stays constant: ~2.8cm/group (matches the 14cm-wide 5-series layout).
    width_cm = max(6.0, 2.8 * n)
    return (
        f"% Auto-generated by scripts/plots/per_frame.py - do not hand-edit.\n"
        f"% Insert with \\input{{{base}}} from a document whose preamble loads\n"
        f"% pgfplots (>= 1.16). Companion data file (same directory):\n"
        f"%   {base}.dat - rows: x cpu_ms gpu_ms (one per series)\n"
        f"\\begin{{tikzpicture}}\n"
        f"  \\definecolor{{cpubar}}{{HTML}}{{4C72B0}}\n"
        f"  \\definecolor{{gpubar}}{{HTML}}{{7FB3D5}}\n"
        f"  \\begin{{axis}}[\n"
        f"    ybar,\n"
        f"    ymode=log,\n"
        f"    log origin=infty,\n"
        f"    bar width=0.35cm,\n"
        f"    width={width_cm:.2f}cm, height=8cm,\n"
        f"    scale only axis=true,\n"
        f"    % `enlarge x limits` expands the x-range -- low values keep\n"
        f"    % the per-group pixel width WIDE so adjacent groups have a\n"
        f"    % big gap. Higher values do the opposite (compress the data).\n"
        f"    enlarge x limits=0.10,\n"
        f"    ymin={y_min:.4g}, ymax={y_max:.4g},\n"
        f"    xtick={{{xticks}}},\n"
        f"    xticklabels={{{xticklabels}}},\n"
        f"    xticklabel style={{rotate=20, anchor=north east,\n"
        f"                       font=\\scriptsize}},\n"
        f"    ylabel={{Mean time per frame (ms, log)}},\n"
        f"    ylabel style={{font=\\small}},\n"
        f"    yticklabel style={{font=\\scriptsize}},\n"
        f"    yticklabel={{1.0e\\pgfmathprintnumber[print sign,fixed,precision=0]{{\\tick}}}},\n"
        f"    title={{CPU vs GPU mean frame time}},\n"
        f"    ymajorgrids=true, yminorgrids=true,\n"
        f"    minor grid style={{densely dotted, gray!50, line width=0.3pt}},\n"
        f"    % `area legend` forces simple filled-square swatches in the legend\n"
        f"    % (the default ybar legend image is a tiny grouped-bar drawing).\n"
        f"    area legend,\n"
        f"    legend style={{at={{(0.5, -0.30)}}, anchor=north,\n"
        f"                   draw=none, fill=none, font=\\scriptsize,\n"
        f"                   /tikz/every even column/.append style="
                            f"{{column sep=0.4cm}}}},\n"
        f"    legend columns=2,\n"
        f"    legend cell align=left,\n"
        f"    % Value labels: `point meta=rawy` keeps them as the raw value\n"
        f"    % (default would be the log-transformed value under ymode=log,\n"
        f"    % which shows negative numbers for any y < 1).\n"
        f"    every axis plot/.append style={{point meta=rawy}},\n"
        f"    nodes near coords={{\\pgfmathprintnumber[fixed, precision=2]\\pgfplotspointmeta}},\n"
        f"    nodes near coords style={{font=\\tiny, anchor=south, yshift=0.5pt}},\n"
        f"  ]\n"
        f"    \\addplot[fill=cpubar, draw=black, line width=0.3pt]\n"
        f"      table[x=x, y=cpu_ms] {{{base}.dat}};\n"
        f"    \\addlegendentry{{CPU}}\n"
        f"    \\addplot[fill=gpubar, draw=black, line width=0.3pt]\n"
        f"      table[x=x, y=gpu_ms] {{{base}.dat}};\n"
        f"    \\addlegendentry{{GPU}}\n"
        f"  \\end{{axis}}\n"
        f"\\end{{tikzpicture}}\n"
    )


def plot_cpu_gpu_bar_tex(series_list: list[Series], out_dir: Path, *,
                         fmt: str = "pdf+png") -> list[Path]:
    """pgfplots/tikz equivalent of `plot_cpu_gpu_bar`.

    Emits four files into `out_dir/cpu_gpu_bar/`:
      - `cpu_gpu_bar.tex`            (\\input{}-able fragment)
      - `cpu_gpu_bar.standalone.tex` (wrapper for pdflatex preview)
      - `cpu_gpu_bar.dat`            (x cpu_ms gpu_ms per series)
      - (cpu_gpu_bar.pdf is produced separately by compiling the wrapper)

    Unlike camera_path_3d, this is a single chart with all series side-by-
    side; one folder, no per-series subfolders.

    `fmt` is accepted for interface uniformity but ignored - output is
    always .tex + .dat."""
    del fmt  # always tex + dat
    if not series_list:
        return []

    chart_dir = out_dir / "cpu_gpu_bar"
    chart_dir.mkdir(parents=True, exist_ok=True)
    base = "cpu_gpu_bar"

    labels = [s.label for s in series_list]
    cpu_means = np.array([s.frames["cpuMs"].dropna().mean()
                          for s in series_list], dtype=float)
    gpu_means = np.array([s.frames["gpuMs"].dropna().mean()
                          for s in series_list], dtype=float)

    written: list[Path] = []
    dat_rows = np.column_stack([
        np.arange(1, len(labels) + 1, dtype=float),
        cpu_means, gpu_means,
    ])
    _write_dat(chart_dir / f"{base}.dat",
               ["x", "cpu_ms", "gpu_ms"], dat_rows)
    written.append(chart_dir / f"{base}.dat")

    # Log-scale Y limits with headroom for nodes-near-coords labels.
    finite = np.concatenate([cpu_means[np.isfinite(cpu_means)],
                             gpu_means[np.isfinite(gpu_means)]])
    if finite.size == 0:
        y_min, y_max = 0.1, 100.0
    else:
        y_min = max(0.05, float(finite.min()) * 0.4)
        # Modest headroom: horizontal value labels sit just above each bar
        # tip and only need ~quarter-decade of log-space clearance.
        y_max = float(finite.max()) * 2.0

    tex = _cpu_gpu_bar_tex(base=base, labels=labels,
                          y_min=y_min, y_max=y_max)
    written.append(_write_tex(chart_dir, base, tex))
    written.append(_write_tex(chart_dir, f"{base}.standalone",
                              _make_standalone_wrapper(base)))
    return written


# ---------------------------------------------------------------------------
# pgfplots / TikZ variant of plot_fps_over_time. Single chart with one line
# per series; one .dat file per series alongside one fragment .tex + wrapper.
# ---------------------------------------------------------------------------

def _line_series_dat_name(base: str, label: str) -> str:
    """Helper: per-series .dat filename used by the multi-line tex plots."""
    return f"{base}__{_safe_filename(label)}.dat"


def _fps_over_time_tex(*, base: str,
                       series_meta: list[tuple[str, str, str]]) -> str:
    """Return the `\\input{}`-able fragment for the FPS-over-time chart.

    `series_meta` is a list of (label, color_hex, dat_filename) tuples; one
    `\\addplot ... \\addlegendentry` is emitted per entry, sharing the same
    axis."""
    color_defs = "\n".join(
        f"  \\definecolor{{series{i}}}{{HTML}}{{{color.lstrip('#').upper()}}}"
        for i, (_, color, _) in enumerate(series_meta)
    )
    plot_lines: list[str] = []
    for i, (label, _, dat) in enumerate(series_meta):
        plot_lines.append(
            f"    \\addplot[draw=series{i}, line width=1pt,\n"
            f"            unbounded coords=discard]\n"
            f"      table[x=time, y=fps] {{{dat}}};\n"
            f"    \\addlegendentry{{{_tex_escape(label)}}}"
        )
    plots_str = "\n".join(plot_lines)
    return (
        f"% Auto-generated by scripts/plots/per_frame.py - do not hand-edit.\n"
        f"% Insert with \\input{{{base}}} from a document whose preamble loads\n"
        f"% pgfplots (>= 1.16). One companion .dat per series, named\n"
        f"% `{base}__<label>.dat`, with columns: time fps.\n"
        f"\\begin{{tikzpicture}}\n"
        f"{color_defs}\n"
        f"  \\begin{{axis}}[\n"
        f"    width=14cm, height=8cm, scale only axis=true,\n"
        f"    xlabel={{Simulation time (s)}},\n"
        f"    ylabel={{FPS (frames/s)}},\n"
        f"    xlabel style={{font=\\small}},\n"
        f"    ylabel style={{font=\\small}},\n"
        f"    tick label style={{font=\\scriptsize}},\n"
        f"    title={{Frame rate over time}},\n"
        f"    grid=major,\n"
        f"    ymajorgrids=true,\n"
        f"    enlarge x limits=false, enlarge y limits={{upper}},\n"
        f"    legend style={{at={{(0.5, -0.18)}}, anchor=north,\n"
        f"                   draw=none, fill=none, font=\\scriptsize,\n"
        f"                   /tikz/every even column/.append style="
                            f"{{column sep=0.5cm}}}},\n"
        f"    legend columns=3,\n"
        f"    legend cell align=left,\n"
        f"  ]\n"
        f"{plots_str}\n"
        f"  \\end{{axis}}\n"
        f"\\end{{tikzpicture}}\n"
    )


def plot_fps_over_time_tex(series_list: list[Series], out_dir: Path, *,
                           fmt: str = "pdf+png") -> list[Path]:
    """pgfplots/tikz equivalent of `plot_fps_over_time`.

    Emits into `out_dir/fps_over_time/`:
      - `fps_over_time.tex`            (\\input{}-able fragment)
      - `fps_over_time.standalone.tex` (wrapper for pdflatex preview)
      - `fps_over_time__<label>.dat`   (one per series: columns `time fps`)
      - `fps_over_time.pdf`            (produced by compiling the wrapper)

    Series colours come from `_series_style` so they match the matplotlib
    `plot_fps_over_time` output exactly (pipeline base colour + variant
    darkening for duplicate pipelines). Pass `--labels` to keep label slugs
    -- and therefore .dat filenames -- clean."""
    del fmt  # always .tex + .dat
    if not series_list:
        return []

    chart_dir = out_dir / "fps_over_time"
    chart_dir.mkdir(parents=True, exist_ok=True)
    base = "fps_over_time"

    styles = _series_style(series_list)
    written: list[Path] = []
    series_meta: list[tuple[str, str, str]] = []

    for i, s in enumerate(series_list):
        df = s.frames
        t = (df["simTimeMs"].to_numpy(dtype=float) / 1000.0)
        fps = df["fps"].to_numpy(dtype=float)
        # `unbounded coords=discard` in the tex side skips NaN rows, but
        # writing them as the literal "nan" keeps row alignment if any
        # downstream tool wants to index by frame.
        rows = np.column_stack([t, fps])
        dat_name = _line_series_dat_name(base, s.label)
        _write_dat(chart_dir / dat_name, ["time", "fps"], rows)
        written.append(chart_dir / dat_name)

        color_hex, _ = styles[i]
        series_meta.append((s.label, color_hex, dat_name))

    tex = _fps_over_time_tex(base=base, series_meta=series_meta)
    written.append(_write_tex(chart_dir, base, tex))
    written.append(_write_tex(chart_dir, f"{base}.standalone",
                              _make_standalone_wrapper(base)))
    return written


# ---------------------------------------------------------------------------
# pgfplots / TikZ variant of plot_frame_time_cdf. Single chart with one CDF
# line per series; each .dat is a (ms, cdf) pair sorted by ms.
# ---------------------------------------------------------------------------

def _frame_time_cdf_tex(*, base: str, x_max: float,
                        series_meta: list[tuple[str, str, str]]) -> str:
    """Return the `\\input{}`-able fragment for the frame-time CDF chart."""
    color_defs = "\n".join(
        f"  \\definecolor{{series{i}}}{{HTML}}{{{color.lstrip('#').upper()}}}"
        for i, (_, color, _) in enumerate(series_meta)
    )
    plot_lines: list[str] = []
    for i, (label, _, dat) in enumerate(series_meta):
        plot_lines.append(
            f"    \\addplot[draw=series{i}, line width=1.2pt]\n"
            f"      table[x=ms, y=cdf] {{{dat}}};\n"
            f"    \\addlegendentry{{{_tex_escape(label)}}}"
        )
    plots_str = "\n".join(plot_lines)
    return (
        f"% Auto-generated by scripts/plots/per_frame.py - do not hand-edit.\n"
        f"% Insert with \\input{{{base}}} from a document whose preamble loads\n"
        f"% pgfplots (>= 1.16). One companion .dat per series, named\n"
        f"% `{base}__<label>.dat`, with columns: ms cdf.\n"
        f"\\begin{{tikzpicture}}\n"
        f"{color_defs}\n"
        f"  \\begin{{axis}}[\n"
        f"    width=14cm, height=8cm, scale only axis=true,\n"
        f"    xlabel={{GPU frame time (ms)}},\n"
        f"    ylabel={{Fraction of frames}},\n"
        f"    xlabel style={{font=\\small}},\n"
        f"    ylabel style={{font=\\small}},\n"
        f"    tick label style={{font=\\scriptsize}},\n"
        f"    title={{Frame-time CDF}},\n"
        f"    grid=major,\n"
        f"    ymin=0, ymax=1.02,\n"
        f"    xmin=0, xmax={x_max:.6g},\n"
        f"    enlarge x limits=false,\n"
        f"    legend style={{at={{(0.5, -0.18)}}, anchor=north,\n"
        f"                   draw=none, fill=none, font=\\scriptsize,\n"
        f"                   /tikz/every even column/.append style="
                            f"{{column sep=0.5cm}}}},\n"
        f"    legend columns=3,\n"
        f"    legend cell align=left,\n"
        f"  ]\n"
        f"{plots_str}\n"
        f"  \\end{{axis}}\n"
        f"\\end{{tikzpicture}}\n"
    )


def plot_frame_time_cdf_tex(series_list: list[Series], out_dir: Path, *,
                            fmt: str = "pdf+png") -> list[Path]:
    """pgfplots/tikz equivalent of `plot_frame_time_cdf`.

    Emits into `out_dir/frame_time_cdf/`:
      - `frame_time_cdf.tex`            (\\input{}-able fragment)
      - `frame_time_cdf.standalone.tex` (wrapper for pdflatex preview)
      - `frame_time_cdf__<label>.dat`   (per series: columns `ms cdf`)
      - `frame_time_cdf.pdf`            (compiled separately via wrapper)

    Per-series .dat is sorted gpuMs vs i/n (empirical CDF). Series with no
    finite gpuMs samples are skipped silently."""
    del fmt
    if not series_list:
        return []

    chart_dir = out_dir / "frame_time_cdf"
    chart_dir.mkdir(parents=True, exist_ok=True)
    base = "frame_time_cdf"

    styles = _series_style(series_list)
    written: list[Path] = []
    series_meta: list[tuple[str, str, str]] = []
    x_max = 0.0

    for i, s in enumerate(series_list):
        ms = s.frames["gpuMs"].dropna().to_numpy(dtype=float)
        if ms.size == 0:
            continue
        ms_sorted = np.sort(ms)
        cdf = np.arange(1, ms_sorted.size + 1) / ms_sorted.size
        x_max = max(x_max, float(ms_sorted[-1]))

        rows = np.column_stack([ms_sorted, cdf])
        dat_name = _line_series_dat_name(base, s.label)
        _write_dat(chart_dir / dat_name, ["ms", "cdf"], rows)
        written.append(chart_dir / dat_name)

        color_hex, _ = styles[i]
        series_meta.append((s.label, color_hex, dat_name))

    if not series_meta:
        return written
    # Round x_max up slightly so the curve doesn't kiss the right axis.
    x_max *= 1.02

    tex = _frame_time_cdf_tex(base=base, x_max=x_max, series_meta=series_meta)
    written.append(_write_tex(chart_dir, base, tex))
    written.append(_write_tex(chart_dir, f"{base}.standalone",
                              _make_standalone_wrapper(base)))
    return written


# ---------------------------------------------------------------------------
# pgfplots / TikZ variant of plot_stage_grouped_bar. Stages on the x-axis,
# one bar per series at each stage. Stages absent from EVERY series are
# dropped (kept the same active-stage logic as the matplotlib version).
# ---------------------------------------------------------------------------

def _stage_grouped_bar_tex(*, base: str, stage_labels: list[str],
                           series_meta: list[tuple[str, str, str]],
                           y_max: float) -> str:
    """Return the `\\input{}`-able fragment for the per-stage grouped bar."""
    color_defs = "\n".join(
        f"  \\definecolor{{series{i}}}{{HTML}}{{{color.lstrip('#').upper()}}}"
        for i, (_, color, _) in enumerate(series_meta)
    )
    n_stages = len(stage_labels)
    xticks = ", ".join(str(i) for i in range(1, n_stages + 1))
    xticklabels = ", ".join("{" + _tex_escape(lbl) + "}" for lbl in stage_labels)
    plot_lines: list[str] = []
    for i, (label, _, dat) in enumerate(series_meta):
        plot_lines.append(
            f"    \\addplot[fill=series{i}, draw=black, line width=0.25pt]\n"
            f"      table[x=stage_idx, y=mean_ms] {{{dat}}};\n"
            f"    \\addlegendentry{{{_tex_escape(label)}}}"
        )
    plots_str = "\n".join(plot_lines)
    return (
        f"% Auto-generated by scripts/plots/per_frame.py - do not hand-edit.\n"
        f"% Insert with \\input{{{base}}} from a document whose preamble loads\n"
        f"% pgfplots (>= 1.16). One companion .dat per series, named\n"
        f"% `{base}__<label>.dat`, with columns: stage_idx mean_ms.\n"
        f"\\begin{{tikzpicture}}\n"
        f"{color_defs}\n"
        f"  \\begin{{axis}}[\n"
        f"    ybar,\n"
        f"    bar width=0.18cm,\n"
        f"    width=14cm, height=8cm, scale only axis=true,\n"
        f"    % Keep enlarge x limits LOW so per-group width stays generous.\n"
        f"    enlarge x limits=0.08,\n"
        f"    ymin=0, ymax={y_max:.6g},\n"
        f"    xtick={{{xticks}}},\n"
        f"    xticklabels={{{xticklabels}}},\n"
        f"    xticklabel style={{rotate=20, anchor=north east,\n"
        f"                       font=\\scriptsize}},\n"
        f"    ylabel={{Mean stage time (ms)}},\n"
        f"    ylabel style={{font=\\small}},\n"
        f"    yticklabel style={{font=\\scriptsize}},\n"
        f"    title={{Per-stage mean GPU time}},\n"
        f"    grid=major, ymajorgrids=true,\n"
        f"    area legend,\n"
        f"    legend style={{at={{(0.5, -0.30)}}, anchor=north,\n"
        f"                   draw=none, fill=none, font=\\scriptsize,\n"
        f"                   /tikz/every even column/.append style="
                            f"{{column sep=0.4cm}}}},\n"
        f"    legend columns=3,\n"
        f"    legend cell align=left,\n"
        f"  ]\n"
        f"{plots_str}\n"
        f"  \\end{{axis}}\n"
        f"\\end{{tikzpicture}}\n"
    )


def plot_stage_grouped_bar_tex(series_list: list[Series], out_dir: Path, *,
                               fmt: str = "pdf+png") -> list[Path]:
    """pgfplots/tikz equivalent of `plot_stage_grouped_bar`.

    Emits into `out_dir/stage_grouped_bar/`:
      - `stage_grouped_bar.tex`            (\\input{}-able fragment)
      - `stage_grouped_bar.standalone.tex` (wrapper for pdflatex preview)
      - `stage_grouped_bar__<label>.dat`   (one per series: stage_idx mean_ms)
      - `stage_grouped_bar.pdf`            (compiled separately)

    Stages absent from EVERY series are dropped; stages absent from a single
    series get a 0-height bar so the visual grouping stays consistent."""
    del fmt
    if not series_list:
        return []
    active = [c for c in STAGE_COLS
              if any(s.frames[c].notna().any() for s in series_list)]
    if not active:
        return []

    chart_dir = out_dir / "stage_grouped_bar"
    chart_dir.mkdir(parents=True, exist_ok=True)
    base = "stage_grouped_bar"

    stage_labels = [c.replace("Ms", "") for c in active]
    styles = _series_style(series_list)
    written: list[Path] = []
    series_meta: list[tuple[str, str, str]] = []
    y_max_val = 0.0
    for i, s in enumerate(series_list):
        means = np.array([
            float(s.frames[c].dropna().mean()) if s.frames[c].notna().any()
            else 0.0
            for c in active
        ], dtype=float)
        rows = np.column_stack([
            np.arange(1, len(active) + 1, dtype=float),
            means,
        ])
        dat_name = _line_series_dat_name(base, s.label)
        _write_dat(chart_dir / dat_name,
                   ["stage_idx", "mean_ms"], rows)
        written.append(chart_dir / dat_name)

        finite = means[np.isfinite(means)]
        if finite.size:
            y_max_val = max(y_max_val, float(finite.max()))

        color_hex, _ = styles[i]
        series_meta.append((s.label, color_hex, dat_name))

    # Modest headroom above the tallest bar so the bar tips don't kiss the
    # title baseline.
    y_max_val = (y_max_val if y_max_val > 0 else 1.0) * 1.15

    tex = _stage_grouped_bar_tex(
        base=base, stage_labels=stage_labels,
        series_meta=series_meta, y_max=y_max_val,
    )
    written.append(_write_tex(chart_dir, base, tex))
    written.append(_write_tex(chart_dir, f"{base}.standalone",
                              _make_standalone_wrapper(base)))
    return written


# ---------------------------------------------------------------------------
# pgfplots / TikZ variant of plot_stage_stacked_area. Per-series subfolders
# (one chart per series, like camera_path_3d). Each chart stacks the
# series's active stages over simulation time with a shared global y-limit.
# ---------------------------------------------------------------------------

def _stage_stacked_area_tex(*, base: str, label: str,
                            active_cols: list[str], y_max: float) -> str:
    """Return the `\\input{}`-able fragment for one series's stacked-area
    panel. `active_cols` is the ordered list of STAGE_COLS entries the
    series has any data for; layers stack bottom-up in that order."""
    # Inline color definitions only for the active stages so the fragment
    # stays minimal and the legend's swatch palette matches the layers.
    color_defs_lines: list[str] = []
    stage_names: list[str] = []  # cleaned (without "Ms" suffix)
    for col in active_cols:
        cleaned = col.replace("Ms", "")
        stage_names.append(cleaned)
        color_defs_lines.append(
            f"  \\definecolor{{stage_{cleaned}}}{{HTML}}"
            f"{{{STAGE_COLORS[col].lstrip('#').upper()}}}"
        )
    color_defs = "\n".join(color_defs_lines)

    # Stack plots: each gets `forget plot` so pgfplots doesn't pair them
    # with the legend entries (otherwise `area legend` paints the swatches
    # with a black stroke via \draw, ignoring our `draw=none` on the plot).
    # The legend entries below are manual `\addlegendimage` with an
    # explicit `\fill` (no stroke) so the swatches are guaranteed borderless.
    plot_lines: list[str] = []
    for cleaned in stage_names:
        plot_lines.append(
            f"    \\addplot[fill=stage_{cleaned}, draw=none, fill opacity=0.85,\n"
            f"            forget plot]\n"
            f"      table[x=time, y={cleaned}] {{{base}.dat}} \\closedcycle;"
        )
    legend_lines: list[str] = []
    for cleaned in stage_names:
        legend_lines.append(
            f"    \\addlegendimage{{\n"
            f"      legend image code/.code={{%\n"
            f"        \\fill[stage_{cleaned}, fill opacity=0.85]\n"
            f"          (0cm,-0.08cm) rectangle (0.55cm,0.17cm);\n"
            f"      }},\n"
            f"    }}\n"
            f"    \\addlegendentry{{{_tex_escape(cleaned)}}}"
        )
    plots_str = "\n".join(plot_lines + legend_lines)

    legend_columns = min(len(stage_names), 4)
    return (
        f"% Auto-generated by scripts/plots/per_frame.py - do not hand-edit.\n"
        f"% Insert with \\input{{{base}}} from a document whose preamble loads\n"
        f"% pgfplots (>= 1.16). Companion data file (same directory):\n"
        f"%   {base}.dat - columns: time " + " ".join(stage_names) + "\n"
        f"%   (one row per frame; NaN cells masked to 0 so the layer stays\n"
        f"%   visible at 0 height when the stage didn't run that frame).\n"
        f"\\begin{{tikzpicture}}\n"
        f"{color_defs}\n"
        f"  \\begin{{axis}}[\n"
        f"    stack plots=y,\n"
        f"    area style,\n"
        f"    width=14cm, height=8cm, scale only axis=true,\n"
        f"    xlabel={{Simulation time (s)}},\n"
        f"    ylabel={{Stage GPU time (ms)}},\n"
        f"    xlabel style={{font=\\small}},\n"
        f"    ylabel style={{font=\\small}},\n"
        f"    tick label style={{font=\\scriptsize}},\n"
        f"    title={{Per-stage GPU time --- {_tex_escape(label)}}},\n"
        f"    ymin=0, ymax={y_max:.6g},\n"
        f"    enlarge x limits=false,\n"
        f"    grid=major,\n"
        f"    area legend,\n"
        f"    legend style={{at={{(0.5, -0.18)}}, anchor=north,\n"
        f"                   draw=none, fill=none, font=\\scriptsize,\n"
        f"                   /tikz/every even column/.append style="
                            f"{{column sep=0.4cm}}}},\n"
        f"    legend columns={legend_columns},\n"
        f"    legend cell align=left,\n"
        f"  ]\n"
        f"{plots_str}\n"
        f"  \\end{{axis}}\n"
        f"\\end{{tikzpicture}}\n"
    )


def plot_stage_stacked_area_tex(series_list: list[Series], out_dir: Path, *,
                                fmt: str = "pdf+png") -> list[Path]:
    """pgfplots/tikz equivalent of `plot_stage_stacked_area`.

    Emits per-series subfolders into `out_dir/stage_stacked_area/<label>/`:
      - `stage_stacked_area.tex`            (\\input{}-able fragment)
      - `stage_stacked_area.standalone.tex` (wrapper for pdflatex preview)
      - `stage_stacked_area.dat`            (one .dat per series; columns
                                             time + per-stage ms)
      - `stage_stacked_area.pdf`            (compiled separately)

    Shared y-axis upper bound across all series (99th-percentile of the per-
    frame stack total) matches the matplotlib version, so a faster pipeline
    is visibly a shorter stack in the same coordinate space."""
    del fmt
    if not series_list:
        return []

    # First pass: shared y_top from the 99th percentile of each series's
    # frame-by-frame stack height (i.e. sum across active stages per frame).
    global_max = 0.0
    for s in series_list:
        df = s.frames
        active = [c for c in STAGE_COLS if df[c].notna().any()]
        if not active:
            continue
        totals = np.sum([df[c].fillna(0).to_numpy() for c in active], axis=0)
        if totals.size:
            global_max = max(global_max, float(np.percentile(totals, 99)))
    y_top = (global_max if global_max > 0 else 1.0) * 1.08

    written: list[Path] = []
    base = "stage_stacked_area"
    parent = out_dir / base

    # Subsample to stay inside pgfplots' TeX main-memory limit. `stack
    # plots=y` + `\closedcycle` blows up memory at >~600 points per layer
    # (`! TeX capacity exceeded` on the 1801-frame raw data); a stride of 3-4
    # is visually identical at the 14cm chart width and trivially fits.
    target_points = 600

    for s in series_list:
        df = s.frames
        active = [c for c in STAGE_COLS if df[c].notna().any()]
        if not active:
            continue
        series_dir = parent / _safe_filename(s.label)
        series_dir.mkdir(parents=True, exist_ok=True)

        time_full = df["simTimeMs"].to_numpy(dtype=float) / 1000.0
        stride = max(1, len(time_full) // target_points)
        time = time_full[::stride]
        stage_arrays = [df[c].fillna(0).to_numpy(dtype=float)[::stride]
                        for c in active]
        rows = np.column_stack([time] + stage_arrays)
        header = ["time"] + [c.replace("Ms", "") for c in active]
        _write_dat(series_dir / f"{base}.dat", header, rows)
        written.append(series_dir / f"{base}.dat")

        tex = _stage_stacked_area_tex(
            base=base, label=s.label, active_cols=active, y_max=y_top,
        )
        written.append(_write_tex(series_dir, base, tex))
        written.append(_write_tex(series_dir, f"{base}.standalone",
                                  _make_standalone_wrapper(base)))
    return written


# ---------------------------------------------------------------------------
# pgfplots / TikZ variant of plot_sun_elevation. Single chart with one line
# per series. Sun phase is reset on `IsFirstRecordingFrame` and advanced by
# a fixed `simulationDtMs`, so every pipeline traces the same curve -- this
# plot is mostly a visual sanity-check that recording is deterministic.
# ---------------------------------------------------------------------------

def _split_sun_moon(t: np.ndarray, elev: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                               np.ndarray, np.ndarray]:
    """Split an elevation series into 4 styled segments.

    Returns (time, sun_solid, sun_dashed, moon_solid, moon_dashed) where:
      - `_solid`  columns hold above-horizon (elev > 0) samples in that
                  phase (the source IS casting shadows)
      - `_dashed` columns hold below-horizon (elev <= 0) samples in that
                  phase (no shadow -- the source is underground)
    Each sample contributes to exactly ONE of the four columns; the
    other three are NaN. At every elevation-sign change, an extra
    linear-interpolated zero row is inserted so consecutive solid and
    dashed segments meet exactly at y=0.

    PHASE detection: the C++ sun-cycle (Scene/SunSky.hpp) walks two
    symmetric phases (day = sun, night = moon). Each phase traces ONE
    full bump of the elevation curve. Within a phase there is exactly
    ONE positive-going zero crossing (the source rising). Each
    subsequent positive-going crossing in the data starts a NEW phase,
    alternating sun/moon. Counting positive-going crossings 1, 2, 3...:
       1st = sunrise within phase 1 (still sun -- predawn was sun too)
       2nd = moonrise (sun -> moon)   [phase boundary]
       3rd = next sunrise (moon -> sun) [phase boundary]
       4th = next moonrise (sun -> moon) [phase boundary]
       ...

    Phase 1 is labelled "sun" by convention -- the user's rule "first
    triangle of the wave is sunlight". So we start in sun phase and
    toggle on every positive-going crossing AFTER the first.

    Note: the model's actual phase teleport sits inside the dashed
    stretch between sunset and the next moonrise (and vice versa), but
    that teleport isn't visible in the elevation signal, so the dashed
    portion immediately before each subsequent rising is coloured with
    the OUTGOING phase. Visually the line stays one colour through the
    whole below-horizon span until the source on the OTHER side rises.

    Returns 5 arrays. Phase boundaries (every positive crossing AFTER
    the first) get an interpolated zero row written into the OUTGOING
    phase's dashed column and the INCOMING phase's solid column. Within
    a phase, a solid<->dashed transition at any zero crossing inserts
    one zero row written into both the solid and dashed columns of that
    phase so the segments meet."""
    nan = float("nan")
    if t.size == 0:
        return (t, np.array([], float), np.array([], float),
                np.array([], float), np.array([], float))
    out_t: list[float] = []
    out_ss: list[float] = []   # sun_solid
    out_sd: list[float] = []   # sun_dashed
    out_ms: list[float] = []   # moon_solid
    out_md: list[float] = []   # moon_dashed

    is_sun = True
    pos_crossings = 0

    for i in range(len(elev)):
        e = float(elev[i])
        ti = float(t[i])
        above = e > 0.0
        out_t.append(ti)
        out_ss.append(e if (is_sun and above) else nan)
        out_sd.append(e if (is_sun and not above) else nan)
        out_ms.append(e if ((not is_sun) and above) else nan)
        out_md.append(e if ((not is_sun) and not above) else nan)

        if i + 1 >= len(elev):
            continue
        e_next = float(elev[i + 1])
        ti_next = float(t[i + 1])

        pos_cross = (e <= 0.0 and e_next > 0.0)
        neg_cross = (e >= 0.0 and e_next < 0.0)
        if not (pos_cross or neg_cross):
            continue

        ratio = -e / (e_next - e)
        t_zero = ti + ratio * (ti_next - ti)

        if pos_cross:
            pos_crossings += 1
            if pos_crossings >= 2:
                # Phase boundary: outgoing-phase DASHED segment ends at
                # y=0, incoming-phase SOLID segment starts at y=0.
                out_t.append(t_zero)
                if is_sun:
                    out_sd.append(0.0); out_ss.append(nan)
                    out_md.append(nan); out_ms.append(0.0)
                else:
                    out_sd.append(nan); out_ss.append(0.0)
                    out_md.append(0.0); out_ms.append(nan)
                is_sun = not is_sun
            else:
                # Within-phase rising edge: this phase's dashed segment
                # ends at 0, its solid segment starts at 0.
                out_t.append(t_zero)
                if is_sun:
                    out_sd.append(0.0); out_ss.append(0.0)
                    out_md.append(nan); out_ms.append(nan)
                else:
                    out_sd.append(nan); out_ss.append(nan)
                    out_md.append(0.0); out_ms.append(0.0)
        else:
            # Negative-going crossing = source setting within current
            # phase: solid ends, dashed begins, both reach y=0.
            out_t.append(t_zero)
            if is_sun:
                out_ss.append(0.0); out_sd.append(0.0)
                out_ms.append(nan); out_md.append(nan)
            else:
                out_ss.append(nan); out_sd.append(nan)
                out_ms.append(0.0); out_md.append(0.0)

    return (np.array(out_t,  dtype=float),
            np.array(out_ss, dtype=float),
            np.array(out_sd, dtype=float),
            np.array(out_ms, dtype=float),
            np.array(out_md, dtype=float))


def _sun_elevation_tex(*, base: str,
                       sun_color: str, moon_color: str,
                       x_min: float, x_max: float) -> str:
    """Return the `\\input{}`-able fragment for the sun/moon elevation chart.

    The companion .dat carries 5 columns -- time, sun_solid, sun_dashed,
    moon_solid, moon_dashed -- so each sample contributes to exactly ONE
    line and the other three render NaN (skipped via `unbounded
    coords=discard`). Dashed segments mark below-horizon ("no shadow")
    stretches in either phase."""
    return (
        f"% Auto-generated by scripts/plots/per_frame.py - do not hand-edit.\n"
        f"% Insert with \\input{{{base}}} from a document whose preamble loads\n"
        f"% pgfplots (>= 1.16). Companion data file (same directory):\n"
        f"%   {base}.dat - columns: time sun_solid sun_dashed moon_solid\n"
        f"%                          moon_dashed. Solid columns hold above-\n"
        f"%                          horizon samples for the named phase;\n"
        f"%                          dashed columns hold below-horizon samples\n"
        f"%                          ('no shadow' since the source is\n"
        f"%                          underground). Phase = sun or moon is\n"
        f"%                          determined by positive-going zero\n"
        f"%                          crossings (see _split_sun_moon).\n"
        f"\\begin{{tikzpicture}}\n"
        f"  \\definecolor{{suncol}}{{HTML}}{{{sun_color.lstrip('#').upper()}}}\n"
        f"  \\definecolor{{mooncol}}{{HTML}}{{{moon_color.lstrip('#').upper()}}}\n"
        f"  \\begin{{axis}}[\n"
        f"    width=14cm, height=8cm, scale only axis=true,\n"
        f"    xlabel={{Simulation time (s)}},\n"
        f"    ylabel={{Sun elevation (deg)}},\n"
        f"    xlabel style={{font=\\small}},\n"
        f"    ylabel style={{font=\\small}},\n"
        f"    tick label style={{font=\\scriptsize}},\n"
        f"    title={{Sun elevation over time}},\n"
        f"    grid=major,\n"
        f"    enlarge x limits=false,\n"
        f"    xmin={x_min:.6g}, xmax={x_max:.6g},\n"
        f"    legend style={{at={{(0.5, -0.18)}}, anchor=north,\n"
        f"                   draw=none, fill=none, font=\\scriptsize,\n"
        f"                   /tikz/every even column/.append style="
                            f"{{column sep=0.5cm}}}},\n"
        f"    legend columns=3,\n"
        f"    legend cell align=left,\n"
        f"  ]\n"
        f"    % Horizon reference (sun elevation = 0 deg).\n"
        f"    \\addplot[mark=none, gray, densely dotted, line width=0.6pt,\n"
        f"            forget plot]\n"
        f"      coordinates {{({x_min:.6g},0) ({x_max:.6g},0)}};\n"
        f"\n"
        f"    % Above-horizon segments: solid lines in the phase colour.\n"
        f"    % `unbounded coords=jump` BREAKS the line at NaN rows (the\n"
        f"    % alternative `discard` drops the row entirely and reconnects\n"
        f"    % the surrounding valid points, which would draw a flat line at\n"
        f"    % y=0 between every sunrise and sunset zero-crossing).\n"
        f"    \\addplot[draw=suncol, line width=1.4pt,\n"
        f"            unbounded coords=jump]\n"
        f"      table[x=time, y=sun_solid] {{{base}.dat}};\n"
        f"    \\addlegendentry{{Sunlight}}\n"
        f"\n"
        f"    \\addplot[draw=mooncol, line width=1.4pt,\n"
        f"            unbounded coords=jump]\n"
        f"      table[x=time, y=moon_solid] {{{base}.dat}};\n"
        f"    \\addlegendentry{{Moonlight}}\n"
        f"\n"
        f"    % Below-horizon segments: gray dashed (matches the 'No shadow'\n"
        f"    % legend swatch). One shared entry covers both phases.\n"
        f"    \\addplot[draw=gray, line width=1.4pt, dashed,\n"
        f"            unbounded coords=jump, forget plot]\n"
        f"      table[x=time, y=sun_dashed] {{{base}.dat}};\n"
        f"    \\addplot[draw=gray, line width=1.4pt, dashed,\n"
        f"            unbounded coords=jump, forget plot]\n"
        f"      table[x=time, y=moon_dashed] {{{base}.dat}};\n"
        f"\n"
        f"    % Manual legend entry: 'No shadow' dashed gray indicator.\n"
        f"    \\addlegendimage{{\n"
        f"      legend image code/.code={{%\n"
        f"        \\draw[gray, dashed, line width=1.4pt]\n"
        f"          (0cm,0cm) -- (0.6cm,0cm);\n"
        f"      }},\n"
        f"    }}\n"
        f"    \\addlegendentry{{No shadow}}\n"
        f"  \\end{{axis}}\n"
        f"\\end{{tikzpicture}}\n"
    )


def plot_sun_elevation_tex(series_list: list[Series], out_dir: Path, *,
                           fmt: str = "pdf+png") -> list[Path]:
    """pgfplots/tikz equivalent of `plot_sun_elevation`.

    Sun phase is deterministic across pipelines (advanced by a fixed
    `simulationDtMs`), so we plot a SINGLE elevation curve drawn from the
    first available series. The line is split at horizon crossings into a
    sunlight segment (warm) and a moonlight segment (cool) -- legend has
    just those two entries.

    Emits into `out_dir/sun_elevation/`:
      - `sun_elevation.tex`            (\\input{}-able fragment)
      - `sun_elevation.standalone.tex` (wrapper for pdflatex preview)
      - `sun_elevation.dat`            (columns: time sun moon, with NaN in
                                        the segment where the sample doesn't
                                        belong)
      - `sun_elevation.pdf`            (compiled separately)"""
    del fmt
    series_list = [s for s in series_list
                   if "sunElevationDeg" in s.frames.columns]
    if not series_list:
        return []

    chart_dir = out_dir / "sun_elevation"
    chart_dir.mkdir(parents=True, exist_ok=True)
    base = "sun_elevation"

    # Use the first series as the canonical curve. All pipelines produce an
    # identical curve by construction; if that ever stops being true,
    # `plot_sun_elevation` (the matplotlib variant) overlays all series and
    # makes the divergence visible.
    df = series_list[0].frames
    t = df["simTimeMs"].to_numpy(dtype=float) / 1000.0
    elev = df["sunElevationDeg"].to_numpy(dtype=float)
    if t.size == 0:
        return []
    t_split, ss, sd, ms, md = _split_sun_moon(t, elev)
    rows = np.column_stack([t_split, ss, sd, ms, md])

    written: list[Path] = []
    _write_dat(chart_dir / f"{base}.dat",
               ["time", "sun_solid", "sun_dashed",
                "moon_solid", "moon_dashed"], rows)
    written.append(chart_dir / f"{base}.dat")

    # Warm/cool palette: golden orange for the above-horizon segment, dusty
    # blue for the below-horizon segment.
    sun_color = "#E4A93C"
    moon_color = "#3D5A80"
    tex = _sun_elevation_tex(
        base=base, sun_color=sun_color, moon_color=moon_color,
        x_min=float(t.min()), x_max=float(t.max()),
    )
    written.append(_write_tex(chart_dir, base, tex))
    written.append(_write_tex(chart_dir, f"{base}.standalone",
                              _make_standalone_wrapper(base)))
    return written


# ---------------------------------------------------------------------------
# pgfplots / TikZ variant of the "instance visibility" comparison plot.
# 2x2 groupplot per (P0 vs P1/P2-with-Hi-Z) pair: top row = main pass with
# per-LOD trunk stack, bottom row = shadow pass (single-LOD trunk + impostor
# stack). Right-column panels overlay P0's stack-top as a dashed reference
# line so the gap visualises how much Hi-Z culled.
# ---------------------------------------------------------------------------

# Visibility colour palette referenced by HEX so the tex fragment doesn't
# depend on import order of the matplotlib palette dicts.
_VIS_TRUNK_HEX     = "55A467"
_VIS_LEAVES_HEX    = "9CCAA0"
_VIS_IMPOSTORS_HEX = "DD8452"

# ColorBrewer "Greens" ramp shifted toward the lighter end so even LOD 0
# reads as a clear medium green rather than near-black. LOD 0 (closest,
# highest detail) is the darkest in the ramp; LOD 7 (farthest, lowest
# detail) is the palest. Last two stops drift toward yellow-green so the
# top of an 8-LOD stack stays distinguishable from white background.
_LOD_COLORS_HEX: tuple[str, ...] = (
    "238B45", "41AB5D", "74C476", "A1D99B",
    "C7E9C0", "E5F5E0", "F7FCB9", "FFEDA0",
)


def _instance_visibility_tex(*, base: str,
                             pipeline_label: str,
                             primary_suffix: str,
                             reference_suffix: str,
                             reference_label: str,
                             show_reference: bool,
                             x_min: float, x_max: float,
                             main_y_max: float, shadow_y_max: float,
                             active_lod_indices: list[int]) -> str:
    """Return the `\\input{}`-able fragment for one pipeline's instance-
    visibility panel pair (main on top, shadow on bottom). When
    `show_reference` is True, the reference pipeline's stack-top is
    overlaid as a dashed black line on both panels and added to the
    legend (labelled `reference_label`).

    `active_lod_indices` lists the LODs to actually plot + show in the
    legend (any LOD that's all-zero across both series is dropped so the
    legend doesn't carry phantom entries)."""
    color_defs_lines = [
        f"  \\definecolor{{lod{i}}}{{HTML}}{{{_LOD_COLORS_HEX[i]}}}"
        for i in active_lod_indices
    ]
    color_defs_lines.append(
        f"  \\definecolor{{visimpostors}}{{HTML}}{{{_VIS_IMPOSTORS_HEX}}}")
    color_defs = "\n".join(color_defs_lines)

    # CRITICAL: every actual `\addplot` carries `forget plot`. Without it,
    # pgfplots allocates a legend slot per addplot in source order and the
    # later `\addlegendentry` calls map to THOSE slots (with the addplot's
    # fill colour), not to the `\addlegendimage` entries we want. With
    # `forget plot`, only the explicit \addlegendimage entries below
    # participate in the legend.
    def main_stack() -> str:
        out: list[str] = []
        for i in active_lod_indices:
            out.append(
                f"      \\addplot[fill=lod{i}, draw=none, fill opacity=0.95,\n"
                f"              forget plot]\n"
                f"        table[x=time, y=lod{i}]\n"
                f"          {{{base}__{primary_suffix}.dat}} \\closedcycle;\n"
            )
        out.append(
            f"      \\addplot[fill=visimpostors, draw=none,\n"
            f"              fill opacity=0.95, forget plot]\n"
            f"        table[x=time, y=impostors]\n"
            f"          {{{base}__{primary_suffix}.dat}} \\closedcycle;\n"
        )
        return "".join(out)

    # Shadow trunk renders at the LOWEST level of detail (= highest active
    # LOD index, the simplest geometry). In this scene that's LOD 2.
    shadow_trunk_lod = (active_lod_indices[-1]
                        if active_lod_indices else 0)
    shadow_trunk_color = f"lod{shadow_trunk_lod}"

    def shadow_stack() -> str:
        return (
            f"      \\addplot[fill={shadow_trunk_color}, draw=none,\n"
            f"              fill opacity=0.95, forget plot]\n"
            f"        table[x=time, y=shadow_trunk]\n"
            f"          {{{base}__{primary_suffix}.dat}} \\closedcycle;\n"
            f"      \\addplot[fill=visimpostors, draw=none,\n"
            f"              fill opacity=0.95, forget plot]\n"
            f"        table[x=time, y=shadow_imp]\n"
            f"          {{{base}__{primary_suffix}.dat}} \\closedcycle;\n"
        )

    def ref_line(metric: str) -> str:
        """Dashed reference line. Placed on the `axis foreground` layer so
        it paints AFTER every stack fill (pgfplots batches stack-plot
        rendering and source-order alone isn't enough; the layer is). The
        axis must have `set layers` for this to take effect."""
        if not show_reference:
            return ""
        return (
            f"      % `on layer=axis foreground` puts the line on a layer\n"
            f"      % that is rendered AFTER the main layer (which is where\n"
            f"      % the stack fills live), so the dashed stroke always\n"
            f"      % paints on top of the orange fill -- not under it. The\n"
            f"      % axis enables this via `set layers` above.\n"
            f"      \\addplot[mark=none, draw=black, dashed, line width=0.9pt,\n"
            f"              stack plots=false, forget plot,\n"
            f"              on layer=axis foreground]\n"
            f"        table[x=time, y={metric}]\n"
            f"          {{{base}__{reference_suffix}.dat}};\n"
        )

    # Bottom-panel legend block. Each entry uses `legend image code` with an
    # explicit tikz `\fill` (for the area swatches) or `\draw` (for the
    # dashed reference). This bypasses pgfplots' cycle list entirely so the
    # swatch colour is guaranteed to match what's on the plot.
    def area_swatch(color_name: str) -> str:
        return (
            f"      \\addlegendimage{{\n"
            f"        legend image code/.code={{%\n"
            f"          \\fill[{color_name}, fill opacity=0.95]\n"
            f"            (0cm,-0.08cm) rectangle (0.55cm,0.17cm);\n"
            f"        }},\n"
            f"      }}\n"
        )

    legend_entries: list[str] = []
    for i in active_lod_indices:
        legend_entries.append(
            area_swatch(f"lod{i}")
            + f"      \\addlegendentry{{LOD {i}}}"
        )
    legend_entries.append(
        area_swatch("visimpostors")
        + f"      \\addlegendentry{{Impostor}}"
    )
    if show_reference:
        legend_entries.append(
            f"      \\addlegendimage{{\n"
            f"        legend image code/.code={{%\n"
            f"          \\draw[black, dashed, line width=0.9pt]\n"
            f"            (0cm,0cm) -- (0.55cm,0cm);\n"
            f"        }},\n"
            f"      }}\n"
            f"      \\addlegendentry{{{_tex_escape(reference_label)}}}"
        )
    legend_block = "\n".join(legend_entries) + "\n"

    ref_files_note = ""
    if show_reference:
        ref_files_note = (
            f"%   {base}__{reference_suffix}.dat - reference data (Traditional\n"
            f"%                                    pipeline) for the dashed\n"
            f"%                                    overlay line.\n"
        )

    n_legend = len(active_lod_indices) + 1 + (1 if show_reference else 0)
    legend_columns = min(n_legend, 5)
    active_lods_note = (
        f"% Active LODs in this run: {', '.join(str(i) for i in active_lod_indices)}\n"
        f"% (LODs with all-zero data across both series are dropped from\n"
        f"% both the stack and the legend so the legend matches what's\n"
        f"% actually visible.)\n"
    )

    return (
        f"% Auto-generated by scripts/plots/per_frame.py - do not hand-edit.\n"
        f"% Insert with \\input{{{base}}} from a document whose preamble loads\n"
        f"% pgfplots (>= 1.16) AND `\\usepgfplotslibrary{{groupplots}}`.\n"
        f"% Companion data files (same directory):\n"
        f"%   {base}__{primary_suffix}.dat - primary data (this pipeline)\n"
        f"%     columns: time lod0..lodN impostors main_total shadow_trunk\n"
        f"%              shadow_imp shadow_total (all 8 lodN columns are\n"
        f"%              written for transparency, but only active LODs\n"
        f"%              show up in the chart and legend).\n"
        f"{ref_files_note}"
        f"{active_lods_note}"
        f"\\begin{{tikzpicture}}\n"
        f"{color_defs}\n"
        f"  \\begin{{groupplot}}[\n"
        f"    group style={{\n"
        f"      group size=1 by 2,\n"
        f"      vertical sep=1.0cm,\n"
        f"      x descriptions at=edge bottom,\n"
        f"    }},\n"
        f"    width=13cm, height=4.6cm, scale only axis=true,\n"
        f"    xmin={x_min:.6g}, xmax={x_max:.6g},\n"
        f"    enlarge x limits=false,\n"
        f"    grid=major, ymajorgrids=true,\n"
        f"    stack plots=y, area style,\n"
        f"    area legend,\n"
        f"    % Enable pgfplots' layered rendering so `on layer=axis\n"
        f"    % foreground` on the dashed reference line works (without\n"
        f"    % `set layers`, every plot lands on the same anonymous layer\n"
        f"    % and the stack fills can paint over the dashed line).\n"
        f"    set layers,\n"
        f"    % Scientific notation on Y so each tick reads as e.g. `5.0e+4`\n"
        f"    % directly (matching the terrain-shadow scatter); `scaled y\n"
        f"    % ticks=false` disables the floating `x10^N` multiplier annotation\n"
        f"    % pgfplots otherwise places at the top of the axis.\n"
        f"    scaled y ticks=false,\n"
        f"    yticklabel style={{\n"
        f"      /pgf/number format/sci,\n"
        f"      /pgf/number format/sci e,\n"
        f"      /pgf/number format/sci zerofill,\n"
        f"      /pgf/number format/precision=1,\n"
        f"    }},\n"
        f"    tick label style={{font=\\scriptsize}},\n"
        f"    xlabel={{Simulation time (s)}},\n"
        f"    xlabel style={{font=\\small}},\n"
        f"    ylabel={{Visible instances}},\n"
        f"    ylabel style={{font=\\small}},\n"
        f"    title style={{font=\\small}},\n"
        f"  ]\n"
        f"\n"
        f"    % Top: main pass. No legend entries here -- the bottom panel\n"
        f"    % owns the unified legend so all colours appear in one block.\n"
        f"    \\nextgroupplot[\n"
        f"      title={{Main pass --- {_tex_escape(pipeline_label)}}},\n"
        f"      ymin=0, ymax={main_y_max:.6g},\n"
        f"    ]\n"
        f"{main_stack()}"
        f"{ref_line('main_total')}"
        f"\n"
        f"    % Bottom: shadow pass. The legend lives on this panel,\n"
        f"    % positioned below it, so it ends up beneath the whole\n"
        f"    % groupplot -- matching the stage_stacked_area legend style.\n"
        f"    \\nextgroupplot[\n"
        f"      title={{Shadow pass --- {_tex_escape(pipeline_label)}}},\n"
        f"      ymin=0, ymax={shadow_y_max:.6g},\n"
        f"      legend style={{at={{(0.5, -0.30)}}, anchor=north,\n"
        f"                     draw=none, fill=none, font=\\scriptsize,\n"
        f"                     /tikz/every even column/.append style="
                              f"{{column sep=0.4cm}}}},\n"
        f"      legend columns={legend_columns},\n"
        f"      legend cell align=left,\n"
        f"    ]\n"
        f"{shadow_stack()}"
        f"{ref_line('shadow_total')}"
        f"\n"
        f"      % Legend swatches (explicit \\addlegendimage entries so\n"
        f"      % every colour in the plot shows up regardless of which\n"
        f"      % panel actually drew it).\n"
        f"{legend_block}"
        f"\n"
        f"  \\end{{groupplot}}\n"
        f"\\end{{tikzpicture}}\n"
    )


def plot_instance_visibility_tex(series_list: list[Series], out_dir: Path, *,
                                 fmt: str = "pdf+png") -> list[Path]:
    """Per-pipeline instance-visibility plot (main + shadow panels).

    Emits two subfolders into `out_dir/instance_visibility/`:
      - `Traditional/` -- Traditional pipeline panels (no reference overlay)
      - `Compute/`     -- Compute pipeline panels with the Traditional
                          stack-top dashed over the top so the gap reads as
                          'this is how much Hi-Z removed'

    Each folder contains:
      - `instance_visibility.tex`              (\\input{}-able fragment)
      - `instance_visibility.standalone.tex`   (pdflatex preview wrapper)
      - `instance_visibility__traditional.dat` (Traditional per-frame data;
                                               always present -- primary in
                                               the Traditional folder,
                                               reference overlay in Compute)
      - `instance_visibility__compute.dat`     (Compute folder only)
      - `instance_visibility.pdf`              (compiled separately)

    MeshShader is intentionally skipped (per user spec); add it back by
    appending its label to `pipeline_specs` below."""
    del fmt

    traditional = next((s for s in series_list
                        if s.pipeline == "Traditional"), None)
    compute = next((s for s in series_list
                    if s.pipeline == "Compute"
                    and "No Hi-Z" not in s.label), None)
    if traditional is None or compute is None:
        return []

    # Per-pipeline spec: (label, suffix, primary, show_ref, ref_series,
    #                    ref_suffix, ref_label)
    # The Compute folder's dashed reference is the Traditional stack-top,
    # so the gap reads as "this is what GPU culling removed vs. CPU
    # frustum-only".
    pipeline_specs: list[tuple[str, str, "Series", bool,
                               "Series | None", str, str]] = [
        ("Traditional", "traditional", traditional,
         False, None,          "",            ""),
        ("Compute",     "compute",     compute,
         True,          traditional,  "traditional", "Traditional"),
    ]

    # How many LOD columns the CSV carries. Discovered dynamically so future
    # builds with more LODs Just Work.
    lod_cols = sorted(
        [c for c in traditional.frames.columns
         if c.startswith("mainVisibleLod")],
        key=lambda c: int(c.replace("mainVisibleLod", "")),
    )
    num_lods = min(len(lod_cols), len(_LOD_COLORS_HEX))
    lod_cols = lod_cols[:num_lods]
    if num_lods == 0:
        return []

    # An LOD is "active" if ANY series has non-zero samples in it. LODs that
    # are all-zero everywhere shouldn't appear in the legend or contribute
    # an addplot to the stack (otherwise the legend lies about what's drawn).
    active_lod_indices: list[int] = []
    for i, col in enumerate(lod_cols):
        if any(float(s.frames[col].fillna(0).sum()) > 0
               for s in [traditional, compute]):
            active_lod_indices.append(i)
    if not active_lod_indices:
        return []

    def percentile_top(df, cols: list[str]) -> float:
        present = [c for c in cols if c in df.columns]
        if not present:
            return 0.0
        total = sum(df[c].fillna(0).to_numpy() for c in present)
        if not isinstance(total, np.ndarray) or total.size == 0:
            return 0.0
        return float(np.percentile(total, 99))

    # Shared y-ranges across both folders so the Traditional and Compute
    # plots use the same coordinate space. Traditional doubles as the
    # dashed reference in the Compute folder, so its samples are already
    # in the pool and the dashed line is guaranteed inside ymax.
    main_top = 0.0
    shadow_top = 0.0
    series_for_y = [traditional, compute]
    for s in series_for_y:
        main_top = max(main_top, percentile_top(
            s.frames, lod_cols + ["mainVisibleImpostors"]))
        shadow_top = max(shadow_top, percentile_top(
            s.frames, ["shadowVisibleTrunk", "shadowVisibleImpostors"]))
    main_top = main_top * 1.08 if main_top > 0 else 1.0
    shadow_top = shadow_top * 1.08 if shadow_top > 0 else 1.0

    base = "instance_visibility"
    parent = out_dir / base
    target_points = 600
    written: list[Path] = []

    def write_series_dat(out_path: Path, df) -> None:
        time_full = df["simTimeMs"].to_numpy(dtype=float) / 1000.0
        if time_full.size == 0:
            return
        stride = max(1, len(time_full) // target_points)
        time = time_full[::stride]
        lod_arrs = [df[c].fillna(0).to_numpy(dtype=float)[::stride]
                    for c in lod_cols]
        impostors = (df["mainVisibleImpostors"].fillna(0)
                     .to_numpy(dtype=float)[::stride])
        main_total = sum(lod_arrs) + impostors
        shadow_trunk = (df.get("shadowVisibleTrunk",
                               pd.Series(np.zeros(len(time_full))))
                        .fillna(0).to_numpy(dtype=float)[::stride])
        shadow_imp = (df.get("shadowVisibleImpostors",
                             pd.Series(np.zeros(len(time_full))))
                      .fillna(0).to_numpy(dtype=float)[::stride])
        shadow_total = shadow_trunk + shadow_imp
        rows = np.column_stack(
            [time] + lod_arrs +
            [impostors, main_total, shadow_trunk, shadow_imp, shadow_total])
        header = (["time"] + [f"lod{i}" for i in range(num_lods)]
                  + ["impostors", "main_total",
                     "shadow_trunk", "shadow_imp", "shadow_total"])
        _write_dat(out_path, header, rows)

    # Canonical timeline (deterministic across pipelines).
    t = traditional.frames["simTimeMs"].to_numpy(dtype=float) / 1000.0
    x_min = float(t.min()) if t.size else 0.0
    x_max = float(t.max()) if t.size else 1.0

    for (label, suffix, series, show_ref,
         ref_series, ref_suffix, ref_label) in pipeline_specs:
        folder = parent / label
        folder.mkdir(parents=True, exist_ok=True)

        # Primary data for this folder's pipeline.
        write_series_dat(folder / f"{base}__{suffix}.dat", series.frames)
        written.append(folder / f"{base}__{suffix}.dat")
        # Reference data (per-spec; Compute uses Compute-No-Hi-Z so the
        # dashed line isolates Hi-Z's contribution rather than mixing in
        # the P0 cull-architecture difference).
        if show_ref and ref_series is not None:
            write_series_dat(folder / f"{base}__{ref_suffix}.dat",
                             ref_series.frames)
            written.append(folder / f"{base}__{ref_suffix}.dat")

        tex = _instance_visibility_tex(
            base=base, pipeline_label=label,
            primary_suffix=suffix, reference_suffix=ref_suffix,
            reference_label=ref_label,
            show_reference=show_ref,
            x_min=x_min, x_max=x_max,
            main_y_max=main_top, shadow_y_max=shadow_top,
            active_lod_indices=active_lod_indices,
        )
        written.append(_write_tex(folder, base, tex))
        written.append(_write_tex(folder, f"{base}.standalone",
                                  _make_standalone_wrapper(
                                      base,
                                      extra_preamble=
                                      r"\usepgfplotslibrary{groupplots}")))
    return written


# ---------------------------------------------------------------------------
# pgfplots / TikZ instance-LOD-fraction plot: top panel reuses the
# instance_visibility main-pass stacked-area (absolute counts of LOD0..N +
# impostors); bottom panel replaces the shadow stack with a normalised
# stack where each layer is lodN / sum(lod0..lodN) -- i.e. the fraction of
# all currently-visible geometric instances that fell into LOD N. Impostors
# are excluded from the denominator (the bottom panel is about how the
# geometric-tier population splits across LODs, not how much of the scene
# went to impostors). When no geometric LODs are visible in a frame, every
# fraction is zero (the column reads as a flat 0, not a NaN).
# ---------------------------------------------------------------------------


def _instance_lod_fraction_tex(*, base: str,
                               pipeline_label: str,
                               primary_suffix: str,
                               reference_suffix: str,
                               reference_label: str,
                               show_reference: bool,
                               x_min: float, x_max: float,
                               main_y_max: float,
                               active_lod_indices: list[int]) -> str:
    """Return the `\\input{}`-able fragment for one pipeline's LOD-fraction
    panel pair: the instance_visibility main-pass stack on top, and a
    normalised LOD-fraction stack (sum-to-1 per frame, 0 when no geometric
    LODs are visible) on the bottom.

    Bottom-panel y is fixed [0, 1.0]; ticks are emitted as fixed-point
    fractions rather than scientific so 0.25 / 0.50 / 0.75 / 1.00 read
    cleanly. `active_lod_indices` is shared with the top panel so the
    legend keys line up colour-for-colour."""
    color_defs_lines = [
        f"  \\definecolor{{lod{i}}}{{HTML}}{{{_LOD_COLORS_HEX[i]}}}"
        for i in active_lod_indices
    ]
    color_defs_lines.append(
        f"  \\definecolor{{visimpostors}}{{HTML}}{{{_VIS_IMPOSTORS_HEX}}}")
    color_defs = "\n".join(color_defs_lines)

    def main_stack() -> str:
        out: list[str] = []
        for i in active_lod_indices:
            out.append(
                f"      \\addplot[fill=lod{i}, draw=none, fill opacity=0.95,\n"
                f"              forget plot]\n"
                f"        table[x=time, y=lod{i}]\n"
                f"          {{{base}__{primary_suffix}.dat}} \\closedcycle;\n"
            )
        out.append(
            f"      \\addplot[fill=visimpostors, draw=none,\n"
            f"              fill opacity=0.95, forget plot]\n"
            f"        table[x=time, y=impostors]\n"
            f"          {{{base}__{primary_suffix}.dat}} \\closedcycle;\n"
        )
        return "".join(out)

    def fraction_stack() -> str:
        out: list[str] = []
        for i in active_lod_indices:
            out.append(
                f"      \\addplot[fill=lod{i}, draw=none, fill opacity=0.95,\n"
                f"              forget plot]\n"
                f"        table[x=time, y=lod{i}_frac]\n"
                f"          {{{base}__{primary_suffix}.dat}} \\closedcycle;\n"
            )
        return "".join(out)

    def ref_line(metric: str) -> str:
        if not show_reference:
            return ""
        return (
            f"      \\addplot[mark=none, draw=black, dashed, line width=0.9pt,\n"
            f"              stack plots=false, forget plot,\n"
            f"              on layer=axis foreground]\n"
            f"        table[x=time, y={metric}]\n"
            f"          {{{base}__{reference_suffix}.dat}};\n"
        )

    def area_swatch(color_name: str) -> str:
        return (
            f"      \\addlegendimage{{\n"
            f"        legend image code/.code={{%\n"
            f"          \\fill[{color_name}, fill opacity=0.95]\n"
            f"            (0cm,-0.08cm) rectangle (0.55cm,0.17cm);\n"
            f"        }},\n"
            f"      }}\n"
        )

    legend_entries: list[str] = []
    for i in active_lod_indices:
        legend_entries.append(
            area_swatch(f"lod{i}")
            + f"      \\addlegendentry{{LOD {i}}}"
        )
    legend_entries.append(
        area_swatch("visimpostors")
        + f"      \\addlegendentry{{Impostor}}"
    )
    if show_reference:
        legend_entries.append(
            f"      \\addlegendimage{{\n"
            f"        legend image code/.code={{%\n"
            f"          \\draw[black, dashed, line width=0.9pt]\n"
            f"            (0cm,0cm) -- (0.55cm,0cm);\n"
            f"        }},\n"
            f"      }}\n"
            f"      \\addlegendentry{{{_tex_escape(reference_label)}}}"
        )
    legend_block = "\n".join(legend_entries) + "\n"

    n_legend = len(active_lod_indices) + 1 + (1 if show_reference else 0)
    legend_columns = min(n_legend, 5)

    ref_files_note = ""
    if show_reference:
        ref_files_note = (
            f"%   {base}__{reference_suffix}.dat - reference data (Traditional\n"
            f"%                                    pipeline) for the dashed\n"
            f"%                                    overlay line on the top panel.\n"
        )
    active_lods_note = (
        f"% Active LODs in this run: {', '.join(str(i) for i in active_lod_indices)}\n"
    )

    return (
        f"% Auto-generated by scripts/plots/per_frame.py - do not hand-edit.\n"
        f"% Insert with \\input{{{base}}} from a document whose preamble loads\n"
        f"% pgfplots (>= 1.16) AND `\\usepgfplotslibrary{{groupplots}}`.\n"
        f"% Companion data files (same directory):\n"
        f"%   {base}__{primary_suffix}.dat - primary data (this pipeline)\n"
        f"%     columns: time lod0..lodN impostors main_total\n"
        f"%              lod0_frac..lodN_frac (geometric-LOD fractions; per row\n"
        f"%              they sum to 1 when any LOD is visible, 0 otherwise).\n"
        f"{ref_files_note}"
        f"{active_lods_note}"
        f"\\begin{{tikzpicture}}\n"
        f"{color_defs}\n"
        f"  \\begin{{groupplot}}[\n"
        f"    group style={{\n"
        f"      group size=1 by 2,\n"
        f"      vertical sep=1.0cm,\n"
        f"      x descriptions at=edge bottom,\n"
        f"    }},\n"
        f"    width=13cm, height=4.6cm, scale only axis=true,\n"
        f"    xmin={x_min:.6g}, xmax={x_max:.6g},\n"
        f"    enlarge x limits=false,\n"
        f"    grid=major, ymajorgrids=true,\n"
        f"    stack plots=y, area style,\n"
        f"    area legend,\n"
        f"    set layers,\n"
        f"    tick label style={{font=\\scriptsize}},\n"
        f"    xlabel={{Simulation time (s)}},\n"
        f"    xlabel style={{font=\\small}},\n"
        f"    title style={{font=\\small}},\n"
        f"  ]\n"
        f"\n"
        f"    % Top: main pass absolute counts. Mirrors instance_visibility.\n"
        f"    \\nextgroupplot[\n"
        f"      title={{Main pass --- {_tex_escape(pipeline_label)}}},\n"
        f"      ymin=0, ymax={main_y_max:.6g},\n"
        f"      ylabel={{Visible instances}},\n"
        f"      ylabel style={{font=\\small}},\n"
        f"      scaled y ticks=false,\n"
        f"      yticklabel style={{\n"
        f"        /pgf/number format/sci,\n"
        f"        /pgf/number format/sci e,\n"
        f"        /pgf/number format/sci zerofill,\n"
        f"        /pgf/number format/precision=1,\n"
        f"      }},\n"
        f"    ]\n"
        f"{main_stack()}"
        f"{ref_line('main_total')}"
        f"\n"
        f"    % Bottom: geometric-LOD fractions (impostors excluded from the\n"
        f"    % denominator). Stack sums to 1 on frames with any geometric\n"
        f"    % LOD visible, and to 0 otherwise (the precomputed *_frac\n"
        f"    % columns clamp the division-by-zero case to 0 per LOD).\n"
        f"    \\nextgroupplot[\n"
        f"      title={{Geometric LOD fraction --- {_tex_escape(pipeline_label)}}},\n"
        f"      ymin=0, ymax=1.0,\n"
        f"      ylabel={{Fraction of geom. LODs}},\n"
        f"      ylabel style={{font=\\small}},\n"
        f"      ytick={{0, 0.25, 0.5, 0.75, 1.0}},\n"
        f"      yticklabel style={{font=\\scriptsize,\n"
        f"        /pgf/number format/.cd, fixed, precision=2,\n"
        f"      }},\n"
        f"      legend style={{at={{(0.5, -0.30)}}, anchor=north,\n"
        f"                     draw=none, fill=none, font=\\scriptsize,\n"
        f"                     /tikz/every even column/.append style="
                              f"{{column sep=0.4cm}}}},\n"
        f"      legend columns={legend_columns},\n"
        f"      legend cell align=left,\n"
        f"    ]\n"
        f"{fraction_stack()}"
        f"\n"
        f"      % Legend swatches (same colours as the top panel; impostor\n"
        f"      % swatch is kept so the legend is a complete colour key for\n"
        f"      % the figure even though the bottom panel itself doesn't\n"
        f"      % draw impostors).\n"
        f"{legend_block}"
        f"\n"
        f"  \\end{{groupplot}}\n"
        f"\\end{{tikzpicture}}\n"
    )


def plot_instance_lod_fraction_tex(series_list: list[Series], out_dir: Path, *,
                                   fmt: str = "pdf+png") -> list[Path]:
    """Per-pipeline LOD-fraction plot (main absolute counts on top, geometric
    LOD fraction stack on the bottom).

    Emits two subfolders into `out_dir/instance_LOD_frac/`:
      - `Traditional/` -- Traditional pipeline panels (no reference overlay)
      - `Compute/`     -- Compute pipeline panels with the Traditional
                          stack-top dashed over the top so the gap reads as
                          'this is how much GPU culling removed'

    The bottom panel renders each geometric LOD's per-frame fraction of the
    total geometric (non-impostor) instance count, stacked. The per-LOD
    fractions are precomputed in Python and stored as `lodN_frac` columns
    in the .dat file -- the pgfplots side just stacks them as-is, so the
    division-by-zero clamp (zero geometric LODs -> all fractions = 0) lives
    in one place."""
    del fmt

    traditional = next((s for s in series_list
                        if s.pipeline == "Traditional"), None)
    compute = next((s for s in series_list
                    if s.pipeline == "Compute"
                    and "No Hi-Z" not in s.label), None)
    if traditional is None or compute is None:
        return []

    pipeline_specs: list[tuple[str, str, "Series", bool,
                               "Series | None", str, str]] = [
        ("Traditional", "traditional", traditional,
         False, None,          "",            ""),
        ("Compute",     "compute",     compute,
         True,          traditional,  "traditional", "Traditional"),
    ]

    lod_cols = sorted(
        [c for c in traditional.frames.columns
         if c.startswith("mainVisibleLod")],
        key=lambda c: int(c.replace("mainVisibleLod", "")),
    )
    num_lods = min(len(lod_cols), len(_LOD_COLORS_HEX))
    lod_cols = lod_cols[:num_lods]
    if num_lods == 0:
        return []

    active_lod_indices: list[int] = []
    for i, col in enumerate(lod_cols):
        if any(float(s.frames[col].fillna(0).sum()) > 0
               for s in [traditional, compute]):
            active_lod_indices.append(i)
    if not active_lod_indices:
        return []

    def percentile_top(df, cols: list[str]) -> float:
        present = [c for c in cols if c in df.columns]
        if not present:
            return 0.0
        total = sum(df[c].fillna(0).to_numpy() for c in present)
        if not isinstance(total, np.ndarray) or total.size == 0:
            return 0.0
        return float(np.percentile(total, 99))

    main_top = 0.0
    for s in [traditional, compute]:
        main_top = max(main_top, percentile_top(
            s.frames, lod_cols + ["mainVisibleImpostors"]))
    main_top = main_top * 1.08 if main_top > 0 else 1.0

    base = "instance_lod_frac"
    parent = out_dir / "instance_LOD_frac"
    target_points = 600
    written: list[Path] = []

    def write_series_dat(out_path: Path, df) -> None:
        time_full = df["simTimeMs"].to_numpy(dtype=float) / 1000.0
        if time_full.size == 0:
            return
        stride = max(1, len(time_full) // target_points)
        time = time_full[::stride]
        lod_arrs = [df[c].fillna(0).to_numpy(dtype=float)[::stride]
                    for c in lod_cols]
        impostors = (df["mainVisibleImpostors"].fillna(0)
                     .to_numpy(dtype=float)[::stride])
        main_total = sum(lod_arrs) + impostors
        # Per-frame geometric-LOD denominator (impostors excluded). Clamp
        # division-by-zero to a zero fraction per LOD so empty frames read
        # as a flat 0 stack rather than NaN.
        geom_sum = sum(lod_arrs)
        lod_fracs = [np.divide(lod, geom_sum,
                               out=np.zeros_like(lod),
                               where=geom_sum > 0)
                     for lod in lod_arrs]
        rows = np.column_stack(
            [time] + lod_arrs + [impostors, main_total] + lod_fracs)
        header = (["time"] + [f"lod{i}" for i in range(num_lods)]
                  + ["impostors", "main_total"]
                  + [f"lod{i}_frac" for i in range(num_lods)])
        _write_dat(out_path, header, rows)

    t = traditional.frames["simTimeMs"].to_numpy(dtype=float) / 1000.0
    x_min = float(t.min()) if t.size else 0.0
    x_max = float(t.max()) if t.size else 1.0

    for (label, suffix, series, show_ref,
         ref_series, ref_suffix, ref_label) in pipeline_specs:
        folder = parent / label
        folder.mkdir(parents=True, exist_ok=True)

        write_series_dat(folder / f"{base}__{suffix}.dat", series.frames)
        written.append(folder / f"{base}__{suffix}.dat")
        if show_ref and ref_series is not None:
            write_series_dat(folder / f"{base}__{ref_suffix}.dat",
                             ref_series.frames)
            written.append(folder / f"{base}__{ref_suffix}.dat")

        tex = _instance_lod_fraction_tex(
            base=base, pipeline_label=label,
            primary_suffix=suffix, reference_suffix=ref_suffix,
            reference_label=ref_label,
            show_reference=show_ref,
            x_min=x_min, x_max=x_max,
            main_y_max=main_top,
            active_lod_indices=active_lod_indices,
        )
        written.append(_write_tex(folder, base, tex))
        written.append(_write_tex(folder, f"{base}.standalone",
                                  _make_standalone_wrapper(
                                      base,
                                      extra_preamble=
                                      r"\usepgfplotslibrary{groupplots}")))
    return written


# ---------------------------------------------------------------------------
# pgfplots / TikZ meshlet-visibility plot: 3x2 groupplot showing the
# MeshShader (P2) per-frame meshlet workload as a 100%-stacked area. Each
# panel sums to y=1; bottom layer = rendered fraction (of dispatched), top
# layer = culled fraction. Rows = Main / Shadow pass, columns = Trunk /
# Leaves / Terrain. For terrain the per-frame "dispatched" isn't logged --
# we use the scene-aggregate constant (sceneStats.terrainMeshlets for main,
# sceneStats.terrainShadowMeshletsDispatched for shadow) as the denominator.
# ---------------------------------------------------------------------------

# Per-panel spec: (pass, category, rendered_col, dispatched_col_or_None,
#                  dat_key)
# When dispatched_col is None the denominator is supplied at runtime from
# the series's scene aggregate (terrain only).
_MESHLET_PANELS: tuple[tuple[str, str, str, "str | None", str], ...] = (
    ("Main",   "Trunk",
     "trunkMainMeshletsRendered",
     "trunkMainMeshletsDispatched",   "main_trunk"),
    ("Main",   "Leaves",
     "leafMainMeshletsRendered",
     "leafMainMeshletsDispatched",    "main_leaves"),
    ("Main",   "Terrain",
     "terrainMainMeshletsRendered",
     None,                            "main_terrain"),
    ("Shadow", "Trunk",
     "trunkShadowMeshletsRendered",
     "trunkShadowMeshletsDispatched", "shadow_trunk"),
    ("Shadow", "Leaves",
     "leafShadowMeshletsRendered",
     "leafShadowMeshletsDispatched",  "shadow_leaves"),
    ("Shadow", "Terrain",
     "terrainShadowMeshletsRendered",
     None,                            "shadow_terrain"),
)

_MESHLET_RENDERED_HEX = "55A467"   # green for the rendered (kept) layer


def _meshlet_visibility_tex(*, base: str, x_min: float, x_max: float) -> str:
    """Return the `\\input{}`-able fragment for the 3x2 rendered-fraction
    area groupplot. Each panel shows the rendered/dispatched fraction as a
    single filled area from y=0 up to the fraction value; the empty space
    above is the culled portion (left intentionally blank, no gray fill)."""
    panel_blocks: list[str] = []
    for pass_name, category, _rend_col, _disp_col, key in _MESHLET_PANELS:
        is_first_col = (category == "Trunk")
        is_legend_panel = (key == "shadow_leaves")  # bottom-centre
        ylabel_block = (
            f"      ylabel={{Rendered / Dispatched}},\n"
            f"      ylabel style={{font=\\small}},\n"
            if is_first_col else "")
        legend_block = ""
        if is_legend_panel:
            legend_block = (
                f"      \\addlegendimage{{\n"
                f"        legend image code/.code={{%\n"
                f"          \\fill[rendcol, fill opacity=0.9]\n"
                f"            (0cm,-0.08cm) rectangle (0.55cm,0.17cm);\n"
                f"        }},\n"
                f"      }}\n"
                f"      \\addlegendentry{{Rendered fraction}}\n"
            )
        panel_blocks.append(
            f"    \\nextgroupplot[\n"
            f"      title={{{pass_name} --- {category}}},\n"
            f"      ymin=0, ymax=1.0,\n"
            f"{ylabel_block}"
            f"    ]\n"
            f"      \\addplot[fill=rendcol, draw=none, fill opacity=0.9,\n"
            f"              forget plot]\n"
            f"        table[x=time, y={key}_rend_frac]\n"
            f"          {{{base}.dat}} \\closedcycle;\n"
            f"{legend_block}"
        )
    panels_str = "\n".join(panel_blocks)

    return (
        f"% Auto-generated by scripts/plots/per_frame.py - do not hand-edit.\n"
        f"% Insert with \\input{{{base}}} from a document whose preamble loads\n"
        f"% pgfplots (>= 1.16) AND `\\usepgfplotslibrary{{groupplots}}`.\n"
        f"% Companion data file (same directory):\n"
        f"%   {base}.dat - columns: time + per-panel <key>_rend_frac\n"
        f"%                in [0, 1] (rendered / dispatched).\n"
        f"%   For terrain panels, the dispatched denominator is the scene\n"
        f"%   aggregate (sceneStats.terrainMeshlets / .terrainShadow\n"
        f"%   MeshletsDispatched) because BenchmarkRunner doesn't log a\n"
        f"%   per-frame terrain dispatched count.\n"
        f"\\begin{{tikzpicture}}\n"
        f"  \\definecolor{{rendcol}}{{HTML}}{{{_MESHLET_RENDERED_HEX}}}\n"
        f"  \\begin{{groupplot}}[\n"
        f"    group style={{\n"
        f"      group size=3 by 2,\n"
        f"      horizontal sep=1.2cm,\n"
        f"      vertical sep=1.2cm,\n"
        f"      x descriptions at=edge bottom,\n"
        f"    }},\n"
        f"    width=4.6cm, height=3.6cm, scale only axis=true,\n"
        f"    xmin={x_min:.6g}, xmax={x_max:.6g},\n"
        f"    enlarge x limits=false,\n"
        f"    grid=major, ymajorgrids=true,\n"
        f"    area style,\n"
        f"    ytick={{0, 0.25, 0.5, 0.75, 1.0}},\n"
        f"    yticklabel style={{font=\\scriptsize,\n"
        f"      /pgf/number format/.cd, fixed, precision=2,\n"
        f"    }},\n"
        f"    tick label style={{font=\\scriptsize}},\n"
        f"    xlabel={{Simulation time (s)}},\n"
        f"    xlabel style={{font=\\small}},\n"
        f"    title style={{font=\\small}},\n"
        f"    % Legend lives on the bottom-centre panel ('shadow_leaves'),\n"
        f"    % at={{(0.5, -0.40)}} anchors it under that panel's centre.\n"
        f"    legend style={{at={{(0.5, -0.40)}}, anchor=north,\n"
        f"                   draw=none, fill=none, font=\\scriptsize}},\n"
        f"    legend columns=1,\n"
        f"    legend cell align=left,\n"
        f"  ]\n"
        f"\n"
        f"{panels_str}\n"
        f"  \\end{{groupplot}}\n"
        f"\\end{{tikzpicture}}\n"
    )


def plot_meshlet_visibility_tex(series_list: list[Series], out_dir: Path, *,
                                fmt: str = "pdf+png") -> list[Path]:
    """pgfplots/tikz 3x2 meshlet-visibility groupplot. Each panel is a
    100%-stacked area of (rendered, culled) where rendered = per-frame
    rendered/dispatched fraction. Bottom layer is rendered, top is culled,
    sum = 1.0 at every x.

    Terrain panels use the SCENE-aggregate dispatched count
    (`sceneStats.terrainMeshlets` for main,
    `sceneStats.terrainShadowMeshletsDispatched` for shadow) as their
    denominator because BenchmarkRunner doesn't log a per-frame terrain
    dispatched. That means the terrain fraction reads as
    'rendered / scene-total', i.e. the full-pipeline cull effectiveness,
    while trunk/leaves read as 'rendered / per-frame dispatched',
    i.e. the AS-shader cull effectiveness specifically. Both are between
    0 and 1; the panel-pair share is the most informative comparison
    within a column."""
    del fmt
    mesh = next((s for s in series_list
                 if s.pipeline == "MeshShader" and "No Hi-Z" not in s.label),
                None)
    if mesh is None:
        return []
    df = mesh.frames
    if "trunkMainMeshletsRendered" not in df.columns:
        return []

    # Scene-aggregate denominators for terrain panels.
    scene_stats = mesh.summary.get("sceneStats", {})
    terrain_main_disp = float(scene_stats.get("terrainMeshlets", 0) or 0)
    terrain_shadow_disp = float(
        scene_stats.get("terrainShadowMeshletsDispatched", 0) or 0)
    # Fall back to max-of-rendered if scene stats missing -- gives an upper
    # bound (so the fraction never exceeds 1) without crashing.
    if terrain_main_disp <= 0:
        terrain_main_disp = float(
            df.get("terrainMainMeshletsRendered",
                   pd.Series([0])).fillna(0).max() or 1.0)
    if terrain_shadow_disp <= 0:
        terrain_shadow_disp = float(
            df.get("terrainShadowMeshletsRendered",
                   pd.Series([0])).fillna(0).max() or 1.0)

    chart_dir = out_dir / "meshlet_visibility"
    chart_dir.mkdir(parents=True, exist_ok=True)
    base = "meshlet_visibility"

    target_points = 600
    time_full = df["simTimeMs"].to_numpy(dtype=float) / 1000.0
    if time_full.size == 0:
        return []
    stride = max(1, len(time_full) // target_points)
    time = time_full[::stride]
    n = len(time)
    zeros = np.zeros(n, dtype=float)

    def col(name: str | None) -> np.ndarray:
        if name is None or name not in df.columns:
            return zeros
        return df[name].fillna(0).to_numpy(dtype=float)[::stride]

    # Build per-panel rendered / dispatched fraction.
    dat_columns: list[str] = ["time"]
    dat_data: list[np.ndarray] = [time]
    for pass_name, category, rend_col, disp_col, key in _MESHLET_PANELS:
        rend = col(rend_col)
        if disp_col is not None:
            disp = col(disp_col)
            with np.errstate(divide="ignore", invalid="ignore"):
                frac = np.where(disp > 0, rend / disp, 0.0)
        else:
            denom = (terrain_main_disp if pass_name == "Main"
                     else terrain_shadow_disp)
            frac = rend / denom if denom > 0 else zeros
        frac = np.clip(frac, 0.0, 1.0)
        dat_columns.append(f"{key}_rend_frac")
        dat_data.append(frac)

    _write_dat(chart_dir / f"{base}.dat", dat_columns,
               np.column_stack(dat_data))
    written: list[Path] = [chart_dir / f"{base}.dat"]

    tex = _meshlet_visibility_tex(
        base=base,
        x_min=float(time.min()), x_max=float(time.max()),
    )
    written.append(_write_tex(chart_dir, base, tex))
    written.append(_write_tex(chart_dir, f"{base}.standalone",
                              _make_standalone_wrapper(
                                  base,
                                  extra_preamble=
                                  r"\usepgfplotslibrary{groupplots}")))
    return written


# ---------------------------------------------------------------------------
# FPS-overlay variants of instance_visibility and meshlet_visibility. Same
# stack panels as the originals, plus a right-hand y-axis overlay carrying
# one or more FPS lines so frame-rate spikes line up visually with the
# visibility / cull data underneath. Implemented as new functions (the
# originals are untouched) so the user can keep both side by side.
# ---------------------------------------------------------------------------

# Pipeline base colours for the FPS overlay lines, matching style.PIPELINE_COLORS.
_PIPELINE_FPS_COLOR_HEX: dict[str, str] = {
    "Traditional": "111827",   # near-black
    "Compute":     "D946EF",   # fuchsia
    "MeshShader":  "DC2626",   # red
}


def _fps_overlay_axis_tex(*, panel_name: str, panel_w_cm: float,
                          panel_h_cm: float, x_min: float, x_max: float,
                          y_max: float, fps_specs: list[tuple[str, str, str]],
                          show_ylabel: bool,
                          y_column: str = "fps",
                          y_axis_label: str = "FPS") -> str:
    """Render one transparent overlay axis sitting on top of `panel_name`,
    using the right-hand y axis for the overlay metric.

    `fps_specs` is a list of (label, color_name, dat_filename) tuples. One
    `\\addplot` is emitted per spec, reading the `y_column` column. Only
    the last overlay axis in a figure should set `show_ylabel=True` (or
    you'll get redundant axis labels). All panels share the same x-range
    with the underlying stack so the lines visually correspond
    frame-for-frame."""
    plots: list[str] = []
    for label, color, dat in fps_specs:
        plots.append(
            f"      \\addplot[draw={color}, line width=0.7pt, forget plot]\n"
            f"        table[x=time, y={y_column}] {{{dat}}};"
        )
    plots_str = "\n".join(plots)
    ylabel_block = (
        f"      ylabel={{{y_axis_label}}},\n"
        f"      ylabel style={{font=\\small}},\n"
        if show_ylabel else
        f"      yticklabel pos=right,\n"
    )
    return (
        f"  % FPS overlay (right y-axis) anchored on top of `{panel_name}`.\n"
        f"  \\begin{{axis}}[\n"
        f"    at=({panel_name}.south west), anchor=south west,\n"
        f"    width={panel_w_cm:.4g}cm, height={panel_h_cm:.4g}cm,\n"
        f"    scale only axis=true,\n"
        f"    axis y line*=right,\n"
        f"    axis x line=none,\n"
        f"    xmin={x_min:.6g}, xmax={x_max:.6g},\n"
        f"    ymin=0, ymax={y_max:.6g},\n"
        f"    enlarge x limits=false,\n"
        f"{ylabel_block}"
        f"    yticklabel style={{font=\\scriptsize}},\n"
        f"  ]\n"
        f"{plots_str}\n"
        f"  \\end{{axis}}\n"
    )


def _instance_visibility_fps_tex(*, base: str,
                                 pipeline_label: str,
                                 primary_suffix: str,
                                 reference_suffix: str,
                                 reference_label: str,
                                 show_reference: bool,
                                 x_min: float, x_max: float,
                                 main_y_max: float, shadow_y_max: float,
                                 active_lod_indices: list[int],
                                 fps_specs: list[tuple[str, str, str]],
                                 fps_y_max: float) -> str:
    """Like `_instance_visibility_tex` but adds an FPS overlay axis on top
    of each panel. `fps_specs` is the list of (legend_label, color_name,
    dat_filename) entries that the FPS overlay should plot.

    All FPS line colours are defined inline as `fps0`, `fps1`, ... in the
    tikzpicture preamble; the caller's `fps_specs` references those names.
    Panel widths/heights match the existing instance_visibility (13cm x
    4.6cm) so the overlay aligns pixel-for-pixel."""
    color_defs_lines = [
        f"  \\definecolor{{lod{i}}}{{HTML}}{{{_LOD_COLORS_HEX[i]}}}"
        for i in active_lod_indices
    ]
    color_defs_lines.append(
        f"  \\definecolor{{visimpostors}}{{HTML}}{{{_VIS_IMPOSTORS_HEX}}}")
    # FPS line colours: emit one \definecolor per spec, named by index.
    for i, (_, color_hex, _) in enumerate(fps_specs):
        color_defs_lines.append(
            f"  \\definecolor{{fps{i}}}{{HTML}}{{{color_hex}}}")
    color_defs = "\n".join(color_defs_lines)

    def main_stack() -> str:
        out: list[str] = []
        for i in active_lod_indices:
            out.append(
                f"      \\addplot[fill=lod{i}, draw=none, fill opacity=0.95,\n"
                f"              forget plot]\n"
                f"        table[x=time, y=lod{i}]\n"
                f"          {{{base}__{primary_suffix}.dat}} \\closedcycle;\n"
            )
        out.append(
            f"      \\addplot[fill=visimpostors, draw=none,\n"
            f"              fill opacity=0.95, forget plot]\n"
            f"        table[x=time, y=impostors]\n"
            f"          {{{base}__{primary_suffix}.dat}} \\closedcycle;\n"
        )
        return "".join(out)

    shadow_trunk_lod = (active_lod_indices[-1]
                        if active_lod_indices else 0)

    def shadow_stack() -> str:
        return (
            f"      \\addplot[fill=lod{shadow_trunk_lod}, draw=none,\n"
            f"              fill opacity=0.95, forget plot]\n"
            f"        table[x=time, y=shadow_trunk]\n"
            f"          {{{base}__{primary_suffix}.dat}} \\closedcycle;\n"
            f"      \\addplot[fill=visimpostors, draw=none,\n"
            f"              fill opacity=0.95, forget plot]\n"
            f"        table[x=time, y=shadow_imp]\n"
            f"          {{{base}__{primary_suffix}.dat}} \\closedcycle;\n"
        )

    def ref_line(metric: str) -> str:
        if not show_reference:
            return ""
        return (
            f"      \\addplot[mark=none, draw=black, dashed, line width=0.9pt,\n"
            f"              stack plots=false, forget plot,\n"
            f"              on layer=axis foreground]\n"
            f"        table[x=time, y={metric}]\n"
            f"          {{{base}__{reference_suffix}.dat}};\n"
        )

    # Build the legend: LOD swatches + Impostor + (optional reference) +
    # FPS line entries.
    def area_swatch(color_name: str) -> str:
        return (
            f"      \\addlegendimage{{\n"
            f"        legend image code/.code={{%\n"
            f"          \\fill[{color_name}, fill opacity=0.95]\n"
            f"            (0cm,-0.08cm) rectangle (0.55cm,0.17cm);\n"
            f"        }},\n"
            f"      }}\n"
        )

    legend_entries: list[str] = []
    for i in active_lod_indices:
        legend_entries.append(
            area_swatch(f"lod{i}")
            + f"      \\addlegendentry{{LOD {i}}}"
        )
    legend_entries.append(
        area_swatch("visimpostors")
        + f"      \\addlegendentry{{Impostor}}"
    )
    if show_reference:
        legend_entries.append(
            f"      \\addlegendimage{{\n"
            f"        legend image code/.code={{%\n"
            f"          \\draw[black, dashed, line width=0.9pt]\n"
            f"            (0cm,0cm) -- (0.55cm,0cm);\n"
            f"        }},\n"
            f"      }}\n"
            f"      \\addlegendentry{{{_tex_escape(reference_label)}}}"
        )
    for i, (label, _hex, _dat) in enumerate(fps_specs):
        legend_entries.append(
            f"      \\addlegendimage{{\n"
            f"        legend image code/.code={{%\n"
            f"          \\draw[fps{i}, line width=0.9pt]\n"
            f"            (0cm,0cm) -- (0.55cm,0cm);\n"
            f"        }},\n"
            f"      }}\n"
            f"      \\addlegendentry{{{_tex_escape(label)} GPU ms}}"
        )
    legend_block = "\n".join(legend_entries) + "\n"

    n_legend = (len(active_lod_indices) + 1
                + (1 if show_reference else 0)
                + len(fps_specs))
    legend_columns = min(n_legend, 6)

    # FPS overlay specs converted to (label, color_name, dat) for the
    # overlay axis helper. The colour names map back to the inline
    # \definecolor entries created above.
    overlay_specs = [(label, f"fps{i}", dat)
                     for i, (label, _, dat) in enumerate(fps_specs)]

    panel_w_cm = 13.0
    panel_h_cm = 4.6
    main_overlay = _fps_overlay_axis_tex(
        panel_name="instg c1r1", panel_w_cm=panel_w_cm,
        panel_h_cm=panel_h_cm, x_min=x_min, x_max=x_max,
        y_max=fps_y_max, fps_specs=overlay_specs, show_ylabel=True,
        y_column="gpuMs", y_axis_label="GPU time (ms)",
    )
    shadow_overlay = _fps_overlay_axis_tex(
        panel_name="instg c1r2", panel_w_cm=panel_w_cm,
        panel_h_cm=panel_h_cm, x_min=x_min, x_max=x_max,
        y_max=fps_y_max, fps_specs=overlay_specs, show_ylabel=True,
        y_column="gpuMs", y_axis_label="GPU time (ms)",
    )

    ref_files_note = ""
    if show_reference:
        ref_files_note = (
            f"%   {base}__{reference_suffix}.dat - reference data for the\n"
            f"%                                    dashed overlay line.\n"
        )
    fps_files_note = ""
    for label, _, dat in fps_specs:
        fps_files_note += f"%   {dat} - FPS data for `{label}` overlay.\n"

    return (
        f"% Auto-generated by scripts/plots/per_frame.py - do not hand-edit.\n"
        f"% Insert with \\input{{{base}}} from a document whose preamble loads\n"
        f"% pgfplots (>= 1.16) AND `\\usepgfplotslibrary{{groupplots}}`.\n"
        f"% Companion data files (same directory):\n"
        f"%   {base}__{primary_suffix}.dat - primary data (this pipeline);\n"
        f"%     columns include time, lod0..lodN, impostors, main_total,\n"
        f"%     shadow_trunk, shadow_imp, shadow_total, gpuMs.\n"
        f"{ref_files_note}"
        f"{fps_files_note}"
        f"\\begin{{tikzpicture}}\n"
        f"{color_defs}\n"
        f"  \\begin{{groupplot}}[\n"
        f"    group style={{\n"
        f"      group size=1 by 2,\n"
        f"      group name=instg,\n"
        f"      vertical sep=1.0cm,\n"
        f"      x descriptions at=edge bottom,\n"
        f"    }},\n"
        f"    width={panel_w_cm:.4g}cm, height={panel_h_cm:.4g}cm,\n"
        f"    scale only axis=true,\n"
        f"    xmin={x_min:.6g}, xmax={x_max:.6g},\n"
        f"    enlarge x limits=false,\n"
        f"    grid=major, ymajorgrids=true,\n"
        f"    stack plots=y, area style,\n"
        f"    area legend,\n"
        f"    scaled y ticks=false,\n"
        f"    yticklabel style={{\n"
        f"      /pgf/number format/sci,\n"
        f"      /pgf/number format/sci e,\n"
        f"      /pgf/number format/sci zerofill,\n"
        f"      /pgf/number format/precision=1,\n"
        f"    }},\n"
        f"    tick label style={{font=\\scriptsize}},\n"
        f"    xlabel={{Simulation time (s)}},\n"
        f"    xlabel style={{font=\\small}},\n"
        f"    ylabel={{Visible instances}},\n"
        f"    ylabel style={{font=\\small}},\n"
        f"    title style={{font=\\small}},\n"
        f"    set layers,\n"
        f"  ]\n"
        f"\n"
        f"    % Top panel: main pass. Accessible via `instg c1r1` for the\n"
        f"    % FPS overlay axis below.\n"
        f"    \\nextgroupplot[\n"
        f"      title={{Main pass --- {_tex_escape(pipeline_label)}}},\n"
        f"      ymin=0, ymax={main_y_max:.6g},\n"
        f"    ]\n"
        f"{main_stack()}"
        f"{ref_line('main_total')}"
        f"\n"
        f"    % Bottom panel: shadow pass (`instg c1r2`). Carries legend.\n"
        f"    \\nextgroupplot[\n"
        f"      title={{Shadow pass --- {_tex_escape(pipeline_label)}}},\n"
        f"      ymin=0, ymax={shadow_y_max:.6g},\n"
        f"      legend style={{at={{(0.5, -0.30)}}, anchor=north,\n"
        f"                     draw=none, fill=none, font=\\scriptsize,\n"
        f"                     /tikz/every even column/.append style="
                              f"{{column sep=0.4cm}}}},\n"
        f"      legend columns={legend_columns},\n"
        f"      legend cell align=left,\n"
        f"    ]\n"
        f"{shadow_stack()}"
        f"{ref_line('shadow_total')}"
        f"\n"
        f"      % Legend swatches.\n"
        f"{legend_block}"
        f"\n"
        f"  \\end{{groupplot}}\n"
        f"\n"
        f"{main_overlay}"
        f"{shadow_overlay}"
        f"\\end{{tikzpicture}}\n"
    )


def plot_instance_visibility_fps_tex(series_list: list[Series], out_dir: Path,
                                     *, fmt: str = "pdf+png") -> list[Path]:
    """instance_visibility but with FPS overlaid on a right-hand y-axis.

    Per-folder FPS line assignment:
      - `Traditional/`: one FPS line, from the Traditional series.
      - `Compute/`:     two FPS lines, from Compute and MeshShader.

    Emits per-folder:
      - `instance_visibility_fps.tex`            (\\input{}-able fragment)
      - `instance_visibility_fps.standalone.tex` (pdflatex wrapper)
      - `instance_visibility_fps__<pipeline>.dat` for each plotted pipeline,
        each carrying time + stack data + an `fps` column.
      - `instance_visibility_fps__traditional.dat` (Compute folder only)
        for the dashed reference line (Traditional stack-top).
      - `instance_visibility_fps.pdf`            (compiled separately)"""
    del fmt
    traditional = next((s for s in series_list
                        if s.pipeline == "Traditional"), None)
    compute = next((s for s in series_list
                    if s.pipeline == "Compute"
                    and "No Hi-Z" not in s.label), None)
    meshshader = next((s for s in series_list
                       if s.pipeline == "MeshShader"
                       and "No Hi-Z" not in s.label), None)
    if traditional is None or compute is None:
        return []

    lod_cols = sorted(
        [c for c in traditional.frames.columns
         if c.startswith("mainVisibleLod")],
        key=lambda c: int(c.replace("mainVisibleLod", "")),
    )
    num_lods = min(len(lod_cols), len(_LOD_COLORS_HEX))
    lod_cols = lod_cols[:num_lods]
    if num_lods == 0:
        return []
    active_lod_indices: list[int] = []
    for i, col in enumerate(lod_cols):
        if any(float(s.frames[col].fillna(0).sum()) > 0
               for s in [traditional, compute]):
            active_lod_indices.append(i)
    if not active_lod_indices:
        return []

    def percentile_top(df, cols: list[str]) -> float:
        present = [c for c in cols if c in df.columns]
        if not present:
            return 0.0
        total = sum(df[c].fillna(0).to_numpy() for c in present)
        if not isinstance(total, np.ndarray) or total.size == 0:
            return 0.0
        return float(np.percentile(total, 99))

    main_top = 0.0
    shadow_top = 0.0
    series_for_y = [traditional, compute]
    for s in series_for_y:
        main_top = max(main_top, percentile_top(
            s.frames, lod_cols + ["mainVisibleImpostors"]))
        shadow_top = max(shadow_top, percentile_top(
            s.frames, ["shadowVisibleTrunk", "shadowVisibleImpostors"]))
    main_top = main_top * 1.08 if main_top > 0 else 1.0
    shadow_top = shadow_top * 1.08 if shadow_top > 0 else 1.0

    # Shared GPU-ms y-range across both folders so the overlay axes use the
    # same scale (99th percentile + 10% headroom across every loaded series).
    # 99th percentile keeps worst-case spikes visible since the GPU-ms
    # distribution is heavier on the upper tail than FPS.
    gpu_ms_pool: list[float] = []
    for s in series_for_y + ([meshshader] if meshshader is not None else []):
        if "gpuMs" in s.frames.columns:
            v = s.frames["gpuMs"].dropna().to_numpy(dtype=float)
            v = v[np.isfinite(v) & (v > 0)]
            if v.size:
                gpu_ms_pool.append(float(np.percentile(v, 99)))
    fps_y_max = (max(gpu_ms_pool) * 1.10) if gpu_ms_pool else 16.7

    base = "instance_visibility_gpums"
    parent = out_dir / base
    target_points = 600
    written: list[Path] = []

    def write_series_dat_fps(out_path: Path, df, include_fps: bool = True) -> None:
        time_full = df["simTimeMs"].to_numpy(dtype=float) / 1000.0
        if time_full.size == 0:
            return
        stride = max(1, len(time_full) // target_points)
        time = time_full[::stride]
        lod_arrs = [df[c].fillna(0).to_numpy(dtype=float)[::stride]
                    for c in lod_cols]
        impostors = (df["mainVisibleImpostors"].fillna(0)
                     .to_numpy(dtype=float)[::stride])
        main_total = sum(lod_arrs) + impostors
        shadow_trunk = (df.get("shadowVisibleTrunk",
                               pd.Series(np.zeros(len(time_full))))
                        .fillna(0).to_numpy(dtype=float)[::stride])
        shadow_imp = (df.get("shadowVisibleImpostors",
                             pd.Series(np.zeros(len(time_full))))
                      .fillna(0).to_numpy(dtype=float)[::stride])
        shadow_total = shadow_trunk + shadow_imp
        cols_out: list[np.ndarray] = (
            [time] + lod_arrs
            + [impostors, main_total, shadow_trunk, shadow_imp, shadow_total])
        header = (["time"] + [f"lod{i}" for i in range(num_lods)]
                  + ["impostors", "main_total",
                     "shadow_trunk", "shadow_imp", "shadow_total"])
        if include_fps:
            gpu_ms = (df.get("gpuMs",
                             pd.Series(np.zeros(len(time_full))))
                      .fillna(0).to_numpy(dtype=float)[::stride])
            cols_out.append(gpu_ms)
            header.append("gpuMs")
        _write_dat(out_path, header, np.column_stack(cols_out))

    t = traditional.frames["simTimeMs"].to_numpy(dtype=float) / 1000.0
    x_min = float(t.min()) if t.size else 0.0
    x_max = float(t.max()) if t.size else 1.0

    # Per-folder spec: (folder_name, primary_suffix, primary_series,
    #   show_ref, ref_series, ref_suffix, ref_label,
    #   fps_specs_factory)
    # `fps_specs_factory` returns the (label, color_hex, dat_filename) list
    # at folder-build time so we can derive the dat names from the suffix.
    def fps_specs_traditional() -> list[tuple[str, str, str]]:
        return [("Traditional",
                 _PIPELINE_FPS_COLOR_HEX["Traditional"],
                 f"{base}__traditional.dat")]

    def fps_specs_compute() -> list[tuple[str, str, str]]:
        specs: list[tuple[str, str, str]] = [
            ("Compute", _PIPELINE_FPS_COLOR_HEX["Compute"],
             f"{base}__compute.dat"),
        ]
        if meshshader is not None:
            specs.append(("MeshShader",
                          _PIPELINE_FPS_COLOR_HEX["MeshShader"],
                          f"{base}__meshshader.dat"))
        return specs

    folder_specs = [
        ("Traditional", "traditional", traditional,
         False, None, "", "",
         fps_specs_traditional),
        ("Compute", "compute", compute,
         True, traditional, "traditional", "Traditional",
         fps_specs_compute),
    ]

    # In the Compute folder we additionally need to write MeshShader's
    # .dat (for the second FPS line) even though MeshShader isn't the
    # primary or reference. Track which dats each folder needs.
    for (label, suffix, series, show_ref, ref_series,
         ref_suffix, ref_label, fps_specs_factory) in folder_specs:
        folder = parent / label
        folder.mkdir(parents=True, exist_ok=True)

        # Primary stack + fps.
        write_series_dat_fps(folder / f"{base}__{suffix}.dat", series.frames)
        written.append(folder / f"{base}__{suffix}.dat")
        # Reference data (dashed overlay).
        if show_ref and ref_series is not None:
            write_series_dat_fps(folder / f"{base}__{ref_suffix}.dat",
                                 ref_series.frames)
            written.append(folder / f"{base}__{ref_suffix}.dat")
        # Resolve fps_specs late, after suffix is known.
        fps_specs = fps_specs_factory()
        # For each fps_spec series that isn't already covered by primary
        # or reference, write its own dat with FPS data so the overlay can
        # read it.
        already_written = {f"{base}__{suffix}.dat"}
        if show_ref:
            already_written.add(f"{base}__{ref_suffix}.dat")
        for _, _, fps_dat in fps_specs:
            if fps_dat in already_written:
                continue
            # Locate the series for this fps spec by matching dat filename
            # back to a known pipeline series.
            if fps_dat == f"{base}__meshshader.dat" and meshshader is not None:
                write_series_dat_fps(folder / fps_dat, meshshader.frames)
                written.append(folder / fps_dat)
                already_written.add(fps_dat)

        tex = _instance_visibility_fps_tex(
            base=base, pipeline_label=label,
            primary_suffix=suffix, reference_suffix=ref_suffix,
            reference_label=ref_label, show_reference=show_ref,
            x_min=x_min, x_max=x_max,
            main_y_max=main_top, shadow_y_max=shadow_top,
            active_lod_indices=active_lod_indices,
            fps_specs=fps_specs, fps_y_max=fps_y_max,
        )
        written.append(_write_tex(folder, base, tex))
        written.append(_write_tex(folder, f"{base}.standalone",
                                  _make_standalone_wrapper(
                                      base,
                                      extra_preamble=
                                      r"\usepgfplotslibrary{groupplots}")))
    return written


def _meshlet_visibility_fps_tex(*, base: str, x_min: float, x_max: float,
                                fps_y_max: float) -> str:
    """Like `_meshlet_visibility_tex` but each of the 6 panels carries an
    FPS overlay (MeshShader pipeline) on a right-hand y axis."""
    panel_w_cm = 4.6
    panel_h_cm = 3.6
    panel_blocks: list[str] = []
    # Track auto-generated panel names in row-major order so we can iterate
    # them later for the FPS overlay. Names come from `group name=meshg`
    # set in the groupplot's group style below.
    panel_row_col: list[tuple[int, int]] = []   # (col, row), 1-indexed
    for i, (pass_name, category, _rend_col, _disp_col, key) in enumerate(_MESHLET_PANELS):
        col = (i % 3) + 1
        row = (i // 3) + 1
        panel_row_col.append((col, row))
        is_first_col = (category == "Trunk")
        is_legend_panel = (key == "shadow_leaves")
        ylabel_block = (
            f"      ylabel={{Rendered / Dispatched}},\n"
            f"      ylabel style={{font=\\small}},\n"
            if is_first_col else "")
        legend_block = ""
        if is_legend_panel:
            legend_block = (
                f"      \\addlegendimage{{\n"
                f"        legend image code/.code={{%\n"
                f"          \\fill[rendcol, fill opacity=0.9]\n"
                f"            (0cm,-0.08cm) rectangle (0.55cm,0.17cm);\n"
                f"        }},\n"
                f"      }}\n"
                f"      \\addlegendentry{{Rendered fraction}}\n"
                f"      \\addlegendimage{{\n"
                f"        legend image code/.code={{%\n"
                f"          \\draw[fpsmesh, line width=0.9pt]\n"
                f"            (0cm,0cm) -- (0.55cm,0cm);\n"
                f"        }},\n"
                f"      }}\n"
                f"      \\addlegendentry{{MeshShader GPU ms}}\n"
            )
        panel_blocks.append(
            f"    \\nextgroupplot[\n"
            f"      title={{{pass_name} --- {category}}},\n"
            f"      ymin=0, ymax=1.0,\n"
            f"{ylabel_block}"
            f"    ]\n"
            f"      \\addplot[fill=rendcol, draw=none, fill opacity=0.9,\n"
            f"              forget plot]\n"
            f"        table[x=time, y={key}_rend_frac]\n"
            f"          {{{base}.dat}} \\closedcycle;\n"
            f"{legend_block}"
        )
    panels_str = "\n".join(panel_blocks)

    fps_specs = [("MeshShader", "fpsmesh",
                  f"{base}.dat")]
    overlay_blocks: list[str] = []
    for i, (col, row) in enumerate(panel_row_col):
        panel_name = f"meshg c{col}r{row}"
        # Only the rightmost-column overlay shows the GPU-ms ylabel.
        is_last_col = (col == 3)
        overlay_blocks.append(_fps_overlay_axis_tex(
            panel_name=panel_name, panel_w_cm=panel_w_cm,
            panel_h_cm=panel_h_cm, x_min=x_min, x_max=x_max,
            y_max=fps_y_max, fps_specs=fps_specs,
            show_ylabel=is_last_col,
            y_column="gpuMs", y_axis_label="GPU time (ms)",
        ))
    overlay_str = "".join(overlay_blocks)

    return (
        f"% Auto-generated by scripts/plots/per_frame.py - do not hand-edit.\n"
        f"% Insert with \\input{{{base}}} from a document whose preamble loads\n"
        f"% pgfplots (>= 1.16) AND `\\usepgfplotslibrary{{groupplots}}`.\n"
        f"% Companion data file (same directory):\n"
        f"%   {base}.dat - columns: time + per-panel <key>_rend_frac + gpuMs.\n"
        f"\\begin{{tikzpicture}}\n"
        f"  \\definecolor{{rendcol}}{{HTML}}{{{_MESHLET_RENDERED_HEX}}}\n"
        f"  \\definecolor{{fpsmesh}}"
        f"{{HTML}}{{{_PIPELINE_FPS_COLOR_HEX['MeshShader']}}}\n"
        f"  \\begin{{groupplot}}[\n"
        f"    group style={{\n"
        f"      group size=3 by 2,\n"
        f"      group name=meshg,\n"
        f"      horizontal sep=1.6cm,\n"
        f"      vertical sep=1.2cm,\n"
        f"      x descriptions at=edge bottom,\n"
        f"    }},\n"
        f"    width={panel_w_cm:.4g}cm, height={panel_h_cm:.4g}cm,\n"
        f"    scale only axis=true,\n"
        f"    xmin={x_min:.6g}, xmax={x_max:.6g},\n"
        f"    enlarge x limits=false,\n"
        f"    grid=major, ymajorgrids=true,\n"
        f"    area style,\n"
        f"    ytick={{0, 0.25, 0.5, 0.75, 1.0}},\n"
        f"    yticklabel style={{font=\\scriptsize,\n"
        f"      /pgf/number format/.cd, fixed, precision=2,\n"
        f"    }},\n"
        f"    tick label style={{font=\\scriptsize}},\n"
        f"    xlabel={{Simulation time (s)}},\n"
        f"    xlabel style={{font=\\small}},\n"
        f"    title style={{font=\\small}},\n"
        f"    legend style={{at={{(0.5, -0.40)}}, anchor=north,\n"
        f"                   draw=none, fill=none, font=\\scriptsize,\n"
        f"                   /tikz/every even column/.append style="
                            f"{{column sep=0.4cm}}}},\n"
        f"    legend columns=2,\n"
        f"    legend cell align=left,\n"
        f"  ]\n"
        f"\n"
        f"{panels_str}\n"
        f"  \\end{{groupplot}}\n"
        f"\n"
        f"{overlay_str}"
        f"\\end{{tikzpicture}}\n"
    )


def plot_meshlet_visibility_fps_tex(series_list: list[Series], out_dir: Path,
                                    *, fmt: str = "pdf+png") -> list[Path]:
    """meshlet_visibility plus a MeshShader FPS overlay on each of the 6
    panels. Same .dat layout as meshlet_visibility plus one `fps` column."""
    del fmt
    mesh = next((s for s in series_list
                 if s.pipeline == "MeshShader" and "No Hi-Z" not in s.label),
                None)
    if mesh is None or "trunkMainMeshletsRendered" not in mesh.frames.columns:
        return []
    df = mesh.frames

    scene_stats = mesh.summary.get("sceneStats", {})
    terrain_main_disp = float(scene_stats.get("terrainMeshlets", 0) or 0)
    terrain_shadow_disp = float(
        scene_stats.get("terrainShadowMeshletsDispatched", 0) or 0)
    if terrain_main_disp <= 0:
        terrain_main_disp = float(
            df.get("terrainMainMeshletsRendered",
                   pd.Series([0])).fillna(0).max() or 1.0)
    if terrain_shadow_disp <= 0:
        terrain_shadow_disp = float(
            df.get("terrainShadowMeshletsRendered",
                   pd.Series([0])).fillna(0).max() or 1.0)

    chart_dir = out_dir / "meshlet_visibility_gpums"
    chart_dir.mkdir(parents=True, exist_ok=True)
    base = "meshlet_visibility_gpums"

    target_points = 600
    time_full = df["simTimeMs"].to_numpy(dtype=float) / 1000.0
    if time_full.size == 0:
        return []
    stride = max(1, len(time_full) // target_points)
    time = time_full[::stride]
    n = len(time)
    zeros = np.zeros(n, dtype=float)

    def col(name: str | None) -> np.ndarray:
        if name is None or name not in df.columns:
            return zeros
        return df[name].fillna(0).to_numpy(dtype=float)[::stride]

    dat_columns: list[str] = ["time"]
    dat_data: list[np.ndarray] = [time]
    for pass_name, category, rend_col, disp_col, key in _MESHLET_PANELS:
        rend = col(rend_col)
        if disp_col is not None:
            disp = col(disp_col)
            with np.errstate(divide="ignore", invalid="ignore"):
                frac = np.where(disp > 0, rend / disp, 0.0)
        else:
            denom = (terrain_main_disp if pass_name == "Main"
                     else terrain_shadow_disp)
            frac = rend / denom if denom > 0 else zeros
        frac = np.clip(frac, 0.0, 1.0)
        dat_columns.append(f"{key}_rend_frac")
        dat_data.append(frac)
    # GPU ms column (same series; one number per frame).
    gpu_ms = (df.get("gpuMs", pd.Series(np.zeros(len(time_full))))
              .fillna(0).to_numpy(dtype=float)[::stride])
    dat_columns.append("gpuMs")
    dat_data.append(gpu_ms)

    _write_dat(chart_dir / f"{base}.dat", dat_columns,
               np.column_stack(dat_data))
    written: list[Path] = [chart_dir / f"{base}.dat"]

    finite = gpu_ms[np.isfinite(gpu_ms) & (gpu_ms > 0)]
    fps_y_max = (float(np.percentile(finite, 99)) * 1.10
                 if finite.size else 16.7)

    tex = _meshlet_visibility_fps_tex(
        base=base, x_min=float(time.min()), x_max=float(time.max()),
        fps_y_max=fps_y_max,
    )
    written.append(_write_tex(chart_dir, base, tex))
    written.append(_write_tex(chart_dir, f"{base}.standalone",
                              _make_standalone_wrapper(
                                  base,
                                  extra_preamble=
                                  r"\usepgfplotslibrary{groupplots}")))
    return written


def _has_cols(df, cols) -> bool:
    return all(c in df.columns for c in cols)


def plot_correlation_scatters(series_list: list[Series], out_dir: Path, *,
                              fmt: str = "pdf+png") -> list[Path]:
    """Suite of per-frame correlation scatters against gpuMs, one PDF per
    metric, all under `out_dir/correlation/`. Panels are one-per-series
    (with linear fit, Pearson r, and slope in title). Pipelines that
    don't populate a given counter -- e.g. terrain meshlets on P0/P1 --
    drop out automatically because `_scatter_metric_vs_gpu` filters series
    whose column is identically zero."""
    target = out_dir / "correlation"
    written: list[Path] = []
    for x_col, x_label, basename, slope_unit, slope_scale in _CORRELATION_METRICS:
        written.extend(_scatter_metric_vs_gpu(
            series_list, target,
            x_col=x_col, x_label=x_label, basename=basename,
            slope_unit=slope_unit, slope_scale=slope_scale, fmt=fmt))
    return written


def _terrain_shadow_scatter_tex(*, base: str, label: str, color_hex: str,
                                dat_name: str, fit_dat_name: str,
                                x_max: float, y_min: float, y_max: float,
                                r: float, slope_ns: float) -> str:
    """pgfplots fragment for the single-panel terrain-shadow-meshlet scatter.

    Uses sci-notation x ticks (raw counts run ~1e5--1e6, so the default
    full-digit labels are unreadable). The fit stats sit in a node anchored
    to `current axis.south` with an em-based yshift so the line lands below
    the x-tick labels and xlabel without overlapping either."""
    safe_label = _tex_escape(label)
    safe_color = color_hex.lstrip("#").upper()
    return (
        f"% Auto-generated by scripts/plots/per_frame.py - do not hand-edit.\n"
        f"% Insert with \\input{{{base}}} from a document whose preamble loads\n"
        f"% pgfplots (>= 1.16). Companion data files (same directory):\n"
        f"%   {dat_name}       - per-frame scatter rows: meshlets gpuMs\n"
        f"%   {fit_dat_name}   - two-row endpoints for the linear fit line\n"
        f"\\begin{{tikzpicture}}\n"
        f"  \\definecolor{{series0}}{{HTML}}{{{safe_color}}}\n"
        f"  \\begin{{axis}}[\n"
        f"    width=11cm, height=7cm, scale only axis=true,\n"
        f"    xlabel={{Meshlets rendered}},\n"
        f"    ylabel={{GPU frame time (ms)}},\n"
        f"    xlabel style={{font=\\small}},\n"
        f"    ylabel style={{font=\\small}},\n"
        f"    tick label style={{font=\\scriptsize}},\n"
        f"    title={{Frame time vs.\\ terrain shadow meshlets --- {safe_label}}},\n"
        f"    grid=major,\n"
        f"    xmin=0, xmax={x_max:.6g},\n"
        f"    ymin={y_min:.6g}, ymax={y_max:.6g},\n"
        f"    % Sci notation on the x axis: raw counts are 1e5-1e6 so default\n"
        f"    % comma-grouped labels are unreadable. `sci e` prints `5e5` style.\n"
        f"    scaled x ticks=false,\n"
        f"    x tick label style={{/pgf/number format/.cd, sci, sci e,\n"
        f"                         precision=1, sci zerofill}},\n"
        f"  ]\n"
        f"    \\addplot[\n"
        f"      only marks, mark=*, mark size=0.5pt,\n"
        f"      color=series0, fill opacity=0.35, draw opacity=0.35,\n"
        f"    ] table[x=meshlets, y=gpuMs] {{{dat_name}}};\n"
        f"    \\addplot[\n"
        f"      no marks, line width=0.9pt, black, dashed, opacity=0.75,\n"
        f"    ] table[x=meshlets, y=gpuMs] {{{fit_dat_name}}};\n"
        f"  \\end{{axis}}\n"
        f"  % Fit stats line, anchored below the x-axis label. yshift in em\n"
        f"  % so the offset scales with font size.\n"
        f"  \\node[anchor=north, font=\\small, align=center, yshift=-3.5em]\n"
        f"    at (current axis.south)\n"
        f"    {{$r = {r:.3f}$ \\quad slope $= {slope_ns:.2f}$ ns/meshlet}};\n"
        f"\\end{{tikzpicture}}\n"
    )


def plot_terrain_shadow_meshlet_scatter_tex(
        series_list: list[Series], out_dir: Path, *,
        fmt: str = "pdf+png") -> list[Path]:
    """pgfplots/tikz equivalent of the terrain-shadow-meshlet scatter that
    used to live in `correlation/terrain_shadow_meshlets_vs_gpu.pdf`.

    Emits into `out_dir/correlation/terrain_shadow_meshlets_vs_gpu/`:
      - `terrain_shadow_meshlets_vs_gpu.tex`            (\\input{}-able)
      - `terrain_shadow_meshlets_vs_gpu.standalone.tex` (pdflatex wrapper)
      - `terrain_shadow_meshlets_vs_gpu__<label>.dat`   (scatter rows)
      - `terrain_shadow_meshlets_vs_gpu_fit.dat`        (fit endpoints)

    Only the MeshShader pipeline writes `terrainShadowMeshletsRendered`,
    so this picks the first series whose counter is nonzero and produces
    a single-panel chart. Pdf compilation of the standalone wrapper is
    the caller's responsibility -- run pdflatex from the chart directory."""
    del fmt  # always .tex + .dat
    col = "terrainShadowMeshletsRendered"
    needed = ["gpuMs", col]
    candidates = [s for s in series_list if _has_cols(s.frames, needed)]
    candidates = [s for s in candidates
                  if float(s.frames[col].fillna(0).to_numpy().sum()) > 0]
    if not candidates:
        return []
    s = candidates[0]  # by convention the first (only) MeshShader series.

    base = "terrain_shadow_meshlets_vs_gpu"
    chart_dir = out_dir / "correlation" / base
    chart_dir.mkdir(parents=True, exist_ok=True)

    df = s.frames
    x = df[col].to_numpy(dtype=float)
    y = df["gpuMs"].to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x, y = x[mask], y[mask]
    if x.size < 2 or x.min() == x.max():
        return []

    slope, intercept = np.polyfit(x, y, 1)
    r = float(np.corrcoef(x, y)[0, 1])
    # ns/meshlet keeps the displayed coefficient in the 1-100 range for
    # readability (slope's raw ms/meshlet is ~2e-5).
    slope_ns = float(slope * 1_000_000.0)

    # Scatter rows: meshlets gpuMs.
    written: list[Path] = []
    dat_name = _line_series_dat_name(base, s.label)
    rows = np.column_stack([x, y])
    _write_dat(chart_dir / dat_name, ["meshlets", "gpuMs"], rows)
    written.append(chart_dir / dat_name)

    # Fit line endpoints (two rows: x.min/intercept-side, x.max/end).
    fit_dat_name = f"{base}_fit.dat"
    fit_rows = np.array([[x.min(), slope * x.min() + intercept],
                         [x.max(), slope * x.max() + intercept]])
    _write_dat(chart_dir / fit_dat_name, ["meshlets", "gpuMs"], fit_rows)
    written.append(chart_dir / fit_dat_name)

    # Pad the y axis a touch so the densest cluster has air around it.
    y_pad = 0.05 * (y.max() - y.min())
    color_hex, _ = _series_style([s])[0]
    tex = _terrain_shadow_scatter_tex(
        base=base, label=s.label, color_hex=color_hex,
        dat_name=dat_name, fit_dat_name=fit_dat_name,
        x_max=float(x.max()) * 1.02,
        y_min=max(0.0, float(y.min()) - y_pad),
        y_max=float(y.max()) + y_pad,
        r=r, slope_ns=slope_ns)
    written.append(_write_tex(chart_dir, base, tex))
    written.append(_write_tex(chart_dir, f"{base}.standalone",
                              _make_standalone_wrapper(base)))
    return written


# Meshlet streams shown by plot_meshlet_cull_over_time. Order is the natural
# read order on the 2x2 grid: (trunk-main, trunk-shadow, leaf-main, leaf-shadow).
# Columns surfaced in plot_vis_stats_table, in display order. Each tuple is
# (df-column, table-row-label). Counters that are all-zero in a series are
# dropped from that series' table.
_VIS_TABLE_COLS: tuple[tuple[str, str], ...] = (
    ("mainVisibleTrunk",     "Main: Trunk"),
    ("mainVisibleLeaves",    "Main: Leaves"),
    ("mainVisibleImpostors", "Main: Impostors"),
    ("shadowVisibleTrunk",     "Shadow: Trunk"),
    ("shadowVisibleLeaves",    "Shadow: Leaves"),
    ("shadowVisibleImpostors", "Shadow: Impostors"),
    ("shadowCascadeDraws",     "Shadow: Cascade Draws"),
    ("trunkMainMeshletsDispatched",   "Meshlets Trunk Main (disp)"),
    ("trunkMainMeshletsRendered",     "Meshlets Trunk Main (rend)"),
    ("trunkShadowMeshletsDispatched", "Meshlets Trunk Shadow (disp)"),
    ("trunkShadowMeshletsRendered",   "Meshlets Trunk Shadow (rend)"),
    ("leafMainMeshletsDispatched",    "Meshlets Leaf Main (disp)"),
    ("leafMainMeshletsRendered",      "Meshlets Leaf Main (rend)"),
    ("leafShadowMeshletsDispatched",  "Meshlets Leaf Shadow (disp)"),
    ("leafShadowMeshletsRendered",    "Meshlets Leaf Shadow (rend)"),
)
def plot_vis_stats_table(series_list: list[Series], out_dir: Path, *,
                         fmt: str = "pdf+png") -> list[Path]:
    """Emit one booktabs LaTeX table per series; rows are visibility / cull
    counters that have any data in that series, columns are summary stats.
    `fmt` is accepted for interface uniformity but ignored — output is
    always a `.tex` source file."""
    del fmt
    written: list[Path] = []
    for s in series_list:
        df = s.frames
        active: list[tuple[str, str]] = []
        for col, label in _VIS_TABLE_COLS:
            if col not in df.columns:
                continue
            arr = df[col].dropna().to_numpy()
            if arr.size == 0 or float(arr.sum()) == 0.0:
                continue
            active.append((col, label))
        if not active:
            continue

        # Sample counts match the per-row gpuMs frame count once the column
        # exists, but be defensive and recompute per row.
        n_values = [int(df[c].dropna().shape[0]) for c, _ in active]
        common_n = n_values[0] if len(set(n_values)) == 1 else None
        header = (["Counter", *_STAT_HEADERS] if common_n is not None
                  else ["Counter", "n", *_STAT_HEADERS])

        rows: list[list[str]] = []
        for col, label in active:
            arr = df[col].dropna().to_numpy().astype(float)
            stats = _summarize_ms(arr)
            row = [label]
            if common_n is None:
                row.append(f"{stats['n']:,}")
            for k in _STAT_KEYS:
                v = stats[k]
                # Integer-valued counters render cleaner without trailing
                # decimals when the underlying value is an integer.
                if not np.isfinite(v):
                    row.append("---")
                elif k == "std":
                    row.append(f"{v:,.2f}")
                elif float(v).is_integer():
                    row.append(f"{int(v):,}")
                else:
                    row.append(f"{v:,.2f}")
            rows.append(row)

        slug = _safe_filename(s.label)
        footer: list[str] | None = None
        if common_n is not None and "Median" in header:
            footer = [""] * len(header)
            footer[0] = r"\textbf{Frames}"
            footer[header.index("Median")] = f"{common_n:,}"
        tex = _booktabs_tex(
            header, rows, label_colors=None,
            caption=f"Visibility & cull counters --- {s.label}.",
            tex_label=f"tab:vis-stats-{slug}", footer_row=footer)
        # One folder per pipeline under vis_stats_table/, mirroring the
        # stats_table layout for stage_stats_table.
        chart_dir = out_dir / "vis_stats_table" / slug
        written.append(_write_tex(chart_dir, "vis_stats_table", tex))
        written.append(_write_tex(chart_dir, "vis_stats_table.standalone",
                                  _make_standalone_table_wrapper("vis_stats_table")))
    return written
