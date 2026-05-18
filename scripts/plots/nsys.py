"""Nsys per-pipeline aggregate plots (consume Series.nsys_dir).

Each plot emits a pgfplots fragment (`<base>.tex`), a standalone wrapper
(`<base>.standalone.tex`), one or more `<base>.dat` data files, and runs
pdflatex on the wrapper to produce `<base>.pdf`. Matches the per_frame.py
`_tex` convention so the manifest can be `\\input`'d straight into a paper.
"""
from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .loader import Series, load_nsys_csv
from .per_frame import (
    _make_standalone_wrapper, _safe_filename, _tex_escape,
    _write_dat, _write_tex,
)
from .style import PIPELINE_COLORS, variant_color

# Locked across every nsys bar chart so the figures sit side-by-side cleanly.
_BAR_WIDTH_CM = 0.4
# Single neutral color for the SM-throughput chart (no per-pipeline hue).
_SM_BAR_COLOR_HEX = "4C72B0"
# Diagonal x-tick label style for charts whose x-axis is pipeline names --
# avoids overlap once "Compute (No Hi-Z)" / "MeshShader (No Hi-Z)" are in play.
# \scriptsize keeps the rotated labels short enough that they don't reach the
# legend row below; charts that also draw a legend should pass
# _LEGEND_ANCHOR_Y_DIAGONAL to keep the legend clear of the label tails.
_DIAGONAL_XTICKLABEL_STYLE = (
    "{font=\\scriptsize, rotate=30, anchor=north east, "
    "inner sep=1pt, yshift=-1pt}"
)
# Legend y-anchor when the x-axis uses diagonal labels (was -0.18 for
# horizontal labels). Pushed down so the legend clears the rotated tick text.
_LEGEND_ANCHOR_Y_DIAGONAL = -0.32


def _nsys_series(series_list: list[Series]) -> list[Series]:
    """Filter to series that actually have an nsys_results/ folder."""
    return [s for s in series_list if s.nsys_dir is not None]


def _compile_standalone(standalone_tex: Path) -> Path | None:
    """Run pdflatex on a `<base>.standalone.tex` and return `<base>.pdf`.

    pdflatex runs with the .tex's directory as CWD so relative `\\input{base}`
    and `table{base.dat}` resolve naturally. Aux/log sidecars are removed on
    success. Returns None (with a printed tail of the log) on failure.
    """
    chart_dir = standalone_tex.parent
    stem = standalone_tex.stem  # e.g. "sm_throughput.standalone"
    result = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
         standalone_tex.name],
        cwd=str(chart_dir),
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        tail = (result.stdout or "")[-1500:]
        print(f"  pdflatex FAILED for {standalone_tex.name}:\n{tail}")
        return None
    produced = chart_dir / f"{stem}.pdf"
    if not produced.is_file():
        print(f"  pdflatex produced no PDF for {standalone_tex.name}")
        return None
    # Strip the `.standalone` suffix from the final PDF name.
    final_name = stem
    if final_name.endswith(".standalone"):
        final_name = final_name[: -len(".standalone")]
    final_pdf = chart_dir / f"{final_name}.pdf"
    if produced != final_pdf:
        shutil.move(str(produced), str(final_pdf))
    for ext in (".aux", ".log", ".out"):
        sidecar = chart_dir / f"{stem}{ext}"
        if sidecar.is_file():
            sidecar.unlink()
    return final_pdf


def _series_variant_colors_hex(series_list: list[Series]) -> list[str]:
    """Per-series fill colors with darker shades for repeated pipelines.

    First occurrence of each pipeline uses its base palette color; the second
    (e.g. the no-Hi-Z variant) is darkened via style.variant_color so the eye
    still groups them by hue but reads them as distinct bars.
    """
    pipeline_count: dict[str, int] = {}
    out: list[str] = []
    for s in series_list:
        idx = pipeline_count.get(s.pipeline, 0)
        pipeline_count[s.pipeline] = idx + 1
        base = PIPELINE_COLORS.get(s.pipeline, "#444444")
        out.append(variant_color(base, idx).lstrip("#").upper())
    return out


def _lighten_hex(hex_color: str, amount: float = 0.25) -> str:
    """Return a lightened version of hex_color (HLS lightness += amount).

    Used to derive each pipeline's pastel `shadow` fill from its actual main
    fill -- which itself may already be a variant_color-darkened shade for the
    no-Hi-Z series, so a static shadow palette wouldn't track variants.
    """
    import colorsys
    from .style import _hex_to_rgb, _rgb_to_hex
    r, g, b = _hex_to_rgb(hex_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    new_l = min(0.95, l + amount)
    return _rgb_to_hex(colorsys.hls_to_rgb(h, new_l, s))


def _format_value(v: float, precision: int = 1) -> str:
    if not np.isfinite(v):
        return "nan"
    return f"{v:.{precision}f}"


# ---------------------------------------------------------------------------
# 1) SM Throughput — one bar per pipeline.
# ---------------------------------------------------------------------------

def _sm_throughput_tex(*, base: str, labels: list[str],
                       values: list[float], y_max: float) -> str:
    n = len(labels)
    xticks = ", ".join(str(i) for i in range(1, n + 1))
    xticklabels = ", ".join("{" + _tex_escape(lbl) + "}" for lbl in labels)
    # Single \addplot so every bar belongs to the same series -- pgfplots then
    # centres each bar on its integer x coordinate instead of side-by-side
    # shifting them (which is what made MeshShader look offset).
    coords = " ".join(
        f"({i}, {('nan' if not np.isfinite(v) else f'{v:.6g}')})"
        for i, v in enumerate(values, start=1)
    )
    return (
        f"% Auto-generated by scripts/plots/nsys.py - do not hand-edit.\n"
        f"% Insert with \\input{{{base}}} from a document whose preamble loads\n"
        f"% pgfplots (>= 1.16).\n"
        f"\\begin{{tikzpicture}}\n"
        f"  \\definecolor{{smbar}}{{HTML}}{{{_SM_BAR_COLOR_HEX}}}\n"
        f"  \\begin{{axis}}[\n"
        f"    ybar, bar width={_BAR_WIDTH_CM}cm,\n"
        f"    width=11cm, height=7cm, scale only axis=true,\n"
        f"    enlarge x limits=0.25,\n"
        f"    ymin=0, ymax={y_max:.6g},\n"
        f"    xtick={{{xticks}}},\n"
        f"    xticklabels={{{xticklabels}}},\n"
        f"    xticklabel style={_DIAGONAL_XTICKLABEL_STYLE},\n"
        f"    ylabel={{SM Throughput (\\%)}},\n"
        f"    ylabel style={{font=\\small}},\n"
        f"    yticklabel style={{font=\\scriptsize}},\n"
        f"    title={{Average SM Throughput per pipeline}},\n"
        f"    ymajorgrids=true,\n"
        f"    every axis plot/.append style={{point meta=rawy}},\n"
        f"    nodes near coords={{\\pgfmathprintnumber[fixed, precision=1]"
        f"\\pgfplotspointmeta}},\n"
        f"    nodes near coords style={{font=\\scriptsize, anchor=south}},\n"
        f"  ]\n"
        f"    \\addplot[fill=smbar, draw=black, line width=0.3pt]\n"
        f"      coordinates {{{coords}}};\n"
        f"  \\end{{axis}}\n"
        f"\\end{{tikzpicture}}\n"
    )


def plot_sm_throughput(series_list: list[Series], out_dir: Path, *,
                       fmt: str = "pdf+png") -> list[Path]:
    """Tex/pdflatex: one bar per pipeline, SM Throughput %."""
    del fmt
    series_list = _nsys_series(series_list)
    if not series_list:
        return []
    values: list[float] = []
    for s in series_list:
        df = load_nsys_csv(s, "05_per_pipeline_metrics.csv")
        if df is None:
            values.append(float("nan"))
            continue
        row = df[df["metricName"] == "SM Throughput [Throughput %]"]
        values.append(float(row["avgVal"].iloc[0]) if not row.empty
                      else float("nan"))
    labels = [s.label for s in series_list]

    chart_dir = out_dir / "sm_throughput"
    chart_dir.mkdir(parents=True, exist_ok=True)
    base = "sm_throughput"

    finite = [v for v in values if np.isfinite(v)]
    y_max = (max(finite) * 1.20) if finite else 100.0

    written: list[Path] = []
    tex = _sm_throughput_tex(base=base, labels=labels,
                             values=values, y_max=y_max)
    written.append(_write_tex(chart_dir, base, tex))
    standalone = _write_tex(chart_dir, f"{base}.standalone",
                            _make_standalone_wrapper(base))
    written.append(standalone)
    pdf = _compile_standalone(standalone)
    if pdf is not None:
        written.append(pdf)
    return written


# ---------------------------------------------------------------------------
# 2) DRAM bandwidth — two bars per pipeline (read + write).
# ---------------------------------------------------------------------------

def _dram_bandwidth_tex(*, base: str, labels: list[str],
                        y_max: float) -> str:
    n = len(labels)
    xticks = ", ".join(str(i) for i in range(1, n + 1))
    xticklabels = ", ".join("{" + _tex_escape(lbl) + "}" for lbl in labels)
    return (
        f"% Auto-generated by scripts/plots/nsys.py - do not hand-edit.\n"
        f"% Insert with \\input{{{base}}} from a document whose preamble loads\n"
        f"% pgfplots (>= 1.16). Companion data file: {base}.dat\n"
        f"%   columns: x read_gbps write_gbps (one row per pipeline)\n"
        f"\\begin{{tikzpicture}}\n"
        f"  \\definecolor{{readbar}}{{HTML}}{{4C72B0}}\n"
        f"  \\definecolor{{writebar}}{{HTML}}{{7FB3D5}}\n"
        f"  \\begin{{axis}}[\n"
        f"    ybar, bar width={_BAR_WIDTH_CM}cm,\n"
        f"    width=12cm, height=7cm, scale only axis=true,\n"
        f"    enlarge x limits=0.20,\n"
        f"    ymin=0, ymax={y_max:.6g},\n"
        f"    xtick={{{xticks}}},\n"
        f"    xticklabels={{{xticklabels}}},\n"
        f"    xticklabel style={_DIAGONAL_XTICKLABEL_STYLE},\n"
        f"    ylabel={{DRAM bandwidth (GB/s)}},\n"
        f"    ylabel style={{font=\\small}},\n"
        f"    yticklabel style={{font=\\scriptsize}},\n"
        f"    title={{Average DRAM read + write bandwidth}},\n"
        f"    ymajorgrids=true,\n"
        f"    area legend,\n"
        f"    legend style={{at={{(0.5, {_LEGEND_ANCHOR_Y_DIAGONAL})}}, anchor=north,\n"
        f"                   draw=none, fill=none, font=\\scriptsize,\n"
        f"                   /tikz/every even column/.append style={{column sep=0.5cm}}}},\n"
        f"    legend columns=2,\n"
        f"    legend cell align=left,\n"
        f"    every axis plot/.append style={{point meta=rawy}},\n"
        f"    nodes near coords={{\\pgfmathprintnumber[fixed, precision=1]"
        f"\\pgfplotspointmeta}},\n"
        f"    nodes near coords style={{font=\\tiny, anchor=south}},\n"
        f"  ]\n"
        f"    \\addplot[fill=readbar, draw=black, line width=0.3pt]\n"
        f"      table[x=x, y=read_gbps] {{{base}.dat}};\n"
        f"    \\addlegendentry{{Read}}\n"
        f"    \\addplot[fill=writebar, draw=black, line width=0.3pt]\n"
        f"      table[x=x, y=write_gbps] {{{base}.dat}};\n"
        f"    \\addlegendentry{{Write}}\n"
        f"  \\end{{axis}}\n"
        f"\\end{{tikzpicture}}\n"
    )


def plot_dram_bandwidth(series_list: list[Series], out_dir: Path, *,
                        fmt: str = "pdf+png") -> list[Path]:
    """Tex/pdflatex: grouped bars per pipeline (read + write GB/s)."""
    del fmt
    series_list = _nsys_series(series_list)
    if not series_list:
        return []
    reads: list[float] = []
    writes: list[float] = []
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
    labels = [s.label for s in series_list]
    n = len(labels)

    chart_dir = out_dir / "dram_bandwidth"
    chart_dir.mkdir(parents=True, exist_ok=True)
    base = "dram_bandwidth"

    rows = np.column_stack([
        np.arange(1, n + 1, dtype=float),
        np.array(reads, dtype=float),
        np.array(writes, dtype=float),
    ])
    written: list[Path] = []
    _write_dat(chart_dir / f"{base}.dat",
               ["x", "read_gbps", "write_gbps"], rows)
    written.append(chart_dir / f"{base}.dat")

    finite = [v for v in (reads + writes) if np.isfinite(v)]
    y_max = (max(finite) * 1.20) if finite else 100.0
    tex = _dram_bandwidth_tex(base=base, labels=labels, y_max=y_max)
    written.append(_write_tex(chart_dir, base, tex))
    standalone = _write_tex(chart_dir, f"{base}.standalone",
                            _make_standalone_wrapper(base))
    written.append(standalone)
    pdf = _compile_standalone(standalone)
    if pdf is not None:
        written.append(pdf)
    return written


# ---------------------------------------------------------------------------
# 3) Cache hit rates — grouped bars (3 categories × N pipelines).
# ---------------------------------------------------------------------------

_CACHE_CATEGORIES = ("L1 Hit Rate [Ratio %]",
                     "L2 Hit Rate [Ratio %]",
                     "L2 Hit Rate from L1 [Ratio %]")
_CACHE_SHORT = ("L1", "L2", "L2 from L1")


def _cache_hit_rates_tex(*, base: str,
                         series_meta: list[tuple[str, str, str]]) -> str:
    """series_meta: list of (label, color_hex, dat_filename) tuples."""
    color_defs = "\n".join(
        f"  \\definecolor{{series{i}}}{{HTML}}{{{color.lstrip('#').upper()}}}"
        for i, (_, color, _) in enumerate(series_meta)
    )
    n_cats = len(_CACHE_SHORT)
    xticks = ", ".join(str(i) for i in range(1, n_cats + 1))
    xticklabels = ", ".join("{" + _tex_escape(s) + "}" for s in _CACHE_SHORT)
    plot_lines: list[str] = []
    for i, (label, _, dat) in enumerate(series_meta):
        plot_lines.append(
            f"    \\addplot[fill=series{i}, draw=black, line width=0.25pt]\n"
            f"      table[x=cat_idx, y=hit_pct] {{{dat}}};\n"
            f"    \\addlegendentry{{{_tex_escape(label)}}}"
        )
    plots_str = "\n".join(plot_lines)
    return (
        f"% Auto-generated by scripts/plots/nsys.py - do not hand-edit.\n"
        f"% Insert with \\input{{{base}}} from a document whose preamble loads\n"
        f"% pgfplots (>= 1.16). One companion .dat per series, named\n"
        f"% `{base}__<label>.dat`, with columns: cat_idx hit_pct.\n"
        f"\\begin{{tikzpicture}}\n"
        f"{color_defs}\n"
        f"  \\begin{{axis}}[\n"
        f"    ybar, bar width={_BAR_WIDTH_CM}cm,\n"
        f"    width=12cm, height=7cm, scale only axis=true,\n"
        f"    enlarge x limits=0.18,\n"
        f"    ymin=0, ymax=100,\n"
        f"    xtick={{{xticks}}},\n"
        f"    xticklabels={{{xticklabels}}},\n"
        f"    xticklabel style={{font=\\small}},\n"
        f"    ylabel={{Hit rate (\\%)}},\n"
        f"    ylabel style={{font=\\small}},\n"
        f"    yticklabel style={{font=\\scriptsize}},\n"
        f"    title={{Cache hit rates per pipeline}},\n"
        f"    ymajorgrids=true,\n"
        f"    area legend,\n"
        f"    legend style={{at={{(0.5, -0.18)}}, anchor=north,\n"
        f"                   draw=none, fill=none, font=\\scriptsize,\n"
        f"                   /tikz/every even column/.append style={{column sep=0.5cm}}}},\n"
        f"    legend columns={min(len(series_meta), 3)},\n"
        f"    legend cell align=left,\n"
        f"  ]\n"
        f"{plots_str}\n"
        f"  \\end{{axis}}\n"
        f"\\end{{tikzpicture}}\n"
    )


def plot_cache_hit_rates(series_list: list[Series], out_dir: Path, *,
                         fmt: str = "pdf+png") -> list[Path]:
    """Tex/pdflatex: L1 / L2 / L2-from-L1 grouped bars per pipeline."""
    del fmt
    series_list = _nsys_series(series_list)
    if not series_list:
        return []
    chart_dir = out_dir / "cache_hit_rates"
    chart_dir.mkdir(parents=True, exist_ok=True)
    base = "cache_hit_rates"

    written: list[Path] = []
    series_meta: list[tuple[str, str, str]] = []
    variant_hex = _series_variant_colors_hex(series_list)
    for s, color_hex in zip(series_list, variant_hex):
        df = load_nsys_csv(s, "08_per_pipeline_cache_hit_rates.csv")
        row_vals: list[float] = []
        for cat in _CACHE_CATEGORIES:
            if df is None:
                row_vals.append(float("nan"))
            else:
                m = df[df["metric"] == cat]
                row_vals.append(float(m["avgPct"].iloc[0]) if not m.empty
                                else float("nan"))
        rows = np.column_stack([
            np.arange(1, len(_CACHE_CATEGORIES) + 1, dtype=float),
            np.array(row_vals, dtype=float),
        ])
        dat_name = f"{base}__{_safe_filename(s.label)}.dat"
        _write_dat(chart_dir / dat_name, ["cat_idx", "hit_pct"], rows)
        written.append(chart_dir / dat_name)
        # `_series_variant_colors_hex` returns hex without leading '#'; the
        # tex helper re-strips/uppercases, so pass it in '#'-prefixed form
        # to match the prior contract.
        series_meta.append((s.label, f"#{color_hex}", dat_name))

    tex = _cache_hit_rates_tex(base=base, series_meta=series_meta)
    written.append(_write_tex(chart_dir, base, tex))
    standalone = _write_tex(chart_dir, f"{base}.standalone",
                            _make_standalone_wrapper(base))
    written.append(standalone)
    pdf = _compile_standalone(standalone)
    if pdf is not None:
        written.append(pdf)
    return written


# ---------------------------------------------------------------------------
# 4) Warp occupancy — stacked bars per pipeline.
# ---------------------------------------------------------------------------

# (metric, short label, color hex)
_WARP_STACKS: tuple[tuple[str, str, str], ...] = (
    ("Vertex/Tess/Geometry Warps [Throughput %]",        "VTG",         "#4C72B0"),
    ("Pixel Warps [Throughput %]",                       "Pixel",       "#DD8452"),
    ("Compute Warps [Throughput %]",                     "Compute",     "#55A467"),
    ("Unallocated Warps in Active SMs [Throughput %]",   "Unallocated", "#C44E52"),
    ("Idle SM Unused Warp Slots [Throughput %]",         "IdleSlots",   "#8172B2"),
)


def _warp_occupancy_tex(*, base: str, labels: list[str],
                        y_max: float) -> str:
    n = len(labels)
    color_defs = "\n".join(
        f"  \\definecolor{{stack_{short}}}{{HTML}}{{{color.lstrip('#').upper()}}}"
        for _, short, color in _WARP_STACKS
    )
    xticks = ", ".join(str(i) for i in range(1, n + 1))
    xticklabels = ", ".join("{" + _tex_escape(lbl) + "}" for lbl in labels)
    plot_lines: list[str] = []
    legend_lines: list[str] = []
    for _, short, _color in _WARP_STACKS:
        plot_lines.append(
            f"    \\addplot[fill=stack_{short}, draw=black, line width=0.25pt,\n"
            f"            forget plot]\n"
            f"      table[x=x, y={short}] {{{base}.dat}};"
        )
        legend_lines.append(
            f"    \\addlegendimage{{\n"
            f"      legend image code/.code={{%\n"
            f"        \\fill[stack_{short}]\n"
            f"          (0cm,-0.08cm) rectangle (0.55cm,0.17cm);\n"
            f"      }},\n"
            f"    }}\n"
            f"    \\addlegendentry{{{_tex_escape(short)}}}"
        )
    plots_str = "\n".join(plot_lines + legend_lines)
    columns = " ".join(short for _, short, _ in _WARP_STACKS)
    return (
        f"% Auto-generated by scripts/plots/nsys.py - do not hand-edit.\n"
        f"% Insert with \\input{{{base}}} from a document whose preamble loads\n"
        f"% pgfplots (>= 1.16). Companion data file: {base}.dat\n"
        f"%   columns: x {columns}\n"
        f"\\begin{{tikzpicture}}\n"
        f"{color_defs}\n"
        f"  \\begin{{axis}}[\n"
        f"    ybar stacked, bar width={_BAR_WIDTH_CM}cm,\n"
        f"    width=12cm, height=7.5cm, scale only axis=true,\n"
        f"    enlarge x limits=0.25,\n"
        f"    ymin=0, ymax={y_max:.6g},\n"
        f"    xtick={{{xticks}}},\n"
        f"    xticklabels={{{xticklabels}}},\n"
        f"    xticklabel style={_DIAGONAL_XTICKLABEL_STYLE},\n"
        f"    ylabel={{Throughput (\\%)}},\n"
        f"    ylabel style={{font=\\small}},\n"
        f"    yticklabel style={{font=\\scriptsize}},\n"
        f"    title={{Warp class occupancy per pipeline}},\n"
        f"    ymajorgrids=true,\n"
        f"    area legend,\n"
        f"    legend style={{at={{(0.5, {_LEGEND_ANCHOR_Y_DIAGONAL})}}, anchor=north,\n"
        f"                   draw=none, fill=none, font=\\scriptsize,\n"
        f"                   /tikz/every even column/.append style={{column sep=0.4cm}}}},\n"
        f"    legend columns={len(_WARP_STACKS)},\n"
        f"    legend cell align=left,\n"
        f"  ]\n"
        f"{plots_str}\n"
        f"  \\end{{axis}}\n"
        f"\\end{{tikzpicture}}\n"
    )


def plot_warp_occupancy(series_list: list[Series], out_dir: Path, *,
                        fmt: str = "pdf+png") -> list[Path]:
    """Tex/pdflatex: stacked warp-class occupancy per pipeline."""
    del fmt
    series_list = _nsys_series(series_list)
    if not series_list:
        return []
    n = len(series_list)
    # values_by_short[short][i] = value for series i
    values_by_short: dict[str, list[float]] = {
        short: [] for _, short, _ in _WARP_STACKS
    }
    for s in series_list:
        df = load_nsys_csv(s, "12_per_pipeline_warp_occupancy.csv")
        for metric, short, _ in _WARP_STACKS:
            if df is None:
                values_by_short[short].append(float("nan"))
            else:
                m = df[df["metric"] == metric]
                values_by_short[short].append(
                    float(m["avgVal"].iloc[0]) if not m.empty
                    else float("nan"))

    labels = [s.label for s in series_list]
    chart_dir = out_dir / "warp_occupancy"
    chart_dir.mkdir(parents=True, exist_ok=True)
    base = "warp_occupancy"

    # Build a single .dat with x and one column per stack short name.
    cols: list[np.ndarray] = [np.arange(1, n + 1, dtype=float)]
    header: list[str] = ["x"]
    for _, short, _ in _WARP_STACKS:
        # NaN -> 0 so pgfplots stacks cleanly.
        col = np.array(values_by_short[short], dtype=float)
        col = np.nan_to_num(col, nan=0.0)
        cols.append(col)
        header.append(short)
    rows = np.column_stack(cols)
    written: list[Path] = []
    _write_dat(chart_dir / f"{base}.dat", header, rows)
    written.append(chart_dir / f"{base}.dat")

    # Y-max = ceil to next 25% above stack total, with a floor of 100.
    totals = rows[:, 1:].sum(axis=1)
    y_max = max(100.0, float(np.ceil(totals.max() / 25.0) * 25.0)) \
        if totals.size else 100.0

    tex = _warp_occupancy_tex(base=base, labels=labels, y_max=y_max)
    written.append(_write_tex(chart_dir, base, tex))
    standalone = _write_tex(chart_dir, f"{base}.standalone",
                            _make_standalone_wrapper(base))
    written.append(standalone)
    pdf = _compile_standalone(standalone)
    if pdf is not None:
        written.append(pdf)
    return written


# ---------------------------------------------------------------------------
# 5) PIX stages — per geometry (Trunk / Leaves / Terrain / Impostors),
#    grouped by pipeline, main+shadow stacked.
# ---------------------------------------------------------------------------

GEOMETRY_KINDS = ("Trunk", "Leaves", "Terrain", "Impostors")


def _pix_stages_tex(*, base: str, active: list[str],
                    series_meta: list[tuple[str, str, str, str]],
                    y_min: float, y_max: float) -> str:
    """series_meta: list of (label, main_color_hex, shadow_color_hex, dat_filename)"""
    color_defs_lines: list[str] = []
    for i, (_, main, shadow, _) in enumerate(series_meta):
        color_defs_lines.append(
            f"  \\definecolor{{main{i}}}{{HTML}}{{{main.lstrip('#').upper()}}}"
        )
        color_defs_lines.append(
            f"  \\definecolor{{shadow{i}}}{{HTML}}{{{shadow.lstrip('#').upper()}}}"
        )
    color_defs = "\n".join(color_defs_lines)
    n_geo = len(active)
    xticks = ", ".join(str(i) for i in range(1, n_geo + 1))
    xticklabels = ", ".join("{" + _tex_escape(s) + "}" for s in active)

    # Per pipeline: a paired set of side-by-side bars (main left, shadow
    # right). bar_width is half of _BAR_WIDTH_CM so the pair occupies roughly
    # one standard slot; pairs are then tiled across each geometry via
    # bar shift. inner_gap separates main from shadow within a pair; the
    # pair_step builds in outer_gap between pipelines.
    sub_width_cm = _BAR_WIDTH_CM / 2.0
    inner_gap_cm = 0.02
    outer_gap_cm = 0.06
    pair_step_cm = 2.0 * sub_width_cm + inner_gap_cm + outer_gap_cm
    plot_lines: list[str] = []
    legend_lines: list[str] = []
    n_series = len(series_meta)
    for i, (label, _main, _shadow, dat) in enumerate(series_meta):
        pair_center_cm = (i - (n_series - 1) / 2.0) * pair_step_cm
        main_shift_cm = (pair_center_cm
                         - sub_width_cm / 2.0 - inner_gap_cm / 2.0)
        shadow_shift_cm = (pair_center_cm
                           + sub_width_cm / 2.0 + inner_gap_cm / 2.0)
        plot_lines.append(
            f"    % {_tex_escape(label)} (main left, shadow right)\n"
            f"    \\addplot[ybar, fill=main{i}, draw=black, line width=0.25pt,\n"
            f"            bar width={sub_width_cm:.3f}cm,"
            f" bar shift={main_shift_cm:.3f}cm,\n"
            f"            forget plot]\n"
            f"      table[x=geo_idx, y=main_ms] {{{dat}}};\n"
            f"    \\addplot[ybar, fill=shadow{i}, draw=black, line width=0.25pt,\n"
            f"            bar width={sub_width_cm:.3f}cm,"
            f" bar shift={shadow_shift_cm:.3f}cm,\n"
            f"            forget plot]\n"
            f"      table[x=geo_idx, y=shadow_ms] {{{dat}}};"
        )
        legend_lines.append(
            f"    \\addlegendimage{{\n"
            f"      legend image code/.code={{%\n"
            f"        \\fill[main{i}]\n"
            f"          (0cm,-0.08cm) rectangle (0.55cm,0.17cm);\n"
            f"      }},\n"
            f"    }}\n"
            f"    \\addlegendentry{{{_tex_escape(label)} (main)}}"
        )
        legend_lines.append(
            f"    \\addlegendimage{{\n"
            f"      legend image code/.code={{%\n"
            f"        \\fill[shadow{i}]\n"
            f"          (0cm,-0.08cm) rectangle (0.55cm,0.17cm);\n"
            f"      }},\n"
            f"    }}\n"
            f"    \\addlegendentry{{{_tex_escape(label)} (shadow)}}"
        )
    plots_str = "\n".join(plot_lines + legend_lines)
    return (
        f"% Auto-generated by scripts/plots/nsys.py - do not hand-edit.\n"
        f"% Insert with \\input{{{base}}} from a document whose preamble loads\n"
        f"% pgfplots (>= 1.16). One companion .dat per series, named\n"
        f"% `{base}__<label>.dat`, with columns: geo_idx main_ms shadow_ms\n"
        f"% total_ms. The chart consumes main_ms and shadow_ms (absolute GPU\n"
        f"% time in ms summed across the run), drawn side-by-side per pipeline.\n"
        f"\\begin{{tikzpicture}}\n"
        f"{color_defs}\n"
        f"  \\begin{{axis}}[\n"
        f"    ybar,\n"
        f"    width=13cm, height=7.5cm, scale only axis=true,\n"
        f"    enlarge x limits=0.15,\n"
        f"    ymode=log, log basis y=10,\n"
        f"    ymin={y_min:.6g}, ymax={y_max:.6g},\n"
        f"    xtick={{{xticks}}},\n"
        f"    xticklabels={{{xticklabels}}},\n"
        f"    xticklabel style={{font=\\small}},\n"
        f"    ylabel={{GPU time (ms, log)}},\n"
        f"    ylabel style={{font=\\small}},\n"
        f"    yticklabel style={{font=\\scriptsize}},\n"
        f"    yticklabel={{1.0e\\pgfmathprintnumber[print sign,fixed,precision=0]{{\\tick}}}},\n"
        f"    title={{GPU time by geometry}},\n"
        f"    ymajorgrids=true,\n"
        f"    yminorgrids=true,\n"
        f"    minor grid style={{densely dotted, gray!50, line width=0.3pt}},\n"
        f"    area legend,\n"
        f"    legend style={{at={{(0.5, -0.20)}}, anchor=north,\n"
        f"                   draw=none, fill=none, font=\\scriptsize,\n"
        f"                   /tikz/every even column/.append style={{column sep=0.4cm}}}},\n"
        f"    legend columns={max(1, n_series)},\n"
        f"    legend cell align=left,\n"
        f"  ]\n"
        f"{plots_str}\n"
        f"  \\end{{axis}}\n"
        f"\\end{{tikzpicture}}\n"
    )


def plot_pix_stages(series_list: list[Series], out_dir: Path, *,
                    fmt: str = "pdf+png") -> list[Path]:
    """Tex/pdflatex: per-geometry total GPU time (ms), grouped per pipeline.

    Per (pipeline, geometry) cell: main bar (left) + shadow bar (right) in
    absolute ms summed across the recorded NVTX window. Plotting ms instead
    of pipeline-window fractions preserves the relative cost between
    pipelines (faster pipelines have shorter bars)."""
    del fmt
    series_list = _nsys_series(series_list)
    if not series_list:
        return []
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

    active = [k for k in GEOMETRY_KINDS
              if any(main_ms[k][i] + shadow_ms[k][i] > 0
                     for i in range(len(series_list)))]
    if not active:
        return []

    chart_dir = out_dir / "pix_stages"
    chart_dir.mkdir(parents=True, exist_ok=True)
    base = "pix_stages"

    variant_hex = _series_variant_colors_hex(series_list)

    written: list[Path] = []
    series_meta: list[tuple[str, str, str, str]] = []
    y_max_val = 0.0
    y_min_val: float | None = None
    for i, s in enumerate(series_list):
        m_ms = np.array([main_ms[k][i] for k in active], dtype=float)
        s_ms = np.array([shadow_ms[k][i] for k in active], dtype=float)
        t_ms = m_ms + s_ms
        rows = np.column_stack([
            np.arange(1, len(active) + 1, dtype=float),
            m_ms, s_ms, t_ms,
        ])
        dat_name = f"{base}__{_safe_filename(s.label)}.dat"
        _write_dat(chart_dir / dat_name,
                   ["geo_idx", "main_ms", "shadow_ms", "total_ms"], rows)
        written.append(chart_dir / dat_name)
        # Bars are side-by-side, so the tallest bar is max(main, shadow)
        # for any (series, geometry) pair -- not the stacked total.
        finite_bars = np.concatenate([m_ms, s_ms])
        finite_bars = finite_bars[np.isfinite(finite_bars)]
        if finite_bars.size:
            y_max_val = max(y_max_val, float(finite_bars.max()))
        pos_bars = finite_bars[finite_bars > 0]
        if pos_bars.size:
            y_min_val = (float(pos_bars.min()) if y_min_val is None
                         else min(y_min_val, float(pos_bars.min())))
        # Main color tracks the variant (no-Hi-Z = darker); shadow color is
        # derived as a lightened version of the actual main so the
        # main/shadow relationship holds across variants too.
        c_main = f"#{variant_hex[i]}"
        c_shadow = _lighten_hex(c_main, amount=0.25)
        series_meta.append((s.label, c_main, c_shadow, dat_name))

    # Snap log-axis bounds to decade boundaries for clean 1eN tick labels.
    y_max_raw = y_max_val if y_max_val > 0 else 1.0
    y_max = 10.0 ** math.ceil(math.log10(y_max_raw))
    y_min_raw = y_min_val if y_min_val is not None else 0.1
    y_min = 10.0 ** math.floor(math.log10(y_min_raw))

    tex = _pix_stages_tex(base=base, active=list(active),
                          series_meta=series_meta,
                          y_min=y_min, y_max=y_max)
    written.append(_write_tex(chart_dir, base, tex))
    standalone = _write_tex(chart_dir, f"{base}.standalone",
                            _make_standalone_wrapper(base))
    written.append(standalone)
    pdf = _compile_standalone(standalone)
    if pdf is not None:
        written.append(pdf)
    return written
