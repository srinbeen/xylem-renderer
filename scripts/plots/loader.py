"""Discovers benchmark series on disk and loads per-frame + nsys CSVs."""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd


# Stage columns the BenchmarkRunner writes; -1 in any of these means
# "this pipeline does not run that stage" and is masked to NaN at load.
STAGE_COLS: tuple[str, ...] = (
    "depthPrepassMs", "hizMs", "sdsmMs",
    "cullMs", "shadowMs", "skyMs", "sceneMs",
)

# Maps CSV filename (lowercase) to the canonical pipeline label used everywhere
# downstream (palette, legends, nsys NVTX marker filtering).
PIPELINE_BY_CSV: dict[str, str] = {
    "traditional.csv": "Traditional",
    "compute.csv":     "Compute",
    "meshshader.csv":  "MeshShader",
}


@dataclass
class Series:
    label: str
    run_dir: Path
    pipeline: str
    frames: pd.DataFrame
    nsys_dir: Optional[Path]
    summary: dict = field(default_factory=dict)


def _load_summary(run_dir: Path) -> dict:
    p = run_dir / "run_summary.json"
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        warnings.warn(f"run_summary.json unreadable in {run_dir}: {e}")
        return {}


def _load_frames(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Mask -1 stage cells to NaN so .mean() / stacked-area auto-ignore them.
    for col in STAGE_COLS:
        if col in df.columns:
            df.loc[df[col] < 0, col] = float("nan")
    # gpuMs is -1 on rows where the late-binding never fired; also mask.
    if "gpuMs" in df.columns:
        df.loc[df["gpuMs"] < 0, "gpuMs"] = float("nan")
    # Add derived `fps` column for convenience (NaN where gpuMs is NaN).
    df["fps"] = 1000.0 / df["gpuMs"]
    return df


def discover(run_dirs: list[Path], *, include_partial: bool = False) -> list[Series]:
    """Walk each run folder, yield one Series per (run × completed pipeline).

    Skips `.partial` CSVs unless include_partial=True. Missing CSVs are
    silently skipped (the run_summary.json's `csv: null` case).
    """
    out: list[Series] = []
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        if not run_dir.is_dir():
            warnings.warn(f"Not a directory, skipping: {run_dir}")
            continue
        summary = _load_summary(run_dir)
        nsys_dir = run_dir / "nsys_results"
        if not nsys_dir.is_dir():
            nsys_dir = None

        for csv_name, pipeline in PIPELINE_BY_CSV.items():
            candidates = [run_dir / csv_name]
            if include_partial:
                candidates.append(run_dir / f"{csv_name}.partial")
            for csv_path in candidates:
                if not csv_path.is_file():
                    continue
                try:
                    frames = _load_frames(csv_path)
                except Exception as e:
                    warnings.warn(f"Failed to read {csv_path}: {e}")
                    continue
                if frames.empty:
                    warnings.warn(f"Empty CSV, skipping: {csv_path}")
                    continue
                label = f"{run_dir.name} / {pipeline}"
                out.append(Series(
                    label=label,
                    run_dir=run_dir,
                    pipeline=pipeline,
                    frames=frames,
                    nsys_dir=nsys_dir,
                    summary=summary,
                ))
                break  # don't double-add the same pipeline from both completed + partial
    return out


def load_nsys_csv(series: Series, name: str) -> Optional[pd.DataFrame]:
    """Read one of the nsys_results/*.csv files and filter to this series' pipeline.

    Returns None if the file is missing or no rows match this pipeline.
    Caller is responsible for downstream column handling; this just opens
    the file and applies the NVTX window filter.
    """
    if series.nsys_dir is None:
        return None
    p = series.nsys_dir / name
    if not p.is_file():
        return None
    df = pd.read_csv(p)
    if "pipeline" not in df.columns:
        # Catalog / overview CSVs (00_, 01_, 02_, 03_) have no pipeline column;
        # caller is expected to use them as-is.
        return df
    nvtx_marker = f"Pipeline:{series.pipeline}"
    filtered = df[df["pipeline"] == nvtx_marker]
    if filtered.empty:
        return None
    return filtered.reset_index(drop=True)
