"""CLI entrypoint: discover benchmark runs and emit the plot catalog."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python scripts/plots/plot_benchmarks.py ...` to find the sibling
# modules without requiring the user to set PYTHONPATH.
_HERE = Path(__file__).resolve().parent
if str(_HERE.parent.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent.parent))

from scripts.plots import per_frame, nsys
from scripts.plots.loader import discover


PLOT_REGISTRY = {
    "fps_tex":         (per_frame.plot_fps_over_time_tex,           "per_frame"),
    "cdf_tex":         (per_frame.plot_frame_time_cdf_tex,          "per_frame"),
    "bar_tex":         (per_frame.plot_cpu_gpu_bar_tex,             "per_frame"),
    "stage_area_tex":  (per_frame.plot_stage_stacked_area_tex,      "per_frame"),
    "stage_bar_tex":   (per_frame.plot_stage_grouped_bar_tex,       "per_frame"),
    "frame_stats":     (per_frame.plot_frame_stats_table,           "per_frame"),
    "stage_stats":     (per_frame.plot_stage_stats_table,           "per_frame"),
    "path3d_tex":      (per_frame.plot_camera_path_3d_tex,          "per_frame"),
    "instance_visibility_tex":     (per_frame.plot_instance_visibility_tex,     "per_frame"),
    "instance_lod_frac":           (per_frame.plot_instance_lod_fraction_tex,   "per_frame"),
    "correlation_terrain_tex":     (per_frame.plot_terrain_shadow_meshlet_scatter_tex, "per_frame"),
    "meshlet_visibility_tex":      (per_frame.plot_meshlet_visibility_tex,      "per_frame"),
    "instance_visibility_fps_tex": (per_frame.plot_instance_visibility_fps_tex, "per_frame"),
    "meshlet_visibility_fps_tex":  (per_frame.plot_meshlet_visibility_fps_tex,  "per_frame"),
    "sun_tex":                     (per_frame.plot_sun_elevation_tex,           "per_frame"),
    "vis_stats":                   (per_frame.plot_vis_stats_table,             "per_frame"),
    "sm":     (nsys.plot_sm_throughput,   "nsys"),
    "dram":   (nsys.plot_dram_bandwidth,  "nsys"),
    "cache":  (nsys.plot_cache_hit_rates, "nsys"),
    "warps":  (nsys.plot_warp_occupancy,  "nsys"),
    "pix":    (nsys.plot_pix_stages,      "nsys"),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path,
                        help="One or more benchmarks/<timestamp>/ directories")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output directory for plots")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Override series labels (must match the discovered count)")
    parser.add_argument("--pipelines", default="traditional,compute,meshshader",
                        help="Comma-separated pipelines to include (default: all three)")
    parser.add_argument("--include-partial", action="store_true",
                        help="Also consume .partial CSVs")
    parser.add_argument("--plots", default=None,
                        help="Comma-separated plot keys to emit (default: all). "
                             f"Valid keys: {','.join(PLOT_REGISTRY.keys())}")
    parser.add_argument("--path-color-by", default="fps",
                        help="Column or derived metric used to color the camera path")
    parser.add_argument("--path-arrow-stride", type=int, default=30,
                        help="Subsample stride for direction arrows on the camera path")
    args = parser.parse_args(argv)

    series_list = discover(args.run_dirs, include_partial=args.include_partial)
    if not series_list:
        print("No series discovered. Check the run folder(s).", file=sys.stderr)
        return 1

    # Pipeline filter (post-discovery so warnings about empty CSVs still fire).
    requested = {p.strip().lower() for p in args.pipelines.split(",")}
    pipeline_lower = {"Traditional": "traditional",
                      "Compute":     "compute",
                      "MeshShader":  "meshshader"}
    series_list = [s for s in series_list if pipeline_lower[s.pipeline] in requested]
    if not series_list:
        print(f"No series matched --pipelines={args.pipelines}", file=sys.stderr)
        return 1

    # Label override.
    if args.labels is not None:
        if len(args.labels) != len(series_list):
            print(f"--labels arity mismatch: got {len(args.labels)} labels for "
                  f"{len(series_list)} series", file=sys.stderr)
            return 2
        for s, lbl in zip(series_list, args.labels):
            s.label = lbl

    # Plot selection.
    plot_keys = (list(PLOT_REGISTRY.keys()) if args.plots is None
                 else [k.strip() for k in args.plots.split(",")])
    unknown = [k for k in plot_keys if k not in PLOT_REGISTRY]
    if unknown:
        print(f"Unknown plot key(s): {unknown}", file=sys.stderr)
        return 2

    print(f"Discovered {len(series_list)} series:")
    for s in series_list:
        nsys_str = "nsys" if s.nsys_dir else "no nsys"
        print(f"  - {s.label} ({len(s.frames)} rows, {nsys_str})")

    manifest: dict = {
        "out": str(args.out),
        "run_dirs": [str(r) for r in args.run_dirs],
        "series": [
            {"label": s.label, "run_dir": str(s.run_dir),
             "pipeline": s.pipeline,
             "frames": int(len(s.frames)),
             "nsys": (s.nsys_dir is not None)}
            for s in series_list
        ],
        "plots": {},
        "options": {
            "pipelines": args.pipelines,
            "include_partial": args.include_partial,
            "path_color_by": args.path_color_by,
            "path_arrow_stride": args.path_arrow_stride,
        },
    }

    for name in plot_keys:
        fn, subdir = PLOT_REGISTRY[name]
        target = args.out / subdir
        kwargs: dict = {}
        if name == "path3d_tex":
            kwargs["color_by"] = args.path_color_by
            kwargs["arrow_stride"] = args.path_arrow_stride
        written = fn(series_list, target, **kwargs)
        manifest["plots"][name] = [str(p) for p in written]
        for p in written:
            print(f"  wrote {p}")

    args.out.mkdir(parents=True, exist_ok=True)
    import json
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest: {args.out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
