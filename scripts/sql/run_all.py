#!/usr/bin/env python3
"""Run every NSight Systems analysis query against a report .sqlite and dump CSVs.

Tailored for the *Graphics Throughput Metrics* preset (~124 counters) — see
NSight Systems > GPU Metrics > Sampling buffer overflow notes if some samples
appear dropped. Queries are filtered by counter *name* (not metricId) so the
script is robust against the GUI re-ordering metrics across captures.

Usage:
    python run_all.py --db <path-to-report.sqlite> --out <output-dir>

The DB is opened read-only. Output directory is created if missing.
One CSV is written per query, named `<query-name>.csv`.
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
import time
from pathlib import Path

# Each query is (name, sql). Order controls CSV file ordering on disk.
QUERIES: list[tuple[str, str]] = [
    (
        "00_session_overview",
        # Capture start time + range derived from GPU_METRICS timestamps.
        """
        SELECT
          s.utcEpochNs                                              AS sessionStartEpochNs,
          s.utcTime                                                 AS utcTime,
          s.localTime                                               AS localTime,
          (SELECT MIN(timestamp) FROM GPU_METRICS)                  AS captureMinTsNs,
          (SELECT MAX(timestamp) FROM GPU_METRICS)                  AS captureMaxTsNs,
          ROUND(((SELECT MAX(timestamp) FROM GPU_METRICS) -
                 (SELECT MIN(timestamp) FROM GPU_METRICS)) / 1e9, 3) AS captureDurationSec
        FROM TARGET_INFO_SESSION_START_TIME s;
        """,
    ),
    (
        "01_gpu_info",
        # Discrete-GPU hardware fields. memoryBandwidth is the denominator for
        # the % → GB/s conversion in 06_per_pipeline_memory_bandwidth.
        """
        SELECT id,
               name,
               busLocation,
               isDiscrete,
               l2CacheSize,
               totalMemory,
               memoryBandwidth                      AS peakMemoryBandwidthBps,
               ROUND(memoryBandwidth / 1e9, 1)      AS peakDRAMBandwidthGBps,
               clockRate                            AS baseClockHz,
               ROUND(clockRate / 1e6, 0)            AS baseClockMHz,
               smCount,
               threadsPerWarp,
               maxWarpsPerSm,
               maxRegistersPerBlock,
               maxShmemPerBlock,
               maxShmemPerSm,
               computeMajor,
               computeMinor,
               smMajor,
               smMinor,
               chipName
        FROM TARGET_INFO_GPU
        WHERE isDiscrete = 1 AND memoryBandwidth > 0;
        """,
    ),
    (
        "02_metric_catalog",
        # All GPU counters configured during this capture. metricName is the
        # FK used by every per-counter query below.
        """
        SELECT metricId,
               metricName,
               typeName,
               typeId,
               sourceId
        FROM TARGET_INFO_GPU_METRICS
        ORDER BY metricId;
        """,
    ),
    (
        "03_pipeline_windows",
        # NVTX ranges the Xylem benchmark emits around each pipeline run.
        # These are the slicing key for every per-pipeline aggregate below.
        """
        SELECT text                              AS marker,
               start                             AS startNs,
               end                               AS endNs,
               ROUND((end - start) / 1e6, 3)     AS durationMs,
               ROUND((end - start) / 1e9, 3)     AS durationSec
        FROM NVTX_EVENTS
        WHERE text LIKE 'Pipeline:%' OR text = 'XylemBenchmark'
        ORDER BY start;
        """,
    ),
    (
        "04_metric_value_ranges",
        # Min/avg/max + sample count per metric across the full capture.
        # Sanity check on scale. Gaps between expected and actual sample
        # counts hint at GPU-Metrics buffer-overflow drops.
        """
        SELECT m.metricId,
               t.metricName,
               MIN(m.value)              AS valMin,
               ROUND(AVG(m.value), 3)    AS valAvg,
               MAX(m.value)              AS valMax,
               COUNT(*)                  AS samples
        FROM GPU_METRICS m
        JOIN TARGET_INFO_GPU_METRICS t ON t.metricId = m.metricId
        GROUP BY m.metricId
        ORDER BY m.metricId;
        """,
    ),
    (
        "05_per_pipeline_metrics",
        # The catch-all aggregate: avg/min/max of every metric per pipeline
        # window. Every more-specialised query below is a filtered view of
        # this same join.
        """
        WITH win(name, t0, t1) AS (
          SELECT text, start, end FROM NVTX_EVENTS WHERE text LIKE 'Pipeline:%'
        ),
        samples AS (
          SELECT w.name AS pipeline, g.metricId, g.value
          FROM GPU_METRICS g
          JOIN win w ON g.timestamp BETWEEN w.t0 AND w.t1
        )
        SELECT s.pipeline,
               s.metricId,
               t.metricName,
               ROUND(AVG(s.value), 3) AS avgVal,
               MIN(s.value)           AS minVal,
               MAX(s.value)           AS maxVal,
               COUNT(*)               AS samples
        FROM samples s
        JOIN TARGET_INFO_GPU_METRICS t ON t.metricId = s.metricId
        GROUP BY s.pipeline, s.metricId
        ORDER BY s.pipeline, s.metricId;
        """,
    ),
    (
        "06_per_pipeline_memory_bandwidth",
        # VRAM / DRAM bandwidth per pipeline. The Graphics preset reports % of
        # peak; we also derive absolute GB/s for the GPU Memory R/W metrics
        # using TARGET_INFO_GPU.memoryBandwidth as the peak denominator.
        """
        WITH gpu AS (
          SELECT memoryBandwidth AS peakBps FROM TARGET_INFO_GPU
           WHERE isDiscrete = 1 AND memoryBandwidth > 0 LIMIT 1
        ),
        win(name, t0, t1) AS (
          SELECT text, start, end FROM NVTX_EVENTS WHERE text LIKE 'Pipeline:%'
        )
        SELECT w.name                                   AS pipeline,
               t.metricName                             AS metric,
               ROUND(AVG(g.value), 3)                   AS avgPct,
               MAX(g.value)                             AS maxPct,
               CASE WHEN t.metricName IN (
                      'GPU Memory Read Bandwidth [Throughput %]',
                      'GPU Memory Write Bandwidth [Throughput %]')
                    THEN ROUND(AVG(g.value) * (SELECT peakBps FROM gpu) / 100.0 / 1e9, 2)
                    ELSE NULL END                       AS avgGBps,
               CASE WHEN t.metricName IN (
                      'GPU Memory Read Bandwidth [Throughput %]',
                      'GPU Memory Write Bandwidth [Throughput %]')
                    THEN ROUND(MAX(g.value) * (SELECT peakBps FROM gpu) / 100.0 / 1e9, 2)
                    ELSE NULL END                       AS maxGBps,
               ROUND((SELECT peakBps FROM gpu) / 1e9, 1) AS peakGBps
        FROM GPU_METRICS g
        JOIN TARGET_INFO_GPU_METRICS t ON t.metricId = g.metricId
        JOIN win w ON g.timestamp BETWEEN w.t0 AND w.t1
        WHERE t.metricName IN (
          'GPU Memory Read Bandwidth [Throughput %]',
          'GPU Memory Write Bandwidth [Throughput %]',
          'VRAM Throughput [Throughput %]',
          'L2 Bandwidth to VRAM [Throughput %]',
          'L2 Bandwidth to PCIe+Peer [Throughput %]'
        )
        GROUP BY w.name, t.metricName
        ORDER BY w.name, t.metricName;
        """,
    ),
    (
        "07_per_pipeline_pcie_bandwidth",
        # PCIe traffic per pipeline. The metric NAMED `[GB/s]` actually stores
        # bytes/s as an integer (Nsight scales up to keep precision), so we
        # divide by 1e9 in the output to surface real GB/s. The [Throughput %]
        # variant is left as-is. BAR1 request counts surface CPU-uploaded
        # resource traffic (instance buffers etc).
        """
        WITH win(name, t0, t1) AS (
          SELECT text, start, end FROM NVTX_EVENTS WHERE text LIKE 'Pipeline:%'
        )
        SELECT w.name                                   AS pipeline,
               t.metricName                             AS metric,
               CASE WHEN t.metricName LIKE '%[GB/s]'
                    THEN ROUND(AVG(g.value) / 1e9, 3)
                    ELSE ROUND(AVG(g.value), 3) END     AS avgVal,
               CASE WHEN t.metricName LIKE '%[GB/s]'
                    THEN ROUND(MAX(g.value) / 1e9, 3)
                    ELSE MAX(g.value) * 1.0 END         AS maxVal
        FROM GPU_METRICS g
        JOIN TARGET_INFO_GPU_METRICS t ON t.metricId = g.metricId
        JOIN win w ON g.timestamp BETWEEN w.t0 AND w.t1
        WHERE t.metricName IN (
          'PCIe Read Bandwidth [Throughput %]',
          'PCIe Read Bandwidth [GB/s]',
          'PCIe Write Bandwidth [Throughput %]',
          'PCIe Write Bandwidth [GB/s]',
          'PCIe Read Requests to BAR1 [Requests]',
          'PCIe Write Requests to BAR1 [Requests]'
        )
        GROUP BY w.name, t.metricName
        ORDER BY w.name, t.metricName;
        """,
    ),
    (
        "08_per_pipeline_cache_hit_rates",
        # L1 + L2 cache hit rates per pipeline. Direct signal on whether the
        # mega-buffer / persistent-instance / visibility-indirection layouts
        # are cache-friendly.
        """
        WITH win(name, t0, t1) AS (
          SELECT text, start, end FROM NVTX_EVENTS WHERE text LIKE 'Pipeline:%'
        )
        SELECT w.name                                   AS pipeline,
               t.metricName                             AS metric,
               ROUND(AVG(g.value), 2)                   AS avgPct,
               MIN(g.value)                             AS minPct,
               MAX(g.value)                             AS maxPct
        FROM GPU_METRICS g
        JOIN TARGET_INFO_GPU_METRICS t ON t.metricId = g.metricId
        JOIN win w ON g.timestamp BETWEEN w.t0 AND w.t1
        WHERE t.metricName IN (
          'L1 Hit Rate [Ratio %]',
          'L2 Hit Rate [Ratio %]',
          'L2 Hit Rate from L1 [Ratio %]'
        )
        GROUP BY w.name, t.metricName
        ORDER BY w.name, t.metricName;
        """,
    ),
    (
        "09_per_pipeline_sm_pipes",
        # SM instruction-issue + per-pipe throughput. ALU = int + bitwise,
        # FMA Light = fp32 add/multiply, FMA Heavy = fma/double, SFU =
        # transcendentals (sin/cos/rsqrt/exp). A high % here = pipe pressure.
        """
        WITH win(name, t0, t1) AS (
          SELECT text, start, end FROM NVTX_EVENTS WHERE text LIKE 'Pipeline:%'
        )
        SELECT w.name                                   AS pipeline,
               t.metricName                             AS metric,
               ROUND(AVG(g.value), 2)                   AS avgPct,
               MAX(g.value)                             AS maxPct
        FROM GPU_METRICS g
        JOIN TARGET_INFO_GPU_METRICS t ON t.metricId = g.metricId
        JOIN win w ON g.timestamp BETWEEN w.t0 AND w.t1
        WHERE t.metricName IN (
          'SM Issue Active [Throughput %]',
          'SM ALU Pipe Throughput [Throughput %]',
          'SM FMA Light Pipe Throughput [Throughput %]',
          'SM FMA Heavy Pipe Throughput [Throughput %]',
          'SM SFU Pipe Throughput [Throughput %]'
        )
        GROUP BY w.name, t.metricName
        ORDER BY w.name, t.metricName;
        """,
    ),
    (
        "10_per_pipeline_fixed_function",
        # Throughput of the fixed-function graphics units:
        #   VAF      = Vertex Attribute Fetch
        #   PD       = Primitive Distributor
        #   PES+VPC  = Polymorph Engine + Viewport-Clip-Cull
        #   RASTER   = rasteriser
        #   PROP     = Pre-ROP (early-Z/blending fetch)
        #   ZROP     = depth-test + write
        #   CROP     = colour-blend + write
        """
        WITH win(name, t0, t1) AS (
          SELECT text, start, end FROM NVTX_EVENTS WHERE text LIKE 'Pipeline:%'
        )
        SELECT w.name                                   AS pipeline,
               t.metricName                             AS metric,
               ROUND(AVG(g.value), 2)                   AS avgPct,
               MAX(g.value)                             AS maxPct
        FROM GPU_METRICS g
        JOIN TARGET_INFO_GPU_METRICS t ON t.metricId = g.metricId
        JOIN win w ON g.timestamp BETWEEN w.t0 AND w.t1
        WHERE t.metricName IN (
          'VAF Throughput [Throughput %]',
          'PD Throughput [Throughput %]',
          'PES+VPC Throughput [Throughput %]',
          'RASTER Throughput [Throughput %]',
          'PROP Throughput [Throughput %]',
          'ZROP Throughput [Throughput %]',
          'CROP Throughput [Throughput %]'
        )
        GROUP BY w.name, t.metricName
        ORDER BY w.name, t.metricName;
        """,
    ),
    (
        "12_per_pipeline_warp_occupancy",
        # Warp-class occupancy per pipeline. VTG / Pixel / Compute = warps in
        # flight by shader class; Unallocated = SM has room but nothing to
        # schedule there; Idle SM Unused = SM not even active. Active Thread
        # Groups = compute CTA occupancy. Active Threads Per Warp = warp
        # divergence (32 = no divergence; lower = some lanes masked off).
        """
        WITH win(name, t0, t1) AS (
          SELECT text, start, end FROM NVTX_EVENTS WHERE text LIKE 'Pipeline:%'
        )
        SELECT w.name                                   AS pipeline,
               t.metricName                             AS metric,
               ROUND(AVG(g.value), 2)                   AS avgVal,
               MAX(g.value)                             AS maxVal
        FROM GPU_METRICS g
        JOIN TARGET_INFO_GPU_METRICS t ON t.metricId = g.metricId
        JOIN win w ON g.timestamp BETWEEN w.t0 AND w.t1
        -- 'Active Threads Per Warp [Threads/Warp]' stores a raw aggregated
        -- counter (~10^9), not a per-warp average. The [Coherence] variant
        -- is the readable %-of-full-coherence number; we keep that one only.
        WHERE t.metricName IN (
          'Vertex/Tess/Geometry Warps [Throughput %]',
          'Pixel Warps [Throughput %]',
          'Compute Warps [Throughput %]',
          'Unallocated Warps in Active SMs [Throughput %]',
          'Idle SM Unused Warp Slots [Throughput %]',
          'Active Thread Groups in SM [Throughput %]',
          'Active Threads Per Warp [Coherence]'
        )
        GROUP BY w.name, t.metricName
        ORDER BY w.name, t.metricName;
        """,
    ),
    (
        "13_per_pipeline_frontend_stalls",
        # GPU front-end stall accounting per pipeline.
        # FE Stalled = command processor waiting (sync = current frame's
        # work; async = parallel queue). Go Idle / Subchannel Switch =
        # context-switch counts that often signal under-fed queues.
        """
        WITH win(name, t0, t1) AS (
          SELECT text, start, end FROM NVTX_EVENTS WHERE text LIKE 'Pipeline:%'
        )
        SELECT w.name                                   AS pipeline,
               t.metricName                             AS metric,
               ROUND(AVG(g.value), 3)                   AS avgVal,
               MAX(g.value)                             AS maxVal
        FROM GPU_METRICS g
        JOIN TARGET_INFO_GPU_METRICS t ON t.metricId = g.metricId
        JOIN win w ON g.timestamp BETWEEN w.t0 AND w.t1
        WHERE t.metricName IN (
          'FE Stalled Sync [Throughput %]',
          'FE Stalled Async [Throughput %]',
          'Go Idle Sync [Commands]',
          'Subchannel Switch Sync [Commands]',
          'Go Idle Async [Commands]',
          'Subchannel Switch Async [Commands]',
          'Pixel Shader Barriers [Commands]'
        )
        GROUP BY w.name, t.metricName
        ORDER BY w.name, t.metricName;
        """,
    ),
    (
        "14_per_pipeline_workloads",
        # GPU workload accounting per pipeline window.
        # workloads          = #DX12_WORKLOAD rows inside the window
        # gpuTimeMs          = sum of (end-start) across those workloads
        # wallMs             = window duration
        # avgConcurrentEng.  = gpuTimeMs / wallMs (>1 = parallel engines)
        """
        WITH win(name, t0, t1) AS (
          SELECT text, start, end FROM NVTX_EVENTS WHERE text LIKE 'Pipeline:%'
        )
        SELECT w.name                                                  AS pipeline,
               COUNT(d.start)                                          AS workloads,
               ROUND(SUM(d.end - d.start) / 1e6, 2)                    AS gpuTimeMs,
               ROUND((w.t1 - w.t0) / 1e6, 2)                           AS wallMs,
               ROUND(1.0 * SUM(d.end - d.start) / (w.t1 - w.t0), 3)    AS avgConcurrentEngines
        FROM win w
        LEFT JOIN DX12_WORKLOAD d
               ON d.start >= w.t0 AND d.end <= w.t1
        GROUP BY w.name
        ORDER BY w.name;
        """,
    ),
    (
        "15_per_pipeline_pix_stages",
        # Per-PIX-marker (per-stage) GPU time within each pipeline window.
        # PIX markers come from the renderer: Draw, Shadow, C0/C1/C2,
        # Main, Trunk, Terrain, Impostors, Leaves, DepthPrepass, HiZ, Cull,
        # *Dispatch, SDSM, Sky, ImGUI, BuildMipChain.
        """
        WITH win(name, t0, t1) AS (
          SELECT text, start, end FROM NVTX_EVENTS WHERE text LIKE 'Pipeline:%'
        )
        SELECT w.name                                AS pipeline,
               s.value                               AS stage,
               COUNT(*)                              AS calls,
               ROUND(SUM(d.end - d.start) / 1e6, 2)  AS totalMs,
               ROUND(AVG(d.end - d.start) / 1e3, 2)  AS avgUs,
               ROUND(MIN(d.end - d.start) / 1e3, 2)  AS minUs,
               ROUND(MAX(d.end - d.start) / 1e3, 2)  AS maxUs
        FROM DX12_WORKLOAD d
        JOIN StringIds s ON s.id = d.textId
        JOIN win w       ON d.start >= w.t0 AND d.end <= w.t1
        WHERE d.textId IS NOT NULL
        GROUP BY w.name, d.textId
        ORDER BY w.name, totalMs DESC;
        """,
    ),
    (
        "16_metric_timeseries",
        # Full-rate GPU counter timeseries with active-pipeline labels.
        # Output is ~1.7M rows under the Graphics preset (124 metrics x 14k
        # samples); well suited to pandas / matplotlib downstream.
        """
        SELECT g.timestamp                         AS tsNs,
               ROUND(g.timestamp / 1e6, 3)         AS tsMs,
               g.metricId,
               t.metricName,
               g.value,
               w.text                              AS pipeline
        FROM GPU_METRICS g
        JOIN TARGET_INFO_GPU_METRICS t ON t.metricId = g.metricId
        LEFT JOIN NVTX_EVENTS w
               ON w.text LIKE 'Pipeline:%'
              AND g.timestamp BETWEEN w.start AND w.end
        ORDER BY g.timestamp, g.metricId;
        """,
    ),
    (
        "17_top_workloads",
        # Top GPU workload API names by cumulative GPU time. nameId resolves
        # to ID3D12* function names like ExecuteCommandLists, BeginEvent.
        """
        SELECT s.value                              AS workloadName,
               COUNT(*)                             AS calls,
               ROUND(SUM(d.end - d.start) / 1e6, 2) AS totalMs,
               ROUND(AVG(d.end - d.start) / 1e3, 2) AS avgUs
        FROM DX12_WORKLOAD d
        JOIN StringIds s ON s.id = d.nameId
        GROUP BY d.nameId
        ORDER BY totalMs DESC;
        """,
    ),
    (
        "21_memory_ops_by_pipeline",
        # DX12 memory operations (resource creation / mapping) per pipeline
        # window, broken down by D3D12 heap type.
        # bytes = rangeEnd - rangeStart, summed.
        # traceEventId joins DX12_MEMORY_OPERATION back to its DX12_API row.
        """
        WITH win(name, t0, t1) AS (
          SELECT text, start, end FROM NVTX_EVENTS WHERE text LIKE 'Pipeline:%'
        ),
        ops AS (
          SELECT a.start                                AS ts,
                 m.heapType,
                 (m.rangeEnd - m.rangeStart)            AS bytes
          FROM DX12_MEMORY_OPERATION m
          JOIN DX12_API a ON a.id = m.traceEventId
        )
        SELECT w.name                                   AS pipeline,
               COALESCE(ht.label, 'Unknown')            AS heapType,
               COUNT(o.bytes)                           AS ops,
               ROUND(SUM(o.bytes) / 1048576.0, 2)       AS totalMB
        FROM win w
        LEFT JOIN ops o            ON o.ts >= w.t0 AND o.ts <= w.t1
        LEFT JOIN ENUM_D3D12_HEAP_TYPE ht ON ht.id = o.heapType
        GROUP BY w.name, o.heapType
        ORDER BY w.name, totalMB DESC;
        """,
    ),
]


def _format_value(v: object) -> str:
    if v is None:
        return ""
    return str(v)


def run_query(conn: sqlite3.Connection, name: str, sql: str, out_dir: Path) -> int:
    cursor = conn.execute(sql)
    cols = [d[0] for d in cursor.description]
    out_file = out_dir / f"{name}.csv"
    n = 0
    with out_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for row in cursor:
            writer.writerow(_format_value(v) for v in row)
            n += 1
    return n


def _fetch_wrapper_bounds(conn: sqlite3.Connection, index_1based: int
                          ) -> tuple[int, int]:
    """Return (startNs, endNs) of the index-1based-th `XylemBenchmark` NVTX
    range. Used by --window-index to scope every per-pipeline CTE to a single
    benchmark run when one .sqlite contains several back-to-back invocations.
    """
    rows = conn.execute(
        "SELECT start, end FROM NVTX_EVENTS "
        "WHERE text = 'XylemBenchmark' ORDER BY start"
    ).fetchall()
    if not rows:
        raise RuntimeError(
            "No 'XylemBenchmark' NVTX windows in DB -- cannot apply "
            "--window-index. Capture must include the outer benchmark marker.")
    if index_1based < 1 or index_1based > len(rows):
        raise RuntimeError(
            f"--window-index {index_1based} out of range; DB has "
            f"{len(rows)} XylemBenchmark window(s).")
    return rows[index_1based - 1]


def _scope_to_wrapper(sql: str, ws: int, we: int) -> str:
    """Inject wrapper-window bounds into every per-pipeline CTE in `sql`.

    Per-pipeline queries use one of two recurring patterns that select the
    `Pipeline:*` NVTX events:
      WHERE text LIKE 'Pipeline:%'           (CTE form, ~10x)
      ON w.text LIKE 'Pipeline:%'            (LEFT JOIN form, query 16)
    plus query 03's diagnostic dump that also pulls in the XylemBenchmark
    wrapper rows. The substitutions below add `AND start/end BETWEEN ws..we`
    so only windows nested in the chosen wrapper survive.
    """
    # Query 03's diagnostic dump: keep the wrapper row itself + only the
    # Pipeline:* rows that sit inside it.
    sql = sql.replace(
        "WHERE text LIKE 'Pipeline:%' OR text = 'XylemBenchmark'",
        f"WHERE (text LIKE 'Pipeline:%' AND start >= {ws} AND end <= {we}) "
        f"OR (text = 'XylemBenchmark' AND start = {ws})",
    )
    # Standard CTE form -- 10+ occurrences.
    sql = sql.replace(
        "WHERE text LIKE 'Pipeline:%'",
        f"WHERE text LIKE 'Pipeline:%' AND start >= {ws} AND end <= {we}",
    )
    # Query 16's LEFT JOIN form.
    sql = sql.replace(
        "ON w.text LIKE 'Pipeline:%'",
        f"ON w.text LIKE 'Pipeline:%' AND w.start >= {ws} AND w.end <= {we}",
    )
    return sql


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", required=True, help="Path to NSight Systems .sqlite report")
    p.add_argument("--out", required=True, help="Output directory for CSVs (created if missing)")
    p.add_argument("--window-index", type=int, default=None,
                   help="Optional 1-based index of the XylemBenchmark wrapper "
                        "to scope every per-pipeline aggregate to. Useful "
                        "when a single capture spans multiple benchmark "
                        "invocations (e.g. Hi-Z vs no-Hi-Z back-to-back).")
    args = p.parse_args()

    db_path = Path(args.db)
    out_dir = Path(args.out)
    if not db_path.is_file():
        print(f"DB not found: {db_path}", file=sys.stderr)
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)

    uri = f"file:{db_path.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        wrapper_bounds: tuple[int, int] | None = None
        if args.window_index is not None:
            wrapper_bounds = _fetch_wrapper_bounds(conn, args.window_index)
            ws, we = wrapper_bounds
            print(f"--window-index {args.window_index}: scoping to "
                  f"XylemBenchmark window [{ws} .. {we}] ns "
                  f"(duration {(we - ws) / 1e9:.2f}s)")

        for name, sql in QUERIES:
            scoped_sql = (_scope_to_wrapper(sql, *wrapper_bounds)
                          if wrapper_bounds is not None else sql)
            t0 = time.perf_counter()
            print(f"[{name}] ...", end="", flush=True)
            n = run_query(conn, name, scoped_sql, out_dir)
            dt_ms = int((time.perf_counter() - t0) * 1000)
            print(f" {n} rows ({dt_ms} ms) -> {out_dir / (name + '.csv')}")
    finally:
        conn.close()

    print(f"\nDone. CSVs in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
