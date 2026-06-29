#!/usr/bin/env python3
"""Experiment Visualizer — Browse every run from the Gemma 4 overnight research loop.

Scans overnight_results/ for all JSONL files, indexes them, and serves a
browser-based visualizer at http://localhost:8899.

Usage:
    python3 scripts/visualizer.py
    # or
    chmod +x scripts/visualizer.py && ./scripts/visualizer.py
"""

import http.server
import json
import math
import threading
import time
import urllib.parse
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HOST = "0.0.0.0"
PORT = 8899

# Resolve the project root (parent of scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "overnight_results"

# ---------------------------------------------------------------------------
# Index builder
# ---------------------------------------------------------------------------

def build_index(results_dir: Path):
    """Walk results_dir, discover every *.jsonl file, and build an index.

    Returns
    -------
    experiments : list[dict]
        Each item: { "id", "dir", "file", "path", "group", "variation",
                     "round", "line_count", "line_offsets" }
    file_map : dict[str, dict]
        Maps relative path -> experiment dict for fast lookup.
    """
    experiments = []
    file_map = {}

    # Discover all JSONL files
    jsonl_files = sorted(results_dir.rglob("*.jsonl"))

    for fpath in jsonl_files:
        rel_path = str(fpath.relative_to(results_dir))
        parent_dir = fpath.parent.name

        # Try to extract a sensible group label from the directory name
        group = parent_dir
        variation = fpath.stem  # e.g. "baseline_r1", "r5_multistep"

        # Build line-offset index so we can seek efficiently
        # Robustly count line offsets; skip on any file error
        line_offsets = []
        try:
            with open(fpath, "rb") as fh:
                while True:
                    offset = fh.tell()
                    line = fh.readline()
                    if not line:
                        break
                    line_offsets.append(offset)
        except OSError:
            continue  # skip unreadable files
        line_count = len(line_offsets)

        # Try to parse the first non-empty line for extra metadata
        first_line_data = {}
        try:
            with open(fpath, encoding="utf-8", errors="replace") as fh:
                for raw_line in fh:
                    raw_line = raw_line.strip()
                    if raw_line:
                        first_line_data = json.loads(raw_line)
                        break
        except (json.JSONDecodeError, OSError, ValueError):
            pass

        # Mark clean reruns (file names starting with "clean_")
        is_clean = fpath.stem.startswith("clean_")

        entry = {
            "id": rel_path,
            "dir": parent_dir,
            "file": fpath.name,
            "path": rel_path,
            "group": group,
            "variation": variation,
            "round": _infer_round(variation, parent_dir),
            "line_count": line_count,
            "line_offsets": line_offsets,
            "sample": first_line_data,
            "is_clean": is_clean,
        }
        experiments.append(entry)
        file_map[rel_path] = entry

    return experiments, file_map


def _infer_round(variation: str, parent_dir: str) -> str:
    """Try to extract a round label from the variation or directory name."""
    import re
    m = re.search(r"r(\d+)", variation)
    if m:
        return f"Round {m.group(1)}"
    m = re.search(r"r(\d+)", parent_dir)
    if m:
        return f"Round {m.group(1)}"
    # Fallback: check parent dir of parent dir
    return ""


def group_experiments(experiments):
    """Group experiments by their directory (experiment type)."""
    groups = {}
    for exp in experiments:
        g = exp["group"]
        if g not in groups:
            groups[g] = []
        groups[g].append(exp)
    return groups


# ---------------------------------------------------------------------------
# Line reader (efficient — seeks to offset and reads one line)
# ---------------------------------------------------------------------------

def read_line_at(results_dir: Path, rel_path: str, line_index: int):
    """Read one JSON line from a JSONL file by its zero-based line index.

    Returns parsed dict, or None on error.
    """
    fpath = results_dir / rel_path
    if not fpath.exists():
        return None
    try:
        with open(fpath) as fh:
            for i, line in enumerate(fh):
                if i == line_index:
                    line = line.strip()
                    if line:
                        return json.loads(line)
                    return None
        return None
    except (OSError, json.JSONDecodeError):
        return None


def read_line_at_offset(fpath: Path, offset: int):
    """Read one JSON line from a file at the given byte offset."""
    try:
        with open(fpath) as fh:
            fh.seek(offset)
            line = fh.readline()
            if line.strip():
                return json.loads(line)
        return None
    except (OSError, json.JSONDecodeError):
        return None



# ---------------------------------------------------------------------------
# Accuracy field detection priority
# ---------------------------------------------------------------------------

ACCURACY_FIELDS = [
    "zone_occupancy_accuracy",
    "zone_accuracy",
    "visual_grounding_accuracy",
    "plan_complete",
    "correct",
    "is_correct",
    "detection_correct",
    "change_correct",
]


def compute_stats(fpath: Path, forced_field: str | None = None) -> dict:
    """Read ALL valid records from a JSONL file and compute honest metrics.

    Never trusts pre-summarised numbers — derives everything from raw records.
    Pass forced_field to override auto-detection (useful for apples-to-apples
    audit comparisons).

    Returns a dict with:
      n, accuracy_field, n_acc, mean, ci_lo, ci_hi (Wilson 95%),
      p50_latency, p95_latency, mean_latency,
      mean_hallucinations, mean_misses  (where present)
    """
    if not fpath.exists():
        return {"n": 0, "error": "file not found"}

    records = []
    try:
        with open(fpath, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if isinstance(rec, dict):
                        records.append(rec)
                except (json.JSONDecodeError, ValueError):
                    continue  # skip malformed lines
    except OSError as exc:
        return {"n": 0, "error": str(exc)}

    n = len(records)
    result: dict = {"n": n}
    if n == 0:
        return result

    # Auto-detect the primary accuracy field (present in >50% of records)
    # If forced_field is set, use it directly (for audit comparisons)
    if forced_field and sum(1 for r in records if forced_field in r) > 0:
        acc_field = forced_field
    else:
        acc_field = None
        for field in ACCURACY_FIELDS:
            present = sum(1 for r in records if field in r)
            if present > n * 0.5:
                acc_field = field
                break

    result["accuracy_field"] = acc_field

    # Compute mean accuracy and Wilson 95% CI
    if acc_field:
        values = []
        for r in records:
            v = r.get(acc_field)
            if v is None:
                continue
            if isinstance(v, bool):
                values.append(1.0 if v else 0.0)
            elif isinstance(v, (int, float)) and not math.isnan(float(v)):
                values.append(float(v))

        m = len(values)
        result["n_acc"] = m
        if m > 0:
            p = sum(values) / m
            result["mean"] = round(p, 4)
            z = 1.96
            denom = 1.0 + z * z / m
            center = (p + z * z / (2 * m)) / denom
            half = z * math.sqrt(p * (1 - p) / m + z * z / (4 * m * m)) / denom
            result["ci_lo"] = round(max(0.0, center - half), 4)
            result["ci_hi"] = round(min(1.0, center + half), 4)
        else:
            result["mean"] = None
            result["ci_lo"] = None
            result["ci_hi"] = None
    else:
        result["n_acc"] = 0
        result["mean"] = None
        result["ci_lo"] = None
        result["ci_hi"] = None

    # Latency percentiles
    latencies = sorted(
        float(r["latency_ms"])
        for r in records
        if "latency_ms" in r and isinstance(r["latency_ms"], (int, float))
    )
    if latencies:
        nl = len(latencies)
        result["p50_latency"] = round(latencies[int(nl * 0.50)], 1)
        result["p95_latency"] = round(latencies[min(int(nl * 0.95), nl - 1)], 1)
        result["mean_latency"] = round(sum(latencies) / nl, 1)

    # Hallucinations / misses
    halls = [
        float(r["hallucinations"]) for r in records
        if "hallucinations" in r and isinstance(r["hallucinations"], (int, float))
    ]
    if halls:
        result["mean_hallucinations"] = round(sum(halls) / len(halls), 3)

    misses = [
        float(r["misses"]) for r in records
        if "misses" in r and isinstance(r["misses"], (int, float))
    ]
    if misses:
        result["mean_misses"] = round(sum(misses) / len(misses), 3)

    return result


# ---------------------------------------------------------------------------
# HTML template (served at /)
# ---------------------------------------------------------------------------

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Gemma 4 Experiment Visualizer</title>
<style>
  :root {
    --bg: #0d1117;
    --surface: #161b22;
    --border: #30363d;
    --text: #c9d1d9;
    --text-secondary: #8b949e;
    --accent: #58a6ff;
    --success: #3fb950;
    --failure: #f85149;
    --warning: #d29922;
    --sidebar-width: 320px;
  }
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, sans-serif;
    background: var(--bg);
    color: var(--text);
    height: 100vh;
    display: flex;
    overflow: hidden;
  }

  /* Sidebar */
  #sidebar {
    width: var(--sidebar-width);
    min-width: var(--sidebar-width);
    background: var(--surface);
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  #sidebar h1 {
    font-size: 14px;
    padding: 16px;
    border-bottom: 1px solid var(--border);
    color: var(--accent);
    letter-spacing: 0.5px;
  }
  #sidebar h1 small { color: var(--text-secondary); font-weight: normal; font-size: 11px; display: block; margin-top: 2px; }
  #sidebar-search {
    padding: 8px 12px;
    border-bottom: 1px solid var(--border);
  }
  #sidebar-search input {
    width: 100%;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 6px 10px;
    color: var(--text);
    font-size: 13px;
    outline: none;
  }
  #sidebar-search input:focus { border-color: var(--accent); }
  #experiment-list {
    flex: 1;
    overflow-y: auto;
    padding: 8px 0;
  }
  .exp-group { margin-bottom: 4px; }
  .exp-group-header {
    padding: 6px 16px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    color: var(--text-secondary);
    letter-spacing: 0.8px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: space-between;
    user-select: none;
  }
  .exp-group-header:hover { color: var(--text); }
  .exp-group-header .count { font-size: 10px; background: var(--bg); padding: 1px 6px; border-radius: 8px; }
  .exp-group-header .arrow { transition: transform 0.15s; font-size: 10px; }
  .exp-group-header.collapsed .arrow { transform: rotate(-90deg); }
  .exp-item {
    padding: 6px 16px 6px 28px;
    font-size: 13px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-left: 2px solid transparent;
    transition: background 0.1s;
  }
  .exp-item:hover { background: rgba(88,166,255,0.08); }
  .exp-item.active { background: rgba(88,166,255,0.12); border-left-color: var(--accent); }
  .exp-item .name { flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .exp-item .badge {
    font-size: 10px;
    padding: 1px 6px;
    border-radius: 8px;
    background: var(--bg);
    color: var(--text-secondary);
    margin-left: 8px;
  }

  /* Main Panel */
  #main {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  /* Toolbar */
  #toolbar {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 16px;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    flex-wrap: wrap;
  }
  #toolbar label { font-size: 12px; color: var(--text-secondary); display: flex; align-items: center; gap: 4px; }
  #toolbar select, #toolbar input {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 4px 8px;
    color: var(--text);
    font-size: 12px;
    outline: none;
  }
  #toolbar select:focus, #toolbar input:focus { border-color: var(--accent); }
  #toolbar button {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 4px 10px;
    color: var(--text);
    font-size: 12px;
    cursor: pointer;
  }
  #toolbar button:hover { border-color: var(--accent); color: var(--accent); }
  #toolbar .nav-buttons { display: flex; gap: 4px; }
  #toolbar .nav-buttons button { min-width: 32px; }
  #run-indicator { font-size: 12px; color: var(--text-secondary); margin-left: auto; white-space: nowrap; }

  /* Content area */
  #content {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
  }

  /* Run header */
  #run-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 16px;
    flex-wrap: wrap;
  }
  #run-header h2 { font-size: 18px; font-weight: 600; }
  #run-header .status-badge {
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 600;
  }
  .status-badge.success { background: rgba(63,185,80,0.15); color: var(--success); }
  .status-badge.failure { background: rgba(248,81,73,0.15); color: var(--failure); }
  .status-badge.unknown { background: rgba(210,153,34,0.15); color: var(--warning); }

  /* Metrics row */
  #metrics {
    display: flex;
    gap: 12px;
    margin-bottom: 20px;
    flex-wrap: wrap;
  }
  .metric {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 14px;
    min-width: 100px;
  }
  .metric .label { font-size: 10px; text-transform: uppercase; color: var(--text-secondary); letter-spacing: 0.5px; }
  .metric .value { font-size: 18px; font-weight: 700; margin-top: 2px; }
  .metric .value.good { color: var(--success); }
  .metric .value.bad { color: var(--failure); }
  .metric .value.mid { color: var(--warning); }

  /* Image */
  #image-container {
    margin-bottom: 20px;
    text-align: center;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px;
    display: none;
  }
  #image-container.visible { display: block; }
  #image-container img {
    max-width: 100%;
    max-height: 500px;
    border-radius: 4px;
  }

  /* Prompt and response */
  .section {
    margin-bottom: 16px;
  }
  .section h3 {
    font-size: 13px;
    font-weight: 600;
    color: var(--text-secondary);
    margin-bottom: 8px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  .section .content {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px;
    font-size: 14px;
    line-height: 1.6;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .section .content.json {
    font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
    font-size: 12px;
    line-height: 1.5;
  }
  .section .content .json-string { color: #a5d6ff; }
  .section .content .json-number { color: #79c0ff; }
  .section .content .json-boolean { color: #ff7b72; }
  .section .content .json-null { color: #ff7b72; }
  .section .content .json-key { color: #7ee787; }
  .section .content .json-bracket { color: #d2a8ff; }

  /* Extra fields table */
  #extra-fields {
    margin-bottom: 16px;
  }
  #extra-fields table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
  }
  #extra-fields th, #extra-fields td {
    padding: 6px 12px;
    text-align: left;
    border-bottom: 1px solid var(--border);
  }
  #extra-fields th {
    color: var(--text-secondary);
    font-weight: 600;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  /* Loading / Error */
  #loading { display: flex; align-items: center; justify-content: center; height: 200px; color: var(--text-secondary); }
  #loading .spinner {
    width: 24px; height: 24px;
    border: 2px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    margin-right: 12px;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  #error {
    display: none;
    background: rgba(248,81,73,0.1);
    border: 1px solid var(--failure);
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 16px;
    color: var(--failure);
  }
  #error.visible { display: block; }

  /* Jump dialog */
  #jump-input { width: 60px; }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 8px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
  ::-webkit-scrollbar-thumb:hover { background: var(--text-secondary); }

  /* Summary stats */
  #summary { font-size: 12px; color: var(--text-secondary); padding: 8px 16px; border-top: 1px solid var(--border); }
  #summary span { margin-right: 12px; }

  /* Collapsible groups */
  .exp-group-children { overflow: hidden; transition: max-height 0.2s; }
  .exp-group-children.collapsed { max-height: 0 !important; }

  /* Metadata in sidebar */
  .exp-item .meta { font-size: 10px; color: var(--text-secondary); }

  /* ---- Clean / Original badges ---- */
  .clean-badge {
    font-size: 9px; padding: 1px 5px; border-radius: 6px;
    background: rgba(63,185,80,0.2); color: var(--success);
    border: 1px solid var(--success); margin-left: 4px;
    font-weight: 700; letter-spacing: 0.3px; white-space: nowrap;
  }
  .orig-badge {
    font-size: 9px; padding: 1px 5px; border-radius: 6px;
    background: rgba(210,153,34,0.2); color: var(--warning);
    border: 1px solid rgba(210,153,34,0.4); margin-left: 4px;
    font-weight: 600; white-space: nowrap;
  }

  /* ---- Group Overview ---- */
  .group-overview-header { display: flex; align-items: center; gap: 12px; margin-bottom: 20px; }
  .group-overview-header h2 { font-size: 20px; font-weight: 700; }

  .exp-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 16px; margin-bottom: 14px;
    cursor: pointer; transition: border-color 0.15s, box-shadow 0.15s;
  }
  .exp-card:hover { border-color: var(--accent); box-shadow: 0 0 0 1px rgba(88,166,255,0.2); }
  .exp-card.is-clean { border-left: 4px solid var(--success); }
  .exp-card.is-original { border-left: 4px solid var(--warning); }
  .exp-card-header { display: flex; align-items: center; gap: 8px; margin-bottom: 12px; flex-wrap: wrap; }
  .exp-card-header h3 { font-size: 15px; font-weight: 600; flex: 1; }

  /* Accuracy bar */
  .acc-bar-wrap { margin: 10px 0; }
  .acc-bar-label { display: flex; justify-content: space-between; font-size: 11px; color: var(--text-secondary); margin-bottom: 4px; }
  .acc-bar-track { position: relative; height: 10px; background: var(--bg); border-radius: 5px; }
  .acc-bar-fill { height: 100%; border-radius: 5px; background: var(--success); transition: width 0.4s; }
  .acc-bar-fill.mid { background: var(--warning); }
  .acc-bar-fill.bad { background: var(--failure); }
  .ci-lo, .ci-hi { position: absolute; top: -3px; width: 2px; height: 16px; background: rgba(255,255,255,0.65); border-radius: 1px; }
  .ci-line { position: absolute; top: 4px; height: 2px; background: rgba(255,255,255,0.35); }

  /* Stat grid in card */
  .stat-grid { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
  .stat-item {
    background: var(--bg); border-radius: 6px; padding: 6px 10px;
    font-size: 11px; display: flex; flex-direction: column; gap: 2px;
  }
  .stat-item .stat-label { color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.4px; }
  .stat-item .stat-val { font-weight: 700; font-size: 13px; color: var(--text); }

  /* N badge */
  .n-badge {
    font-size: 10px; color: var(--text-secondary);
    background: var(--bg); padding: 2px 6px; border-radius: 6px; border: 1px solid var(--border);
  }

  /* Pending badge */
  .pending-badge {
    font-size: 10px; padding: 2px 7px; border-radius: 8px;
    background: rgba(88,166,255,0.12); color: var(--accent); border: 1px solid rgba(88,166,255,0.3);
  }

  /* ---- Claims Audit Panel ---- */
  .audit-panel {
    background: var(--surface); border: 1px solid rgba(210,153,34,0.4);
    border-radius: 10px; padding: 16px; margin-bottom: 20px;
  }
  .audit-panel h3 {
    font-size: 13px; font-weight: 700; color: var(--warning);
    text-transform: uppercase; letter-spacing: 0.5px;
    margin-bottom: 12px; display: flex; align-items: center; gap: 6px;
  }
  .audit-table { width: 100%; border-collapse: collapse; font-size: 13px; }
  .audit-table th {
    text-align: left; padding: 6px 10px; font-size: 10px; font-weight: 700;
    text-transform: uppercase; color: var(--text-secondary); letter-spacing: 0.5px;
    border-bottom: 1px solid var(--border);
  }
  .audit-table td { padding: 8px 10px; border-bottom: 1px solid rgba(48,54,61,0.5); vertical-align: middle; }
  .audit-table tr:last-child td { border-bottom: none; }
  .verdict-match { color: var(--success); font-weight: 700; }
  .verdict-diff { color: var(--failure); font-weight: 700; }
  .verdict-pending { color: var(--text-secondary); font-style: italic; }
  .claimed-num { color: var(--text-secondary); }
  .recomputed-num { font-weight: 700; }
  .recomputed-num.good { color: var(--success); }
  .recomputed-num.bad { color: var(--failure); }
  .recomputed-num.mid { color: var(--warning); }

  /* ---- Auto-refresh indicator ---- */
  #refresh-info {
    font-size: 11px; color: var(--text-secondary);
    padding: 6px 16px; border-top: 1px solid var(--border);
    display: flex; align-items: center; gap: 6px;
  }
  .refresh-dot {
    width: 6px; height: 6px; border-radius: 50%; background: var(--success);
    animation: pulse 2s ease-in-out infinite; flex-shrink: 0;
  }
  @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
  #refresh-info.stale .refresh-dot { background: var(--warning); }

  /* ---- Group overview container ---- */
  #group-overview { padding: 0; }

</style>
</head>
<body>

<!-- Sidebar -->
<div id="sidebar">
  <h1>
    Gemma 4 Experiment Browser
    <small>Overnight Research Loop</small>
  </h1>
  <div id="sidebar-search">
    <input type="text" id="filter-input" placeholder="Filter experiments..." oninput="filterExperiments()">
  </div>
  <div id="experiment-list"></div>
  <div id="summary"></div>
  <div id="refresh-info"><span class="refresh-dot"></span> Connecting...</div>
</div>

<!-- Main -->
<div id="main">
  <div id="toolbar">
    <div class="nav-buttons">
      <button onclick="navigateRun(-10)" title="-10 runs">--</button>
      <button onclick="navigateRun(-1)" title="Previous run">&larr;</button>
      <button onclick="navigateRun(1)" title="Next run">&rarr;</button>
      <button onclick="navigateRun(10)" title="+10 runs">++</button>
    </div>
    <label>
      Jump to #:
      <input type="number" id="jump-input" min="1" value="1" onkeydown="if(event.key==='Enter')jumpToRun()">
      <button onclick="jumpToRun()" style="padding:2px 6px;font-size:11px;">Go</button>
    </label>
    <label>
      Sort:
      <select id="sort-select" onchange="applyFilters()">
        <option value="run">Run #</option>
        <option value="latency">Latency</option>
        <option value="accuracy">Zone Accuracy</option>
      </select>
    </label>
    <label>
      Status:
      <select id="status-filter" onchange="applyFilters()">
        <option value="all">All</option>
        <option value="success">Success</option>
        <option value="failure">Failure</option>
      </select>
    </label>
    <label>
      Min accuracy:
      <select id="accuracy-filter" onchange="applyFilters()">
        <option value="0">Any</option>
        <option value="0.5">0.5+</option>
        <option value="0.8">0.8+</option>
        <option value="0.9">0.9+</option>
        <option value="1.0">1.0</option>
      </select>
    </label>
    <span id="run-indicator"></span>
  </div>

  <div id="content">
    <div id="loading">
      <div class="spinner"></div>
      <span>Loading experiments...</span>
    </div>
    <div id="error"></div>
    <div id="run-header"></div>
    <div id="metrics"></div>
    <div id="image-container"></div>
    <div id="extra-fields"></div>
    <div class="section" id="prompt-section">
      <h3>Prompt Sent</h3>
      <div class="content" id="prompt-content"></div>
    </div>
    <div class="section" id="response-section">
      <h3>Raw Response</h3>
      <div class="content json" id="response-content"></div>
    </div>
    <div style="display:none" id="empty-state">
      <p style="text-align:center;padding:60px 20px;color:var(--text-secondary);">
        Select an experiment from the sidebar to begin browsing.
      </p>
    </div>
    <div id="group-overview" style="display:none"></div>
  </div>
</div>

<script>
// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
const state = {
  experiments: [],         // all experiments
  fileMap: {},             // rel_path -> experiment
  currentFile: null,       // currently selected experiment rel_path
  currentLine: 0,          // currently displayed line number (0-based)
  currentData: null,       // parsed JSON of currently displayed run
  currentIndex: -1,        // index into the sorted runs array
  sortedRuns: [],          // [{file, line}] — current filtered+sorted list
  currentGroup: null,      // currently viewed group name (for group overview)
  currentView: 'run',      // 'run' | 'group'
};

let _refreshTimer = null;

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', init);

async function init() {
  try {
    const resp = await fetch('/api/experiments');
    const data = await resp.json();
    state.experiments = data.experiments;
    state.fileMap = {};
    data.experiments.forEach(e => { state.fileMap[e.path] = e; });
    renderSidebar();
    document.getElementById('summary').innerHTML =
      `<span>${data.experiments.length} exps</span>` +
      `<span>${data.total_runs.toLocaleString()} runs</span>` +
      `<span>${data.files} files</span>`;
    startAutoRefresh();
    updateRefreshInfo(true);
    // Show group overview for first group on load
    if (data.experiments.length > 0) {
      const firstGroup = data.experiments[0].group;
      await loadGroupOverview(firstGroup);
    }
  } catch (e) {
    showError('Failed to load experiments: ' + e.message);
  }
}

function startAutoRefresh() {
  if (_refreshTimer) clearInterval(_refreshTimer);
  _refreshTimer = setInterval(async () => {
    try {
      const resp = await fetch('/api/experiments?t=' + Date.now());
      const data = await resp.json();
      state.experiments = data.experiments;
      state.fileMap = {};
      data.experiments.forEach(e => { state.fileMap[e.path] = e; });
      renderSidebar();
      document.getElementById('summary').innerHTML =
        `<span>${data.experiments.length} exps</span>` +
        `<span>${data.total_runs.toLocaleString()} runs</span>` +
        `<span>${data.files} files</span>`;
      updateRefreshInfo(true);
      // If in group overview, refresh it silently
      if (state.currentView === 'group' && state.currentGroup) {
        await loadGroupOverview(state.currentGroup, false);
      }
    } catch(e) { updateRefreshInfo(false); }
  }, 10000);
}

function updateRefreshInfo(ok) {
  const el = document.getElementById('refresh-info');
  if (!el) return;
  const t = new Date().toLocaleTimeString();
  el.className = ok ? '' : 'stale';
  el.innerHTML = `<span class="refresh-dot"></span> Live &#x2022; ${t}`;
}

// ---------------------------------------------------------------------------
// Sidebar
// ---------------------------------------------------------------------------
function renderSidebar() {
  const groups = {};
  state.experiments.forEach(exp => {
    const g = exp.group || 'other';
    if (!groups[g]) groups[g] = [];
    groups[g].push(exp);
  });

  const container = document.getElementById('experiment-list');
  container.innerHTML = '';

  const groupOrder = ['vision','vision_no_grid','zone_boundary','occlusion','monochrome','no_gripper',
    'jpeg','relational','multistep','chaos','blank','mixed_sizes','background','high_count',
    'thermal','gauntlet','perturb','temperature'];
  const sortedGroups = Object.keys(groups).sort(
    (a,b) => {
      const ia = groupOrder.indexOf(a);
      const ib = groupOrder.indexOf(b);
      if (ia >= 0 && ib >= 0) return ia - ib;
      if (ia >= 0) return -1;
      if (ib >= 0) return 1;
      return a.localeCompare(b);
    }
  );

  sortedGroups.forEach(groupName => {
    const exps = groups[groupName];
    const div = document.createElement('div');
    div.className = 'exp-group';

    const header = document.createElement('div');
    header.className = 'exp-group-header';
    header.innerHTML = `<span><span class="arrow">&#9660;</span> ${groupName.replace(/_/g,' ')}</span><span class="count">${exps.length}</span>`;
    header.onclick = () => {
      header.classList.toggle('collapsed');
      children.classList.toggle('collapsed');
      loadGroupOverview(groupName);
    };

    const children = document.createElement('div');
    children.className = 'exp-group-children';

    // Sort experiments within group
    exps.sort((a,b) => {
      const ra = a.round || '';
      const rb = b.round || '';
      if (ra !== rb) return ra.localeCompare(rb);
      return a.variation.localeCompare(b.variation);
    });

    exps.forEach(exp => {
      const item = document.createElement('div');
      item.className = 'exp-item';
      item.dataset.path = exp.path;
      const name = exp.variation.replace(/_/g, ' ');
      const round = exp.round ? exp.round + ' ' : '';
      const cleanBadge = exp.is_clean ? '<span class="clean-badge">CLEAN</span>' : '';
      item.innerHTML = `<span class="name">${round}${name}${cleanBadge}</span><span class="badge">${exp.line_count}</span>`;
      item.onclick = (ev) => {
        ev.stopPropagation();
        state.currentView = 'run';
        showRunView();
        selectExperiment(exp.path);
      };
      children.appendChild(item);
    });

    div.appendChild(header);
    div.appendChild(children);
    container.appendChild(div);
  });
}

function filterExperiments() {
  const query = document.getElementById('filter-input').value.toLowerCase();
  document.querySelectorAll('.exp-group').forEach(group => {
    let visible = false;
    group.querySelectorAll('.exp-item').forEach(item => {
      const match = item.textContent.toLowerCase().includes(query);
      item.style.display = match ? 'flex' : 'none';
      if (match) visible = true;
    });
    group.querySelector('.exp-group-header').style.display = visible || !query ? 'flex' : 'none';
    if (query) {
      group.querySelector('.exp-group-children').classList.remove('collapsed');
      group.querySelector('.exp-group-header').classList.remove('collapsed');
    }
  });
}

// ---------------------------------------------------------------------------
// View management
// ---------------------------------------------------------------------------
function showRunView() {
  document.getElementById('group-overview').style.display = 'none';
  ['run-header','metrics','image-container','extra-fields','prompt-section','response-section'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.style.display = '';
  });
}

function showGroupView() {
  document.getElementById('loading').style.display = 'none';
  document.getElementById('error').classList.remove('visible');
  const emptyEl = document.getElementById('empty-state');
  if (emptyEl) emptyEl.style.display = 'none';
  ['run-header','metrics','image-container','extra-fields','prompt-section','response-section'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.style.display = 'none';
  });
  document.getElementById('group-overview').style.display = 'block';
}

async function loadGroupOverview(groupName, scroll=true) {
  state.currentGroup = groupName;
  state.currentView = 'group';
  document.querySelectorAll('.exp-item.active').forEach(el => el.classList.remove('active'));
  showGroupView();

  const groupExps = state.experiments.filter(e => e.group === groupName);
  if (groupExps.length === 0) {
    document.getElementById('group-overview').innerHTML =
      '<p style="padding:40px;color:var(--text-secondary);text-align:center">No experiments in this group.</p>';
    return;
  }

  document.getElementById('group-overview').innerHTML =
    '<div style="display:flex;align-items:center;justify-content:center;height:200px;color:var(--text-secondary)">' +
    '<div class="spinner"></div>&nbsp; Loading stats...</div>';

  const results = await Promise.all(groupExps.map(async exp => {
    try {
      const r = await fetch('/api/stats?file=' + encodeURIComponent(exp.path));
      return { exp, stats: r.ok ? await r.json() : { n: 0 } };
    } catch { return { exp, stats: { n: 0 } }; }
  }));

  renderGroupOverview(groupName, results);
  if (scroll) document.getElementById('content').scrollTop = 0;
}

function renderGroupOverview(groupName, results) {
  const ov = document.getElementById('group-overview');
  let html = `<div class="group-overview-header">
    <h2>${groupName.replace(/_/g,' ')} &mdash; ${results.length} file${results.length!==1?'s':''}</h2>
  </div>`;

  // Audit panel for suspect experiments
  if (['monochrome','multistep','jpeg'].includes(groupName)) {
    html += buildAuditSection(groupName, results);
  }

  // Sort: clean files first, then alphabetical
  const sorted = [...results].sort((a,b) => {
    if (a.exp.is_clean && !b.exp.is_clean) return -1;
    if (!a.exp.is_clean && b.exp.is_clean) return 1;
    return a.exp.variation.localeCompare(b.exp.variation);
  });

  sorted.forEach(({ exp, stats }) => { html += buildExpCard(exp, stats); });
  ov.innerHTML = html;

  ov.querySelectorAll('.exp-card[data-path]').forEach(card => {
    card.addEventListener('click', () => {
      const path = card.dataset.path;
      state.currentView = 'run';
      showRunView();
      document.querySelectorAll('.exp-item.active').forEach(el => el.classList.remove('active'));
      const sideItem = document.querySelector(`.exp-item[data-path="${path}"]`);
      if (sideItem) sideItem.classList.add('active');
      selectExperiment(path);
    });
  });
}

function buildAuditSection(groupName, results) {
  const claimed = {
    monochrome: { label: 'zone_accuracy', value: 1.0, note: 'Final report claimed 100%' },
    multistep:  { label: 'plan_complete',  value: 1.0, note: 'Final report claimed 100%' },
    jpeg:       { label: 'zone_accuracy', value: 1.0, note: 'Final report claimed 100% at all quality levels' },
  };
  const claim = claimed[groupName];
  if (!claim) return '';

  let rows = '';
  results.forEach(({ exp, stats }) => {
    const m = stats.mean;
    const nAcc = typeof stats.n_acc === 'number' ? stats.n_acc : stats.n || 0;
    const badge = exp.is_clean
      ? '<span class="clean-badge">CLEAN</span>'
      : '<span class="orig-badge">ORIGINAL</span>';

    let recomputed = '&mdash;';
    let verdictClass = 'verdict-pending';
    let verdictText = 'Pending';

    if (m !== null && m !== undefined && nAcc > 0) {
      const pct = (m * 100).toFixed(1) + '%';
      const ci = (stats.ci_lo != null && stats.ci_hi != null)
        ? ` [${(stats.ci_lo*100).toFixed(0)}&ndash;${(stats.ci_hi*100).toFixed(0)}%]` : '';
      const numClass = m >= 0.95 ? 'good' : m >= 0.7 ? 'mid' : 'bad';
      recomputed = `<span class="recomputed-num ${numClass}">${pct}</span>` +
        `<span style="font-size:10px;color:var(--text-secondary);margin-left:4px">${ci} N=${nAcc}</span>`;
      const diff = Math.abs(m - claim.value);
      verdictClass = diff < 0.02 ? 'verdict-match' : 'verdict-diff';
      verdictText = diff < 0.02 ? 'Confirmed ✓' : `Off by ${(diff*100).toFixed(0)}pp`;
    } else if (nAcc === 0 && exp.is_clean) {
      recomputed = '<span class="pending-badge">Pending rerun</span>';
    }

    rows += `<tr>
      <td>${exp.variation.replace(/_/g,' ')} ${badge}</td>
      <td class="claimed-num">${(claim.value*100).toFixed(0)}%</td>
      <td>${recomputed}</td>
      <td class="${verdictClass}">${verdictText}</td>
    </tr>`;
  });

  return `<div class="audit-panel">
    <h3>&#9888; Claims Audit &mdash; ${claim.note}</h3>
    <table class="audit-table">
      <thead><tr>
        <th>File</th><th>Original Claimed</th><th>Recomputed from Raw Data</th><th>Verdict</th>
      </tr></thead>
      <tbody>${rows}</tbody>
    </table>
  </div>`;
}

function buildExpCard(exp, stats) {
  const isClean = exp.is_clean;
  const cardClass = 'exp-card ' + (isClean ? 'is-clean' : 'is-original');
  const badge = isClean
    ? '<span class="clean-badge">CLEAN RERUN</span>'
    : '<span class="orig-badge">ORIGINAL</span>';

  const m = stats.mean;
  const nAcc = typeof stats.n_acc === 'number' ? stats.n_acc : stats.n || 0;

  // Accuracy bar with CI whiskers
  let barHtml = '';
  if (m !== null && m !== undefined && nAcc > 0) {
    const pct = (m * 100).toFixed(1);
    const pctNum = parseFloat(pct);
    const barClass = pctNum >= 90 ? '' : pctNum >= 60 ? 'mid' : 'bad';
    let ciHtml = '';
    if (stats.ci_lo != null && stats.ci_hi != null) {
      const loW = (stats.ci_lo * 100).toFixed(1);
      const hiW = (stats.ci_hi * 100).toFixed(1);
      const loPos = parseFloat(loW);
      const hiPos = parseFloat(hiW);
      ciHtml = `<div class="ci-lo" style="left:${loPos}%"></div>` +
        `<div class="ci-hi" style="left:${hiPos}%"></div>` +
        `<div class="ci-line" style="left:${loPos}%;width:${(hiPos-loPos).toFixed(1)}%"></div>` +
        `<div style="font-size:10px;color:var(--text-secondary);margin-top:3px">` +
        `95% CI: ${loW}%&ndash;${hiW}% &nbsp; N=${nAcc}</div>`;
    }
    barHtml = `<div class="acc-bar-wrap">
      <div class="acc-bar-label">
        <span>${stats.accuracy_field || 'accuracy'}</span>
        <span style="font-weight:700;font-size:14px;color:var(--text)">${pct}%</span>
      </div>
      <div class="acc-bar-track">
        <div class="acc-bar-fill ${barClass}" style="width:${pct}%"></div>
      </div>
      ${ciHtml}
    </div>`;
  } else if (nAcc === 0) {
    barHtml = `<div class="acc-bar-wrap"><span class="pending-badge">No data yet</span></div>`;
  }

  // Stat chips
  let statHtml = '';
  const addStat = (label, val) => {
    if (val !== null && val !== undefined) {
      statHtml += `<div class="stat-item"><span class="stat-label">${label}</span><span class="stat-val">${val}</span></div>`;
    }
  };
  addStat('Runs', exp.line_count || stats.n);
  if (stats.p50_latency != null) addStat('p50 lat', stats.p50_latency + 'ms');
  if (stats.p95_latency != null) addStat('p95 lat', stats.p95_latency + 'ms');
  if (stats.mean_hallucinations != null) addStat('Halls/run', stats.mean_hallucinations.toFixed(2));
  if (stats.mean_misses != null) addStat('Misses/run', stats.mean_misses.toFixed(2));

  return `<div class="${cardClass}" data-path="${exp.path}">
    <div class="exp-card-header">
      <h3>${exp.variation.replace(/_/g,' ')}</h3>
      ${badge}
      <span class="n-badge">${exp.line_count} lines</span>
    </div>
    ${barHtml}
    <div class="stat-grid">${statHtml}</div>
  </div>`;
}

// ---------------------------------------------------------------------------
// Experiment selection
// ---------------------------------------------------------------------------
async function selectExperiment(filePath) {
  // Update sidebar active state
  document.querySelectorAll('.exp-item.active').forEach(el => el.classList.remove('active'));
  document.querySelector(`.exp-item[data-path="${filePath}"]`)?.classList.add('active');

  state.currentFile = filePath;
  state.currentLine = 0;

  await loadAllRuns(filePath);
  if (state.sortedRuns.length > 0) {
    state.currentIndex = 0;
    await loadRun(state.sortedRuns[0].file, state.sortedRuns[0].line);
  } else {
    showEmptyState();
  }
}

async function loadAllRuns(filePath) {
  const exp = state.fileMap[filePath];
  if (!exp) return;

  // Load all line numbers for this file
  const lines = [];
  for (let i = 0; i < exp.line_count; i++) {
    lines.push({ file: filePath, line: i });
  }
  state.sortedRuns = lines;
  applyFilters();
}

async function loadRun(filePath, lineIndex) {
  if (!filePath) return;

  document.getElementById('loading').style.display = 'flex';
  document.getElementById('loading').querySelector('span').textContent = 'Loading run...';
  document.getElementById('error').classList.remove('visible');
  document.getElementById('empty-state') && (document.getElementById('empty-state').style.display = 'none');

  try {
    const resp = await fetch(`/api/run?file=${encodeURIComponent(filePath)}&line=${lineIndex}`);
    if (!resp.ok) {
      const errData = await resp.json();
      throw new Error(errData.error || `HTTP ${resp.status}`);
    }
    const data = await resp.json();
    state.currentData = data;
    state.currentLine = lineIndex;
    state.currentFile = filePath;
    renderRun(data, filePath, lineIndex);
    document.getElementById('loading').style.display = 'none';
  } catch (e) {
    document.getElementById('loading').style.display = 'none';
    showError('Failed to load run: ' + e.message);
  }
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------
function renderRun(data, filePath, lineIndex) {
  const exp = state.fileMap[filePath] || {};
  const lineNum = lineIndex + 1;
  const totalLines = exp.line_count || '?';

  // Update indicator
  document.getElementById('run-indicator').textContent =
    `Run #${lineNum} of ${totalLines} — ${exp.variation || ''}`;

  // Header
  const header = document.getElementById('run-header');
  const runNum = data.run || data.run_number || lineNum;
  const expName = data.experiment || exp.group || '';
  const variation = data.variation || data.variation_name || exp.variation || '';
  const success = data.success;
  const statusClass = success === true ? 'success' : (success === false ? 'failure' : 'unknown');
  const statusText = success === true ? 'SUCCESS' : (success === false ? 'FAILURE' : 'UNKNOWN');

  header.innerHTML = `
    <h2>${expName} / ${variation.replace(/_/g, ' ')}</h2>
    <span class="status-badge ${statusClass}">${statusText}</span>
    <span style="font-size:13px;color:var(--text-secondary)">Run ${runNum} — Line ${lineNum}</span>
  `;

  // Metrics
  const metricsContainer = document.getElementById('metrics');
  let metricsHtml = '';

  const addMetric = (label, value, cls='') => {
    if (value !== undefined && value !== null) {
      const valStr = typeof value === 'number' ? (Number.isInteger(value) ? value : value.toFixed(4)) : String(value);
      metricsHtml += `<div class="metric"><div class="label">${label}</div><div class="value ${cls}">${valStr}</div></div>`;
    }
  };

  const za = data.zone_accuracy;
  addMetric('Zone Accuracy', za, za >= 0.9 ? 'good' : za >= 0.5 ? 'mid' : 'bad');

  if (data.hallucinations !== undefined) {
    const h = data.hallucinations;
    addMetric('Hallucinations', h, h === 0 ? 'good' : h <= 1 ? 'mid' : 'bad');
  }
  if (data.misses !== undefined) {
    const m = data.misses;
    addMetric('Misses', m, m === 0 ? 'good' : 'bad');
  }
  if (data.n_hallucinated !== undefined) {
    const h = data.n_hallucinated;
    addMetric('Hallucinated', h, h === 0 ? 'good' : 'bad');
  }
  addMetric('Latency', data.latency_ms ? data.latency_ms.toFixed(1) + ' ms' : undefined);
  addMetric('Temperature', data.temperature);
  addMetric('JPEG Quality', data.jpeg_quality);
  addMetric('Objects', data.n_objects || data.n_objects_before || data.n_expected);
  addMetric('Reported', data.reported_count || data.n_reported);

  if (data.plan_complete !== undefined) {
    addMetric('Plan Complete', data.plan_complete ? 'Yes' : 'No', data.plan_complete ? 'good' : 'bad');
  }
  if (data.obs_accuracy !== undefined) {
    addMetric('Obs Accuracy', data.obs_accuracy, data.obs_accuracy >= 0.9 ? 'good' : 'bad');
  }
  if (data.shape_accuracy !== undefined) {
    addMetric('Shape Accuracy', data.shape_accuracy, data.shape_accuracy >= 0.9 ? 'good' : 'bad');
  }
  if (data.pre_chaos_accuracy !== undefined) {
    addMetric('Pre Chaos Acc', data.pre_chaos_accuracy, data.pre_chaos_accuracy >= 0.9 ? 'good' : 'mid');
    addMetric('Post Chaos Acc', data.post_chaos_accuracy, data.post_chaos_accuracy >= 0.9 ? 'good' : 'mid');
  }
  if (data.change_detected !== undefined) {
    addMetric('Change Detected', data.change_detected ? 'Yes' : 'No');
    addMetric('Change Correct', data.change_correct !== undefined ? (data.change_correct ? 'Yes' : 'No') : undefined);
  }
  if (data.count_error !== undefined) {
    addMetric('Count Error', data.count_error, data.count_error === 0 ? 'good' : 'bad');
  }
  if (data.detection_delay !== undefined) {
    addMetric('Detection Delay', data.detection_delay);
  }

  metricsContainer.innerHTML = metricsHtml;

  // Image
  const imgContainer = document.getElementById('image-container');
  if (data.image_b64) {
    imgContainer.className = 'visible';
    let src = data.image_b64;
    if (src.startsWith('data:')) {
      imgContainer.innerHTML = `<img src="${src}" alt="Experiment scene">`;
    } else {
      imgContainer.innerHTML = `<img src="data:image/png;base64,${src}" alt="Experiment scene">`;
    }
  } else if (data.image) {
    imgContainer.className = 'visible';
    imgContainer.innerHTML = `<img src="${data.image}" alt="Experiment scene">`;
  } else {
    imgContainer.className = '';
    imgContainer.innerHTML = '';
  }

  // Prompt
  document.getElementById('prompt-content').textContent = data.prompt_sent || data.instruction || '(No prompt)';

  // Raw response
  const responseEl = document.getElementById('response-content');
  if (data.raw_response) {
    let raw = data.raw_response;
    if (typeof raw === 'string') {
      try {
        const parsed = JSON.parse(raw);
        responseEl.innerHTML = syntaxHighlight(JSON.stringify(parsed, null, 2));
      } catch {
        responseEl.textContent = raw;
      }
    } else {
      responseEl.innerHTML = syntaxHighlight(JSON.stringify(raw, null, 2));
    }
  } else {
    responseEl.textContent = '(No raw response)';
  }

  // Extra fields (everything not already shown)
  const knownKeys = new Set([
    'run','run_number','experiment','variation','variation_name','scene','n_objects','n_objects_before','n_objects_after',
    'reported_count','count_error','zone_accuracy','zone_matches','zone_total','hallucinations','misses',
    'latency_ms','temperature','jpeg_quality','monochrome','show_grid','success','error','prompt_sent','instruction',
    'raw_response','timestamp','image_b64','image','shape_accuracy','size_accuracy','size_matches','size_total',
    'plan_complete','plan_length','first_step_correct','second_step_correct','obs_accuracy',
    'pre_chaos_accuracy','post_chaos_accuracy','dropped','chaos_type','any_change_detected','reacquire_at_tick',
    'change_detected','change_correct','expected_change','reported_change_type','detection_correct',
    'left_zone_accuracy','right_zone_accuracy','left_hallucinations','right_hallucinations',
    'left_misses','right_misses','description','n_hallucinated','hallucinated_objects','gripper_status',
    'n_boundary_objects','double_hallucinations','boundary_results',
    'background','background_rgb','info',
    'n_expected','n_reported','detected_overlap_correct','merged_single_object','n_observed_listings','scenario','overlap_pct',
    'mean_pre_chaos_accuracy','mean_post_chaos_accuracy','post_vs_pre_drop',
    'pre_perturb_accuracy','post_perturb_accuracy','detection_delay','ticks',
  ]);

  const extraFields = {};
  for (const [key, value] of Object.entries(data)) {
    if (!knownKeys.has(key) && !['image_b64','image','raw_response','prompt_sent','instruction'].includes(key)) {
      extraFields[key] = value;
    }
  }

  const extraContainer = document.getElementById('extra-fields');
  const extraKeys = Object.keys(extraFields);
  if (extraKeys.length > 0) {
    let tableHtml = '<table><thead><tr><th>Field</th><th>Value</th></tr></thead><tbody>';
    extraKeys.forEach(k => {
      let val = extraFields[k];
      if (typeof val === 'object') {
        val = JSON.stringify(val, null, 2);
      }
      tableHtml += `<tr><td style="font-weight:600;color:var(--accent)">${k}</td><td style="white-space:pre-wrap;font-family:monospace;font-size:12px;">${escHtml(String(val))}</td></tr>`;
    });
    tableHtml += '</tbody></table>';
    extraContainer.innerHTML = tableHtml;
  } else {
    extraContainer.innerHTML = '';
  }
}

function showEmptyState() {
  document.getElementById('loading').style.display = 'none';
  document.getElementById('error').classList.remove('visible');
  document.getElementById('empty-state').style.display = 'block';
  document.getElementById('run-header').innerHTML = '';
  document.getElementById('metrics').innerHTML = '';
  document.getElementById('image-container').className = '';
  document.getElementById('prompt-content').textContent = '';
  document.getElementById('response-content').textContent = '';
  document.getElementById('extra-fields').innerHTML = '';
  document.getElementById('run-indicator').textContent = '';
}

function showError(msg) {
  const el = document.getElementById('error');
  el.textContent = msg;
  el.classList.add('visible');
}

// ---------------------------------------------------------------------------
// Navigation
// ---------------------------------------------------------------------------
function navigateRun(delta) {
  if (state.sortedRuns.length === 0) return;
  state.currentIndex = Math.max(0, Math.min(state.sortedRuns.length - 1, state.currentIndex + delta));
  const target = state.sortedRuns[state.currentIndex];
  loadRun(target.file, target.line);
}

function jumpToRun() {
  const inp = document.getElementById('jump-input');
  const target = parseInt(inp.value);
  if (isNaN(target) || target < 1) return;
  // Find the run with this run number
  // We need to search through the data. First, try the current file.
  if (state.sortedRuns.length === 0) return;

  // Since sortedRuns are in file order, we need to find the right line.
  // This is an approximation: seek in the current file.
  // Actually, let's just navigate to the (target-1)-th line of the current file if possible.
  const exp = state.fileMap[state.currentFile];
  if (!exp) return;

  // Find runs with matching run number or just navigate by line index
  const lineIdx = Math.min(target - 1, exp.line_count - 1);
  // Find this in sortedRuns
  let idx = state.sortedRuns.findIndex(r => r.file === state.currentFile && r.line === lineIdx);
  if (idx >= 0) {
    state.currentIndex = idx;
    loadRun(state.sortedRuns[idx].file, state.sortedRuns[idx].line);
  } else {
    // Direct load
    state.currentIndex = 0;
    loadRun(state.currentFile, lineIdx);
  }
}

// ---------------------------------------------------------------------------
// Filtering & Sorting
// ---------------------------------------------------------------------------
function applyFilters() {
  const statusFilter = document.getElementById('status-filter').value;
  const accuracyFilter = parseFloat(document.getElementById('accuracy-filter').value);
  const sortBy = document.getElementById('sort-select').value;

  // We need to re-evaluate based on current data. Since we have line indices,
  // we need to fetch metadata for each run to filter. That's expensive.
  // Instead, let's do a simpler approach: re-fetch the first page after filter change.

  // For now, just reload the current view
  if (state.currentFile) {
    selectExperiment(state.currentFile);
  }
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------
function syntaxHighlight(json) {
  json = json.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  return json.replace(
    /("(\\u[a-fA-F0-9]{4}|\\[^u]|[^"\\])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g,
    function (match) {
      let cls = 'json-number';
      if (/^"/.test(match)) {
        if (/:$/.test(match)) {
          cls = 'json-key';
          match = match.replace(/"/g, '');
        } else {
          cls = 'json-string';
        }
      } else if (/true|false/.test(match)) {
        cls = 'json-boolean';
      } else if (/null/.test(match)) {
        cls = 'json-null';
      }
      return '<span class="' + cls + '">' + match + '</span>';
    }
  );
}

function escHtml(str) {
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# HTTP Request Handler
# ---------------------------------------------------------------------------

class VisualizerHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler that serves the visualizer."""

    # Shared state set from outside
    experiments = []
    file_map = {}
    results_dir = RESULTS_DIR

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        params = urllib.parse.parse_qs(parsed.query)

        try:
            if path == "/" or path == "/index.html":
                self._serve_html()
            elif path == "/api/experiments":
                self._handle_experiments()
            elif path == "/api/run":
                self._handle_run(params)
            elif path == "/api/file_info":
                self._handle_file_info(params)
            elif path == "/api/stats":
                self._handle_stats(params)
            elif path == "/api/audit":
                self._handle_audit(params)
            elif path.startswith("/api/"):
                self._send_json({"error": "Unknown endpoint"}, 404)
            else:
                self._send_error(404, "Not Found")
        except Exception as e:
            self._send_json({"error": str(e)}, 500)

    def _serve_html(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(HTML_PAGE.encode("utf-8"))

    def _handle_experiments(self):
        # Rebuild index on each request so the client gets live line counts
        # (files may be actively appended to by reruns)
        try:
            experiments, file_map = build_index(self.results_dir)
            VisualizerHandler.experiments = experiments
            VisualizerHandler.file_map = file_map
        except Exception:
            experiments = self.experiments  # fall back to cached

        total_runs = sum(e["line_count"] for e in self.experiments)
        self._send_json({
            "experiments": [{
                "id": e["id"],
                "dir": e["dir"],
                "file": e["file"],
                "path": e["path"],
                "group": e["group"],
                "variation": e["variation"],
                "round": e["round"],
                "line_count": e["line_count"],
                "is_clean": e.get("is_clean", False),
            } for e in self.experiments],
            "files": len(self.experiments),
            "total_runs": total_runs,
        })

    def _handle_run(self, params):
        file_path = params.get("file", [None])[0]
        line_str = params.get("line", [None])[0]

        if not file_path:
            self._send_json({"error": "Missing 'file' parameter"}, 400)
            return

        if line_str is None:
            self._send_json({"error": "Missing 'line' parameter"}, 400)
            return

        try:
            line_idx = int(line_str)
        except ValueError:
            self._send_json({"error": f"Invalid line number: {line_str}"}, 400)
            return

        exp = self.file_map.get(file_path)
        if not exp:
            self._send_json({"error": f"Unknown file: {file_path}"}, 404)
            return

        if line_idx < 0 or line_idx >= exp["line_count"]:
            self._send_json({
                "error": f"Line {line_idx} out of range (0-{exp['line_count'] - 1})"
            }, 400)
            return

        # Efficient read using line offsets
        offsets = exp["line_offsets"]
        fpath = self.results_dir / file_path
        data = read_line_at_offset(fpath, offsets[line_idx])

        if data is None:
            self._send_json({"error": f"Failed to read line {line_idx}"}, 500)
            return

        # Truncate image_b64 for the listing (send only first 100 chars)
        # Actually for the visualizer we want the full image, so send it as-is
        self._send_json(data)

    def _handle_file_info(self, params):
        file_path = params.get("file", [None])[0]
        if not file_path:
            self._send_json({"error": "Missing 'file' parameter"}, 400)
            return

        exp = self.file_map.get(file_path)
        if not exp:
            self._send_json({"error": f"Unknown file: {file_path}"}, 404)
            return

        self._send_json({
            "path": exp["path"],
            "group": exp["group"],
            "variation": exp["variation"],
            "round": exp["round"],
            "line_count": exp["line_count"],
        })

    def _handle_stats(self, params):
        """Compute honest metrics from raw JSONL records for a file."""
        file_path = params.get("file", [None])[0]
        if not file_path:
            self._send_json({"error": "Missing 'file' parameter"}, 400)
            return
        fpath = self.results_dir / file_path
        stats = compute_stats(fpath)
        self._send_json(stats)

    def _handle_audit(self, params):
        """Return audit data: original claimed vs recomputed from files.

        Uses the claim's specified field for both original and clean files so
        the comparison is always apples-to-apples.
        """
        # Original claims from the final report (hardcoded as reference labels only)
        CLAIMS = {
            "monochrome": {"claimed": 1.0, "field": "zone_accuracy", "note": "Final report claimed 100%"},
            "multistep":  {"claimed": 1.0, "field": "plan_complete",  "note": "Final report claimed 100%"},
            "jpeg":       {"claimed": 1.0, "field": "zone_accuracy",  "note": "Final report claimed 100% at all quality levels"},
        }
        audit = {}
        for group_name, claim in CLAIMS.items():
            group_files = [e for e in self.experiments if e["group"] == group_name]
            files_data = []
            for exp in group_files:
                # Force the same field as the claim for fair comparison
                stats = compute_stats(self.results_dir / exp["path"],
                                      forced_field=claim["field"])
                files_data.append({
                    "path": exp["path"],
                    "variation": exp["variation"],
                    "is_clean": exp.get("is_clean", False),
                    "stats": stats,
                    "claimed": claim["claimed"],
                    "field": claim["field"],
                })
            audit[group_name] = {
                "claim": claim,
                "files": files_data,
            }
        self._send_json(audit)

    def _send_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def _send_error(self, status, message):
        self.send_response(status)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(message.encode("utf-8"))

    def log_message(self, format, *args):
        """Quiet logging: only log non-static requests."""
        if not args[0].startswith("/api/"):
            return
        super().log_message(format, *args)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Build index
    print(f"Scanning {RESULTS_DIR} for experiment data...")
    experiments, file_map = build_index(RESULTS_DIR)
    print(f"Found {len(experiments)} experiment files with "
          f"{sum(e['line_count'] for e in experiments):,} total runs.")

    # Attach to handler
    VisualizerHandler.experiments = experiments
    VisualizerHandler.file_map = file_map
    VisualizerHandler.results_dir = RESULTS_DIR

    # Start server
    server = http.server.HTTPServer((HOST, PORT), VisualizerHandler)
    print(f"\n  Visualizer running at http://localhost:{PORT}")
    print(f"  Press Ctrl+C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
