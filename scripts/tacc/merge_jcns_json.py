#!/usr/bin/env python3
"""Merge per-seed JCNS benchmark JSON files into a single summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any


def _extract_runs(results: Dict[str, Any]) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    out: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for scenario, methods in results.items():
        if scenario == "metadata":
            continue
        out[scenario] = {}
        for method, data in methods.items():
            out[scenario].setdefault(method, [])
            out[scenario][method].extend(data.get("runs", []))
    return out


def _aggregate(runs: Dict[str, Dict[str, List[Dict[str, Any]]]], metadata: Dict[str, Any]) -> Dict[str, Any]:
    aggregated: Dict[str, Any] = {}
    for scenario, methods in runs.items():
        aggregated[scenario] = {}
        for method, run_list in methods.items():
            converge_times = [r["measurements_to_converge"] for r in run_list if r["measurements_to_converge"] is not None]
            final_errors = [r["final_error"] for r in run_list]
            aggregated[scenario][method] = {
                "mean_converge": float(sum(converge_times) / len(converge_times)) if converge_times else float("inf"),
                "std_converge": float((sum((x - (sum(converge_times) / len(converge_times))) ** 2 for x in converge_times) / len(converge_times)) ** 0.5) if len(converge_times) > 1 else 0.0,
                "converge_rate": len(converge_times) / len(run_list) if run_list else 0.0,
                "mean_final_error": float(sum(final_errors) / len(final_errors)) if final_errors else float("inf"),
                "std_final_error": float((sum((x - (sum(final_errors) / len(final_errors))) ** 2 for x in final_errors) / len(final_errors)) ** 0.5) if len(final_errors) > 1 else 0.0,
                "runs": run_list,
            }
    aggregated["metadata"] = metadata
    return aggregated


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge per-seed JCNS benchmark JSONs.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input JSON files")
    parser.add_argument("--output", required=True, help="Output JSON file")
    args = parser.parse_args()

    inputs = [Path(p) for p in args.inputs]
    merged_runs: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    metadata: Dict[str, Any] = {}

    for path in inputs:
        data = json.loads(path.read_text())
        metadata = data.get("metadata", metadata)
        runs = _extract_runs(data)
        for scenario, methods in runs.items():
            merged_runs.setdefault(scenario, {})
            for method, run_list in methods.items():
                merged_runs[scenario].setdefault(method, [])
                merged_runs[scenario][method].extend(run_list)

    aggregated = _aggregate(merged_runs, metadata)
    Path(args.output).write_text(json.dumps(aggregated, indent=2))


if __name__ == "__main__":
    main()
