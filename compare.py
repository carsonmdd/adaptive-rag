"""
Compare evaluation results across systems.

Usage
-----
python compare.py results/rqrag_simple.json results/treehop_simple.json results/adaptive_simple.json
"""

import argparse
import json
import sys


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def print_table(results: list[dict]):
    all_types = sorted(
        {qtype for r in results for qtype in r["by_type"]}
    )

    col_w = 14
    header_w = 22

    # Header row
    systems = [r["system"] for r in results]
    header = f"{'Subset/Type':{header_w}}" + "".join(
        f"{'F1':>{col_w//2}}{'Lat(s)':>{col_w//2}}" for _ in systems
    )
    sys_header = f"{'':{header_w}}" + "".join(
        f"{s:>{col_w}}" for s in systems
    )
    print(sys_header)
    print(f"{'Subset/Type':{header_w}}" + "".join(
        f"{'F1':>{col_w//2}}{'Lat(s)':>{col_w//2}}" for _ in systems
    ))
    print("-" * (header_w + col_w * len(systems)))

    def row(label, f1s, lats):
        line = f"{label:{header_w}}"
        for f1, lat in zip(f1s, lats):
            line += f"{f1:>{col_w//2}.3f}{lat:>{col_w//2}.1f}"
        print(line)

    # Overall
    row(
        "Overall",
        [r["overall_f1"] for r in results],
        [r["avg_latency"] for r in results],
    )
    print()

    # Per type
    for qtype in all_types:
        f1s = [r["by_type"].get(qtype, {}).get("f1", float("nan")) for r in results]
        lats = [r["by_type"].get(qtype, {}).get("avg_latency", float("nan")) for r in results]
        ns = [r["by_type"].get(qtype, {}).get("n", 0) for r in results]
        row(f"{qtype} (n={ns[0]})", f1s, lats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", help="Result JSON files from evaluate.py")
    args = parser.parse_args()

    results = [load(p) for p in args.results]
    print_table(results)
