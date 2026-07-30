"""
Summarize experiment metric JSON files into CSV and Markdown tables.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path


FIELDS = [
    "name",
    "protocol",
    "model_type",
    "degradation",
    "severity",
    "rank1",
    "rank5",
    "rank10",
    "rank20",
    "mAP",
    "valid_queries",
]


def read_metric(path: Path):
    with path.open("r", encoding="utf-8") as f:
        row = json.load(f)
    row["name"] = path.stem
    row.setdefault("degradation", "clean")
    row.setdefault("severity", "")
    return row


def pct(value):
    if value == "" or value is None:
        return ""
    return f"{float(value) * 100:.2f}"


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            out = {field: row.get(field, "") for field in FIELDS}
            for field in ["rank1", "rank5", "rank10", "rank20", "mAP"]:
                out[field] = pct(out[field])
            writer.writerow(out)


def write_markdown(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["Name", "Model", "Degradation", "Severity", "Rank-1", "Rank-5", "Rank-10", "mAP"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [
            row.get("name", ""),
            row.get("model_type", ""),
            row.get("degradation", "clean"),
            str(row.get("severity", "")),
            pct(row.get("rank1", "")),
            pct(row.get("rank5", "")),
            pct(row.get("rank10", "")),
            pct(row.get("mAP", "")),
        ]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize metric JSON files.")
    parser.add_argument("metrics", nargs="+", type=Path,
                        help="Metric JSON files from evaluators.")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    return parser.parse_args()


def expand_metric_paths(paths: list[Path]):
    expanded = []
    for path in paths:
        text = str(path)
        if "*" in text or "?" in text:
            expanded.extend(Path(p) for p in sorted(glob.glob(text)))
        else:
            expanded.append(path)
    return expanded


def main():
    args = parse_args()
    metric_paths = expand_metric_paths(args.metrics)
    if not metric_paths:
        raise FileNotFoundError("No metric JSON files matched the provided paths.")
    rows = [read_metric(path) for path in metric_paths]
    write_csv(args.csv, rows)
    write_markdown(args.markdown, rows)
    print(f"Saved CSV: {args.csv}")
    print(f"Saved Markdown: {args.markdown}")


if __name__ == "__main__":
    main()
