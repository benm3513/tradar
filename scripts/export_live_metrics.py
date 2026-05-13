#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Optional

from tradarbot.storage.sqlite_store import SQLiteStore


def parse_since(value: Optional[str]) -> Optional[int]:
    if not value:
        return None

    text = str(value).strip().lower()
    now = int(time.time() * 1000)

    if text.endswith("m"):
        return now - int(float(text[:-1]) * 60 * 1000)
    if text.endswith("h"):
        return now - int(float(text[:-1]) * 3600 * 1000)
    if text.endswith("d"):
        return now - int(float(text[:-1]) * 86400 * 1000)

    return int(float(text))


def normalize_row(row: dict) -> dict:
    labels = row.get("labels")
    if labels is None:
        labels = row.get("labels_json")

    if isinstance(labels, str):
        try:
            labels_obj = json.loads(labels) if labels else {}
        except json.JSONDecodeError:
            labels_obj = {"raw": labels}
    else:
        labels_obj = labels or {}

    return {
        "ts_ms": row.get("ts_ms"),
        "metric_group": row.get("metric_group"),
        "metric_name": row.get("metric_name"),
        "metric_numeric_value": row.get("metric_numeric_value", row.get("metric_value")),
        "metric_text": row.get("metric_text"),
        "labels": labels_obj,
    }


def write_csv(path: Path, rows) -> None:
    fieldnames = [
        "ts_ms",
        "metric_group",
        "metric_name",
        "metric_numeric_value",
        "metric_text",
        "labels",
    ]

    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for row in rows:
            out = normalize_row(dict(row))
            out["labels"] = json.dumps(out.get("labels") or {}, sort_keys=True)
            writer.writerow(out)


def aggregate_rows(rows, window_seconds: int):
    window_ms = int(window_seconds) * 1000
    buckets = {}

    for raw in rows:
        row = normalize_row(dict(raw))
        value = row.get("metric_numeric_value")

        if value is None:
            continue

        bucket_ts = (int(row["ts_ms"]) // window_ms) * window_ms
        key = (bucket_ts, row["metric_group"], row["metric_name"])
        cur = buckets.setdefault(key, {"sum": 0.0, "count": 0})
        cur["sum"] += float(value)
        cur["count"] += 1

    return [
        {
            "ts_ms": key[0],
            "metric_group": key[1],
            "metric_name": key[2],
            "metric_numeric_value": val["sum"] / max(val["count"], 1),
            "metric_text": None,
            "labels": {
                "aggregation": "avg",
                "window_seconds": window_seconds,
                "count": val["count"],
            },
        }
        for key, val in sorted(buckets.items())
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Tradar runtime metrics from SQLite")
    parser.add_argument("--db-path", default="tradarbot.db")
    parser.add_argument("--output", required=True)
    parser.add_argument("--format", choices=["csv", "json"], default=None)
    parser.add_argument("--metric-group", default=None)
    parser.add_argument("--metric-name", default=None)
    parser.add_argument("--since", default=None, help="ts_ms or relative window like 30m, 6h, 2d")
    parser.add_argument("--until-ts-ms", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--aggregate-window-seconds",
        type=int,
        default=None,
        help="Average numeric metrics into fixed windows",
    )
    args = parser.parse_args()

    store = SQLiteStore(args.db_path)
    store.init_schema()

    rows = store.query_runtime_metrics(
        metric_group=args.metric_group,
        metric_name=args.metric_name,
        since_ts_ms=parse_since(args.since),
        until_ts_ms=args.until_ts_ms,
        limit=args.limit,
    )

    if args.aggregate_window_seconds:
        rows = aggregate_rows(rows, args.aggregate_window_seconds)
    else:
        rows = [normalize_row(dict(row)) for row in rows]

    output = Path(args.output)
    fmt = args.format or output.suffix.lower().lstrip(".") or "csv"

    if fmt == "json":
        output.write_text(json.dumps(rows, indent=2, sort_keys=True, default=str))
    else:
        write_csv(output, rows)

    print(f"EXPORTED_METRICS rows={len(rows)} output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())