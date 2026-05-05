#!/usr/bin/env python3
"""Convenience wrapper for exporting Phase 5.3 model artifacts.

This intentionally reuses scripts/train_spike_model.py so the artifact export
path cannot drift from the research training path.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train and export Tradar ML model artifacts.")
    p.add_argument("--db-path", required=True)
    p.add_argument("--input-table", required=True)
    p.add_argument("--model-version", required=True)
    p.add_argument("--artifact-dir", default=None)
    p.add_argument("--train-all-horizons", action="store_true", default=True)
    p.add_argument("--model", choices=["hgb", "logistic"], default="hgb")
    p.add_argument("--metrics-table", default=None)
    p.add_argument("--predictions-table-prefix", default=None)
    p.add_argument("--feature-importance-table-prefix", default=None)
    p.add_argument("--enable-permutation-importance", action="store_true")
    p.add_argument("--permutation-importance-table-prefix", default=None)
    p.add_argument("--if-exists", choices=["fail", "replace", "append"], default="replace")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> int:
    args = parse_args()
    artifact_dir = args.artifact_dir or f"artifacts/models/{args.model_version}"
    metrics_table = args.metrics_table or f"spike_model_metrics_{args.model_version}"
    predictions_prefix = args.predictions_table_prefix or f"spike_model_predictions_{args.model_version}"
    fi_prefix = args.feature_importance_table_prefix or f"spike_feature_importance_{args.model_version}"
    pi_prefix = args.permutation_importance_table_prefix or f"spike_permutation_importance_{args.model_version}"

    cmd = [
        sys.executable,
        str(Path(__file__).with_name("train_spike_model.py")),
        "--db-path", args.db_path,
        "--input-table", args.input_table,
        "--train-all-horizons",
        "--model", args.model,
        "--metrics-table", metrics_table,
        "--predictions-table-prefix", predictions_prefix,
        "--feature-importance-table-prefix", fi_prefix,
        "--save-artifacts",
        "--artifact-dir", artifact_dir,
        "--model-version", args.model_version,
        "--if-exists", args.if_exists,
        "--log-level", args.log_level,
    ]
    if args.enable_permutation_importance:
        cmd.extend([
            "--enable-permutation-importance",
            "--permutation-importance-table-prefix", pi_prefix,
        ])

    print("Running:")
    print(" ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
