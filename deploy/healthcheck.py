#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

ENV_PATTERN = re.compile(r"\$\{([^}:]+)(?::-([^}]*))?\}")


def expand_env(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        name = match.group(1)
        default = match.group(2) if match.group(2) is not None else ""
        return os.environ.get(name, default)
    return ENV_PATTERN.sub(repl, text)


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"config_not_found path={path}")
    return yaml.safe_load(expand_env(path.read_text())) or {}


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def fail(reason: str) -> int:
    print(f"HEALTHCHECK_FAIL reason={reason}")
    return 1


def ok(message: str = "ok") -> int:
    print(f"HEALTHCHECK_OK {message}")
    return 0


def import_check() -> None:
    for module in (
        "tradarbot.app.main",
        "tradarbot.storage.sqlite_store",
        "tradarbot.execution.exchange_factory",
        "tradarbot.monitoring.metrics",
        "tradarbot.monitoring.heartbeat",
        "tradarbot.monitoring.reporting",
    ):
        importlib.import_module(module)


def check_db(cfg: Dict[str, Any]) -> None:
    from tradarbot.storage.sqlite_store import SQLiteStore

    runtime = cfg.get("runtime", {}) or {}
    db_path = Path(os.environ.get("TRADAR_DB_PATH") or runtime.get("db_path") or "tradarbot.db")
    if not db_path.is_absolute():
        db_path = Path.cwd() / db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    store = SQLiteStore(str(db_path))
    store.init_schema()
    required_tables = ["runtime_heartbeat", "runtime_metrics", "runtime_status_events"]
    missing = [
        t for t in required_tables
        if store.conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (t,)).fetchone() is None
    ]
    store.conn.close()
    if missing:
        raise RuntimeError(f"monitoring_tables_missing tables={','.join(missing)}")


def check_artifacts(cfg: Dict[str, Any]) -> None:
    ml_live = cfg.get("ml_live", {}) or {}
    mode = str(ml_live.get("predictor_mode", ml_live.get("mode", "")) or "").lower()
    deployment = cfg.get("deployment", {}) or {}
    health = deployment.get("healthcheck", {}) or {}
    artifacts_required = as_bool(health.get("artifacts_required", True))
    if mode not in {"artifact", "artifacts"} or not artifacts_required:
        return

    runtime = cfg.get("runtime", {}) or {}
    artifact_dir = Path(os.environ.get("TRADAR_ARTIFACT_DIR") or runtime.get("artifact_dir") or ml_live.get("artifact_dir") or "artifacts/models/v50_sig_importance")
    if not artifact_dir.exists():
        raise RuntimeError(f"artifact_dir_missing path={artifact_dir}")

    required = ml_live.get("required_artifacts") or [
        "model_6h.joblib",
        "model_24h.joblib",
        "model_72h.joblib",
    ]
    missing = [name for name in required if not (artifact_dir / str(name)).exists()]
    if missing:
        raise RuntimeError(f"artifact_files_missing dir={artifact_dir} files={','.join(missing)}")


def check_provider_env(cfg: Dict[str, Any]) -> None:
    exec_cfg = cfg.get("execution_live", {}) or {}
    profile = str(os.environ.get("TRADAR_PROFILE") or (cfg.get("runtime", {}) or {}).get("profile") or "paper").lower()
    broker = str(exec_cfg.get("broker", "paper") or "paper").lower().replace("-", "_")
    mode = str(os.environ.get("EXECUTION_MODE") or exec_cfg.get("mode", "paper") or "paper").lower().replace("-", "_")
    provider = str(os.environ.get("EXECUTION_PROVIDER") or exec_cfg.get("provider", "alpaca") or "alpaca").lower().replace("-", "_")

    if profile == "live" and not as_bool(os.environ.get("TRADAR_CONFIRM_LIVE", "false")):
        raise RuntimeError("live_profile_requires_TRADAR_CONFIRM_LIVE=true")

    requires_keys = broker in {"live", "dry_run_live", "dryrun"} or mode in {"live", "paper", "testnet"}
    if broker == "paper":
        requires_keys = False
    if not requires_keys:
        return

    if provider.startswith("alpaca"):
        required = [exec_cfg.get("api_key_env", "ALPACA_API_KEY"), exec_cfg.get("api_secret_env", "ALPACA_API_SECRET")]
    elif provider.startswith("binance"):
        required = [exec_cfg.get("api_key_env", "BINANCE_API_KEY"), exec_cfg.get("api_secret_env", "BINANCE_API_SECRET")]
    else:
        required = []
    missing = [name for name in required if not os.environ.get(str(name))]
    if missing:
        raise RuntimeError(f"missing_provider_env vars={','.join(map(str, missing))}")


def optional_exchange_ping(cfg: Dict[str, Any]) -> None:
    deployment = cfg.get("deployment", {}) or {}
    health = deployment.get("healthcheck", {}) or {}
    if not as_bool(health.get("exchange_ping", False)):
        return
    from tradarbot.execution.exchange_factory import build_exchange_client
    client = build_exchange_client(cfg)
    try:
        client.ping()
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Tradar deployment healthcheck")
    parser.add_argument("--config", default=os.environ.get("TRADAR_CONFIG", "config/tradar.yaml"))
    args = parser.parse_args()

    try:
        cfg = load_config(Path(args.config))
        import_check()
        if as_bool(((cfg.get("deployment", {}) or {}).get("healthcheck", {}) or {}).get("db_required", True)):
            check_db(cfg)
        check_artifacts(cfg)
        check_provider_env(cfg)
        optional_exchange_ping(cfg)
        return ok(f"config={args.config}")
    except Exception as exc:
        return fail(str(exc).replace(" ", "_"))


if __name__ == "__main__":
    sys.exit(main())
