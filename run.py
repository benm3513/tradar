from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

from tradarbot.app.main import main


VALID_PROFILES = {"paper", "shadow", "dry_run", "dry_run_live", "live"}
TRUTHY = {"1", "true", "yes", "y", "on"}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in TRUTHY


def _expand_env(text: str) -> str:
    import re
    pattern = re.compile(r"\$\{([^}:]+)(?::-([^}]*))?\}")

    def repl(match: re.Match[str]) -> str:
        name = match.group(1)
        default = match.group(2) if match.group(2) is not None else ""
        return os.environ.get(name, default)

    return pattern.sub(repl, text)


def load_config_for_guardrails(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise SystemExit(f"LIVE_MODE_BLOCKED reason=config_not_found path={config_path}")
    return yaml.safe_load(_expand_env(path.read_text())) or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tradar runtime entrypoint")
    parser.add_argument("--config", default=os.environ.get("TRADAR_CONFIG", "config/tradar.yaml"))
    parser.add_argument("--profile", choices=sorted(VALID_PROFILES), default=os.environ.get("TRADAR_PROFILE", "paper"))
    parser.add_argument("--live-confirm", action="store_true", help="Required explicit confirmation for live execution")
    parser.add_argument("--healthcheck", action="store_true", help="Run deploy healthcheck and exit")
    parser.add_argument("--print-status", action="store_true", help="Print current runtime status from SQLite and exit")
    parser.add_argument("--metrics-once", action="store_true", help="Alias for --print-status --json")
    parser.add_argument("--export-metrics", default=None, help="Export runtime metrics to CSV/JSON and exit")
    parser.add_argument("--db-path", default=os.environ.get("TRADAR_DB_PATH", "tradarbot.db"), help="SQLite DB path for operator commands")
    parser.add_argument("--log-level", default=None, help="Override runtime log level")
    return parser.parse_args()


def run_healthcheck(config_path: str, profile: str | None = None) -> int:
    healthcheck = Path(__file__).resolve().parent / "deploy" / "healthcheck.py"
    if not healthcheck.exists():
        print(f"HEALTHCHECK_FAIL reason=healthcheck_missing path={healthcheck}", file=sys.stderr)
        return 1
    env = dict(os.environ)
    if profile:
        env["TRADAR_PROFILE"] = profile
    return subprocess.call([sys.executable, str(healthcheck), "--config", config_path], env=env)


def _required_env_vars(cfg: Dict[str, Any], profile: str) -> List[str]:
    startup = cfg.get("live_startup", {}) or {}
    explicit = list(startup.get("required_env_vars") or [])
    exec_cfg = cfg.get("execution_live", {}) or {}
    provider = str(exec_cfg.get("provider", "alpaca") or "alpaca").lower()
    broker = str(exec_cfg.get("broker", "") or "").lower().replace("-", "_")
    mode = str(exec_cfg.get("mode", "") or "").lower().replace("-", "_")

    if profile == "live" or broker == "live" or mode == "live":
        if provider.startswith("alpaca"):
            explicit.extend([
                str(exec_cfg.get("api_key_env", "ALPACA_API_KEY") or "ALPACA_API_KEY"),
                str(exec_cfg.get("api_secret_env", "ALPACA_API_SECRET") or "ALPACA_API_SECRET"),
            ])
    return sorted(set(v for v in explicit if v))


def _print_startup_summary(cfg: Dict[str, Any], profile: str, config_path: str) -> None:
    exec_cfg = cfg.get("execution_live", {}) or {}
    ml_live = cfg.get("ml_live", {}) or {}
    safety = cfg.get("safety", {}) or {}
    rollout = cfg.get("rollout", {}) or {}
    print(
        "STARTUP_SUMMARY "
        f"config={config_path} profile={profile} "
        f"rollout_stage={rollout.get('stage', 'n/a')} "
        f"ml_live_enabled={ml_live.get('enabled')} ml_live_mode={ml_live.get('mode')} "
        f"execution_live_enabled={exec_cfg.get('enabled')} "
        f"broker={exec_cfg.get('broker')} provider={exec_cfg.get('provider')} mode={exec_cfg.get('mode')} "
        f"safety_enabled={safety.get('enabled', True)}"
    )


def validate_live_startup(args: argparse.Namespace, cfg: Dict[str, Any]) -> None:
    profile = str(args.profile).lower().replace("-", "_")
    exec_cfg = cfg.get("execution_live", {}) or {}
    runtime_profile = str((cfg.get("runtime", {}) or {}).get("profile", profile) or profile).lower().replace("-", "_")
    startup = cfg.get("live_startup", {}) or {}

    if profile != "live":
        return

    reasons: List[str] = []

    if runtime_profile != "live":
        reasons.append(f"runtime_profile_not_live:{runtime_profile}")

    if not _as_bool(exec_cfg.get("enabled", False)):
        reasons.append("execution_live_disabled")

    broker = str(exec_cfg.get("broker", "") or "").lower().replace("-", "_")
    mode = str(exec_cfg.get("mode", "") or "").lower().replace("-", "_")
    if broker != "live":
        reasons.append(f"broker_not_live:{broker}")
    if mode != "live":
        reasons.append(f"mode_not_live:{mode}")

    env_confirm_var = str(startup.get("confirm_env_var", "TRADAR_CONFIRM_LIVE") or "TRADAR_CONFIRM_LIVE")
    env_confirmed = _as_bool(os.environ.get(env_confirm_var, "false"))
    if not args.live_confirm:
        reasons.append("missing_--live-confirm")
    if not env_confirmed:
        reasons.append(f"{env_confirm_var}_not_true")

    ack_env = startup.get("require_ack_env_var", "TRADAR_LIVE_ACK")
    expected_ack = startup.get("required_ack_value", "I_UNDERSTAND_LIVE_RISK")
    if ack_env and expected_ack and os.environ.get(str(ack_env)) != str(expected_ack):
        reasons.append(f"{ack_env}_missing_or_invalid")

    missing_env = [name for name in _required_env_vars(cfg, profile) if not os.environ.get(name)]
    if missing_env:
        reasons.append("missing_env:" + ",".join(missing_env))

    if reasons:
        print("LIVE_MODE_BLOCKED reason=" + ";".join(reasons), file=sys.stderr)
        raise SystemExit(2)

    if _as_bool(startup.get("require_healthcheck", True)):
        rc = run_healthcheck(args.config, profile="live")
        if rc != 0:
            print(f"LIVE_MODE_BLOCKED reason=healthcheck_failed exit_code={rc}", file=sys.stderr)
            raise SystemExit(rc)

    print(
        "LIVE_MODE_CONFIRMED "
        f"profile={profile} provider={exec_cfg.get('provider')} broker={broker} mode={mode} "
        f"config={args.config}"
    )


def configure_environment(args: argparse.Namespace, cfg: Dict[str, Any]) -> None:
    profile = str(args.profile).lower().replace("-", "_")
    os.environ["TRADAR_CONFIG"] = str(args.config)
    os.environ["TRADAR_PROFILE"] = profile
    if args.log_level:
        os.environ["TRADAR_LOG_LEVEL"] = str(args.log_level).upper()
    if args.live_confirm:
        os.environ["TRADAR_CLI_LIVE_CONFIRM"] = "true"

    validate_live_startup(args, cfg)


def cli() -> int:
    args = parse_args()
    cfg = load_config_for_guardrails(args.config)
    _print_startup_summary(cfg, args.profile, args.config)
    configure_environment(args, cfg)

    if args.healthcheck:
        return run_healthcheck(args.config, profile=args.profile)

    if args.print_status or args.metrics_once:
        from scripts.print_live_status import load_status, print_human
        import json
        payload = load_status(args.db_path, tail_events=10, tail_fills=10)
        if args.metrics_once:
            print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        else:
            print_human(payload)
        return 0

    if args.export_metrics:
        from scripts.export_live_metrics import main as export_main
        sys.argv = [sys.argv[0], "--db-path", args.db_path, "--output", args.export_metrics]
        return export_main()

    try:
        asyncio.run(main())
        return 0
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(cli())
