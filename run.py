from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
from pathlib import Path

from tradarbot.app.main import main


VALID_PROFILES = {"paper", "dry_run", "live"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tradar runtime entrypoint")
    parser.add_argument("--config", default=os.environ.get("TRADAR_CONFIG", "config/tradar.yaml"))
    parser.add_argument("--profile", choices=sorted(VALID_PROFILES), default=os.environ.get("TRADAR_PROFILE", "paper"))
    parser.add_argument("--healthcheck", action="store_true", help="Run deploy healthcheck and exit")
    parser.add_argument("--log-level", default=None, help="Override runtime log level")
    return parser.parse_args()


def run_healthcheck(config_path: str) -> int:
    healthcheck = Path(__file__).resolve().parent / "deploy" / "healthcheck.py"
    if not healthcheck.exists():
        print(f"HEALTHCHECK_FAIL reason=healthcheck_missing path={healthcheck}", file=sys.stderr)
        return 1
    return subprocess.call([sys.executable, str(healthcheck), "--config", config_path])


def configure_environment(args: argparse.Namespace) -> None:
    os.environ["TRADAR_CONFIG"] = str(args.config)
    os.environ["TRADAR_PROFILE"] = str(args.profile)
    if args.log_level:
        os.environ["TRADAR_LOG_LEVEL"] = str(args.log_level).upper()

    if args.profile == "live" and os.environ.get("TRADAR_CONFIRM_LIVE", "false").lower() not in {"1", "true", "yes", "y", "on"}:
        raise SystemExit("Refusing live profile without TRADAR_CONFIRM_LIVE=true")


def cli() -> int:
    args = parse_args()
    configure_environment(args)

    if args.healthcheck:
        return run_healthcheck(args.config)

    try:
        asyncio.run(main())
        return 0
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(cli())
