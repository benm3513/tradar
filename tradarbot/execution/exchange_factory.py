from __future__ import annotations

from typing import Any, Dict

from tradarbot.execution.alpaca_client import AlpacaClient
from tradarbot.execution.dry_run_client import DryRunExchangeClient


def _provider(value: Any) -> str:
    return str(value or "binance_us").lower().replace("-", "_")


def build_exchange_client(cfg: Dict[str, Any]):
    exec_cfg = dict((cfg or {}).get("execution_live", {}) or {})
    provider = _provider(exec_cfg.get("provider", exec_cfg.get("exchange", "binance_us")))
    broker = str(exec_cfg.get("broker", "paper") or "paper").lower().replace("-", "_")
    mode = str(exec_cfg.get("mode", "paper") or "paper").lower().replace("-", "_")

    if broker in {"dry_run_live", "dryrun"} or mode in {"dry_run", "dryrun"}:
        return DryRunExchangeClient(cfg)

    if provider in {"alpaca", "alpaca_paper", "alpaca_live"}:
        return AlpacaClient(cfg)

    if provider in {"binance_us", "binanceus"}:
        from tradarbot.execution.binance_us_client import BinanceUSClient

        return BinanceUSClient(cfg)

    if provider in {"binance", "binance_spot", "binance_spot_testnet"}:
        from tradarbot.execution.binance_client import BinanceClient

        return BinanceClient(cfg)

    raise ValueError(
        "Unsupported execution_live.provider={!r}. Supported providers: alpaca, binance_us, binance.".format(
            provider
        )
    )
