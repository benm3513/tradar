from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional


def _fmt_money(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except Exception:
        return "0.00"


def _fmt_bool(value: Any) -> str:
    return "yes" if bool(value) else "no"


class RuntimeReporter:
    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.cfg = dict((cfg or {}).get("monitoring", {}) or {})

    def build_runtime_summary(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "runtime": snapshot.get("runtime", {}),
            "health": self.build_health_summary(snapshot),
            "market_data": self.build_market_data_summary(snapshot),
            "ml": self.build_ml_summary(snapshot),
            "execution": self.build_execution_summary(snapshot),
            "system": snapshot.get("system", {}),
        }

    def build_health_summary(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        safety = snapshot.get("safety", {}) or {}
        return {
            "status": safety.get("runtime_health_status", "UNKNOWN"),
            "safe_mode": bool(safety.get("safe_mode", False)),
            "kill_switch": bool(safety.get("kill_switch", False)),
            "api_errors": safety.get("api_errors", 0),
            "order_rejections": safety.get("order_rejections", 0),
            "stale_data_violations": safety.get("stale_data_violations", 0),
            "messages": safety.get("health_messages", []),
        }

    def build_execution_summary(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        exe = snapshot.get("execution", {}) or {}
        return {
            "cash": exe.get("cash", 0.0),
            "equity": exe.get("equity", 0.0),
            "realized_pnl": exe.get("realized_pnl", 0.0),
            "unrealized_pnl": exe.get("unrealized_pnl", 0.0),
            "open_positions": exe.get("open_positions", 0),
            "exposure": exe.get("exposure", 0.0),
            "fills": exe.get("fills", 0),
            "rejected_orders": exe.get("rejected_orders", 0),
        }

    def build_market_data_summary(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        md = snapshot.get("market_data", {}) or {}
        return {
            "ws_connected": md.get("ws_connected", False),
            "ws_disconnect_count": md.get("ws_disconnect_count", 0),
            "poll_ok": md.get("poll_ok", 0),
            "poll_err": md.get("poll_err", 0),
            "active_symbols": md.get("active_symbols", 0),
            "ready_symbols": md.get("ready_symbols", 0),
            "stale_symbols": md.get("stale_symbols", []),
            "stale_global": md.get("stale_global", False),
        }

    def build_ml_summary(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        ml = snapshot.get("ml", {}) or {}
        return {
            "feature_updates": ml.get("feature_updates", 0),
            "prediction_count": ml.get("prediction_count", 0),
            "candidate_count": ml.get("candidate_count", 0),
            "ranking_count": ml.get("ranking_count", 0),
            "signal_count": ml.get("signal_count", 0),
            "inference_failures": ml.get("inference_failures", 0),
            "fallback_predictions": ml.get("fallback_predictions", 0),
            "top_symbols": ml.get("top_symbols", []),
        }

    def log_line(self, label: str, summary: Dict[str, Any]) -> str:
        return f"{label} {json.dumps(summary, sort_keys=True, default=str)}"

    def compact_status_line(self, snapshot: Dict[str, Any]) -> str:
        runtime = snapshot.get("runtime", {}) or {}
        health = self.build_health_summary(snapshot)
        exe = self.build_execution_summary(snapshot)
        md = self.build_market_data_summary(snapshot)
        ml = self.build_ml_summary(snapshot)
        return (
            f"profile={runtime.get('runtime_profile')} uptime_s={runtime.get('uptime_seconds')} "
            f"health={health.get('status')} safe_mode={_fmt_bool(health.get('safe_mode'))} "
            f"kill_switch={_fmt_bool(health.get('kill_switch'))} "
            f"equity={_fmt_money(exe.get('equity'))} cash={_fmt_money(exe.get('cash'))} "
            f"exposure={_fmt_money(exe.get('exposure'))} positions={exe.get('open_positions')} "
            f"poll_ok={md.get('poll_ok')} poll_err={md.get('poll_err')} "
            f"ready={md.get('ready_symbols')}/{md.get('active_symbols')} "
            f"ml_features={ml.get('feature_updates')} ml_signals={ml.get('signal_count')}"
        )
