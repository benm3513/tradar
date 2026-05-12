from __future__ import annotations

import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional

import pandas as pd

from tradarbot.core.events import CandleEvent
from tradarbot.ml.live_regime import compute_live_regime
from tradarbot.ml.context_snapshot import LiveContextSnapshot


class RollingFeatureState:
    """In-memory live context table builder.

    Stores recent CandleEvents by symbol and exposes pandas frames that match the
    inputs expected by tradarbot.ml.live_features.build_live_feature_frame().
    """

    def __init__(
        self,
        lookback_bars: int,
        min_ready_bars: int = 24,
        max_symbols: Optional[int] = None,
        interval_s: int = 1,
    ):
        self.lookback_bars = max(1, int(lookback_bars or 168))
        self.min_ready_bars = max(1, int(min_ready_bars or 24))
        self.max_symbols = int(max_symbols) if max_symbols is not None else None
        self.interval_s = int(interval_s or 1)
        self._candles: Dict[str, Deque[CandleEvent]] = {}
        self.last_update_ts_ms: Optional[int] = None
        self.update_count = 0
        self.last_regime: Dict[str, float] = {}
        self.last_snapshot_metadata: Dict[str, Any] = {}

    def update_candle(self, ev: CandleEvent) -> None:
        if self.max_symbols is not None and ev.symbol not in self._candles and len(self._candles) >= self.max_symbols:
            # Keep the object bounded; ignore unexpected symbols beyond cap.
            return
        dq = self._candles.setdefault(ev.symbol, deque(maxlen=self.lookback_bars))
        dq.append(ev)
        self.last_update_ts_ms = int(ev.ts_ms)
        self.update_count += 1

    def symbols(self) -> List[str]:
        return sorted(self._candles.keys())

    def ready_symbols(self) -> List[str]:
        return sorted(sym for sym, rows in self._candles.items() if len(rows) >= self.min_ready_bars)

    def is_ready(self, symbol: str) -> bool:
        return len(self._candles.get(symbol, [])) >= self.min_ready_bars

    def get_symbol_frame(self, symbol: str) -> pd.DataFrame:
        rows = list(self._candles.get(symbol, []))
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(
            {
                "symbol": [r.symbol for r in rows],
                "ts_ms": [int(r.ts_ms) for r in rows],
                "timestamp": pd.to_datetime([int(r.ts_ms) for r in rows], unit="ms", utc=True),
                "open": [float(r.open) for r in rows],
                "high": [float(r.high) for r in rows],
                "low": [float(r.low) for r in rows],
                "close": [float(r.close) for r in rows],
                "volume": [float(r.volume) for r in rows],
            }
        )

    def frames_by_symbol(self, ready_only: bool = True) -> Dict[str, pd.DataFrame]:
        symbols = self.ready_symbols() if ready_only else self.symbols()
        return {sym: self.get_symbol_frame(sym) for sym in symbols}

    def compute_regime(self, ready_only: bool = True) -> Dict[str, float]:
        frames = self.frames_by_symbol(ready_only=ready_only)
        self.last_regime = compute_live_regime(frames)
        return dict(self.last_regime)

    def context_snapshot(self, ts_ms: Optional[int] = None) -> LiveContextSnapshot:
        from tradarbot.ml.live_features import build_live_feature_frame

        ready = self.ready_symbols()
        frames = self.frames_by_symbol(ready_only=True)
        regime = compute_live_regime(frames)
        feature_frame = build_live_feature_frame(
            symbols=ready,
            ctx=None,
            lookback_bars=self.lookback_bars,
            interval_s=self.interval_s,
            candles_by_symbol=frames,
            market_regime=regime,
        )
        metadata = self.health_snapshot()
        metadata["created_at_s"] = time.time()
        self.last_regime = dict(regime)
        self.last_snapshot_metadata = dict(metadata)
        return LiveContextSnapshot(
            ts_ms=int(ts_ms if ts_ms is not None else (self.last_update_ts_ms or int(time.time() * 1000))),
            symbols=self.symbols(),
            feature_frame=feature_frame,
            regime=regime,
            ready_symbols=ready,
            metadata=metadata,
        )

    def health_snapshot(self) -> Dict[str, Any]:
        counts = {sym: len(rows) for sym, rows in self._candles.items()}
        return {
            "enabled": True,
            "symbols": len(self._candles),
            "ready_symbols": len(self.ready_symbols()),
            "lookback_bars": self.lookback_bars,
            "min_ready_bars": self.min_ready_bars,
            "interval_s": self.interval_s,
            "last_update_ts_ms": self.last_update_ts_ms,
            "update_count": self.update_count,
            "bars_by_symbol": counts,
        }
