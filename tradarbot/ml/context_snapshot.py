from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import pandas as pd


@dataclass(frozen=True)
class LiveContextSnapshot:
    """Normalized live equivalent of replay's joined context/prediction rows."""

    ts_ms: int
    symbols: List[str]
    feature_frame: pd.DataFrame
    regime: Dict[str, float] = field(default_factory=dict)
    ready_symbols: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_ready(self) -> bool:
        return bool(self.ready_symbols) and self.feature_frame is not None and not self.feature_frame.empty

    def summary(self) -> Dict[str, Any]:
        return {
            "ts_ms": self.ts_ms,
            "symbols": len(self.symbols),
            "ready_symbols": len(self.ready_symbols),
            "feature_rows": 0 if self.feature_frame is None else len(self.feature_frame),
            "regime": dict(self.regime or {}),
            "metadata": dict(self.metadata or {}),
        }
