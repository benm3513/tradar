from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

from tradarbot.ml.ensemble import compute_ensemble_frame, validate_ensemble_weights

LOGGER = logging.getLogger("tradarbot.live_ensemble")


@dataclass(frozen=True)
class LiveEnsembleConfig:
    weight_6h: float = 0.25
    weight_24h: float = 0.65
    weight_72h: float = 0.10
    agreement_threshold: float = 0.80
    agreement_boost: float = 0.05
    require_all_horizons: bool = True
    renormalize_missing_horizons: bool = True

    @classmethod
    def from_config(cls, cfg: Optional[Dict[str, Any]] = None) -> "LiveEnsembleConfig":
        cfg = dict(cfg or {})
        ens = dict(cfg.get("ensemble") or {})
        return cls(
            weight_6h=float(ens.get("weight_6h", cfg.get("weight_6h", 0.25))),
            weight_24h=float(ens.get("weight_24h", cfg.get("weight_24h", 0.65))),
            weight_72h=float(ens.get("weight_72h", cfg.get("weight_72h", 0.10))),
            agreement_threshold=float(ens.get("agreement_threshold", cfg.get("agreement_threshold", 0.80))),
            agreement_boost=float(ens.get("agreement_boost", cfg.get("agreement_boost", 0.05))),
            require_all_horizons=bool(ens.get("require_all_horizons", cfg.get("require_all_horizons", True))),
            renormalize_missing_horizons=bool(ens.get("renormalize_missing_horizons", cfg.get("renormalize_missing_horizons", True))),
        )


def combine_horizon_probs(df: pd.DataFrame, cfg: LiveEnsembleConfig | Dict[str, Any] | None = None) -> pd.DataFrame:
    if not isinstance(cfg, LiveEnsembleConfig):
        cfg = LiveEnsembleConfig.from_config(cfg or {})
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    weights = {"6h": cfg.weight_6h, "24h": cfg.weight_24h, "72h": cfg.weight_72h}
    present = [h for h in ("6h", "24h", "72h") if f"prob_{h}" in out.columns]
    missing = [h for h in ("6h", "24h", "72h") if h not in present]
    if missing and cfg.require_all_horizons:
        raise KeyError(f"missing required horizon probability columns: {missing}")
    if not present:
        raise KeyError("no horizon probability columns found for live ensemble")

    if missing and cfg.renormalize_missing_horizons:
        total = sum(weights[h] for h in present)
        if total <= 0:
            raise ValueError("cannot renormalize live ensemble weights; present horizon weights sum to zero")
        weights = {h: (weights[h] / total if h in present else 0.0) for h in weights}

    validate_ensemble_weights(weights["6h"], weights["24h"], weights["72h"], allow_partial=bool(missing))
    out = compute_ensemble_frame(
        out,
        w6=weights["6h"],
        w24=weights["24h"],
        w72=weights["72h"],
        threshold=cfg.agreement_threshold,
        agreement_boost=cfg.agreement_boost,
        require_all_horizons=cfg.require_all_horizons,
    )
    out["prediction_source"] = "ensemble_live"
    LOGGER.info(
        "LIVE_ENSEMBLE rows=%d w6=%.3f w24=%.3f w72=%.3f",
        len(out), weights["6h"], weights["24h"], weights["72h"],
    )
    return out


__all__ = ["LiveEnsembleConfig", "combine_horizon_probs"]
