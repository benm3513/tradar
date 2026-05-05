from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from tradarbot.ml.model_registry import ModelRegistry, ModelArtifactSpec

LOGGER = logging.getLogger("tradarbot.model_inference")

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None


class ModelInferenceService:
    """Loads trained sklearn/joblib artifacts and scores live feature frames."""

    def __init__(self, registry: ModelRegistry, cfg: Optional[Dict[str, Any]] = None):
        self.registry = registry
        self.cfg = dict(cfg or {})
        self.strict_feature_columns = bool(
            self.cfg.get("strict_feature_columns", registry.strict_feature_columns)
        )
        self.missing_feature_fill_value = self.cfg.get("missing_feature_fill_value", 0.0)
        self.eager_load = bool(self.cfg.get("eager_load", True))
        self._models: Dict[str, Any] = {}
        if self.eager_load:
            for horizon in self.registry.horizons():
                self.load_model(horizon)

    def load_model(self, horizon: str) -> Any:
        if horizon in self._models:
            return self._models[horizon]
        if joblib is None:
            raise RuntimeError("joblib is required for model artifact loading. Run: pip install joblib")
        spec = self.registry.get(horizon)
        path = Path(spec.model_path)
        if not path.exists():
            raise FileNotFoundError(f"model artifact not found for horizon={horizon}: {path}")
        model = joblib.load(path)
        self._models[horizon] = model
        LOGGER.info(
            "MODEL_INFERENCE loaded horizon=%s features=%d path=%s",
            horizon,
            len(spec.feature_columns or []),
            path,
        )
        return model

    def _prepare_features(self, feature_frame: pd.DataFrame, spec: ModelArtifactSpec) -> pd.DataFrame:
        if feature_frame is None or feature_frame.empty:
            return pd.DataFrame()
        feature_columns = list(spec.feature_columns or [])
        if not feature_columns:
            # Fallback: use all numeric-ish columns except identifiers/labels.
            excluded = {"symbol", "timestamp", "ts_ms", "horizon", "target_column"}
            feature_columns = [c for c in feature_frame.columns if c not in excluded]
        missing = [c for c in feature_columns if c not in feature_frame.columns]
        if missing and self.strict_feature_columns:
            raise KeyError(f"missing live feature columns for horizon={spec.horizon}: {missing}")
        out = feature_frame.copy()
        for col in missing:
            out[col] = self.missing_feature_fill_value
        X = out[feature_columns].copy()
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")
        if not self.strict_feature_columns:
            X = X.fillna(self.missing_feature_fill_value)
        return X

    def predict_horizon(self, feature_frame: pd.DataFrame, horizon: str) -> pd.Series:
        spec = self.registry.get(horizon)
        model = self.load_model(horizon)
        X = self._prepare_features(feature_frame, spec)
        if X.empty:
            return pd.Series(dtype=float, name=f"prob_{horizon}")
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X)[:, 1]
        elif hasattr(model, "decision_function"):
            raw = model.decision_function(X)
            prob = 1.0 / (1.0 + pd.Series(raw).mul(-1).map(__import__("math").exp))
        else:
            prob = model.predict(X)
        return pd.Series(prob, index=feature_frame.index, name=f"prob_{horizon}").astype(float).clip(0.0, 1.0)

    def predict_all(self, feature_frame: pd.DataFrame) -> pd.DataFrame:
        if feature_frame is None or feature_frame.empty:
            return pd.DataFrame()
        out = pd.DataFrame(index=feature_frame.index)
        if "symbol" in feature_frame.columns:
            out["symbol"] = feature_frame["symbol"].astype(str).values
        if "timestamp" in feature_frame.columns:
            out["timestamp"] = feature_frame["timestamp"].values
        elif "ts_ms" in feature_frame.columns:
            out["ts_ms"] = feature_frame["ts_ms"].values

        errors: Dict[str, str] = {}
        for horizon in self.registry.horizons():
            try:
                out[f"prob_{horizon}"] = self.predict_horizon(feature_frame, horizon).values
            except Exception as exc:
                errors[horizon] = str(exc)
                LOGGER.warning("MODEL_INFERENCE horizon=%s failed: %s", horizon, exc)
                if self.registry.require_all_horizons:
                    raise

        if errors and len(errors) == len(self.registry.horizons()):
            raise RuntimeError(f"all horizon model inference failed: {errors}")
        out.attrs["inference_errors"] = errors
        return out.reset_index(drop=True)


__all__ = ["ModelInferenceService"]
