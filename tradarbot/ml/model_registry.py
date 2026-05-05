from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

LOGGER = logging.getLogger("tradarbot.model_registry")

DEFAULT_HORIZONS = ("6h", "24h", "72h")


@dataclass(frozen=True)
class ModelArtifactSpec:
    """Runtime description of one trained horizon model artifact."""

    horizon: str
    model_path: str
    metadata_path: Optional[str] = None
    feature_columns: List[str] = field(default_factory=list)
    target_column: Optional[str] = None
    model_name: Optional[str] = None
    version: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelRegistry:
    """Config-backed model artifact registry for Phase 5.3 live inference."""

    def __init__(
        self,
        *,
        specs: Dict[str, ModelArtifactSpec],
        active_version: Optional[str] = None,
        artifacts_dir: Optional[str] = None,
        allow_missing_artifacts: bool = False,
        strict_feature_columns: bool = False,
        fallback_mode: Optional[str] = None,
        require_all_horizons: bool = True,
    ):
        self._specs = dict(specs)
        self._active_version = active_version
        self.artifacts_dir = artifacts_dir
        self.allow_missing_artifacts = bool(allow_missing_artifacts)
        self.strict_feature_columns = bool(strict_feature_columns)
        self.fallback_mode = fallback_mode
        self.require_all_horizons = bool(require_all_horizons)
        self._validate()

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "ModelRegistry":
        cfg = dict(cfg or {})
        registry_cfg = dict(cfg.get("model_registry") or {})

        active_version = registry_cfg.get("active_version") or cfg.get("model_version")
        artifacts_dir = registry_cfg.get("artifacts_dir") or cfg.get("artifact_dir")
        allow_missing = bool(registry_cfg.get("allow_missing_artifacts", False))
        strict_features = bool(registry_cfg.get("strict_feature_columns", cfg.get("strict_feature_columns", False)))
        fallback_mode = registry_cfg.get("fallback_mode", cfg.get("fallback_mode"))
        require_all = bool(registry_cfg.get("require_all_horizons", cfg.get("require_all_horizons", True)))

        horizon_cfg = dict(registry_cfg.get("horizons") or {})
        if not horizon_cfg and artifacts_dir:
            horizon_cfg = {
                h: {
                    "model_path": str(Path(artifacts_dir) / f"model_{h}.joblib"),
                    "metadata_path": str(Path(artifacts_dir) / f"model_{h}_metadata.json"),
                }
                for h in DEFAULT_HORIZONS
            }

        specs: Dict[str, ModelArtifactSpec] = {}
        for horizon, item in horizon_cfg.items():
            item = dict(item or {})
            model_path = item.get("model_path")
            if not model_path and artifacts_dir:
                model_path = str(Path(artifacts_dir) / f"model_{horizon}.joblib")
            metadata_path = item.get("metadata_path")
            if not metadata_path and artifacts_dir:
                metadata_path = str(Path(artifacts_dir) / f"model_{horizon}_metadata.json")
            if not model_path:
                raise ValueError(f"model_registry.horizons.{horizon}.model_path is required")

            metadata = cls._load_metadata(metadata_path, allow_missing=allow_missing)
            feature_columns = list(
                item.get("feature_columns")
                or metadata.get("feature_columns")
                or metadata.get("features")
                or []
            )
            specs[str(horizon)] = ModelArtifactSpec(
                horizon=str(horizon),
                model_path=str(model_path),
                metadata_path=str(metadata_path) if metadata_path else None,
                feature_columns=feature_columns,
                target_column=item.get("target_column") or metadata.get("target_column"),
                model_name=item.get("model_name") or metadata.get("model_name") or metadata.get("model_type"),
                version=item.get("version") or metadata.get("model_version") or active_version,
                metadata=metadata,
            )

        return cls(
            specs=specs,
            active_version=active_version,
            artifacts_dir=artifacts_dir,
            allow_missing_artifacts=allow_missing,
            strict_feature_columns=strict_features,
            fallback_mode=fallback_mode,
            require_all_horizons=require_all,
        )

    @staticmethod
    def _load_metadata(metadata_path: Optional[str], *, allow_missing: bool) -> Dict[str, Any]:
        if not metadata_path:
            return {}
        path = Path(metadata_path)
        if not path.exists():
            if allow_missing:
                return {}
            raise FileNotFoundError(f"model metadata artifact not found: {metadata_path}")
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _validate(self) -> None:
        if not self._specs:
            raise ValueError("model registry contains no horizon specs")
        for horizon, spec in self._specs.items():
            model_path = Path(spec.model_path)
            if not model_path.exists() and not self.allow_missing_artifacts:
                raise FileNotFoundError(f"model artifact not found for horizon={horizon}: {model_path}")
        LOGGER.info(
            "MODEL_REGISTRY loaded version=%s horizons=%s",
            self._active_version,
            ",".join(self.horizons()),
        )

    def get(self, horizon: str) -> ModelArtifactSpec:
        key = str(horizon)
        if key not in self._specs:
            raise KeyError(f"no model artifact registered for horizon={horizon}")
        return self._specs[key]

    def horizons(self) -> List[str]:
        order = {"6h": 0, "24h": 1, "72h": 2}
        return sorted(self._specs.keys(), key=lambda h: order.get(h, 99))

    def active_version(self) -> Optional[str]:
        return self._active_version


__all__ = ["ModelArtifactSpec", "ModelRegistry"]
