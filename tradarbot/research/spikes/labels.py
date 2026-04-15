"""Time-aware label generation utilities for Phase 4 spike-regime research.

This module computes forward-looking spike labels from historical price series
using elapsed-time horizons rather than row counts. A "6h" label inspects rows
whose timestamps fall in ``(t, t + 6 hours]`` regardless of irregular spacing.

Phase 4.7 upgrades
------------------
- Native multi-horizon support (6h / 24h / 72h / 7d by default)
- Optional percentile-based thresholding derived from future-return distributions
- Stable alias columns for downstream training/replay code
- Pure, testable functions with explicit handling of sparse history gaps

Main entry points
-----------------
- ``build_label_frame`` for raw aligned price/timestamp sequences
- ``merge_labels_onto_frame`` for attaching labels to an existing DataFrame
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None  # type: ignore[assignment]


@dataclass(frozen=True)
class SpikeLabelConfig:
    """Threshold configuration for one spike-label horizon.

    Parameters
    ----------
    horizon:
        Time horizon to inspect, expressed as a pandas-compatible duration such
        as ``"6h"`` or ``pd.Timedelta(hours=6)``.
    spike_threshold_return:
        Absolute minimum future max return required for a binary spike label.
        Ignored when ``spike_threshold_percentile`` is provided and
        ``resolve_label_thresholds`` is used.
    tradeable_min_return:
        Absolute minimum future max return required for the tradeable
        pre-spike label. Ignored when ``tradeable_min_return_percentile`` is
        provided and ``resolve_label_thresholds`` is used.
    tradeable_entry_to_peak_ratio:
        Current price must be <= this fraction of the eventual future peak
        price to count as an early enough entry.
    tradeable_max_pre_peak_drawdown:
        Maximum allowed drawdown before the future peak.
    spike_threshold_percentile:
        Optional percentile in [0, 1]. When provided, the spike threshold is
        resolved from the observed future-return distribution for this horizon.
    tradeable_min_return_percentile:
        Optional percentile in [0, 1]. When provided, the tradeable min return
        is resolved from the observed future-return distribution for this horizon.
    min_resolved_rows:
        Minimum number of non-null future-return rows required before a
        percentile threshold is considered valid.
    """

    horizon: object
    spike_threshold_return: float
    tradeable_min_return: float
    tradeable_entry_to_peak_ratio: float
    tradeable_max_pre_peak_drawdown: float
    spike_threshold_percentile: Optional[float] = None
    tradeable_min_return_percentile: Optional[float] = None
    min_resolved_rows: int = 25

    def __post_init__(self) -> None:
        if pd is None:  # pragma: no cover
            return
        horizon_td = pd.to_timedelta(self.horizon)
        if horizon_td <= pd.Timedelta(0):
            raise ValueError("horizon must be > 0")

        for name, value in (
            ("spike_threshold_return", self.spike_threshold_return),
            ("tradeable_min_return", self.tradeable_min_return),
            ("tradeable_entry_to_peak_ratio", self.tradeable_entry_to_peak_ratio),
            ("tradeable_max_pre_peak_drawdown", self.tradeable_max_pre_peak_drawdown),
        ):
            if value < 0:
                raise ValueError(f"{name} must be >= 0")

        for name, value in (
            ("spike_threshold_percentile", self.spike_threshold_percentile),
            ("tradeable_min_return_percentile", self.tradeable_min_return_percentile),
        ):
            if value is not None and not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be in [0, 1] when provided")

        if self.min_resolved_rows <= 0:
            raise ValueError("min_resolved_rows must be > 0")

    @property
    def horizon_timedelta(self):
        if pd is None:  # pragma: no cover
            raise ImportError("pandas is required to resolve label horizons")
        return pd.to_timedelta(self.horizon)


@dataclass(frozen=True)
class ResolvedSpikeLabelConfig:
    """Concrete per-horizon thresholds after percentile resolution."""

    horizon: object
    spike_threshold_return: float
    tradeable_min_return: float
    tradeable_entry_to_peak_ratio: float
    tradeable_max_pre_peak_drawdown: float

    @property
    def horizon_timedelta(self):
        if pd is None:  # pragma: no cover
            raise ImportError("pandas is required to resolve label horizons")
        return pd.to_timedelta(self.horizon)


@dataclass(frozen=True)
class FutureWindowStats:
    """Forward-looking price statistics for one timestamp and one horizon."""

    current_price: float
    future_peak_price: Optional[float]
    future_peak_offset_steps: Optional[int]
    future_time_to_peak_seconds: Optional[float]
    future_max_return: Optional[float]
    pre_peak_drawdown: Optional[float]
    has_full_horizon: bool


DEFAULT_LABEL_CONFIGS: dict[str, SpikeLabelConfig] = {
    "6h": SpikeLabelConfig(
        horizon="6h",
        spike_threshold_return=0.04,
        tradeable_min_return=0.04,
        tradeable_entry_to_peak_ratio=0.92,
        tradeable_max_pre_peak_drawdown=0.45,
    ),
    "24h": SpikeLabelConfig(
        horizon="24h",
        spike_threshold_return=0.18,
        tradeable_min_return=0.18,
        tradeable_entry_to_peak_ratio=0.92,
        tradeable_max_pre_peak_drawdown=0.45,
    ),
    "72h": SpikeLabelConfig(
        horizon="72h",
        spike_threshold_return=0.25,
        tradeable_min_return=0.25,
        tradeable_entry_to_peak_ratio=0.92,
        tradeable_max_pre_peak_drawdown=0.45,
    ),
    "7d": SpikeLabelConfig(
        horizon="7d",
        spike_threshold_return=0.30,
        tradeable_min_return=0.30,
        tradeable_entry_to_peak_ratio=0.92,
        tradeable_max_pre_peak_drawdown=0.45,
    ),
}

PERCENTILE_LABEL_CONFIGS: dict[str, SpikeLabelConfig] = {
    "6h": SpikeLabelConfig(
        horizon="6h",
        spike_threshold_return=0.04,
        tradeable_min_return=0.04,
        tradeable_entry_to_peak_ratio=0.92,
        tradeable_max_pre_peak_drawdown=0.45,
        spike_threshold_percentile=0.97,
        tradeable_min_return_percentile=0.97,
    ),
    "24h": SpikeLabelConfig(
        horizon="24h",
        spike_threshold_return=0.18,
        tradeable_min_return=0.18,
        tradeable_entry_to_peak_ratio=0.92,
        tradeable_max_pre_peak_drawdown=0.45,
        spike_threshold_percentile=0.985,
        tradeable_min_return_percentile=0.985,
    ),
    "72h": SpikeLabelConfig(
        horizon="72h",
        spike_threshold_return=0.25,
        tradeable_min_return=0.25,
        tradeable_entry_to_peak_ratio=0.92,
        tradeable_max_pre_peak_drawdown=0.45,
        spike_threshold_percentile=0.985,
        tradeable_min_return_percentile=0.985,
    ),
    "7d": SpikeLabelConfig(
        horizon="7d",
        spike_threshold_return=0.30,
        tradeable_min_return=0.30,
        tradeable_entry_to_peak_ratio=0.92,
        tradeable_max_pre_peak_drawdown=0.45,
        spike_threshold_percentile=0.99,
        tradeable_min_return_percentile=0.99,
    ),
}


def validate_prices(prices: Sequence[float]) -> None:
    if not prices:
        raise ValueError("prices must not be empty")
    for idx, price in enumerate(prices):
        if price is None:
            raise ValueError(f"prices[{idx}] is None")
        if price <= 0:
            raise ValueError(f"prices[{idx}] must be > 0, got {price}")


def normalize_timestamps(timestamps: Sequence[object]):
    """Convert timestamps to a strictly non-decreasing UTC DatetimeIndex."""
    if pd is None:  # pragma: no cover
        raise ImportError("pandas is required to normalize timestamps")

    if timestamps is None:
        return pd.DatetimeIndex([], tz="UTC")

    if isinstance(timestamps, (str, bytes)) or not hasattr(timestamps, "__len__"):
        raise TypeError("timestamps must be a sequence of timestamp-like values, not a scalar")

    if len(timestamps) == 0:
        return pd.DatetimeIndex([], tz="UTC")

    if isinstance(timestamps, pd.DatetimeIndex):
        ts = timestamps
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
    else:
        ts = pd.to_datetime(timestamps, utc=True, errors="coerce")
        ts = pd.DatetimeIndex(ts)

    if ts.isna().any():
        bad_count = int(ts.isna().sum())
        raise ValueError(f"timestamps contain {bad_count} null/invalid value(s)")

    if not ts.is_monotonic_increasing:
        raise ValueError("timestamps must be sorted in non-decreasing order")

    return ts


def validate_aligned_inputs(prices: Sequence[float], timestamps: Sequence[object]) -> None:
    validate_prices(prices)
    if len(prices) != len(timestamps):
        raise ValueError(
            f"prices and timestamps must have the same length; got {len(prices)} and {len(timestamps)}"
        )
    normalize_timestamps(timestamps)


def compute_future_window_stats(
    prices: Sequence[float],
    timestamps: Sequence[object],
    horizon,
) -> list[FutureWindowStats]:
    """Compute forward targets using a real elapsed-time horizon."""
    if pd is None:  # pragma: no cover
        raise ImportError("pandas is required to compute time-aware labels")

    validate_aligned_inputs(prices, timestamps)
    horizon_td = pd.to_timedelta(horizon)
    if horizon_td <= pd.Timedelta(0):
        raise ValueError("horizon must be > 0")

    price_list = list(prices)
    ts_index = normalize_timestamps(timestamps)
    stats: list[FutureWindowStats] = []

    for idx, current_price in enumerate(price_list):
        current_ts = ts_index[idx]
        horizon_end_ts = current_ts + horizon_td

        has_full_horizon = bool(ts_index[-1] >= horizon_end_ts)
        if not has_full_horizon:
            stats.append(
                FutureWindowStats(
                    current_price=current_price,
                    future_peak_price=None,
                    future_peak_offset_steps=None,
                    future_time_to_peak_seconds=None,
                    future_max_return=None,
                    pre_peak_drawdown=None,
                    has_full_horizon=False,
                )
            )
            continue

        end_exclusive = int(ts_index.searchsorted(horizon_end_ts, side="right"))
        start = idx + 1
        stop = end_exclusive

        if start >= stop:
            stats.append(
                FutureWindowStats(
                    current_price=current_price,
                    future_peak_price=None,
                    future_peak_offset_steps=None,
                    future_time_to_peak_seconds=None,
                    future_max_return=None,
                    pre_peak_drawdown=None,
                    has_full_horizon=False,
                )
            )
            continue

        future_window_prices = price_list[start:stop]

        peak_price = max(future_window_prices)
        peak_rel_index = future_window_prices.index(peak_price)
        peak_abs_index = start + peak_rel_index
        peak_offset_steps = peak_abs_index - idx
        peak_ts = ts_index[peak_abs_index]
        time_to_peak_seconds = (peak_ts - current_ts).total_seconds()

        path_to_peak = future_window_prices[: peak_rel_index + 1]
        trough_before_or_at_peak = min(path_to_peak)
        pre_peak_drawdown = (current_price - trough_before_or_at_peak) / current_price
        future_max_return = (peak_price - current_price) / current_price

        stats.append(
            FutureWindowStats(
                current_price=current_price,
                future_peak_price=peak_price,
                future_peak_offset_steps=peak_offset_steps,
                future_time_to_peak_seconds=time_to_peak_seconds,
                future_max_return=future_max_return,
                pre_peak_drawdown=pre_peak_drawdown,
                has_full_horizon=True,
            )
        )

    return stats


def assign_binary_spike_label(
    future_max_return: Optional[float],
    threshold_return: float,
) -> Optional[int]:
    if future_max_return is None:
        return None
    return int(future_max_return >= threshold_return)


def assign_tradeable_pre_spike_label(
    *,
    current_price: float,
    future_peak_price: Optional[float],
    future_max_return: Optional[float],
    pre_peak_drawdown: Optional[float],
    min_future_return: float,
    max_entry_to_peak_ratio: float,
    max_pre_peak_drawdown: float,
) -> Optional[int]:
    if future_peak_price is None or future_max_return is None or pre_peak_drawdown is None:
        return None

    early_enough = current_price <= (max_entry_to_peak_ratio * future_peak_price)
    return_ok = future_max_return >= min_future_return
    drawdown_ok = pre_peak_drawdown <= max_pre_peak_drawdown
    return int(return_ok and early_enough and drawdown_ok)


def _resolve_threshold_from_percentile(
    future_returns: Sequence[Optional[float]],
    percentile: Optional[float],
    fallback_value: float,
    min_rows: int,
) -> float:
    if pd is None:  # pragma: no cover
        raise ImportError("pandas is required to resolve percentile thresholds")

    if percentile is None:
        return fallback_value

    valid = pd.Series([value for value in future_returns if value is not None], dtype="float64")
    if len(valid) < min_rows:
        return fallback_value

    resolved = float(valid.quantile(percentile))
    return max(resolved, 0.0)


def resolve_label_thresholds(
    price_returns_by_horizon: Mapping[str, Sequence[Optional[float]]],
    configs: Optional[Mapping[str, SpikeLabelConfig]] = None,
) -> dict[str, ResolvedSpikeLabelConfig]:
    """Resolve concrete thresholds, optionally using per-horizon percentiles."""
    active_configs = dict(configs or DEFAULT_LABEL_CONFIGS)
    resolved: dict[str, ResolvedSpikeLabelConfig] = {}

    for horizon_name, config in active_configs.items():
        future_returns = price_returns_by_horizon.get(horizon_name, [])
        spike_threshold_return = _resolve_threshold_from_percentile(
            future_returns=future_returns,
            percentile=config.spike_threshold_percentile,
            fallback_value=config.spike_threshold_return,
            min_rows=config.min_resolved_rows,
        )
        tradeable_min_return = _resolve_threshold_from_percentile(
            future_returns=future_returns,
            percentile=config.tradeable_min_return_percentile,
            fallback_value=config.tradeable_min_return,
            min_rows=config.min_resolved_rows,
        )
        resolved[horizon_name] = ResolvedSpikeLabelConfig(
            horizon=config.horizon,
            spike_threshold_return=spike_threshold_return,
            tradeable_min_return=tradeable_min_return,
            tradeable_entry_to_peak_ratio=config.tradeable_entry_to_peak_ratio,
            tradeable_max_pre_peak_drawdown=config.tradeable_max_pre_peak_drawdown,
        )

    return resolved


def build_label_frame(
    prices: Sequence[float],
    timestamps: Sequence[object],
    configs: Optional[Mapping[str, SpikeLabelConfig]] = None,
    *,
    resolve_percentiles: bool = False,
):
    """Build a time-aware label table for one ordered price series."""
    if pd is None:  # pragma: no cover
        raise ImportError("pandas is required to build a label frame")

    active_configs = dict(configs or DEFAULT_LABEL_CONFIGS)
    validate_aligned_inputs(prices, timestamps)
    ts_index = normalize_timestamps(timestamps)

    rows: dict[str, list[Any]] = {
        "timestamp": list(ts_index),
        "price_close": list(prices),
    }
    stats_by_horizon: dict[str, list[FutureWindowStats]] = {}
    future_returns_by_horizon: dict[str, list[Optional[float]]] = {}

    for horizon_name, config in active_configs.items():
        stats = compute_future_window_stats(
            prices=prices,
            timestamps=ts_index,
            horizon=config.horizon_timedelta,
        )
        stats_by_horizon[horizon_name] = stats
        future_returns_by_horizon[horizon_name] = [s.future_max_return for s in stats]

    if resolve_percentiles:
        resolved_configs = resolve_label_thresholds(
            price_returns_by_horizon=future_returns_by_horizon,
            configs=active_configs,
        )
    else:
        resolved_configs = {
            horizon_name: ResolvedSpikeLabelConfig(
                horizon=config.horizon,
                spike_threshold_return=config.spike_threshold_return,
                tradeable_min_return=config.tradeable_min_return,
                tradeable_entry_to_peak_ratio=config.tradeable_entry_to_peak_ratio,
                tradeable_max_pre_peak_drawdown=config.tradeable_max_pre_peak_drawdown,
            )
            for horizon_name, config in active_configs.items()
        }

    for horizon_name, config in resolved_configs.items():
        stats = stats_by_horizon[horizon_name]

        rows[f"target_future_max_return_{horizon_name}"] = [s.future_max_return for s in stats]
        rows[f"target_future_peak_price_{horizon_name}"] = [s.future_peak_price for s in stats]
        rows[f"target_time_to_peak_steps_{horizon_name}"] = [s.future_peak_offset_steps for s in stats]
        rows[f"target_time_to_peak_seconds_{horizon_name}"] = [s.future_time_to_peak_seconds for s in stats]
        rows[f"target_pre_peak_drawdown_{horizon_name}"] = [s.pre_peak_drawdown for s in stats]
        rows[f"has_full_horizon_{horizon_name}"] = [int(s.has_full_horizon) for s in stats]
        rows[f"threshold_spike_return_{horizon_name}"] = [config.spike_threshold_return] * len(stats)
        rows[f"threshold_tradeable_return_{horizon_name}"] = [config.tradeable_min_return] * len(stats)

        rows[f"label_spike_{horizon_name}"] = [
            assign_binary_spike_label(
                future_max_return=s.future_max_return,
                threshold_return=config.spike_threshold_return,
            )
            for s in stats
        ]

        rows[f"label_tradeable_pre_spike_{horizon_name}"] = [
            assign_tradeable_pre_spike_label(
                current_price=s.current_price,
                future_peak_price=s.future_peak_price,
                future_max_return=s.future_max_return,
                pre_peak_drawdown=s.pre_peak_drawdown,
                min_future_return=config.tradeable_min_return,
                max_entry_to_peak_ratio=config.tradeable_entry_to_peak_ratio,
                max_pre_peak_drawdown=config.tradeable_max_pre_peak_drawdown,
            )
            for s in stats
        ]

        # Stable alias columns for downstream training code
        rows[f"spike_{horizon_name}_label"] = rows[f"label_spike_{horizon_name}"]
        rows[f"tradeable_pre_spike_{horizon_name}_label"] = rows[f"label_tradeable_pre_spike_{horizon_name}"]

    return pd.DataFrame(rows)


def merge_labels_onto_frame(
    frame,
    price_column: str = "price_close",
    timestamp_column: str = "timestamp",
    configs: Optional[Mapping[str, SpikeLabelConfig]] = None,
    *,
    resolve_percentiles: bool = False,
):
    """Attach time-aware label columns to an existing pandas DataFrame."""
    if pd is None:  # pragma: no cover
        raise ImportError("pandas is required to merge labels onto a frame")
    if price_column not in frame.columns:
        raise KeyError(f"price_column '{price_column}' not found in frame")
    if timestamp_column not in frame.columns:
        raise KeyError(f"timestamp_column '{timestamp_column}' not found in frame")

    temp = frame.copy()
    temp[timestamp_column] = pd.to_datetime(temp[timestamp_column], utc=True, errors="raise")
    temp = temp.sort_values(timestamp_column).reset_index(drop=True)

    label_frame = build_label_frame(
        prices=temp[price_column].tolist(),
        timestamps=temp[timestamp_column].tolist(),
        configs=configs,
        resolve_percentiles=resolve_percentiles,
    )
    extra_cols = [c for c in label_frame.columns if c not in {"price_close", "timestamp"}]

    result = temp.copy()
    for col in extra_cols:
        result[col] = label_frame[col].values
    return result


__all__ = [
    "DEFAULT_LABEL_CONFIGS",
    "PERCENTILE_LABEL_CONFIGS",
    "FutureWindowStats",
    "ResolvedSpikeLabelConfig",
    "SpikeLabelConfig",
    "assign_binary_spike_label",
    "assign_tradeable_pre_spike_label",
    "build_label_frame",
    "compute_future_window_stats",
    "merge_labels_onto_frame",
    "normalize_timestamps",
    "resolve_label_thresholds",
    "validate_aligned_inputs",
    "validate_prices",
]