"""Time-aware label generation utilities for Phase 4 spike-regime research.

This module computes forward-looking spike labels from historical price series
using *elapsed time horizons* rather than row counts. That means a "6h" label
looks at rows whose timestamps fall in ``(t, t + 6 hours]`` regardless of how
sparse or irregular the series is.

Design goals
------------
- Pure, testable functions
- No dependency on the live trading engine
- Elapsed-time horizons instead of observation-count horizons
- Explicit handling of insufficient future history
- Clear separation between continuous targets and binary labels

Main entry points
-----------------
- ``build_label_frame`` for raw sequences
- ``merge_labels_onto_frame`` for attaching labels to a pandas DataFrame
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None  # type: ignore[assignment]


@dataclass(frozen=True)
class SpikeLabelConfig:
    """Threshold configuration for one spike-label horizon.

    Attributes
    ----------
    horizon:
        Time horizon to inspect, expressed as a pandas-compatible duration such
        as ``"6h"`` or ``pd.Timedelta(hours=6)``.
    spike_threshold_return:
        Minimum future max return required for a binary spike label.
    tradeable_min_return:
        Minimum future max return required for the tradeable pre-spike label.
    tradeable_entry_to_peak_ratio:
        Current price must be <= this fraction of the eventual future peak
        price to count as an early enough entry.
    tradeable_max_pre_peak_drawdown:
        Maximum allowed drawdown before the future peak.
    """

    horizon: object
    spike_threshold_return: float
    tradeable_min_return: float
    tradeable_entry_to_peak_ratio: float
    tradeable_max_pre_peak_drawdown: float

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

def validate_prices(prices: Sequence[float]) -> None:
    if not prices:
        raise ValueError("prices must not be empty")
    for idx, price in enumerate(prices):
        if price is None:
            raise ValueError(f"prices[{idx}] is None")
        if price <= 0:
            raise ValueError(f"prices[{idx}] must be > 0, got {price}")


from typing import Sequence, Any

def normalize_timestamps(timestamps: Sequence[object]):
    """Convert timestamps to a strictly non-decreasing UTC DatetimeIndex.

    Accepts pandas DatetimeIndex, Series, NumPy arrays, and generic sequences.
    Returns a UTC-normalized DatetimeIndex.

    Raises:
        ImportError: if pandas is unavailable.
        TypeError: if a scalar/non-sequence is provided.
        ValueError: if timestamps contain null/invalid values or are not sorted.
    """
    if pd is None:  # pragma: no cover
        raise ImportError("pandas is required to normalize timestamps")

    if timestamps is None:
        return pd.DatetimeIndex([], tz="UTC")

    # Reject scalar-like inputs early (e.g. a single Timestamp/string/int).
    if isinstance(timestamps, (str, bytes)) or not hasattr(timestamps, "__len__"):
        raise TypeError("timestamps must be a sequence of timestamp-like values, not a scalar")

    if len(timestamps) == 0:
        return pd.DatetimeIndex([], tz="UTC")

    # Fast path: already a DatetimeIndex
    if isinstance(timestamps, pd.DatetimeIndex):
        ts = timestamps
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
    else:
        # Avoid list(...) so pandas/numpy containers stay efficient.
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
) -> List[FutureWindowStats]:
    """Compute forward targets using a real elapsed-time horizon.

    Parameters
    ----------
    prices:
        Ordered close prices for one asset.
    timestamps:
        Ordered timestamps aligned to ``prices``.
    horizon:
        Time horizon inspectable by ``pandas.to_timedelta``.

    Returns
    -------
    list[FutureWindowStats]
        One object per input row.

    Notes
    -----
    The future window is strictly forward-looking: rows with timestamps in
    ``(t, t + horizon]``. A row only has ``has_full_horizon = True`` if the
    series contains at least one observation at or beyond ``t + horizon``.
    """
    if pd is None:  # pragma: no cover
        raise ImportError("pandas is required to compute time-aware labels")

    validate_aligned_inputs(prices, timestamps)
    horizon_td = pd.to_timedelta(horizon)
    if horizon_td <= pd.Timedelta(0):
        raise ValueError("horizon must be > 0")

    price_list = list(prices)
    ts_index = normalize_timestamps(timestamps)
    n = len(price_list)
    stats: List[FutureWindowStats] = []

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

        end_exclusive = ts_index.searchsorted(horizon_end_ts, side="right")
        start = idx + 1
        stop = int(end_exclusive)

        if start >= stop:
            # We have enough *overall* future history to cover the horizon, but
            # there are no observed rows within (t, t+horizon]. This makes the
            # label uncomputable for sparse gaps, so mark it unavailable.
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
        future_window_timestamps = ts_index[start:stop]

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


def build_label_frame(
    prices: Sequence[float],
    timestamps: Sequence[object],
    configs: Optional[dict[str, SpikeLabelConfig]] = None,
):
    """Build a time-aware label table for one ordered price series."""
    if pd is None:  # pragma: no cover
        raise ImportError("pandas is required to build a label frame")

    active_configs = configs or DEFAULT_LABEL_CONFIGS
    validate_aligned_inputs(prices, timestamps)
    ts_index = normalize_timestamps(timestamps)

    rows: dict[str, List[Optional[float] | Optional[int] | float | object]] = {
        "timestamp": list(ts_index),
        "price_close": list(prices),
    }

    for horizon_name, config in active_configs.items():
        stats = compute_future_window_stats(
            prices=prices,
            timestamps=ts_index,
            horizon=config.horizon_timedelta,
        )

        rows[f"target_future_max_return_{horizon_name}"] = [s.future_max_return for s in stats]
        rows[f"target_future_peak_price_{horizon_name}"] = [s.future_peak_price for s in stats]
        rows[f"target_time_to_peak_steps_{horizon_name}"] = [s.future_peak_offset_steps for s in stats]
        rows[f"target_time_to_peak_seconds_{horizon_name}"] = [s.future_time_to_peak_seconds for s in stats]
        rows[f"target_pre_peak_drawdown_{horizon_name}"] = [s.pre_peak_drawdown for s in stats]
        rows[f"has_full_horizon_{horizon_name}"] = [int(s.has_full_horizon) for s in stats]

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

    return pd.DataFrame(rows)


def merge_labels_onto_frame(
    frame,
    price_column: str = "price_close",
    timestamp_column: str = "timestamp",
    configs: Optional[dict[str, SpikeLabelConfig]] = None,
):
    """Attach time-aware label columns to an existing pandas DataFrame.

    Parameters
    ----------
    frame:
        Input pandas DataFrame for one asset, already sorted by timestamp.
    price_column:
        Name of the close-price column.
    timestamp_column:
        Name of the timestamp column. Values must be parseable by pandas.
    configs:
        Optional custom horizon config mapping.
    """
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
    )
    extra_cols = [c for c in label_frame.columns if c not in {"price_close", "timestamp"}]

    result = temp.copy()
    for col in extra_cols:
        result[col] = label_frame[col].values
    return result


__all__ = [
    "DEFAULT_LABEL_CONFIGS",
    "FutureWindowStats",
    "SpikeLabelConfig",
    "assign_binary_spike_label",
    "assign_tradeable_pre_spike_label",
    "build_label_frame",
    "compute_future_window_stats",
    "merge_labels_onto_frame",
    "normalize_timestamps",
    "validate_aligned_inputs",
    "validate_prices",
]