"""
source_registry.py

Canonical source registry for Phase 4 spike-dataset construction.

Purpose
-------
This module defines the available raw data sources that may feed the
spike-regime research pipeline. It is intentionally lightweight and does
not perform network I/O itself.

The registry is used to answer questions like:
- Which source provides market candles?
- Which source provides listing events?
- Which source provides metadata or external attention features?
- Which source should be preferred first for a given capability?

Design goals
------------
- Keep Phase 4 data ingestion separate from the trading engine
- Make source selection explicit and testable
- Allow incremental rollout of providers over time
- Avoid hard-coding source names all over the codebase

Typical usage
-------------
from tradarbot.research.spikes.source_registry import (
    get_default_registry,
    select_sources_for_capability,
)

registry = get_default_registry()
market_sources = select_sources_for_capability(registry, "market_bars")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# -----------------------------
# Capability constants
# -----------------------------

CAP_MARKET_BARS = "market_bars"
CAP_LISTING_EVENTS = "listing_events"
CAP_ASSET_METADATA = "asset_metadata"
CAP_MARKET_SNAPSHOTS = "market_snapshots"
CAP_ATTENTION_TIMESERIES = "attention_timeseries"
CAP_NEWS_EVENTS = "news_events"
CAP_DERIVATIVES = "derivatives"
CAP_ONCHAIN = "onchain"


ALL_CAPABILITIES = {
    CAP_MARKET_BARS,
    CAP_LISTING_EVENTS,
    CAP_ASSET_METADATA,
    CAP_MARKET_SNAPSHOTS,
    CAP_ATTENTION_TIMESERIES,
    CAP_NEWS_EVENTS,
    CAP_DERIVATIVES,
    CAP_ONCHAIN,
}


# -----------------------------
# Source model
# -----------------------------

@dataclass(frozen=True)
class SourceSpec:
    """
    Defines one external or internal data source.

    Fields
    ------
    source_id:
        Stable machine-readable identifier.
    display_name:
        Human-readable label.
    capabilities:
        Set of supported capability strings.
    priority:
        Lower number means higher preference when multiple sources can
        satisfy the same capability.
    enabled:
        Whether this source is currently enabled for selection.
    description:
        Short explanation of what the source is expected to provide.
    supported_intervals_s:
        Optional tuple of bar intervals in seconds for market-bar sources.
        Example: (60, 300, 3600, 86400)
    requires_api_key:
        Whether the provider normally requires credentials.
    notes:
        Optional freeform implementation notes.
    """

    source_id: str
    display_name: str
    capabilities: Tuple[str, ...]
    priority: int = 100
    enabled: bool = True
    description: str = ""
    supported_intervals_s: Optional[Tuple[int, ...]] = None
    requires_api_key: bool = False
    notes: str = ""

    def supports(self, capability: str) -> bool:
        return capability in self.capabilities

    def supports_interval(self, interval_s: int) -> bool:
        if self.supported_intervals_s is None:
            return True
        return interval_s in self.supported_intervals_s


@dataclass
class SourceRegistry:
    """
    Container for registered data sources.

    This class is intentionally simple:
    - deterministic ordering
    - explicit validation
    - easy to inspect in tests
    """

    _sources: Dict[str, SourceSpec] = field(default_factory=dict)

    def register(self, spec: SourceSpec) -> None:
        validate_source_spec(spec)
        if spec.source_id in self._sources:
            raise ValueError(f"duplicate source_id: {spec.source_id}")
        self._sources[spec.source_id] = spec

    def get(self, source_id: str) -> SourceSpec:
        try:
            return self._sources[source_id]
        except KeyError as exc:
            raise KeyError(f"unknown source_id: {source_id}") from exc

    def all_sources(self, *, enabled_only: bool = False) -> List[SourceSpec]:
        sources = list(self._sources.values())
        if enabled_only:
            sources = [s for s in sources if s.enabled]
        return sorted(sources, key=_sort_key)

    def sources_for_capability(
        self,
        capability: str,
        *,
        enabled_only: bool = True,
        interval_s: Optional[int] = None,
    ) -> List[SourceSpec]:
        validate_capability(capability)
        matches: List[SourceSpec] = []
        for spec in self._sources.values():
            if enabled_only and not spec.enabled:
                continue
            if not spec.supports(capability):
                continue
            if interval_s is not None and not spec.supports_interval(interval_s):
                continue
            matches.append(spec)
        return sorted(matches, key=_sort_key)

    def first_for_capability(
        self,
        capability: str,
        *,
        enabled_only: bool = True,
        interval_s: Optional[int] = None,
    ) -> Optional[SourceSpec]:
        matches = self.sources_for_capability(
            capability,
            enabled_only=enabled_only,
            interval_s=interval_s,
        )
        return matches[0] if matches else None

    def to_rows(self) -> List[dict]:
        """
        Convert registry contents into plain dict rows for debugging,
        logging, or DataFrame creation.
        """
        rows: List[dict] = []
        for spec in self.all_sources(enabled_only=False):
            rows.append(
                {
                    "source_id": spec.source_id,
                    "display_name": spec.display_name,
                    "capabilities": ",".join(spec.capabilities),
                    "priority": spec.priority,
                    "enabled": int(spec.enabled),
                    "requires_api_key": int(spec.requires_api_key),
                    "supported_intervals_s": (
                        None
                        if spec.supported_intervals_s is None
                        else ",".join(str(x) for x in spec.supported_intervals_s)
                    ),
                    "description": spec.description,
                    "notes": spec.notes,
                }
            )
        return rows


# -----------------------------
# Validation helpers
# -----------------------------

def validate_capability(capability: str) -> None:
    if capability not in ALL_CAPABILITIES:
        allowed = ", ".join(sorted(ALL_CAPABILITIES))
        raise ValueError(f"unsupported capability '{capability}'. Allowed: {allowed}")


def validate_source_spec(spec: SourceSpec) -> None:
    if not spec.source_id or not spec.source_id.strip():
        raise ValueError("source_id must be non-empty")
    if not spec.display_name or not spec.display_name.strip():
        raise ValueError("display_name must be non-empty")
    if not spec.capabilities:
        raise ValueError(f"source '{spec.source_id}' must define at least one capability")
    if spec.priority < 0:
        raise ValueError(f"source '{spec.source_id}' priority must be >= 0")

    for capability in spec.capabilities:
        validate_capability(capability)

    if spec.supported_intervals_s is not None:
        if len(spec.supported_intervals_s) == 0:
            raise ValueError(
                f"source '{spec.source_id}' supported_intervals_s cannot be empty"
            )
        bad = [x for x in spec.supported_intervals_s if x <= 0]
        if bad:
            raise ValueError(
                f"source '{spec.source_id}' has invalid intervals: {bad}"
            )


def _sort_key(spec: SourceSpec) -> Tuple[int, str]:
    return (spec.priority, spec.source_id)


# -----------------------------
# Default registry
# -----------------------------

def get_default_registry() -> SourceRegistry:
    """
    Build the default source registry for Phase 4.

    This registry is intentionally practical:
    - starts with sources most useful for the first dataset versions
    - includes placeholders for later external enrichment
    - keeps source preferences centralized
    """
    registry = SourceRegistry()

    # Internal / existing project data
    registry.register(
        SourceSpec(
            source_id="sqlite_candles",
            display_name="SQLite Candles",
            capabilities=(
                CAP_MARKET_BARS,
            ),
            priority=10,
            enabled=True,
            description=(
                "Existing Tradar SQLite candle storage. Best for initial "
                "market-only dataset bootstrapping."
            ),
            supported_intervals_s=(1, 60, 300, 900, 3600, 14400, 86400),
            requires_api_key=False,
            notes="Primary source for V1 base dataset if sufficient history exists.",
        )
    )

    registry.register(
        SourceSpec(
            source_id="sqlite_listing_events",
            display_name="SQLite Listing Events",
            capabilities=(
                CAP_LISTING_EVENTS,
                CAP_ASSET_METADATA,
            ),
            priority=15,
            enabled=True,
            description=(
                "Internal listing detection outputs and asset-first-seen metadata."
            ),
            requires_api_key=False,
            notes="Useful for age-since-listing and listing-catalyst features.",
        )
    )

    # Public market / metadata providers
    registry.register(
        SourceSpec(
            source_id="coingecko",
            display_name="CoinGecko",
            capabilities=(
                CAP_ASSET_METADATA,
                CAP_MARKET_SNAPSHOTS,
                CAP_MARKET_BARS,
            ),
            priority=20,
            enabled=True,
            description=(
                "Broad crypto asset coverage for metadata, rankings, market cap, "
                "volume snapshots, and some historical market data."
            ),
            supported_intervals_s=(300, 900, 3600, 14400, 86400),
            requires_api_key=False,
            notes="Strong candidate for V1/V2 enrichment if rate limits are manageable.",
        )
    )

    registry.register(
        SourceSpec(
            source_id="coinmarketcap",
            display_name="CoinMarketCap",
            capabilities=(
                CAP_ASSET_METADATA,
                CAP_MARKET_SNAPSHOTS,
            ),
            priority=30,
            enabled=False,
            description=(
                "Alternative metadata / market snapshot provider with broad coverage."
            ),
            requires_api_key=True,
            notes="Optional secondary provider for rankings and metadata reconciliation.",
        )
    )

    # Exchange-level sources
    registry.register(
        SourceSpec(
            source_id="binance",
            display_name="Binance",
            capabilities=(
                CAP_MARKET_BARS,
                CAP_LISTING_EVENTS,
                CAP_DERIVATIVES,
            ),
            priority=25,
            enabled=True,
            description=(
                "Exchange-native price bars, listing-related availability, and "
                "derivatives context where supported."
            ),
            supported_intervals_s=(1, 60, 180, 300, 900, 1800, 3600, 7200, 14400, 21600, 43200, 86400),
            requires_api_key=False,
            notes="Good for richer market history and exchange-specific listing context.",
        )
    )

    registry.register(
        SourceSpec(
            source_id="bybit",
            display_name="Bybit",
            capabilities=(
                CAP_MARKET_BARS,
                CAP_DERIVATIVES,
            ),
            priority=35,
            enabled=False,
            description=(
                "Supplemental exchange source for bars and derivatives context."
            ),
            supported_intervals_s=(60, 180, 300, 900, 1800, 3600, 14400, 86400),
            requires_api_key=False,
            notes="Optional secondary exchange source.",
        )
    )

    # Attention / trends / news
    registry.register(
        SourceSpec(
            source_id="google_trends",
            display_name="Google Trends",
            capabilities=(
                CAP_ATTENTION_TIMESERIES,
            ),
            priority=40,
            enabled=False,
            description=(
                "Search-interest timeseries for narrative and hype detection."
            ),
            requires_api_key=False,
            notes="Likely useful once asset keyword normalization is defined.",
        )
    )

    registry.register(
        SourceSpec(
            source_id="crypto_news_api",
            display_name="Crypto News API",
            capabilities=(
                CAP_NEWS_EVENTS,
            ),
            priority=50,
            enabled=False,
            description=(
                "News-event counts and article timestamps for catalyst detection."
            ),
            requires_api_key=True,
            notes="Start with article counts before attempting sentiment.",
        )
    )

    registry.register(
        SourceSpec(
            source_id="social_mentions",
            display_name="Social Mentions",
            capabilities=(
                CAP_ATTENTION_TIMESERIES,
            ),
            priority=60,
            enabled=False,
            description=(
                "Placeholder provider for mention counts from social/community channels."
            ),
            requires_api_key=True,
            notes="Keep disabled until a concrete provider is chosen.",
        )
    )

    # Advanced future-stage sources
    registry.register(
        SourceSpec(
            source_id="coinglass",
            display_name="Coinglass",
            capabilities=(
                CAP_DERIVATIVES,
                CAP_MARKET_SNAPSHOTS,
            ),
            priority=70,
            enabled=False,
            description=(
                "Derivatives context such as OI, funding, liquidation-related metrics."
            ),
            requires_api_key=True,
            notes="Useful later for perp-driven spike features.",
        )
    )

    registry.register(
        SourceSpec(
            source_id="onchain_metrics",
            display_name="On-chain Metrics",
            capabilities=(
                CAP_ONCHAIN,
            ),
            priority=80,
            enabled=False,
            description=(
                "Placeholder provider for token-holder, transfer, and chain-level metrics."
            ),
            requires_api_key=True,
            notes="Not needed for initial market-only and listing-aware dataset versions.",
        )
    )

    return registry


# -----------------------------
# Convenience selectors
# -----------------------------

def select_sources_for_capability(
    registry: SourceRegistry,
    capability: str,
    *,
    interval_s: Optional[int] = None,
    enabled_only: bool = True,
) -> List[SourceSpec]:
    """
    Return all matching sources, ordered by registry priority.
    """
    return registry.sources_for_capability(
        capability,
        enabled_only=enabled_only,
        interval_s=interval_s,
    )


def select_primary_source(
    registry: SourceRegistry,
    capability: str,
    *,
    interval_s: Optional[int] = None,
    enabled_only: bool = True,
) -> Optional[SourceSpec]:
    """
    Return the preferred source for a given capability.
    """
    return registry.first_for_capability(
        capability,
        enabled_only=enabled_only,
        interval_s=interval_s,
    )


def capability_plan_v1() -> Dict[str, List[str]]:
    """
    Recommended source plan for the first practical dataset versions.

    V1:
    - market-only dataset
    - listing-aware metadata if available
    - no heavy external attention stack yet
    """
    return {
        CAP_MARKET_BARS: ["sqlite_candles", "binance", "coingecko"],
        CAP_LISTING_EVENTS: ["sqlite_listing_events", "binance"],
        CAP_ASSET_METADATA: ["sqlite_listing_events", "coingecko"],
        CAP_MARKET_SNAPSHOTS: ["coingecko"],
    }


def capability_plan_v2() -> Dict[str, List[str]]:
    """
    Recommended source plan for V2 external enrichment.

    V2:
    - add market snapshots
    - add attention/news counts
    - keep sentiment / on-chain for later
    """
    return {
        CAP_MARKET_BARS: ["binance", "coingecko"],
        CAP_LISTING_EVENTS: ["sqlite_listing_events", "binance"],
        CAP_ASSET_METADATA: ["coingecko", "coinmarketcap"],
        CAP_MARKET_SNAPSHOTS: ["coingecko", "coinmarketcap"],
        CAP_ATTENTION_TIMESERIES: ["google_trends", "social_mentions"],
        CAP_NEWS_EVENTS: ["crypto_news_api"],
        CAP_DERIVATIVES: ["coinglass", "binance"],
    }


# -----------------------------
# Introspection helpers
# -----------------------------

def summarize_registry(registry: Optional[SourceRegistry] = None) -> str:
    """
    Human-readable summary of the current registry contents.
    """
    if registry is None:
        registry = get_default_registry()

    lines: List[str] = []
    for spec in registry.all_sources(enabled_only=False):
        caps = ", ".join(spec.capabilities)
        lines.append(
            f"{spec.source_id} | enabled={spec.enabled} | "
            f"priority={spec.priority} | caps=[{caps}]"
        )
    return "\n".join(lines)


def validate_registry(registry: SourceRegistry) -> None:
    """
    Extra whole-registry validation.

    Useful in tests to ensure the registry is internally consistent.
    """
    seen = set()
    for spec in registry.all_sources(enabled_only=False):
        if spec.source_id in seen:
            raise ValueError(f"duplicate source_id in registry: {spec.source_id}")
        seen.add(spec.source_id)
        validate_source_spec(spec)


__all__ = [
    "CAP_MARKET_BARS",
    "CAP_LISTING_EVENTS",
    "CAP_ASSET_METADATA",
    "CAP_MARKET_SNAPSHOTS",
    "CAP_ATTENTION_TIMESERIES",
    "CAP_NEWS_EVENTS",
    "CAP_DERIVATIVES",
    "CAP_ONCHAIN",
    "ALL_CAPABILITIES",
    "SourceSpec",
    "SourceRegistry",
    "get_default_registry",
    "select_sources_for_capability",
    "select_primary_source",
    "capability_plan_v1",
    "capability_plan_v2",
    "summarize_registry",
    "validate_registry",
]