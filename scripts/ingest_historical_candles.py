#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import math
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

LOGGER = logging.getLogger("ingest_historical_candles")

BINANCE_BASE_URLS = [
    "https://api.binance.com",
    "https://api-gcp.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://api4.binance.com",
]
BYBIT_BASE_URLS = ["https://api.bybit.com", "https://api.bytick.com"]
COINBASE_BASE_URLS = ["https://api.exchange.coinbase.com"]

SUPPORTED_INTERVALS: Dict[str, int] = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
    "12h": 43200,
    "1d": 86400,
}

COINBASE_GRANULARITIES = {60, 300, 900, 3600, 21600, 86400}
COINBASE_MAX_CANDLES_PER_REQUEST = 300
DEFAULT_MIN_ROWS_PER_SYMBOL = 168
DEFAULT_RETRY_ATTEMPTS = 5
DEFAULT_TIMEOUT_S = 30.0
DEFAULT_SLEEP_S = 0.25
DEFAULT_RETRY_SLEEP_S = 1.5
DEFAULT_BATCH_LIMIT = 1000


class ProviderError(RuntimeError):
    pass


class HTTPProviderError(ProviderError):
    pass


class RestrictedLocationError(ProviderError):
    pass


@dataclass(frozen=True)
class CandleRow:
    symbol: str
    interval_s: int
    ts_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class IngestionResult:
    provider: str
    symbol: str
    rows_fetched: int
    rows_written: int
    first_ts_ms: Optional[int]
    last_ts_ms: Optional[int]


class HttpJsonClient:
    def __init__(
        self,
        provider_name: str,
        base_urls: Sequence[str],
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_attempts: int = DEFAULT_RETRY_ATTEMPTS,
        retry_sleep_s: float = DEFAULT_RETRY_SLEEP_S,
        user_agent: str = "TradarResearch/1.0",
    ) -> None:
        if not base_urls:
            raise ValueError("base_urls must not be empty")
        self.provider_name = provider_name
        self.base_urls = list(base_urls)
        self.timeout_s = timeout_s
        self.max_attempts = max_attempts
        self.retry_sleep_s = retry_sleep_s
        self.user_agent = user_agent

    def get_json(self, path: str, params: Dict[str, Any]) -> Any:
        query = urllib.parse.urlencode(params)
        last_error: Optional[Exception] = None

        for base_url in self.base_urls:
            url = f"{base_url}{path}?{query}" if query else f"{base_url}{path}"
            for attempt in range(1, self.max_attempts + 1):
                request = urllib.request.Request(
                    url,
                    headers={
                        "Accept": "application/json",
                        "User-Agent": self.user_agent,
                    },
                )
                try:
                    with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                        payload = response.read().decode("utf-8")
                        return json.loads(payload)
                except urllib.error.HTTPError as exc:
                    body = ""
                    try:
                        body = exc.read().decode("utf-8", errors="replace")
                    except Exception:
                        body = "<unreadable body>"
                    lowered = body.lower()
                    LOGGER.warning(
                        "HTTP error from %s on attempt %s/%s via %s: status=%s body=%s",
                        self.provider_name,
                        attempt,
                        self.max_attempts,
                        base_url,
                        exc.code,
                        body[:1000],
                    )
                    if exc.code in (403, 451) or "restricted location" in lowered or "blocked access from your country" in lowered:
                        raise RestrictedLocationError(
                            f"{self.provider_name} HTTP error {exc.code}: {body}"
                        ) from exc
                    last_error = HTTPProviderError(f"{self.provider_name} HTTP error {exc.code}: {body}")
                except urllib.error.URLError as exc:
                    LOGGER.warning(
                        "URL error from %s on attempt %s/%s via %s: %s",
                        self.provider_name,
                        attempt,
                        self.max_attempts,
                        base_url,
                        exc,
                    )
                    last_error = ProviderError(f"{self.provider_name} URL error: {exc}")
                except json.JSONDecodeError as exc:
                    LOGGER.warning(
                        "JSON decode error from %s on attempt %s/%s via %s: %s",
                        self.provider_name,
                        attempt,
                        self.max_attempts,
                        base_url,
                        exc,
                    )
                    last_error = ProviderError(f"{self.provider_name} invalid JSON: {exc}")

                if attempt < self.max_attempts:
                    time.sleep(self.retry_sleep_s * attempt)

        if last_error is None:
            last_error = ProviderError(f"{self.provider_name} request failed for unknown reason")
        raise last_error


class BaseProvider:
    name = "base"

    def __init__(self, timeout_s: float, max_attempts: int, retry_sleep_s: float, request_sleep_s: float) -> None:
        self.timeout_s = timeout_s
        self.max_attempts = max_attempts
        self.retry_sleep_s = retry_sleep_s
        self.request_sleep_s = request_sleep_s

    def fetch_all_history(
        self,
        symbol: str,
        interval: str,
        interval_s: int,
        start_ms: int,
        end_ms: int,
    ) -> List[CandleRow]:
        raise NotImplementedError


class BinanceProvider(BaseProvider):
    name = "binance"

    def __init__(self, timeout_s: float, max_attempts: int, retry_sleep_s: float, request_sleep_s: float) -> None:
        super().__init__(timeout_s, max_attempts, retry_sleep_s, request_sleep_s)
        self.client = HttpJsonClient(
            provider_name=self.name,
            base_urls=BINANCE_BASE_URLS,
            timeout_s=timeout_s,
            max_attempts=max_attempts,
            retry_sleep_s=retry_sleep_s,
        )

    def fetch_klines(
        self,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
        limit: int,
    ) -> List[List[Any]]:
        payload = self.client.get_json(
            "/api/v3/klines",
            {
                "symbol": symbol,
                "interval": interval,
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": min(limit, 1000),
            },
        )
        if not isinstance(payload, list):
            raise ProviderError(f"Unexpected Binance payload for {symbol}: {type(payload).__name__}")
        return payload

    def fetch_all_history(self, symbol: str, interval: str, interval_s: int, start_ms: int, end_ms: int) -> List[CandleRow]:
        rows: List[CandleRow] = []
        cursor_ms = start_ms

        while cursor_ms < end_ms:
            page = self.fetch_klines(symbol, interval, cursor_ms, end_ms, DEFAULT_BATCH_LIMIT)
            if not page:
                break

            accepted = 0
            last_open_ms = None
            for entry in page:
                open_ms = int(entry[0])
                if open_ms < start_ms or open_ms >= end_ms:
                    continue
                rows.append(
                    CandleRow(
                        symbol=symbol,
                        interval_s=interval_s,
                        ts_ms=open_ms,
                        open=float(entry[1]),
                        high=float(entry[2]),
                        low=float(entry[3]),
                        close=float(entry[4]),
                        volume=float(entry[5]),
                    )
                )
                accepted += 1
                last_open_ms = open_ms

            if last_open_ms is None:
                break
            cursor_ms = last_open_ms + interval_s * 1000
            if accepted < DEFAULT_BATCH_LIMIT:
                break
            time.sleep(self.request_sleep_s)

        return dedupe_sort_rows(rows)


class BybitProvider(BaseProvider):
    name = "bybit"
    INTERVAL_MAP = {
        "1m": "1",
        "3m": "3",
        "5m": "5",
        "15m": "15",
        "30m": "30",
        "1h": "60",
        "2h": "120",
        "4h": "240",
        "6h": "360",
        "12h": "720",
        "1d": "D",
    }

    def __init__(self, timeout_s: float, max_attempts: int, retry_sleep_s: float, request_sleep_s: float) -> None:
        super().__init__(timeout_s, max_attempts, retry_sleep_s, request_sleep_s)
        self.client = HttpJsonClient(
            provider_name=self.name,
            base_urls=BYBIT_BASE_URLS,
            timeout_s=timeout_s,
            max_attempts=max_attempts,
            retry_sleep_s=retry_sleep_s,
        )

    def fetch_klines(self, symbol: str, interval: str, start_ms: int, end_ms: int, limit: int) -> List[List[Any]]:
        bybit_interval = self.INTERVAL_MAP.get(interval)
        if bybit_interval is None:
            raise ValueError(f"Unsupported Bybit interval: {interval}")
        payload = self.client.get_json(
            "/v5/market/kline",
            {
                "category": "spot",
                "symbol": symbol,
                "interval": bybit_interval,
                "start": start_ms,
                "end": end_ms,
                "limit": min(limit, 1000),
            },
        )
        if not isinstance(payload, dict) or payload.get("retCode") != 0:
            raise ProviderError(f"Unexpected Bybit payload for {symbol}: {payload}")
        result = payload.get("result", {})
        page = result.get("list", [])
        if not isinstance(page, list):
            raise ProviderError(f"Unexpected Bybit list payload for {symbol}: {payload}")
        return page

    def fetch_all_history(self, symbol: str, interval: str, interval_s: int, start_ms: int, end_ms: int) -> List[CandleRow]:
        rows: List[CandleRow] = []
        cursor_ms = start_ms

        while cursor_ms < end_ms:
            page = self.fetch_klines(symbol, interval, cursor_ms, end_ms, DEFAULT_BATCH_LIMIT)
            if not page:
                break
            last_open_ms = None
            accepted = 0
            for entry in sorted(page, key=lambda x: int(x[0])):
                open_ms = int(entry[0])
                if open_ms < start_ms or open_ms >= end_ms:
                    continue
                rows.append(
                    CandleRow(
                        symbol=symbol,
                        interval_s=interval_s,
                        ts_ms=open_ms,
                        open=float(entry[1]),
                        high=float(entry[2]),
                        low=float(entry[3]),
                        close=float(entry[4]),
                        volume=float(entry[5]),
                    )
                )
                accepted += 1
                last_open_ms = open_ms
            if last_open_ms is None:
                break
            cursor_ms = last_open_ms + interval_s * 1000
            if accepted < DEFAULT_BATCH_LIMIT:
                break
            time.sleep(self.request_sleep_s)

        return dedupe_sort_rows(rows)


class CoinbaseProvider(BaseProvider):
    name = "coinbase"

    QUOTE_FALLBACKS = {
        "USDT": ["USD", "USDC"],
        "USD": ["USD"],
        "USDC": ["USDC", "USD"],
    }

    BASE_OVERRIDES = {
        "XBT": "BTC",
    }

    def __init__(
        self,
        timeout_s: float,
        max_attempts: int,
        retry_sleep_s: float,
        request_sleep_s: float,
        alias_requested_symbol: bool = True,
    ) -> None:
        super().__init__(timeout_s, max_attempts, retry_sleep_s, request_sleep_s)
        self.alias_requested_symbol = alias_requested_symbol
        self.client = HttpJsonClient(
            provider_name=self.name,
            base_urls=COINBASE_BASE_URLS,
            timeout_s=timeout_s,
            max_attempts=max_attempts,
            retry_sleep_s=retry_sleep_s,
        )

    def product_candidates(self, symbol: str) -> List[str]:
        if not symbol.endswith(("USDT", "USD", "USDC")):
            base = symbol[:-3]
            quote = symbol[-3:]
        else:
            if symbol.endswith("USDT"):
                base, quote = symbol[:-4], "USDT"
            elif symbol.endswith("USDC"):
                base, quote = symbol[:-4], "USDC"
            else:
                base, quote = symbol[:-3], "USD"

        base = self.BASE_OVERRIDES.get(base, base)
        quotes = self.QUOTE_FALLBACKS.get(quote, [quote])
        candidates = [f"{base}-{q}" for q in quotes]
        # De-duplicate while preserving order.
        out: List[str] = []
        seen = set()
        for item in candidates:
            if item not in seen:
                seen.add(item)
                out.append(item)
        return out

    def fetch_klines(self, product_id: str, start_iso: str, end_iso: str, granularity: int) -> List[List[Any]]:
        payload = self.client.get_json(
            f"/products/{product_id}/candles",
            {
                "start": start_iso,
                "end": end_iso,
                "granularity": granularity,
            },
        )
        if not isinstance(payload, list):
            raise ProviderError(f"Unexpected Coinbase payload for {product_id}: {type(payload).__name__}")
        return payload

    def fetch_all_history(self, symbol: str, interval: str, interval_s: int, start_ms: int, end_ms: int) -> List[CandleRow]:
        if interval_s not in COINBASE_GRANULARITIES:
            raise ValueError(
                f"Coinbase only supports granularities {sorted(COINBASE_GRANULARITIES)} seconds; got {interval_s}"
            )

        errors: List[str] = []
        for product_id in self.product_candidates(symbol):
            LOGGER.info("Trying Coinbase product mapping %s -> %s", symbol, product_id)
            try:
                rows = self._fetch_product_history(symbol, product_id, interval_s, start_ms, end_ms)
                if rows:
                    return rows
                LOGGER.warning("Coinbase returned no candles for %s via %s", symbol, product_id)
            except ProviderError as exc:
                errors.append(f"{product_id}: {exc}")
                LOGGER.warning("Coinbase product %s failed for %s: %s", product_id, symbol, exc)
        raise ProviderError(f"No Coinbase product mapping succeeded for {symbol}. Errors: {'; '.join(errors) or 'no data'}")

    def _fetch_product_history(
        self,
        requested_symbol: str,
        product_id: str,
        interval_s: int,
        start_ms: int,
        end_ms: int,
    ) -> List[CandleRow]:
        rows: List[CandleRow] = []
        chunk_ms = COINBASE_MAX_CANDLES_PER_REQUEST * interval_s * 1000
        cursor_ms = start_ms
        stored_symbol = requested_symbol if self.alias_requested_symbol else product_id.replace("-", "")

        while cursor_ms < end_ms:
            chunk_end_ms = min(end_ms, cursor_ms + chunk_ms)
            start_iso = isoformat_ms(cursor_ms)
            end_iso = isoformat_ms(chunk_end_ms)
            page = self.fetch_klines(product_id, start_iso, end_iso, interval_s)
            if page:
                for entry in sorted(page, key=lambda x: int(x[0])):
                    ts_ms = int(entry[0]) * 1000
                    if ts_ms < start_ms or ts_ms >= end_ms:
                        continue
                    rows.append(
                        CandleRow(
                            symbol=stored_symbol,
                            interval_s=interval_s,
                            ts_ms=ts_ms,
                            open=float(entry[3]),
                            high=float(entry[2]),
                            low=float(entry[1]),
                            close=float(entry[4]),
                            volume=float(entry[5]),
                        )
                    )
            cursor_ms = chunk_end_ms
            time.sleep(self.request_sleep_s)

        return dedupe_sort_rows(rows)


def isoformat_ms(ts_ms: int) -> str:
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def dedupe_sort_rows(rows: Iterable[CandleRow]) -> List[CandleRow]:
    deduped: Dict[Tuple[str, int, int], CandleRow] = {}
    for row in rows:
        deduped[(row.symbol, row.interval_s, row.ts_ms)] = row
    return sorted(deduped.values(), key=lambda r: (r.symbol, r.ts_ms))


def create_table_if_needed(conn: sqlite3.Connection, table_name: str) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {quote_identifier(table_name)} (
            symbol TEXT NOT NULL,
            interval_s INTEGER NOT NULL,
            ts_ms INTEGER NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL NOT NULL,
            PRIMARY KEY (symbol, interval_s, ts_ms)
        )
        """
    )
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS {quote_identifier(f'idx_{table_name}_symbol_interval_ts')} ON {quote_identifier(table_name)} (symbol, interval_s, ts_ms)"
    )
    conn.commit()


def write_to_sqlite(conn: sqlite3.Connection, table_name: str, rows: Sequence[CandleRow]) -> int:
    if not rows:
        return 0
    before = conn.total_changes
    conn.executemany(
        f"""
        INSERT OR IGNORE INTO {quote_identifier(table_name)}
        (symbol, interval_s, ts_ms, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [(r.symbol, r.interval_s, r.ts_ms, r.open, r.high, r.low, r.close, r.volume) for r in rows],
    )
    conn.commit()
    return conn.total_changes - before


def quote_identifier(name: str) -> str:
    if not name:
        raise ValueError("SQL identifier cannot be empty")
    return '"' + name.replace('"', '""') + '"'


def make_provider(name: str, args: argparse.Namespace) -> BaseProvider:
    if name == "binance":
        return BinanceProvider(args.timeout_s, args.max_attempts, args.retry_sleep_s, args.request_sleep_s)
    if name == "bybit":
        return BybitProvider(args.timeout_s, args.max_attempts, args.retry_sleep_s, args.request_sleep_s)
    if name == "coinbase":
        return CoinbaseProvider(
            args.timeout_s,
            args.max_attempts,
            args.retry_sleep_s,
            args.request_sleep_s,
            alias_requested_symbol=not args.store_provider_symbol,
        )
    raise ValueError(f"Unsupported provider: {name}")


def provider_sequence(provider_arg: str) -> List[str]:
    if provider_arg == "auto":
        return ["binance", "bybit", "coinbase"]
    return [provider_arg]


def fetch_history_with_fallback(
    providers: Sequence[BaseProvider],
    symbol: str,
    interval: str,
    interval_s: int,
    start_ms: int,
    end_ms: int,
) -> Tuple[str, List[CandleRow]]:
    errors: List[str] = []
    for provider in providers:
        LOGGER.info(
            "Fetching %s %s candles from %s to %s via %s",
            symbol,
            interval,
            start_ms,
            end_ms,
            provider.name,
        )
        try:
            rows = provider.fetch_all_history(symbol, interval, interval_s, start_ms, end_ms)
            LOGGER.info("Provider %s returned %s rows for %s", provider.name, len(rows), symbol)
            return provider.name, rows
        except RestrictedLocationError as exc:
            errors.append(f"{provider.name}: restricted ({exc})")
            LOGGER.warning("Provider %s is restricted for %s: %s", provider.name, symbol, exc)
        except Exception as exc:  # pragma: no cover - best effort operational guard.
            errors.append(f"{provider.name}: {exc}")
            LOGGER.warning("Provider %s failed for %s: %s", provider.name, symbol, exc)
    raise RuntimeError(f"Failed to ingest {symbol} from providers {[p.name for p in providers]}. Errors: {'; '.join(errors)}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest historical OHLCV candles into research_candles.")
    parser.add_argument("--provider", choices=["auto", "binance", "bybit", "coinbase"], default="auto")
    parser.add_argument("--db-path", required=True)
    parser.add_argument("--table-name", default="research_candles")
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--interval", default="1h", choices=sorted(SUPPORTED_INTERVALS.keys(), key=lambda x: SUPPORTED_INTERVALS[x]))
    parser.add_argument("--lookback-days", type=float, required=True)
    parser.add_argument("--end-time-ms", type=int, default=None)
    parser.add_argument("--min-rows-per-symbol", type=int, default=DEFAULT_MIN_ROWS_PER_SYMBOL)
    parser.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_RETRY_ATTEMPTS)
    parser.add_argument("--retry-sleep-s", type=float, default=DEFAULT_RETRY_SLEEP_S)
    parser.add_argument("--request-sleep-s", type=float, default=DEFAULT_SLEEP_S)
    parser.add_argument("--store-provider-symbol", action="store_true", help="For Coinbase fallback, store the provider product symbol instead of the requested symbol alias.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args(argv)


def configure_logging(level_name: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level_name.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)

    interval_s = SUPPORTED_INTERVALS[args.interval]
    end_ms = args.end_time_ms if args.end_time_ms is not None else int(time.time() * 1000)
    start_ms = end_ms - int(args.lookback_days * 86400 * 1000)
    provider_names = provider_sequence(args.provider)
    providers = [make_provider(name, args) for name in provider_names]

    LOGGER.info("Opening SQLite database: %s", args.db_path)
    conn = sqlite3.connect(args.db_path)
    try:
        create_table_if_needed(conn, args.table_name)
        LOGGER.info(
            "Ingestion plan: providers=%s symbols=%s interval=%s interval_s=%s lookback_days=%s start_ms=%s end_ms=%s",
            provider_names,
            args.symbols,
            args.interval,
            interval_s,
            args.lookback_days,
            start_ms,
            end_ms,
        )

        results: List[IngestionResult] = []
        failures: List[str] = []
        for symbol in args.symbols:
            try:
                provider_name, rows = fetch_history_with_fallback(providers, symbol, args.interval, interval_s, start_ms, end_ms)
                written = write_to_sqlite(conn, args.table_name, rows)
                first_ts_ms = rows[0].ts_ms if rows else None
                last_ts_ms = rows[-1].ts_ms if rows else None
                results.append(
                    IngestionResult(
                        provider=provider_name,
                        symbol=symbol,
                        rows_fetched=len(rows),
                        rows_written=written,
                        first_ts_ms=first_ts_ms,
                        last_ts_ms=last_ts_ms,
                    )
                )
                LOGGER.info(
                    "Completed %s via %s: fetched=%s written=%s first_ts_ms=%s last_ts_ms=%s",
                    symbol,
                    provider_name,
                    len(rows),
                    written,
                    first_ts_ms,
                    last_ts_ms,
                )
                if len(rows) < args.min_rows_per_symbol:
                    LOGGER.warning(
                        "Symbol %s only has %s rows, below minimum target %s",
                        symbol,
                        len(rows),
                        args.min_rows_per_symbol,
                    )
            except Exception as exc:
                failures.append(f"{symbol}: {exc}")
                LOGGER.error("Failed to ingest %s: %s", symbol, exc)

        total_fetched = sum(item.rows_fetched for item in results)
        total_written = sum(item.rows_written for item in results)
        LOGGER.info(
            "Ingestion complete: success_symbols=%s failed_symbols=%s total_fetched=%s total_written=%s table=%s",
            len(results),
            len(failures),
            total_fetched,
            total_written,
            args.table_name,
        )

        for item in results:
            LOGGER.info(
                "Result symbol=%s provider=%s fetched=%s written=%s first_ts_ms=%s last_ts_ms=%s",
                item.symbol,
                item.provider,
                item.rows_fetched,
                item.rows_written,
                item.first_ts_ms,
                item.last_ts_ms,
            )

        if failures:
            for failure in failures:
                LOGGER.error("Ingestion failure: %s", failure)
            return 1
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
