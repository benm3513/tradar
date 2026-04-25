from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_DOWN, InvalidOperation
from typing import Any, Dict, Optional

from tradarbot.core.events import OrderIntent
from tradarbot.execution.symbol_mapper import SymbolMapper


@dataclass
class RoutedOrder:
    symbol: str
    side: str
    order_type: str
    quantity: float
    price: Optional[float]
    tif: Optional[str]
    client_order_id: str
    provider: str = "generic"
    venue_symbol: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)


class OrderRouter:
    """Provider-aware OrderIntent -> exchange order payload router.

    Binance-style exchanges need strict symbol filters from exchangeInfo.
    Alpaca does not expose Binance-like filters, so this router uses simple
    configured/default rounding and maps Tradar symbols like ETHUSDT -> ETH/USD.
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = dict(cfg or {})
        self.exec_cfg = dict(self.cfg.get("execution_live", {}) or {})
        self.provider = self._normalize_provider(
            self.exec_cfg.get("provider", self.exec_cfg.get("exchange", "binance"))
        )
        self.default_order_type = str(self.exec_cfg.get("default_order_type", "limit") or "limit").upper()
        self.default_tif = str(self.exec_cfg.get("default_tif", "IOC") or "IOC")
        self.client_order_id_prefix = str(self.exec_cfg.get("client_order_id_prefix", "tradar") or "tradar")
        self.symbol_rules = dict(self.exec_cfg.get("symbol_rules", {}) or {})
        self.symbol_mapper = SymbolMapper(self.cfg)

    def route_intent(self, intent: OrderIntent) -> RoutedOrder:
        symbol = str(intent.symbol).upper()
        side = str(intent.side).upper()
        if side not in {"BUY", "SELL"}:
            raise ValueError(f"Unsupported side={side!r}")

        venue_symbol = self.to_venue_symbol(symbol)
        order_type = self._resolve_order_type(intent)
        tif = self._resolve_tif(intent, order_type)

        quantity = self._round_qty(symbol=symbol, venue_symbol=venue_symbol, qty=float(intent.qty))
        if quantity <= 0.0:
            raise ValueError(f"Routed quantity is non-positive for {symbol}: {quantity}")

        price: Optional[float] = None
        if order_type == "LIMIT":
            limit_px = float(getattr(intent, "limit_px", 0.0) or 0.0)
            if limit_px <= 0.0:
                raise ValueError(f"Limit order requires positive limit_px for {symbol}")
            price = self._round_price(symbol=symbol, venue_symbol=venue_symbol, price=limit_px)
            if price <= 0.0:
                raise ValueError(f"Routed price is non-positive for {symbol}: {price}")

        client_order_id = self._client_order_id(symbol=symbol, side=side)

        raw = {
            "source_symbol": symbol,
            "venue_symbol": venue_symbol,
            "provider": self.provider,
            "source_qty": float(intent.qty),
            "source_limit_px": float(getattr(intent, "limit_px", 0.0) or 0.0),
            "symbol_rule": self._rules_for(symbol, venue_symbol),
        }

        return RoutedOrder(
            symbol=venue_symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            tif=tif,
            client_order_id=client_order_id,
            provider=self.provider,
            venue_symbol=venue_symbol,
            raw=raw,
        )

    def to_venue_symbol(self, symbol: str) -> str:
        return self.symbol_mapper.to_venue_symbol(symbol)

    def to_source_symbol(self, symbol: str) -> str:
        return self.symbol_mapper.to_source_symbol(symbol)

    def market_symbol_candidates(self, symbol: str):
        return self.symbol_mapper.market_symbol_candidates(symbol)

    def _resolve_order_type(self, intent: OrderIntent) -> str:
        raw_type = getattr(intent, "order_type", None) or self.default_order_type
        order_type = str(raw_type).upper()
        if order_type not in {"LIMIT", "MARKET"}:
            # First pass Phase 5.1 supports market + limit only.
            raise ValueError(f"Unsupported order_type={order_type!r}")
        return order_type

    def _resolve_tif(self, intent: OrderIntent, order_type: str) -> Optional[str]:
        raw_tif = getattr(intent, "tif", None) or self.default_tif
        if order_type == "MARKET" and self.provider != "alpaca":
            return None if raw_tif is None else str(raw_tif).upper()
        if raw_tif is None:
            return None
        tif = str(raw_tif).upper()
        if self.provider == "alpaca":
            # Alpaca accepts lower-case in JSON, but keep RoutedOrder standardized;
            # AlpacaClient converts to lower-case on send.
            if tif == "GTC":
                return "GTC"
            if tif == "IOC":
                return "IOC"
            # Day is useful for equities; crypto supports gtc/ioc.
            if tif == "DAY":
                return "DAY"
            return tif
        return tif

    def _rules_for(self, symbol: str, venue_symbol: str) -> Dict[str, Any]:
        candidates = [symbol, venue_symbol, venue_symbol.replace("/", "")]
        for key in candidates:
            if key in self.symbol_rules:
                return dict(self.symbol_rules[key] or {})

        if self.provider == "alpaca":
            return {
                "qty_step": self.exec_cfg.get("alpaca_qty_step", 0.000001),
                "price_tick": self.exec_cfg.get("alpaca_price_tick", 0.01),
                "min_qty": self.exec_cfg.get("alpaca_min_qty", 0.000001),
            }

        return {
            "qty_step": self.exec_cfg.get("default_qty_step", 0.0001),
            "price_tick": self.exec_cfg.get("default_price_tick", 0.01),
            "min_qty": self.exec_cfg.get("default_min_qty", 0.0),
        }

    def _round_qty(self, *, symbol: str, venue_symbol: str, qty: float) -> float:
        rules = self._rules_for(symbol, venue_symbol)
        step = float(rules.get("qty_step", rules.get("stepSize", 0.000001)) or 0.000001)
        min_qty = float(rules.get("min_qty", rules.get("minQty", 0.0)) or 0.0)
        rounded = self._floor_to_step(qty, step)
        if rounded < min_qty:
            return 0.0
        return rounded

    def _round_price(self, *, symbol: str, venue_symbol: str, price: float) -> float:
        rules = self._rules_for(symbol, venue_symbol)
        tick = float(rules.get("price_tick", rules.get("tickSize", 0.01)) or 0.01)
        return self._floor_to_step(price, tick)

    @staticmethod
    def _floor_to_step(value: float, step: float) -> float:
        if step <= 0.0:
            return float(value)
        if value <= 0.0:
            return 0.0
        try:
            d_value = Decimal(str(value))
            d_step = Decimal(str(step))
            units = (d_value / d_step).to_integral_value(rounding=ROUND_DOWN)
            result = units * d_step
            return float(result)
        except (InvalidOperation, ValueError):
            return math.floor(value / step) * step


    # ------------------------------------------------------------------
    # Unified sizing / precision helpers
    # ------------------------------------------------------------------
    def normalize_quantity(self, symbol: str, qty: float) -> float:
        """Round a source or venue quantity exactly as route_intent() will.

        Use this before creating OrderIntent objects and again in LiveBroker before
        submission so strategy -> router -> broker share one precision policy.
        """
        source_symbol = self.to_source_symbol(str(symbol).upper())
        venue_symbol = self.to_venue_symbol(source_symbol)
        return self._round_qty(symbol=source_symbol, venue_symbol=venue_symbol, qty=float(qty or 0.0))

    def normalize_price(self, symbol: str, price: float) -> float:
        """Round a source or venue price exactly as route_intent() will."""
        source_symbol = self.to_source_symbol(str(symbol).upper())
        venue_symbol = self.to_venue_symbol(source_symbol)
        return self._round_price(symbol=source_symbol, venue_symbol=venue_symbol, price=float(price or 0.0))

    def max_buy_quantity_for_cash(
        self,
        *,
        symbol: str,
        price: float,
        cash: float,
        fee_bps: float = 0.0,
        cash_buffer: float = 0.995,
    ) -> float:
        """Largest routed BUY quantity that should fit available cash.

        The calculation uses the routed/rounded limit price, fee estimate, and a
        configurable cash buffer. The returned quantity is floored to the same
        provider precision as route_intent().
        """
        price = self.normalize_price(symbol, float(price or 0.0))
        cash = float(cash or 0.0)
        if price <= 0.0 or cash <= 0.0:
            return 0.0
        buffer = max(0.0, min(1.0, float(cash_buffer or 1.0)))
        fee_mult = 1.0 + max(0.0, float(fee_bps or 0.0)) / 10_000.0
        raw_qty = (cash * buffer) / max(price * fee_mult, 1e-12)
        return self.normalize_quantity(symbol, raw_qty)

    def clamp_buy_quantity_to_cash(
        self,
        *,
        symbol: str,
        desired_qty: float,
        price: float,
        cash: float,
        fee_bps: float = 0.0,
        cash_buffer: float = 0.995,
    ) -> float:
        routed_desired = self.normalize_quantity(symbol, float(desired_qty or 0.0))
        max_qty = self.max_buy_quantity_for_cash(
            symbol=symbol,
            price=price,
            cash=cash,
            fee_bps=fee_bps,
            cash_buffer=cash_buffer,
        )
        return self.normalize_quantity(symbol, min(routed_desired, max_qty))

    def clamp_sell_quantity_to_position(
        self,
        *,
        symbol: str,
        desired_qty: float,
        available_qty: float,
        position_buffer: float = 0.995,
    ) -> float:
        """Largest routed SELL quantity that should fit available position.

        Alpaca paper can report a slightly smaller available crypto balance than
        the just-filled local position quantity. The buffer avoids full-size SELL
        rejects while Phase 5.4 adds exchange reconciliation.
        """
        available_qty = max(0.0, float(available_qty or 0.0))
        desired_qty = max(0.0, float(desired_qty or 0.0))
        buffer = max(0.0, min(1.0, float(position_buffer or 1.0)))
        raw_qty = min(desired_qty, available_qty * buffer)
        return self.normalize_quantity(symbol, raw_qty)

    def _client_order_id(self, *, symbol: str, side: str) -> str:
        clean_symbol = symbol.lower().replace("/", "").replace("_", "")
        clean_side = side.lower()
        now_ms = int(time.time() * 1000)
        return f"{self.client_order_id_prefix}-{clean_symbol}-{clean_side}-{now_ms}"

    @staticmethod
    def _normalize_provider(value: Any) -> str:
        provider = str(value or "binance").lower().replace("-", "_")
        if provider in {"alpaca_paper", "alpaca_live"}:
            return "alpaca"
        if provider in {"binanceus", "binance_us"}:
            return "binance_us"
        return provider
