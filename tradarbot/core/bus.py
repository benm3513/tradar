import asyncio
import logging
from typing import Any, Callable, Dict, List, Type

log = logging.getLogger("tradar.bus")
Handler = Callable[[Any], Any]

class EventBus:
    """
    Minimal async event bus.
    - publish(event) puts the event on a queue
    - run() dispatches events to handlers registered for the event's type
    """
    def __init__(self):
        self._q: "asyncio.Queue[Any]" = asyncio.Queue()
        self._subs: Dict[Type[Any], List[Handler]] = {}

    def subscribe(self, event_type: Type[Any], handler: Handler) -> None:
        self._subs.setdefault(event_type, []).append(handler)

    def publish(self, event: Any) -> None:
        self._q.put_nowait(event)

    async def run(self) -> None:
        while True:
            ev = await self._q.get()
            for h in self._subs.get(type(ev), []):
                try:
                    out = h(ev)
                    if asyncio.iscoroutine(out):
                        await out
                except Exception:
                    log.exception("handler failed event=%s handler=%s", type(ev).__name__, h)
