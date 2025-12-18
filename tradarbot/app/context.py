from dataclasses import dataclass
from typing import Any, Dict

from tradarbot.core.state import State
from tradarbot.execution.paper_broker import PaperBroker
from tradarbot.risk.risk_manager import RiskManager
from tradarbot.storage.sqlite_store import SQLiteStore

@dataclass
class Ctx:
    cfg: Dict[str, Any]
    state: State
    store: SQLiteStore
    broker: PaperBroker
    risk: RiskManager
