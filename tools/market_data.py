from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class PriceBar:
    timestamp: str
    symbol: str
    open: float
    high: float
    low: float
    close: float

class HistoricalPriceFeed:
    def __init__(self, prices: Dict[str, List[PriceBar]]) -> None:
        self.prices = prices
        self.index = 0

    def has_next(self) -> bool:
        # assumes all symbols have same lenght

