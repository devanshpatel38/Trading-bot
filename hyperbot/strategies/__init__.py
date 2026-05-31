from .base import Strategy, StrategySignal, last_timestamp
from .ema_trend import EmaTrendStrategy
from .rsi_meanrev import RsiMeanRevStrategy
from .bb_squeeze import BbSqueezeStrategy
from .fvg import FvgStrategy
from .macd_momentum import MacdMomentumStrategy

REGISTRY = {
    "ema_trend": EmaTrendStrategy,
    "rsi_meanrev": RsiMeanRevStrategy,
    "bb_squeeze": BbSqueezeStrategy,
    "fvg": FvgStrategy,
    "macd_momentum": MacdMomentumStrategy,
}

__all__ = ["Strategy", "StrategySignal", "last_timestamp", "REGISTRY"]