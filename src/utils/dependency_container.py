"""A typed dependency container for the application."""

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from src.core.feature_calculator import FeatureCalculator
    from src.core.historical_aggregator import HistoricalAggregator
    from src.core.labeler import OptimizedTripleBarrierLabeler
    from src.database.instrument_repo import InstrumentRepository
    from src.database.ohlcv_repo import OHLCVRepository
    from src.historical.fetcher import HistoricalFetcher
    from src.historical.processor import HistoricalProcessor
    from src.model.lgbm_trainer import LGBMTrainer


class HistoricalPipelineDependencies:
    """Typed container for historical pipeline dependencies."""

    def __init__(self, **kwargs: Any) -> None:
        # Dependencies injected via kwargs are validated at the container level
        self.instrument_repo: InstrumentRepository = cast("InstrumentRepository", kwargs.get("instrument_repo"))
        self.historical_fetcher: HistoricalFetcher = cast("HistoricalFetcher", kwargs.get("historical_fetcher"))
        self.historical_processor: HistoricalProcessor = cast("HistoricalProcessor", kwargs.get("historical_processor"))
        self.ohlcv_repo: OHLCVRepository = cast("OHLCVRepository", kwargs.get("ohlcv_repo"))
        self.historical_aggregator: HistoricalAggregator = cast(
            "HistoricalAggregator", kwargs.get("historical_aggregator")
        )
        self.feature_calculator: FeatureCalculator = cast("FeatureCalculator", kwargs.get("feature_calculator"))
        self.labeler: OptimizedTripleBarrierLabeler = cast("OptimizedTripleBarrierLabeler", kwargs.get("labeler"))
        self.lgbm_trainer: LGBMTrainer = cast("LGBMTrainer", kwargs.get("lgbm_trainer"))
