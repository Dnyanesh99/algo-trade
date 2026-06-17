import json

import asyncpg

from src.database.db_utils import DatabaseManager
from src.database.models import LabelingStatsData
from src.utils.config_loader import ConfigLoader
from src.utils.logger import LOGGER as logger

config = ConfigLoader().get_config()


class LabelingStatsRepository:
    """
    Repository for persisting and retrieving labeling statistics.
    """

    def __init__(self) -> None:
        self.queries = ConfigLoader().get_queries().label_stats_repo
        self.db_manager = DatabaseManager()
        logger.info("LabelingStatsRepository initialized.")

    async def insert_stats(self, stats_data: LabelingStatsData) -> None:
        """
        Insert a new labeling statistics record into the database.

        Raises:
            ValueError: If stats_data is invalid.
            RuntimeError: If the database insertion fails.
        """
        if not isinstance(stats_data, LabelingStatsData):
            raise ValueError("Invalid LabelingStatsData object provided.")

        query = self.queries["insert_stats"]
        logger.debug(f"Inserting labeling statistics for {stats_data.symbol} at {stats_data.run_timestamp}.")
        try:
            await self.db_manager.execute(
                query,
                stats_data.symbol,
                stats_data.timeframe,
                stats_data.total_bars,
                stats_data.labeled_bars,
                json.dumps(stats_data.label_distribution)
                if isinstance(stats_data.label_distribution, dict)
                else stats_data.label_distribution,
                json.dumps(stats_data.avg_return_by_label)
                if isinstance(stats_data.avg_return_by_label, dict)
                else stats_data.avg_return_by_label,
                json.dumps(stats_data.exit_reasons)
                if isinstance(stats_data.exit_reasons, dict)
                else stats_data.exit_reasons,
                stats_data.avg_holding_period,
                stats_data.processing_time_ms,
                stats_data.data_quality_score,
                stats_data.sharpe_ratio,
                stats_data.win_rate,
                stats_data.profit_factor,
                stats_data.run_timestamp,
            )
            logger.info(f"Successfully inserted labeling statistics for {stats_data.symbol}.")
        except asyncpg.UniqueViolationError:
            logger.warning(
                f"A labeling statistics record for {stats_data.symbol} at {stats_data.run_timestamp} already exists. Skipping insertion."
            )
        except Exception as e:
            safe_error = str(e).replace("{", "{{").replace("}", "}}")
            logger.error(
                f"Database error inserting labeling statistics for {stats_data.symbol}: {safe_error}", exc_info=True
            )
            raise RuntimeError("Failed to insert labeling statistics.") from e
