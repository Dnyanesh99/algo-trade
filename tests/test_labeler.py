import asyncio
import logging

import numpy as np
import pandas as pd

from src.core.labeler import BarrierConfig, OptimizedTripleBarrierLabeler
from src.database.label_repo import LabelRepository
from src.utils.config_loader import config_loader
from src.utils.logger import LOGGER as logger

# Load configuration
config = config_loader.get_config()

# Configure logging for the test file
logging.basicConfig(level=getattr(logging, config.logging.level), format=config.logging.format)


async def test_labeler_functionality():
    """
    Tests the functionality of the OptimizedTripleBarrierLabeler with sample data.
    """
    logger.info("\n--- Starting Labeler Functionality Test ---")

    # Initialize LabelRepository (can be mocked for isolated testing)
    label_repo = LabelRepository()

    # Initialize with custom configuration
    barrier_config = BarrierConfig(
        atr_period=config.example.data_generation.atr_period,
        tp_atr_multiplier=config.example.data_generation.tp_atr_multiplier,
        sl_atr_multiplier=config.example.data_generation.sl_atr_multiplier,
        max_holding_periods=config.example.data_generation.max_holding_periods,
        close_open_threshold=config.example.data_generation.close_open_threshold,
    )

    labeler = OptimizedTripleBarrierLabeler(barrier_config, label_repo)

    # Generate sample data for testing
    example_config = config.example.data_generation
    np.random.seed(example_config.random_seed)
    dates = pd.date_range(example_config.start_date, example_config.end_date, freq=example_config.frequency)
    n = len(dates)

    # Simulate price with trend and volatility
    returns = np.random.normal(example_config.return_mean, example_config.return_std, n)
    price = example_config.initial_price * np.exp(np.cumsum(returns))

    # Create OHLCV data
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": price * (1 + np.random.normal(0, example_config.price_noise_std, n)),
            "high": price * (1 + np.abs(np.random.normal(0, example_config.high_noise_std, n))),
            "low": price * (1 - np.abs(np.random.normal(0, example_config.low_noise_std, n))),
            "close": price,
            "volume": np.random.lognormal(example_config.volume_log_mean, example_config.volume_log_std, n),
        }
    )

    # Ensure OHLC relationships
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)

    # Process data
    with labeler.performance_monitor("Full processing"):
        result = await labeler.process_symbol(1, df, "TEST_SYMBOL")  # Pass instrument_id and symbol

    if result is not None:
        # Display results
        logger.info("\n=== Processing Results ===")
        logger.info(f"Total rows: {len(result):,}")
        logger.info(f"Labeled rows: {result['label'].notna().sum():,}")
        logger.info("\nLabel distribution:")
        logger.info(result["label"].value_counts().sort_index())
        logger.info("\nExit reasons:")
        logger.info(result["exit_reason"].value_counts())
        logger.info("\nAverage returns by label:")
        for label_val in [-1, 0, 1]:
            data = result[result["label"] == label_val]
            if len(data) > 0:
                avg_return = data["barrier_return"].mean() * 100
                logger.info(f"  Label {label_val}: {avg_return:.4f}%")

        # Show summary statistics
        logger.info("\n=== Summary Statistics ===")
        summary = labeler.get_stats_summary()
        logger.info(summary.to_string(index=False))

        # Assertions for basic validation
        assert not result.empty, "Result DataFrame should not be empty"
        assert result["label"].notna().sum() > 0, "At least some rows should be labeled"
        assert "label" in result.columns, "'label' column should be in the result"

    logger.info("--- Labeler Functionality Test Completed ---")


if __name__ == "__main__":
    asyncio.run(test_labeler_functionality())
