"""
Central metrics registry for Prometheus metrics.
Thread-safe singleton that manages all application metrics based on configuration.
"""

import threading
from typing import Optional, Union

from prometheus_client import Counter, Gauge, Histogram, start_http_server
from prometheus_client.core import CollectorRegistry

from src.utils.config_loader import MetricsConfig
from src.utils.logger import LOGGER as logger


class MetricsRegistry:
    """
    Central registry for all Prometheus metrics.
    """

    _instance: Optional["MetricsRegistry"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "MetricsRegistry":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def initialize(self, metrics_config: Optional[MetricsConfig]) -> None:
        """Initialize the metrics registry"""
        if getattr(self, "_initialized", False):
            return

        self._registry = CollectorRegistry()
        self._metrics: dict[str, Union[Counter, Gauge, Histogram]] = {}
        self._server_started = False
        self._metrics_config = metrics_config

        if self._metrics_config and self._metrics_config.enabled:
            self._create_metrics_from_config()
            logger.info(f"Created {len(self._metrics)} metrics from configuration")
        else:
            logger.info("Metrics disabled in configuration")

        self._initialized = True
        logger.info("MetricsRegistry initialized")

    def _create_metrics_from_config(self) -> None:
        """Create Prometheus metrics based on configuration"""
        if not self._metrics_config:
            return

        for _module_name, metric_definitions in self._metrics_config.definitions.items():
            for metric_def in metric_definitions:
                metric_name = f"algo_trading_{metric_def.name}"

                metric: Union[Counter, Gauge, Histogram]
                if metric_def.type == "counter":
                    metric = Counter(metric_name, metric_def.description, metric_def.labels, registry=self._registry)
                elif metric_def.type == "gauge":
                    metric = Gauge(metric_name, metric_def.description, metric_def.labels, registry=self._registry)
                elif metric_def.type == "histogram":
                    buckets = metric_def.buckets or self._metrics_config.default_histogram_buckets
                    metric = Histogram(
                        metric_name, metric_def.description, metric_def.labels, buckets=buckets, registry=self._registry
                    )

                self._metrics[metric_def.name] = metric
                logger.info(f"Registered metric: {metric_name} of type {metric_def.type}")

    def start_metrics_server(self, port: Optional[int] = None) -> None:
        """Start the Prometheus metrics HTTP server"""
        if self._server_started:
            logger.warning("Metrics server already started")
            return

        if not self._metrics_config or not self._metrics_config.enabled:
            logger.info("Metrics disabled, not starting server")
            return

        server_port = port or self._metrics_config.endpoint_port

        try:
            start_http_server(server_port, registry=self._registry)
            self._server_started = True
            logger.info(f"Prometheus metrics server started on port {server_port}")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")

    def is_enabled(self) -> bool:
        """Check if metrics are enabled"""
        return hasattr(self, "_metrics_config") and self._metrics_config is not None and self._metrics_config.enabled

    def increment_counter(self, metric_name: str, labels: Optional[dict[str, str]] = None, value: float = 1.0) -> None:
        """Increment a counter metric"""
        if not self.is_enabled():
            return

        if not hasattr(self, "_metrics"):
            logger.debug(f"Metrics not initialized, skipping counter increment for: {metric_name}")
            return

        metric = self._metrics.get(metric_name)
        if metric and isinstance(metric, Counter):
            if labels:
                metric.labels(**labels).inc(value)
            else:
                metric.inc(value)
        else:
            logger.warning(f"Counter metric not found: {metric_name}")

    def set_gauge(self, metric_name: str, value: float, labels: Optional[dict[str, str]] = None) -> None:
        """Set a gauge metric value"""
        if not self.is_enabled():
            return

        if not hasattr(self, "_metrics"):
            logger.debug(f"Metrics not initialized, skipping gauge set for: {metric_name}")
            return

        metric = self._metrics.get(metric_name)
        if metric and isinstance(metric, Gauge):
            if labels:
                metric.labels(**labels).set(value)
            else:
                metric.set(value)
        else:
            logger.warning(f"Gauge metric not found: {metric_name}")

    def observe_histogram(self, metric_name: str, value: float, labels: Optional[dict[str, str]] = None) -> None:
        """Observe a value in a histogram metric"""
        if not self.is_enabled():
            return

        if not hasattr(self, "_metrics"):
            logger.debug(f"Metrics not initialized, skipping histogram observation for: {metric_name}")
            return

        metric = self._metrics.get(metric_name)
        if metric and isinstance(metric, Histogram):
            if labels:
                metric.labels(**labels).observe(value)
            else:
                metric.observe(value)
        else:
            logger.warning(f"Histogram metric not found: {metric_name}")

    def record_db_query(self, operation: str, table: str, success: bool, duration: float, rows: int) -> None:
        """Record database query metrics."""
        if not self.is_enabled():
            return

        query_total_labels = {"operation": operation, "table": table, "status": str(success)}
        logger.debug(f"Recording db_queries_total with labels: {query_total_labels}")
        self.increment_counter("db_queries_total", query_total_labels)

        duration_rows_labels = {"operation": operation, "table": table}
        logger.debug(f"Recording db_query_duration_seconds with labels: {duration_rows_labels}")
        self.observe_histogram("db_query_duration_seconds", duration, duration_rows_labels)
        logger.debug(f"Recording db_rows_processed_total with labels: {duration_rows_labels}")
        self.increment_counter("db_rows_processed_total", duration_rows_labels, value=float(rows))

    def record_broker_api_request(self, api_name: str, success: bool, duration: float) -> None:
        """Record broker API request metrics."""
        if not self.is_enabled():
            return

        # Labels for broker_api_requests_total: ["endpoint", "status"]
        api_labels = {"endpoint": api_name, "status": str(success)}
        self.increment_counter("broker_api_requests_total", api_labels)

        # Labels for broker_api_duration_seconds: ["endpoint"]
        duration_labels = {"endpoint": api_name}
        self.observe_histogram("broker_api_duration_seconds", duration, duration_labels)

    def record_auth_request(self, auth_type: str, success: bool, duration: float) -> None:
        """Record authentication request metrics."""
        if not self.is_enabled():
            return

        # Labels for auth_requests_total: ["method", "status"]
        auth_labels = {"method": auth_type, "status": str(success)}
        self.increment_counter("auth_requests_total", auth_labels)

        # Labels for auth_request_duration_seconds: ["method"]
        duration_labels = {"method": auth_type}
        self.observe_histogram("auth_request_duration_seconds", duration, duration_labels)

    def record_model_training(
        self, instrument: str, timeframe: str, model_type: str, success: bool, duration: float
    ) -> None:
        """Record model training metrics."""
        if not self.is_enabled():
            return

        labels = {"instrument": instrument, "timeframe": timeframe, "model_type": model_type, "success": str(success)}
        self.increment_counter("model_training_total", labels)
        self.observe_histogram("model_training_duration_seconds", duration, labels)

    def record_model_prediction(
        self,
        instrument: str,
        timeframe: str,
        model_type: str,
        prediction_label: str,
        confidence: float,
        duration: float,
        success: bool = True,
    ) -> None:
        """Record model prediction metrics including confidence distribution."""
        if not self.is_enabled():
            return

        # Record prediction count and duration
        pred_labels = {
            "instrument": instrument,
            "timeframe": timeframe,
            "model_type": model_type,
            "prediction_label": prediction_label,
            "status": str(success),
        }
        self.increment_counter("model_prediction_total", pred_labels)

        duration_labels = {"instrument": instrument, "timeframe": timeframe, "model_type": model_type}
        self.observe_histogram("model_prediction_duration_seconds", duration, duration_labels)

        # Record confidence distribution
        conf_labels = {"instrument": instrument, "timeframe": timeframe, "prediction_label": prediction_label}
        self.observe_histogram("model_prediction_confidence_distribution", confidence, conf_labels)

    def record_model_performance(
        self, instrument: str, timeframe: str, metric_type: str, class_label: str, value: float
    ) -> None:
        """Record model performance metrics (precision, recall, f1, accuracy)."""
        if not self.is_enabled():
            return

        labels = {
            "instrument": instrument,
            "timeframe": timeframe,
            "metric_type": metric_type,
            "class_label": class_label,
        }
        self.set_gauge("model_performance_metrics", value, labels)

    def record_feature_drift(
        self, instrument: str, timeframe: str, feature_name: str, z_score: float, severity: str = "info"
    ) -> None:
        """Record feature drift metrics."""
        if not self.is_enabled():
            return

        # Record z-score
        drift_labels = {"instrument": instrument, "timeframe": timeframe, "feature_name": feature_name}
        self.set_gauge("feature_drift_z_score", z_score, drift_labels)

        # Record alert if drift is significant
        if abs(z_score) > 2.0:  # Configurable threshold
            alert_labels = drift_labels.copy()
            alert_labels["severity"] = severity
            self.increment_counter("feature_drift_alerts_total", alert_labels)

    def record_data_quality(self, instrument: str, timeframe: str, quality_score: float) -> None:
        """Record data quality metrics."""
        if not self.is_enabled():
            return

        labels = {"instrument": instrument, "timeframe": timeframe}
        self.set_gauge("data_quality_score", quality_score, labels)

    def record_prediction_accuracy(self, instrument: str, timeframe: str, window_size: str, accuracy: float) -> None:
        """Record real-time prediction accuracy."""
        if not self.is_enabled():
            return

        labels = {"instrument": instrument, "timeframe": timeframe, "window_size": window_size}
        self.set_gauge("model_accuracy_real_time", accuracy, labels)

    def record_prediction_bias(self, instrument: str, timeframe: str, bias_type: str, bias_score: float) -> None:
        """Record prediction bias metrics."""
        if not self.is_enabled():
            return

        labels = {"instrument": instrument, "timeframe": timeframe, "bias_type": bias_type}
        self.set_gauge("prediction_bias_score", bias_score, labels)

    def record_feature_importance_shift(
        self, instrument: str, timeframe: str, feature_name: str, shift_value: float
    ) -> None:
        """Record feature importance drift."""
        if not self.is_enabled():
            return

        labels = {"instrument": instrument, "timeframe": timeframe, "feature_name": feature_name}
        self.set_gauge("feature_importance_shift", shift_value, labels)

    def record_signal_effectiveness(
        self, instrument: str, signal_type: str, timeframe: str, effectiveness_ratio: float
    ) -> None:
        """Record signal effectiveness metrics."""
        if not self.is_enabled():
            return

        labels = {"instrument": instrument, "signal_type": signal_type, "timeframe": timeframe}
        self.set_gauge("signal_effectiveness_ratio", effectiveness_ratio, labels)

    def record_signal_pnl(self, instrument: str, signal_type: str, pnl_value: float) -> None:
        """Record signal PnL attribution."""
        if not self.is_enabled():
            return

        labels = {"instrument": instrument, "signal_type": signal_type}
        self.observe_histogram("signal_pnl_attribution", pnl_value, labels)

    def record_signal_timing(self, instrument: str, signal_type: str, latency_seconds: float) -> None:
        """Record signal timing latency."""
        if not self.is_enabled():
            return

        labels = {"instrument": instrument, "signal_type": signal_type}
        self.observe_histogram("signal_timing_latency", latency_seconds, labels)

    def record_feature_generation(
        self, instrument: str, timeframe: str, pattern: str, duration: float, success: bool = True
    ) -> None:
        """Record feature generation metrics."""
        if not self.is_enabled():
            return

        # Record generation count
        gen_labels = {"instrument": instrument, "timeframe": timeframe, "pattern": pattern, "status": str(success)}
        self.increment_counter("features_generated_total", gen_labels)

        # Record generation duration
        duration_labels = {"instrument": instrument, "timeframe": timeframe, "pattern": pattern}
        self.observe_histogram("feature_generation_duration_seconds", duration, duration_labels)

    def record_feature_selection_change(self, instrument: str, timeframe: str, change_type: str) -> None:
        """Record feature selection changes."""
        if not self.is_enabled():
            return

        labels = {"instrument": instrument, "timeframe": timeframe, "change_type": change_type}
        self.increment_counter("feature_selection_changes_total", labels)

    def record_selected_features_count(self, instrument: str, timeframe: str, count: int) -> None:
        """Record current number of selected features."""
        if not self.is_enabled():
            return

        labels = {"instrument": instrument, "timeframe": timeframe}
        self.set_gauge("selected_features_count", float(count), labels)

    def record_feature_importance_distribution(self, instrument: str, timeframe: str, importance_score: float) -> None:
        """Record feature importance score distribution."""
        if not self.is_enabled():
            return

        labels = {"instrument": instrument, "timeframe": timeframe}
        self.observe_histogram("feature_importance_distribution", importance_score, labels)

    def record_cross_asset_correlation(
        self, primary_instrument: str, related_instrument: str, timeframe: str, correlation: float
    ) -> None:
        """Record cross-asset correlation strength."""
        if not self.is_enabled():
            return

        labels = {
            "primary_instrument": primary_instrument,
            "related_instrument": related_instrument,
            "timeframe": timeframe,
        }
        self.set_gauge("cross_asset_correlation_score", correlation, labels)

    def record_feature_lineage_depth(self, feature_type: str, depth: int) -> None:
        """Record feature dependency chain depth."""
        if not self.is_enabled():
            return

        labels = {"feature_type": feature_type}
        self.set_gauge("feature_lineage_depth", float(depth), labels)


# Singleton instance to be created and initialized in main.py
metrics_registry = MetricsRegistry()
