# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Model Monitoring System for TensorFlow Production Models."""

import time
import threading
from collections import deque, defaultdict
import numpy as np
from typing import Dict, List, Optional, Union, Callable, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.python.eager import monitoring


@dataclass
class MetricThreshold:
    """Threshold configuration for metrics."""
    warning_threshold: float
    critical_threshold: float
    comparison: str  # "greater_than", "less_than"


@dataclass
class ModelMetrics:
    """Container for model performance metrics."""
    timestamp: datetime
    latency_p50: float
    latency_p95: float
    latency_p99: float
    throughput: float
    error_rate: float
    prediction_confidence: float
    memory_usage: float
    cpu_usage: float
    custom_metrics: Dict[str, float]


class ModelMonitor:
    """Comprehensive monitoring system for production TensorFlow models.
    
    Tracks performance, quality, and business metrics with real-time alerting.
    
    Example usage:
        monitor = ModelMonitor(
            model=deployed_model,
            metrics={
                "performance": ["latency_p95", "throughput", "error_rate"],
                "quality": ["prediction_confidence", "input_drift"],
                "business": ["conversion_rate", "revenue_impact"]
            },
            alerts={
                "slack": "#ml-alerts",
                "email": ["ml-team@company.com"],
                "pagerduty": "ml-on-call"
            },
            thresholds={
                "latency_p95": MetricThreshold(100, 200, "greater_than"),
                "error_rate": MetricThreshold(1.0, 5.0, "greater_than")
            }
        )
        
        # Monitor predictions
        for batch in production_stream:
            start_time = time.time()
            predictions = model(batch)
            monitor.record_prediction(batch, predictions, time.time() - start_time)
    """
    
    def __init__(self,
                 model: Any,
                 metrics: Dict[str, List[str]] = None,
                 alerts: Dict[str, Union[str, List[str]]] = None,
                 thresholds: Dict[str, MetricThreshold] = None,
                 monitoring_window_minutes: int = 5,
                 history_retention_hours: int = 24):
        """Initialize model monitor.
        
        Args:
            model: TensorFlow model to monitor
            metrics: Dictionary of metric categories and metric names
            alerts: Alert configuration (slack, email, pagerduty)
            thresholds: Metric thresholds for warnings/alerts
            monitoring_window_minutes: Window size for metric aggregation
            history_retention_hours: How long to retain metric history
        """
        self.model = model
        self.metrics_config = metrics or {
            "performance": ["latency_p95", "throughput", "error_rate"],
            "quality": ["prediction_confidence"],
            "system": ["memory_usage", "cpu_usage"]
        }
        self.alerts_config = alerts or {}
        self.thresholds = thresholds or {}
        self.monitoring_window = timedelta(minutes=monitoring_window_minutes)
        self.history_retention = timedelta(hours=history_retention_hours)
        
        # Metric storage
        self.latencies = deque(maxlen=10000)
        self.predictions_count = 0
        self.errors_count = 0
        self.prediction_confidences = deque(maxlen=1000)
        self.custom_metrics = defaultdict(lambda: deque(maxlen=1000))
        
        # Time-series metrics
        self.metrics_history = deque(maxlen=int(history_retention_hours * 12))  # 5-min intervals
        self.current_window_start = datetime.now()
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Background monitoring
        self._monitor_thread = None
        self._stop_monitoring = False
        
        # TensorFlow monitoring integration
        self._setup_tf_monitoring()
        
        # Start monitoring
        self.start_monitoring()
    
    def record_prediction(self,
                         inputs: tf.Tensor,
                         predictions: tf.Tensor,
                         latency_seconds: float,
                         ground_truth: Optional[tf.Tensor] = None,
                         custom_metrics: Dict[str, float] = None) -> None:
        """Record a prediction and its associated metrics.
        
        Args:
            inputs: Input tensor
            predictions: Model predictions
            latency_seconds: Prediction latency in seconds
            ground_truth: True labels (if available)
            custom_metrics: Additional custom metrics
        """
        timestamp = datetime.now()
        
        with self._lock:
            # Record latency
            self.latencies.append(latency_seconds * 1000)  # Convert to ms
            
            # Count predictions
            self.predictions_count += 1
            
            # Calculate prediction confidence
            if len(predictions.shape) > 1 and predictions.shape[-1] > 1:
                # Classification: use max probability
                confidence = float(tf.reduce_max(tf.nn.softmax(predictions)))
            else:
                # Regression: use inverse of prediction variance (simplified)
                confidence = 1.0 / (1.0 + float(tf.reduce_std(predictions)))
            
            self.prediction_confidences.append(confidence)
            
            # Record custom metrics
            if custom_metrics:
                for metric_name, metric_value in custom_metrics.items():
                    self.custom_metrics[metric_name].append(metric_value)
            
            # Update TensorFlow monitoring
            self._tf_latency_gauge.get_cell().set(int(latency_seconds * 1000))
            self._tf_predictions_counter.get_cell().increase_by(1)
            self._tf_confidence_gauge.get_cell().set(str(f"{confidence:.3f}"))
    
    def record_error(self, error_type: str = "prediction_error") -> None:
        """Record a prediction error.
        
        Args:
            error_type: Type of error that occurred
        """
        with self._lock:
            self.errors_count += 1
            self._tf_errors_counter.get_cell(error_type).increase_by(1)
    
    def get_current_metrics(self) -> ModelMetrics:
        """Get current aggregated metrics."""
        with self._lock:
            # Calculate latency percentiles
            if self.latencies:
                latencies_array = np.array(list(self.latencies))
                latency_p50 = float(np.percentile(latencies_array, 50))
                latency_p95 = float(np.percentile(latencies_array, 95))
                latency_p99 = float(np.percentile(latencies_array, 99))
            else:
                latency_p50 = latency_p95 = latency_p99 = 0.0
            
            # Calculate throughput (predictions per second)
            window_duration = (datetime.now() - self.current_window_start).total_seconds()
            throughput = self.predictions_count / max(window_duration, 1.0)
            
            # Calculate error rate
            error_rate = (self.errors_count / max(self.predictions_count, 1)) * 100
            
            # Calculate average prediction confidence
            if self.prediction_confidences:
                avg_confidence = float(np.mean(list(self.prediction_confidences)))
            else:
                avg_confidence = 0.0
            
            # Get system metrics
            memory_usage = self._get_memory_usage()
            cpu_usage = self._get_cpu_usage()
            
            # Aggregate custom metrics
            custom_aggregated = {}
            for metric_name, values in self.custom_metrics.items():
                if values:
                    custom_aggregated[metric_name] = float(np.mean(list(values)))
        
        return ModelMetrics(
            timestamp=datetime.now(),
            latency_p50=latency_p50,
            latency_p95=latency_p95,
            latency_p99=latency_p99,
            throughput=throughput,
            error_rate=error_rate,
            prediction_confidence=avg_confidence,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            custom_metrics=custom_aggregated
        )
    
    def get_metrics_history(self, hours: int = 1) -> List[ModelMetrics]:
        """Get historical metrics for the specified time period.
        
        Args:
            hours: Number of hours of history to return
            
        Returns:
            List of historical metrics
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            metrics for metrics in self.metrics_history
            if metrics.timestamp >= cutoff_time
        ]
    
    def check_thresholds(self) -> Dict[str, str]:
        """Check current metrics against configured thresholds.
        
        Returns:
            Dictionary of metric_name -> alert_level for metrics exceeding thresholds
        """
        current_metrics = self.get_current_metrics()
        alerts = {}
        
        for metric_name, threshold in self.thresholds.items():
            # Get metric value
            if hasattr(current_metrics, metric_name):
                metric_value = getattr(current_metrics, metric_name)
            elif metric_name in current_metrics.custom_metrics:
                metric_value = current_metrics.custom_metrics[metric_name]
            else:
                continue
            
            # Check thresholds
            if threshold.comparison == "greater_than":
                if metric_value >= threshold.critical_threshold:
                    alerts[metric_name] = "critical"
                elif metric_value >= threshold.warning_threshold:
                    alerts[metric_name] = "warning"
            elif threshold.comparison == "less_than":
                if metric_value <= threshold.critical_threshold:
                    alerts[metric_name] = "critical"
                elif metric_value <= threshold.warning_threshold:
                    alerts[metric_name] = "warning"
        
        return alerts
    
    def start_monitoring(self) -> None:
        """Start background monitoring."""
        if self._monitor_thread and self._monitor_thread.is_alive():
            return
        
        self._stop_monitoring = False
        self._monitor_thread = threading.Thread(target=self._monitoring_loop)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
        
        tf.get_logger().info("Started model monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self._stop_monitoring = True
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        
        tf.get_logger().info("Stopped model monitoring")
    
    def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while not self._stop_monitoring:
            try:
                # Check if monitoring window is complete
                if datetime.now() - self.current_window_start >= self.monitoring_window:
                    # Get current metrics and store in history
                    current_metrics = self.get_current_metrics()
                    self.metrics_history.append(current_metrics)
                    
                    # Check thresholds and send alerts
                    alerts = self.check_thresholds()
                    if alerts:
                        self._send_alerts(alerts, current_metrics)
                    
                    # Reset window
                    self._reset_current_window()
                    
                    # Clean old history
                    self._cleanup_old_metrics()
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                tf.get_logger().error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _reset_current_window(self) -> None:
        """Reset current monitoring window."""
        with self._lock:
            self.current_window_start = datetime.now()
            # Keep some recent data for smoothing, but reset counters
            self.predictions_count = 0
            self.errors_count = 0
    
    def _cleanup_old_metrics(self) -> None:
        """Remove old metrics beyond retention period."""
        cutoff_time = datetime.now() - self.history_retention
        
        while (self.metrics_history and 
               self.metrics_history[0].timestamp < cutoff_time):
            self.metrics_history.popleft()
    
    def _send_alerts(self, alerts: Dict[str, str], metrics: ModelMetrics) -> None:
        """Send alerts for threshold violations."""
        for metric_name, alert_level in alerts.items():
            message = self._format_alert_message(metric_name, alert_level, metrics)
            
            tf.get_logger().warning(f"Model monitoring alert: {message}")
            
            # Send to configured alert channels
            if "slack" in self.alerts_config:
                self._send_slack_alert(message)
            
            if "email" in self.alerts_config:
                self._send_email_alert(message)
            
            if "pagerduty" in self.alerts_config:
                self._send_pagerduty_alert(message, alert_level)
    
    def _format_alert_message(self, metric_name: str, alert_level: str, metrics: ModelMetrics) -> str:
        """Format alert message."""
        if hasattr(metrics, metric_name):
            metric_value = getattr(metrics, metric_name)
        else:
            metric_value = metrics.custom_metrics.get(metric_name, "N/A")
        
        threshold = self.thresholds[metric_name]
        threshold_value = (threshold.critical_threshold if alert_level == "critical" 
                         else threshold.warning_threshold)
        
        return (
            f"🚨 {alert_level.upper()} Alert: {metric_name} = {metric_value:.2f} "
            f"(threshold: {threshold_value})\n"
            f"Model: {self.model.__class__.__name__}\n"
            f"Time: {metrics.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
        )
    
    def _send_slack_alert(self, message: str) -> None:
        """Send alert to Slack (placeholder implementation)."""
        # In production, would integrate with Slack API
        tf.get_logger().info(f"Would send Slack alert: {message}")
    
    def _send_email_alert(self, message: str) -> None:
        """Send alert via email (placeholder implementation)."""
        # In production, would integrate with email service
        tf.get_logger().info(f"Would send email alert: {message}")
    
    def _send_pagerduty_alert(self, message: str, severity: str) -> None:
        """Send alert to PagerDuty (placeholder implementation)."""
        # In production, would integrate with PagerDuty API
        tf.get_logger().info(f"Would send PagerDuty alert ({severity}): {message}")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage (placeholder)."""
        # In production, would get actual memory usage
        return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage (placeholder)."""
        # In production, would get actual CPU usage
        return 0.0
    
    def _setup_tf_monitoring(self) -> None:
        """Setup TensorFlow monitoring metrics."""
        self._tf_latency_gauge = monitoring.IntGauge(
            'model_latency_ms',
            'Model prediction latency in milliseconds'
        )
        
        self._tf_predictions_counter = monitoring.Counter(
            'model_predictions_total',
            'Total number of predictions made'
        )
        
        self._tf_errors_counter = monitoring.Counter(
            'model_errors_total',
            'Total number of prediction errors',
            'error_type'
        )
        
        self._tf_confidence_gauge = monitoring.StringGauge(
            'model_prediction_confidence',
            'Average prediction confidence'
        )
        
        self._tf_throughput_gauge = monitoring.IntGauge(
            'model_throughput_rps',
            'Model throughput in requests per second'
        ) 