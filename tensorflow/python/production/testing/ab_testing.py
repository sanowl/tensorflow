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
"""A/B Testing Framework for TensorFlow Production Models."""

import time
import threading
import numpy as np
from typing import Dict, List, Optional, Union, Callable, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.python.eager import monitoring


@dataclass
class ABTestResult:
    """Results from an A/B test."""
    model_a_metric: float
    model_b_metric: float
    statistical_significance: float
    confidence_interval: tuple
    sample_size_a: int
    sample_size_b: int
    test_duration: timedelta
    winner: Optional[str]
    recommendation: str


@dataclass
class ABTestConfig:
    """Configuration for A/B test."""
    name: str
    model_a: Any  # TensorFlow model
    model_b: Any  # TensorFlow model
    traffic_split: float = 0.1  # Percentage for model_b (0.1 = 10%)
    success_metrics: List[str] = None
    duration_days: int = 7
    min_sample_size: int = 1000
    significance_threshold: float = 0.95
    auto_promote: bool = False


class ABTest:
    """A/B Testing framework for comparing model performance in production.
    
    Example usage:
        ab_test = ABTest(
            name="recommendation_model_v2",
            model_a=current_model,
            model_b=new_model,
            traffic_split=0.1,
            success_metrics=["click_rate", "conversion_rate"],
            duration_days=7
        )
        
        ab_test.start()
        
        # Check results
        if ab_test.is_significant():
            winner = ab_test.get_winner()
            ab_test.promote_winner()
    """
    
    def __init__(self, 
                 name: str,
                 model_a: Any,
                 model_b: Any,
                 traffic_split: float = 0.1,
                 success_metrics: List[str] = None,
                 duration_days: int = 7,
                 min_sample_size: int = 1000,
                 significance_threshold: float = 0.95,
                 auto_promote: bool = False):
        """Initialize A/B test.
        
        Args:
            name: Unique name for the test
            model_a: Control model (baseline)
            model_b: Treatment model (new version)
            traffic_split: Fraction of traffic for model_b (0.0-1.0)
            success_metrics: List of metrics to track
            duration_days: Maximum test duration
            min_sample_size: Minimum samples before statistical testing
            significance_threshold: Required confidence level (0.95 = 95%)
            auto_promote: Whether to automatically promote winner
        """
        self.config = ABTestConfig(
            name=name,
            model_a=model_a,
            model_b=model_b,
            traffic_split=traffic_split,
            success_metrics=success_metrics or ["accuracy"],
            duration_days=duration_days,
            min_sample_size=min_sample_size,
            significance_threshold=significance_threshold,
            auto_promote=auto_promote
        )
        
        self.start_time = None
        self.is_running = False
        self._results_a = {metric: [] for metric in self.config.success_metrics}
        self._results_b = {metric: [] for metric in self.config.success_metrics}
        self._sample_count_a = 0
        self._sample_count_b = 0
        self._lock = threading.Lock()
        
        # Monitoring metrics
        self._ab_test_counter = monitoring.Counter(
            f'ab_test_requests', 
            'Number of requests processed by A/B test',
            'test_name', 'model_variant'
        )
        
        self._ab_test_metric_gauge = monitoring.StringGauge(
            f'ab_test_metrics',
            'Current A/B test metric values',
            'test_name', 'model_variant', 'metric_name'
        )
    
    def start(self) -> None:
        """Start the A/B test."""
        if self.is_running:
            raise ValueError(f"A/B test '{self.config.name}' is already running")
        
        self.start_time = datetime.now()
        self.is_running = True
        
        tf.get_logger().info(
            f"Started A/B test '{self.config.name}' with {self.config.traffic_split*100:.1f}% "
            f"traffic to model B for {self.config.duration_days} days"
        )
    
    def stop(self) -> None:
        """Stop the A/B test."""
        self.is_running = False
        tf.get_logger().info(f"Stopped A/B test '{self.config.name}'")
    
    def predict(self, inputs: tf.Tensor) -> tf.Tensor:
        """Route prediction through A/B test.
        
        Args:
            inputs: Input tensor for prediction
            
        Returns:
            Prediction from selected model
        """
        if not self.is_running:
            return self.config.model_a(inputs)
        
        # Route traffic based on hash of input (for consistency)
        input_hash = hash(str(inputs.numpy())) % 100
        use_model_b = input_hash < (self.config.traffic_split * 100)
        
        if use_model_b:
            prediction = self.config.model_b(inputs)
            self._ab_test_counter.get_cell(self.config.name, 'model_b').increase_by(1)
            return prediction
        else:
            prediction = self.config.model_a(inputs)
            self._ab_test_counter.get_cell(self.config.name, 'model_a').increase_by(1)
            return prediction
    
    def record_outcome(self, 
                      inputs: tf.Tensor, 
                      outcome_metrics: Dict[str, float]) -> None:
        """Record outcome metrics for the prediction.
        
        Args:
            inputs: Original input tensor (to determine which model was used)
            outcome_metrics: Dictionary of metric_name -> value
        """
        if not self.is_running:
            return
        
        # Determine which model was used
        input_hash = hash(str(inputs.numpy())) % 100
        use_model_b = input_hash < (self.config.traffic_split * 100)
        
        with self._lock:
            if use_model_b:
                self._sample_count_b += 1
                for metric, value in outcome_metrics.items():
                    if metric in self._results_b:
                        self._results_b[metric].append(value)
                        self._ab_test_metric_gauge.get_cell(
                            self.config.name, 'model_b', metric
                        ).set(str(np.mean(self._results_b[metric])))
            else:
                self._sample_count_a += 1
                for metric, value in outcome_metrics.items():
                    if metric in self._results_a:
                        self._results_a[metric].append(value)
                        self._ab_test_metric_gauge.get_cell(
                            self.config.name, 'model_a', metric
                        ).set(str(np.mean(self._results_a[metric])))
    
    def get_results(self) -> ABTestResult:
        """Get current A/B test results with statistical analysis."""
        if not self._results_a or not self._results_b:
            return ABTestResult(
                model_a_metric=0.0,
                model_b_metric=0.0,
                statistical_significance=0.0,
                confidence_interval=(0.0, 0.0),
                sample_size_a=self._sample_count_a,
                sample_size_b=self._sample_count_b,
                test_duration=datetime.now() - self.start_time if self.start_time else timedelta(0),
                winner=None,
                recommendation="Insufficient data"
            )
        
        # Use primary metric (first in list) for winner determination
        primary_metric = self.config.success_metrics[0]
        
        if primary_metric not in self._results_a or primary_metric not in self._results_b:
            return ABTestResult(
                model_a_metric=0.0,
                model_b_metric=0.0,
                statistical_significance=0.0,
                confidence_interval=(0.0, 0.0),
                sample_size_a=self._sample_count_a,
                sample_size_b=self._sample_count_b,
                test_duration=datetime.now() - self.start_time if self.start_time else timedelta(0),
                winner=None,
                recommendation="Primary metric not available"
            )
        
        metric_a = np.mean(self._results_a[primary_metric])
        metric_b = np.mean(self._results_b[primary_metric])
        
        # Perform t-test for statistical significance
        significance, ci = self._calculate_significance(
            self._results_a[primary_metric],
            self._results_b[primary_metric]
        )
        
        # Determine winner and recommendation
        winner = None
        recommendation = "Continue test"
        
        if (self._sample_count_a >= self.config.min_sample_size and 
            self._sample_count_b >= self.config.min_sample_size):
            
            if significance >= self.config.significance_threshold:
                winner = "model_b" if metric_b > metric_a else "model_a"
                recommendation = f"Significant result: {winner} wins"
            elif self._is_test_expired():
                winner = "model_b" if metric_b > metric_a else "model_a"
                recommendation = f"Test expired: {winner} performs better"
        
        return ABTestResult(
            model_a_metric=float(metric_a),
            model_b_metric=float(metric_b),
            statistical_significance=significance,
            confidence_interval=ci,
            sample_size_a=self._sample_count_a,
            sample_size_b=self._sample_count_b,
            test_duration=datetime.now() - self.start_time if self.start_time else timedelta(0),
            winner=winner,
            recommendation=recommendation
        )
    
    def is_significant(self) -> bool:
        """Check if results are statistically significant."""
        results = self.get_results()
        return results.statistical_significance >= self.config.significance_threshold
    
    def get_winner(self) -> Optional[str]:
        """Get the winning model if test is conclusive."""
        return self.get_results().winner
    
    def promote_winner(self) -> None:
        """Promote the winning model to receive 100% traffic."""
        winner = self.get_winner()
        if not winner:
            raise ValueError("No clear winner yet - test inconclusive")
        
        if winner == "model_b":
            self.config.traffic_split = 1.0
            tf.get_logger().info(f"Promoted model_b to 100% traffic in test '{self.config.name}'")
        else:
            self.config.traffic_split = 0.0
            tf.get_logger().info(f"Keeping model_a at 100% traffic in test '{self.config.name}'")
        
        self.stop()
    
    def _calculate_significance(self, samples_a: List[float], samples_b: List[float]) -> tuple:
        """Calculate statistical significance using Welch's t-test."""
        if len(samples_a) < 2 or len(samples_b) < 2:
            return 0.0, (0.0, 0.0)
        
        mean_a = np.mean(samples_a)
        mean_b = np.mean(samples_b)
        var_a = np.var(samples_a, ddof=1)
        var_b = np.var(samples_b, ddof=1)
        n_a = len(samples_a)
        n_b = len(samples_b)
        
        # Welch's t-test
        pooled_se = np.sqrt(var_a/n_a + var_b/n_b)
        
        if pooled_se == 0:
            return 0.0, (0.0, 0.0)
        
        t_stat = (mean_b - mean_a) / pooled_se
        
        # Degrees of freedom (Welch-Satterthwaite equation)
        df = ((var_a/n_a + var_b/n_b)**2 / 
              ((var_a/n_a)**2/(n_a-1) + (var_b/n_b)**2/(n_b-1)))
        
        # Convert t-statistic to p-value (simplified)
        # In production, would use scipy.stats.t.cdf
        p_value = 2 * (1 - self._t_cdf(abs(t_stat), df))
        confidence = 1 - p_value
        
        # Confidence interval for difference
        margin_error = 1.96 * pooled_se  # 95% CI
        ci_lower = (mean_b - mean_a) - margin_error
        ci_upper = (mean_b - mean_a) + margin_error
        
        return float(confidence), (float(ci_lower), float(ci_upper))
    
    def _t_cdf(self, t: float, df: float) -> float:
        """Simplified t-distribution CDF approximation."""
        # Simple approximation - in production use scipy.stats
        if df > 30:
            # Use normal approximation for large df
            return 0.5 * (1 + np.tanh(t / np.sqrt(2)))
        else:
            # Very rough approximation
            return 0.5 + 0.3 * np.tanh(t)
    
    def _is_test_expired(self) -> bool:
        """Check if test has exceeded maximum duration."""
        if not self.start_time:
            return False
        return datetime.now() - self.start_time > timedelta(days=self.config.duration_days) 