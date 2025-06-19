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
"""Canary Deployment System for TensorFlow Production Models."""

import time
import threading
import numpy as np
from typing import Dict, List, Optional, Union, Callable, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import tensorflow as tf
from tensorflow.python.eager import monitoring


class CanaryStatus(Enum):
    """Status of canary deployment."""
    INITIALIZING = "initializing"
    RAMPING_UP = "ramping_up"
    MONITORING = "monitoring"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"


@dataclass
class CanaryConfig:
    """Configuration for canary deployment."""
    name: str
    new_model: Any
    baseline_model: Any
    initial_traffic_percentage: float = 5.0
    final_traffic_percentage: float = 100.0
    ramp_up_duration_minutes: int = 30
    monitoring_duration_minutes: int = 60
    success_criteria: Dict[str, str] = None  # e.g., {"error_rate": "< 1%", "latency_p95": "< 100ms"}
    failure_criteria: Dict[str, str] = None  # e.g., {"error_rate": "> 5%"}
    auto_rollback: bool = True
    ramp_up_schedule: str = "linear"  # "linear", "exponential", "manual"


@dataclass
class CanaryMetrics:
    """Current metrics for canary deployment."""
    traffic_percentage: float
    requests_new: int
    requests_baseline: int
    error_rate_new: float
    error_rate_baseline: float
    latency_p95_new: float
    latency_p95_baseline: float
    success_criteria_met: bool
    failure_criteria_met: bool


class CanaryDeployment:
    """Canary deployment system for gradual model rollouts.
    
    Example usage:
        canary = CanaryDeployment(
            name="recommendation_v3",
            new_model=new_model,
            baseline_model=current_model,
            initial_traffic_percentage=5,
            ramp_up_duration_minutes=30,
            success_criteria={
                "error_rate": "< 1%",
                "latency_p95": "< 100ms"
            }
        )
        
        canary.start()
        
        # Monitor progress
        while canary.is_active():
            status = canary.get_status()
            metrics = canary.get_metrics()
            time.sleep(60)
        
        if canary.get_status() == CanaryStatus.SUCCEEDED:
            print("Canary deployment successful!")
    """
    
    def __init__(self,
                 name: str,
                 new_model: Any,
                 baseline_model: Any,
                 initial_traffic_percentage: float = 5.0,
                 final_traffic_percentage: float = 100.0,
                 ramp_up_duration_minutes: int = 30,
                 monitoring_duration_minutes: int = 60,
                 success_criteria: Dict[str, str] = None,
                 failure_criteria: Dict[str, str] = None,
                 auto_rollback: bool = True,
                 ramp_up_schedule: str = "linear"):
        """Initialize canary deployment."""
        self.config = CanaryConfig(
            name=name,
            new_model=new_model,
            baseline_model=baseline_model,
            initial_traffic_percentage=initial_traffic_percentage,
            final_traffic_percentage=final_traffic_percentage,
            ramp_up_duration_minutes=ramp_up_duration_minutes,
            monitoring_duration_minutes=monitoring_duration_minutes,
            success_criteria=success_criteria or {"error_rate": "< 2%"},
            failure_criteria=failure_criteria or {"error_rate": "> 10%"},
            auto_rollback=auto_rollback,
            ramp_up_schedule=ramp_up_schedule
        )
        
        self.status = CanaryStatus.INITIALIZING
        self.current_traffic_percentage = 0.0
        self.start_time = None
        
        # Metrics tracking
        self._requests_new = 0
        self._requests_baseline = 0
        self._errors_new = 0
        self._errors_baseline = 0
        self._latencies_new = []
        self._latencies_baseline = []
        self._lock = threading.Lock()
        
        # Background monitoring
        self._monitor_thread = None
        self._stop_monitoring = False
    
    def start(self) -> None:
        """Start the canary deployment."""
        if self.status != CanaryStatus.INITIALIZING:
            raise ValueError(f"Canary '{self.config.name}' is already started")
        
        self.start_time = datetime.now()
        self.current_traffic_percentage = self.config.initial_traffic_percentage
        self.status = CanaryStatus.RAMPING_UP
        
        # Start background monitoring thread
        self._stop_monitoring = False
        self._monitor_thread = threading.Thread(target=self._monitor_deployment)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
        
        tf.get_logger().info(
            f"Started canary deployment '{self.config.name}' with "
            f"{self.current_traffic_percentage}% initial traffic"
        )
    
    def predict(self, inputs: tf.Tensor) -> tf.Tensor:
        """Route prediction through canary deployment."""
        start_time = time.time()
        
        # Route traffic based on current percentage
        input_hash = hash(str(inputs.numpy())) % 100
        use_new_model = input_hash < self.current_traffic_percentage
        
        try:
            if use_new_model and self.status in [CanaryStatus.RAMPING_UP, CanaryStatus.MONITORING]:
                prediction = self.config.new_model(inputs)
                self._record_request('new', start_time, success=True)
                return prediction
            else:
                prediction = self.config.baseline_model(inputs)
                self._record_request('baseline', start_time, success=True)
                return prediction
        except Exception as e:
            if use_new_model and self.status in [CanaryStatus.RAMPING_UP, CanaryStatus.MONITORING]:
                self._record_request('new', start_time, success=False)
            else:
                self._record_request('baseline', start_time, success=False)
            raise e
    
    def get_status(self) -> CanaryStatus:
        """Get current canary deployment status."""
        return self.status
    
    def get_metrics(self) -> CanaryMetrics:
        """Get current canary deployment metrics."""
        with self._lock:
            error_rate_new = (self._errors_new / max(self._requests_new, 1)) * 100
            error_rate_baseline = (self._errors_baseline / max(self._requests_baseline, 1)) * 100
            
            latency_p95_new = np.percentile(self._latencies_new, 95) if self._latencies_new else 0
            latency_p95_baseline = np.percentile(self._latencies_baseline, 95) if self._latencies_baseline else 0
            
            success_criteria_met = self._check_success_criteria(error_rate_new, latency_p95_new)
            failure_criteria_met = self._check_failure_criteria(error_rate_new, latency_p95_new)
            
            return CanaryMetrics(
                traffic_percentage=self.current_traffic_percentage,
                requests_new=self._requests_new,
                requests_baseline=self._requests_baseline,
                error_rate_new=error_rate_new,
                error_rate_baseline=error_rate_baseline,
                latency_p95_new=latency_p95_new,
                latency_p95_baseline=latency_p95_baseline,
                success_criteria_met=success_criteria_met,
                failure_criteria_met=failure_criteria_met
            )
    
    def is_active(self) -> bool:
        """Check if canary deployment is currently active."""
        return self.status in [CanaryStatus.RAMPING_UP, CanaryStatus.MONITORING]
    
    def _monitor_deployment(self) -> None:
        """Background monitoring thread."""
        ramp_start_time = self.start_time
        monitoring_start_time = None
        
        while not self._stop_monitoring:
            time.sleep(10)  # Check every 10 seconds
            
            try:
                if self.status == CanaryStatus.RAMPING_UP:
                    elapsed_minutes = (datetime.now() - ramp_start_time).total_seconds() / 60
                    
                    if elapsed_minutes >= self.config.ramp_up_duration_minutes:
                        # Ramp up complete
                        self.current_traffic_percentage = self.config.final_traffic_percentage
                        self.status = CanaryStatus.MONITORING
                        monitoring_start_time = datetime.now()
                        
                        tf.get_logger().info(
                            f"Canary '{self.config.name}' completed ramp-up, entering monitoring"
                        )
                    else:
                        # Update traffic percentage
                        progress = elapsed_minutes / self.config.ramp_up_duration_minutes
                        
                        if self.config.ramp_up_schedule == "linear":
                            target_percentage = (
                                self.config.initial_traffic_percentage + 
                                progress * (self.config.final_traffic_percentage - self.config.initial_traffic_percentage)
                            )
                        elif self.config.ramp_up_schedule == "exponential":
                            exp_progress = progress ** 2
                            target_percentage = (
                                self.config.initial_traffic_percentage + 
                                exp_progress * (self.config.final_traffic_percentage - self.config.initial_traffic_percentage)
                            )
                        else:
                            target_percentage = self.current_traffic_percentage
                        
                        self.current_traffic_percentage = target_percentage
                    
                    # Check failure criteria during ramp up
                    metrics = self.get_metrics()
                    if metrics.failure_criteria_met and self.config.auto_rollback:
                        tf.get_logger().warning(f"Failure criteria met for canary '{self.config.name}', rolling back")
                        self._rollback()
                        break
                
                elif self.status == CanaryStatus.MONITORING and monitoring_start_time:
                    elapsed_minutes = (datetime.now() - monitoring_start_time).total_seconds() / 60
                    
                    metrics = self.get_metrics()
                    
                    # Check failure criteria
                    if metrics.failure_criteria_met and self.config.auto_rollback:
                        tf.get_logger().warning(f"Failure criteria met for canary '{self.config.name}', rolling back")
                        self._rollback()
                        break
                    
                    # Check if monitoring period is complete
                    if elapsed_minutes >= self.config.monitoring_duration_minutes:
                        if metrics.success_criteria_met:
                            self.status = CanaryStatus.SUCCEEDED
                            tf.get_logger().info(f"Canary deployment '{self.config.name}' succeeded!")
                        else:
                            tf.get_logger().warning(f"Canary deployment '{self.config.name}' monitoring complete but success criteria not met")
                            if self.config.auto_rollback:
                                self._rollback()
                            else:
                                self.status = CanaryStatus.FAILED
                        break
                        
            except Exception as e:
                tf.get_logger().error(f"Error in canary monitoring: {e}")
                if self.config.auto_rollback:
                    self._rollback()
                break
    
    def _rollback(self) -> None:
        """Rollback the canary deployment."""
        self.status = CanaryStatus.ROLLING_BACK
        self.current_traffic_percentage = 0.0
        tf.get_logger().warning(f"Rolling back canary deployment '{self.config.name}'")
        self.status = CanaryStatus.FAILED
    
    def _record_request(self, model_type: str, start_time: float, success: bool) -> None:
        """Record request metrics."""
        latency_ms = (time.time() - start_time) * 1000
        
        with self._lock:
            if model_type == 'new':
                self._requests_new += 1
                self._latencies_new.append(latency_ms)
                if not success:
                    self._errors_new += 1
            else:
                self._requests_baseline += 1
                self._latencies_baseline.append(latency_ms)
                if not success:
                    self._errors_baseline += 1
            
            # Keep only recent latencies (last 1000 requests)
            if len(self._latencies_new) > 1000:
                self._latencies_new = self._latencies_new[-1000:]
            if len(self._latencies_baseline) > 1000:
                self._latencies_baseline = self._latencies_baseline[-1000:]
    
    def _check_success_criteria(self, error_rate: float, latency_p95: float) -> bool:
        """Check if success criteria are met."""
        for criterion, threshold in self.config.success_criteria.items():
            if not self._evaluate_criterion(criterion, threshold, error_rate, latency_p95):
                return False
        return True
    
    def _check_failure_criteria(self, error_rate: float, latency_p95: float) -> bool:
        """Check if failure criteria are met."""
        for criterion, threshold in self.config.failure_criteria.items():
            if self._evaluate_criterion(criterion, threshold, error_rate, latency_p95):
                return True
        return False
    
    def _evaluate_criterion(self, criterion: str, threshold: str, error_rate: float, latency_p95: float) -> bool:
        """Evaluate a single criterion."""
        if criterion == "error_rate":
            current_value = error_rate
        elif criterion == "latency_p95":
            current_value = latency_p95
        else:
            return False
        
        threshold = threshold.strip()
        if threshold.startswith("< "):
            target_value = float(threshold[2:].replace('%', '').replace('ms', ''))
            return current_value < target_value
        elif threshold.startswith("> "):
            target_value = float(threshold[2:].replace('%', '').replace('ms', ''))
            return current_value > target_value
        
        return False 