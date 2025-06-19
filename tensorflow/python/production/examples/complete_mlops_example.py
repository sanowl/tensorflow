#!/usr/bin/env python3
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
"""Complete MLOps Example using TensorFlow Production Toolkit.

This example demonstrates a full production ML pipeline including:
- Model training and versioning
- A/B testing between model versions
- Canary deployment with gradual rollout
- Real-time monitoring and drift detection
- Automated rollback on failure

Use case: E-commerce recommendation system
"""

import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Dict, List, Any

# Import TensorFlow Production toolkit
import tensorflow.python.production as tf_prod


class RecommendationModel(keras.Model):
    """Simple recommendation model for demonstration."""
    
    def __init__(self, embedding_dim: int = 64, num_items: int = 1000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_items = num_items
        
        # User and item embeddings
        self.user_embedding = keras.layers.Embedding(1000, embedding_dim)
        self.item_embedding = keras.layers.Embedding(num_items, embedding_dim)
        
        # Dense layers
        self.dense1 = keras.layers.Dense(128, activation='relu')
        self.dense2 = keras.layers.Dense(64, activation='relu')
        self.output_layer = keras.layers.Dense(1, activation='sigmoid')
    
    def call(self, inputs):
        user_id, item_id = inputs[:, 0], inputs[:, 1]
        
        user_emb = self.user_embedding(user_id)
        item_emb = self.item_embedding(item_id)
        
        # Concatenate embeddings
        concat = tf.concat([user_emb, item_emb], axis=-1)
        
        # Pass through dense layers
        x = self.dense1(concat)
        x = self.dense2(x)
        
        return self.output_layer(x)


def create_sample_data(num_samples: int = 1000) -> Dict[str, np.ndarray]:
    """Create sample training data."""
    # Generate random user-item interactions
    user_ids = np.random.randint(0, 1000, num_samples)
    item_ids = np.random.randint(0, 1000, num_samples)
    
    # Create features (user_id, item_id)
    X = np.column_stack([user_ids, item_ids])
    
    # Generate synthetic ratings (0-1)
    # Higher ratings for user-item pairs with similar IDs (simplified)
    similarity = 1.0 - (np.abs(user_ids - item_ids) / 1000.0)
    y = (similarity + np.random.normal(0, 0.1, num_samples)).clip(0, 1)
    
    return {"features": X, "labels": y}


def train_models() -> tuple:
    """Train two model versions for A/B testing."""
    print("🏋️ Training models...")
    
    # Create training data
    train_data = create_sample_data(5000)
    val_data = create_sample_data(1000)
    
    # Model V1 (smaller)
    model_v1 = RecommendationModel(embedding_dim=32)
    model_v1.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    model_v1.fit(
        train_data["features"],
        train_data["labels"],
        validation_data=(val_data["features"], val_data["labels"]),
        epochs=5,
        batch_size=64,
        verbose=0
    )
    
    # Model V2 (larger, hopefully better)
    model_v2 = RecommendationModel(embedding_dim=64)
    model_v2.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    model_v2.fit(
        train_data["features"],
        train_data["labels"],
        validation_data=(val_data["features"], val_data["labels"]),
        epochs=5,
        batch_size=64,
        verbose=0
    )
    
    print("✅ Models trained successfully!")
    return model_v1, model_v2, train_data


def setup_monitoring_and_drift_detection(model, reference_data) -> tuple:
    """Set up monitoring and drift detection."""
    print("📊 Setting up monitoring and drift detection...")
    
    # Setup model monitoring
    monitor = tf_prod.ModelMonitor(
        model=model,
        metrics={
            "performance": ["latency_p95", "throughput", "error_rate"],
            "quality": ["prediction_confidence"],
            "business": ["click_rate", "conversion_rate"]
        },
        thresholds={
            "latency_p95": tf_prod.monitoring.MetricThreshold(100, 200, "greater_than"),
            "error_rate": tf_prod.monitoring.MetricThreshold(1.0, 5.0, "greater_than"),
            "conversion_rate": tf_prod.monitoring.MetricThreshold(2.0, 1.0, "less_than")
        },
        alerts={
            "slack": "#ml-alerts",
            "email": ["ml-team@company.com"]
        }
    )
    
    # Setup drift detection
    drift_detector = tf_prod.DriftDetector(
        reference_data=reference_data["features"],
        alert_threshold=0.1,
        warning_threshold=0.05,
        detection_window="30s",  # Short window for demo
        methods=["ks_test", "psi"]
    )
    
    print("✅ Monitoring and drift detection configured!")
    return monitor, drift_detector


def run_ab_test(model_v1, model_v2) -> str:
    """Run A/B test between two model versions."""
    print("🧪 Starting A/B test...")
    
    # Setup A/B test
    ab_test = tf_prod.ABTest(
        name="recommendation_model_v2",
        model_a=model_v1,
        model_b=model_v2,
        traffic_split=0.2,  # 20% traffic to new model
        success_metrics=["click_rate", "conversion_rate"],
        duration_days=1,  # Short for demo
        min_sample_size=100,
        significance_threshold=0.95
    )
    
    ab_test.start()
    
    # Simulate production traffic with outcomes
    print("📈 Simulating production traffic...")
    
    for i in range(200):  # Simulate 200 requests
        # Generate random request
        user_id = np.random.randint(0, 1000)
        item_id = np.random.randint(0, 1000)
        inputs = tf.constant([[user_id, item_id]], dtype=tf.float32)
        
        # Get prediction through A/B test
        prediction = ab_test.predict(inputs)
        
        # Simulate user interaction outcome
        # Higher predictions generally lead to better outcomes
        predicted_score = float(prediction[0][0])
        
        # Simulate click and conversion rates based on prediction quality
        clicked = np.random.random() < (predicted_score * 0.8 + 0.1)  # Base 10% + prediction boost
        converted = clicked and (np.random.random() < (predicted_score * 0.5 + 0.05))  # Base 5% + prediction boost
        
        # Record outcomes
        outcomes = {
            "click_rate": 1.0 if clicked else 0.0,
            "conversion_rate": 1.0 if converted else 0.0
        }
        
        ab_test.record_outcome(inputs, outcomes)
        
        # Progress indicator
        if (i + 1) % 50 == 0:
            results = ab_test.get_results()
            print(f"  Progress: {i+1}/200 requests | "
                  f"Model A: {results.model_a_metric:.3f} | "
                  f"Model B: {results.model_b_metric:.3f} | "
                  f"Significance: {results.statistical_significance:.3f}")
    
    # Get final results
    final_results = ab_test.get_results()
    
    print(f"\n🎯 A/B Test Results:")
    print(f"   Model A (v1) performance: {final_results.model_a_metric:.3f}")
    print(f"   Model B (v2) performance: {final_results.model_b_metric:.3f}")
    print(f"   Statistical significance: {final_results.statistical_significance:.3f}")
    print(f"   Winner: {final_results.winner}")
    print(f"   Recommendation: {final_results.recommendation}")
    
    if ab_test.is_significant():
        winner_model = model_v2 if final_results.winner == "model_b" else model_v1
        print(f"✅ A/B test conclusive - promoting {final_results.winner}")
        ab_test.promote_winner()
        return final_results.winner, winner_model
    else:
        print("⚠️  A/B test inconclusive - keeping current model")
        return "model_a", model_v1


def run_canary_deployment(winning_model, baseline_model):
    """Run canary deployment of winning model."""
    print("🐦 Starting canary deployment...")
    
    # Setup canary deployment
    canary = tf_prod.CanaryDeployment(
        name="recommendation_v2_canary",
        new_model=winning_model,
        baseline_model=baseline_model,
        initial_traffic_percentage=5,
        final_traffic_percentage=100,
        ramp_up_duration_minutes=2,  # Short for demo
        monitoring_duration_minutes=1,  # Short for demo
        success_criteria={
            "error_rate": "< 2%",
            "latency_p95": "< 100ms"
        },
        failure_criteria={
            "error_rate": "> 10%"
        },
        auto_rollback=True
    )
    
    canary.start()
    
    # Monitor canary deployment
    print("📊 Monitoring canary deployment...")
    
    start_time = time.time()
    while canary.is_active() and (time.time() - start_time) < 300:  # Max 5 minutes
        status = canary.get_status()
        metrics = canary.get_metrics()
        
        print(f"  Status: {status.value} | "
              f"Traffic: {metrics.traffic_percentage:.1f}% | "
              f"Requests (new/baseline): {metrics.requests_new}/{metrics.requests_baseline} | "
              f"Error rate: {metrics.error_rate_new:.2f}%")
        
        # Simulate some traffic
        for _ in range(10):
            user_id = np.random.randint(0, 1000)
            item_id = np.random.randint(0, 1000)
            inputs = tf.constant([[user_id, item_id]], dtype=tf.float32)
            
            try:
                prediction = canary.predict(inputs)
            except Exception as e:
                print(f"  ❌ Prediction error: {e}")
        
        time.sleep(10)  # Check every 10 seconds
    
    final_status = canary.get_status()
    final_metrics = canary.get_metrics()
    
    print(f"\n🎯 Canary Deployment Results:")
    print(f"   Final status: {final_status.value}")
    print(f"   Total requests (new): {final_metrics.requests_new}")
    print(f"   Error rate (new): {final_metrics.error_rate_new:.2f}%")
    print(f"   Success criteria met: {final_metrics.success_criteria_met}")
    
    if final_status.value == "succeeded":
        print("✅ Canary deployment successful - model promoted to production!")
    else:
        print("❌ Canary deployment failed - rolled back to baseline")
    
    return final_status.value == "succeeded"


def simulate_production_with_monitoring(model, monitor, drift_detector):
    """Simulate production traffic with monitoring and drift detection."""
    print("🚀 Simulating production traffic with monitoring...")
    
    # Simulate normal traffic for a while
    print("  Phase 1: Normal traffic...")
    for i in range(50):
        # Normal user behavior
        user_id = np.random.randint(0, 1000)
        item_id = np.random.randint(0, 1000)
        inputs = tf.constant([[user_id, item_id]], dtype=tf.float32)
        
        start_time = time.time()
        prediction = model(inputs)
        latency = time.time() - start_time
        
        # Record metrics
        monitor.record_prediction(
            inputs=inputs,
            predictions=prediction,
            latency_seconds=latency,
            custom_metrics={
                "click_rate": np.random.random() * 0.3,  # 0-30% click rate
                "conversion_rate": np.random.random() * 0.1,  # 0-10% conversion rate
            }
        )
        
        # Check for drift
        drift_results = drift_detector.detect_drift(inputs)
        for result in drift_results:
            if result.status.value != "stable":
                print(f"  ⚠️  Drift detected in {result.feature_name}: {result.drift_score:.3f} ({result.status.value})")
    
    # Simulate data drift (different user behavior)
    print("  Phase 2: Introducing data drift...")
    for i in range(50):
        # Shifted user behavior (different user ID range)
        user_id = np.random.randint(800, 1000)  # Concentrated in high IDs
        item_id = np.random.randint(0, 200)     # Concentrated in low IDs
        inputs = tf.constant([[user_id, item_id]], dtype=tf.float32)
        
        start_time = time.time()
        prediction = model(inputs)
        latency = time.time() - start_time
        
        # Record metrics (worse performance due to drift)
        monitor.record_prediction(
            inputs=inputs,
            predictions=prediction,
            latency_seconds=latency,
            custom_metrics={
                "click_rate": np.random.random() * 0.15,  # Lower click rate
                "conversion_rate": np.random.random() * 0.05,  # Lower conversion rate
            }
        )
        
        # Check for drift
        drift_results = drift_detector.detect_drift(inputs)
        for result in drift_results:
            if result.status.value != "stable":
                print(f"  🚨 Drift detected in {result.feature_name}: {result.drift_score:.3f} ({result.status.value})")
    
    # Get monitoring summary
    current_metrics = monitor.get_current_metrics()
    print(f"\n📊 Production Monitoring Summary:")
    print(f"   Total predictions: {100}")
    print(f"   Average latency (P95): {current_metrics.latency_p95:.2f}ms")
    print(f"   Error rate: {current_metrics.error_rate:.2f}%")
    print(f"   Prediction confidence: {current_metrics.prediction_confidence:.3f}")
    print(f"   Average click rate: {current_metrics.custom_metrics.get('click_rate', 0):.3f}")
    print(f"   Average conversion rate: {current_metrics.custom_metrics.get('conversion_rate', 0):.3f}")
    
    # Check for threshold violations
    alerts = monitor.check_thresholds()
    if alerts:
        print(f"   🚨 Active alerts: {alerts}")
    else:
        print(f"   ✅ All metrics within thresholds")


def main():
    """Run the complete MLOps example."""
    print("🚀 TensorFlow Production MLOps Complete Example")
    print("=" * 60)
    
    # Step 1: Train models
    model_v1, model_v2, training_data = train_models()
    
    # Step 2: Run A/B test
    winner_name, winning_model = run_ab_test(model_v1, model_v2)
    baseline_model = model_v1  # Keep V1 as baseline
    
    print("\n" + "=" * 60)
    
    # Step 3: Setup monitoring and drift detection
    monitor, drift_detector = setup_monitoring_and_drift_detection(winning_model, training_data)
    
    # Step 4: Run canary deployment
    canary_success = run_canary_deployment(winning_model, baseline_model)
    
    print("\n" + "=" * 60)
    
    # Step 5: Simulate production with monitoring
    production_model = winning_model if canary_success else baseline_model
    simulate_production_with_monitoring(production_model, monitor, drift_detector)
    
    print("\n" + "=" * 60)
    print("🎉 Complete MLOps pipeline demonstration finished!")
    print("\nWhat we demonstrated:")
    print("  ✅ Model training and versioning")
    print("  ✅ A/B testing with statistical significance")
    print("  ✅ Canary deployment with gradual rollout")
    print("  ✅ Real-time monitoring and alerting")
    print("  ✅ Data drift detection")
    print("  ✅ Automated rollback on failure")
    
    # Cleanup
    monitor.stop_monitoring()
    drift_detector.stop()


if __name__ == "__main__":
    main() 