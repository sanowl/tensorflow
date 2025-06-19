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
"""Tests for A/B Testing Framework."""

import time
import unittest
import numpy as np
import tensorflow as tf
from tensorflow.python.production.testing.ab_testing import ABTest, ABTestResult


class MockModel:
    """Mock model for testing."""
    
    def __init__(self, base_performance: float = 0.5):
        self.base_performance = base_performance
    
    def __call__(self, inputs):
        # Return constant prediction based on base performance
        batch_size = inputs.shape[0]
        return tf.constant([[self.base_performance]] * batch_size, dtype=tf.float32)


class ABTestingTest(unittest.TestCase):
    """Test cases for A/B testing framework."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model_a = MockModel(base_performance=0.5)  # Baseline model
        self.model_b = MockModel(base_performance=0.6)  # Better model
        
        self.test_inputs = tf.constant([[1, 2], [3, 4], [5, 6]], dtype=tf.float32)
    
    def test_ab_test_initialization(self):
        """Test A/B test initialization."""
        ab_test = ABTest(
            name="test_experiment",
            model_a=self.model_a,
            model_b=self.model_b,
            traffic_split=0.2,
            success_metrics=["accuracy", "conversion"],
            duration_days=7
        )
        
        self.assertEqual(ab_test.config.name, "test_experiment")
        self.assertEqual(ab_test.config.traffic_split, 0.2)
        self.assertEqual(ab_test.config.success_metrics, ["accuracy", "conversion"])
        self.assertFalse(ab_test.is_running)
    
    def test_ab_test_start_stop(self):
        """Test starting and stopping A/B test."""
        ab_test = ABTest(
            name="test_start_stop",
            model_a=self.model_a,
            model_b=self.model_b
        )
        
        # Test start
        ab_test.start()
        self.assertTrue(ab_test.is_running)
        self.assertIsNotNone(ab_test.start_time)
        
        # Test stop
        ab_test.stop()
        self.assertFalse(ab_test.is_running)
    
    def test_ab_test_prediction_routing(self):
        """Test that predictions are routed correctly."""
        ab_test = ABTest(
            name="test_routing",
            model_a=self.model_a,
            model_b=self.model_b,
            traffic_split=0.5  # 50-50 split for testing
        )
        
        ab_test.start()
        
        # Make multiple predictions and check routing
        model_a_count = 0
        model_b_count = 0
        
        for i in range(100):
            inputs = tf.constant([[i, i+1]], dtype=tf.float32)
            prediction = ab_test.predict(inputs)
            
            # Based on our mock models, we can determine which was used
            if float(prediction[0][0]) == 0.5:
                model_a_count += 1
            else:
                model_b_count += 1
        
        # With 50-50 split and consistent hashing, we should get both models
        self.assertGreater(model_a_count, 0)
        self.assertGreater(model_b_count, 0)
        self.assertEqual(model_a_count + model_b_count, 100)
    
    def test_outcome_recording(self):
        """Test recording of outcome metrics."""
        ab_test = ABTest(
            name="test_outcomes",
            model_a=self.model_a,
            model_b=self.model_b,
            success_metrics=["accuracy", "conversion"]
        )
        
        ab_test.start()
        
        # Record some outcomes
        for i in range(10):
            inputs = tf.constant([[i, i+1]], dtype=tf.float32)
            ab_test.predict(inputs)  # Make prediction first
            
            outcomes = {
                "accuracy": 0.8,
                "conversion": 0.1
            }
            ab_test.record_outcome(inputs, outcomes)
        
        # Check that outcomes were recorded
        self.assertGreater(ab_test._sample_count_a + ab_test._sample_count_b, 0)
        self.assertGreater(len(ab_test._results_a["accuracy"]) + len(ab_test._results_b["accuracy"]), 0)
    
    def test_statistical_significance_calculation(self):
        """Test statistical significance calculation."""
        ab_test = ABTest(
            name="test_significance",
            model_a=self.model_a,
            model_b=self.model_b,
            min_sample_size=10,
            significance_threshold=0.8
        )
        
        ab_test.start()
        
        # Simulate experiment with clear winner (model B)
        for i in range(50):
            inputs = tf.constant([[i, i+1]], dtype=tf.float32)
            prediction = ab_test.predict(inputs)
            
            # Model B should perform better
            if float(prediction[0][0]) == 0.6:  # Model B
                outcomes = {"accuracy": 0.9}  # High performance
            else:  # Model A
                outcomes = {"accuracy": 0.3}  # Low performance
            
            ab_test.record_outcome(inputs, outcomes)
        
        results = ab_test.get_results()
        
        # Check that we have results
        self.assertIsInstance(results, ABTestResult)
        self.assertGreater(results.sample_size_a + results.sample_size_b, 0)
        self.assertIsNotNone(results.winner)
    
    def test_winner_promotion(self):
        """Test promoting the winning model."""
        ab_test = ABTest(
            name="test_promotion",
            model_a=self.model_a,
            model_b=self.model_b,
            traffic_split=0.1,
            min_sample_size=5,
            significance_threshold=0.5  # Low threshold for testing
        )
        
        ab_test.start()
        
        # Create clear winner scenario
        for i in range(20):
            inputs = tf.constant([[i, i+1]], dtype=tf.float32)
            prediction = ab_test.predict(inputs)
            
            # Always give model B better outcomes
            if float(prediction[0][0]) == 0.6:  # Model B
                outcomes = {"accuracy": 1.0}
            else:  # Model A
                outcomes = {"accuracy": 0.0}
            
            ab_test.record_outcome(inputs, outcomes)
        
        # Check if we can promote winner
        if ab_test.is_significant():
            winner = ab_test.get_winner()
            self.assertIsNotNone(winner)
            
            original_split = ab_test.config.traffic_split
            ab_test.promote_winner()
            
            # Check that traffic split changed
            if winner == "model_b":
                self.assertEqual(ab_test.config.traffic_split, 1.0)
            else:
                self.assertEqual(ab_test.config.traffic_split, 0.0)
    
    def test_insufficient_data_handling(self):
        """Test handling of insufficient data."""
        ab_test = ABTest(
            name="test_insufficient_data",
            model_a=self.model_a,
            model_b=self.model_b,
            min_sample_size=1000  # High threshold
        )
        
        ab_test.start()
        
        # Record minimal data
        inputs = tf.constant([[1, 2]], dtype=tf.float32)
        ab_test.predict(inputs)
        ab_test.record_outcome(inputs, {"accuracy": 0.5})
        
        results = ab_test.get_results()
        
        # Should indicate insufficient data
        self.assertIn("Insufficient data", results.recommendation)
        self.assertIsNone(results.winner)
    
    def test_multiple_metrics(self):
        """Test handling of multiple success metrics."""
        ab_test = ABTest(
            name="test_multiple_metrics",
            model_a=self.model_a,
            model_b=self.model_b,
            success_metrics=["accuracy", "precision", "recall", "f1_score"]
        )
        
        ab_test.start()
        
        # Record outcomes with multiple metrics
        for i in range(10):
            inputs = tf.constant([[i, i+1]], dtype=tf.float32)
            ab_test.predict(inputs)
            
            outcomes = {
                "accuracy": np.random.random(),
                "precision": np.random.random(),
                "recall": np.random.random(),
                "f1_score": np.random.random()
            }
            ab_test.record_outcome(inputs, outcomes)
        
        results = ab_test.get_results()
        
        # Should use first metric (accuracy) as primary
        self.assertGreater(results.sample_size_a + results.sample_size_b, 0)
    
    def test_edge_cases(self):
        """Test various edge cases."""
        # Test with 0% traffic split
        ab_test_zero = ABTest(
            name="test_zero_split",
            model_a=self.model_a,
            model_b=self.model_b,
            traffic_split=0.0
        )
        
        ab_test_zero.start()
        
        # All traffic should go to model A
        for i in range(10):
            inputs = tf.constant([[i, i+1]], dtype=tf.float32)
            prediction = ab_test_zero.predict(inputs)
            self.assertEqual(float(prediction[0][0]), 0.5)  # Model A's output
        
        # Test with 100% traffic split
        ab_test_full = ABTest(
            name="test_full_split",
            model_a=self.model_a,
            model_b=self.model_b,
            traffic_split=1.0
        )
        
        ab_test_full.start()
        
        # All traffic should go to model B
        for i in range(10):
            inputs = tf.constant([[i, i+1]], dtype=tf.float32)
            prediction = ab_test_full.predict(inputs)
            self.assertEqual(float(prediction[0][0]), 0.6)  # Model B's output
    
    def test_consistency_across_calls(self):
        """Test that same input consistently routes to same model."""
        ab_test = ABTest(
            name="test_consistency",
            model_a=self.model_a,
            model_b=self.model_b,
            traffic_split=0.5
        )
        
        ab_test.start()
        
        # Same input should always route to same model
        test_input = tf.constant([[42, 43]], dtype=tf.float32)
        
        first_prediction = ab_test.predict(test_input)
        
        for _ in range(10):
            prediction = ab_test.predict(test_input)
            self.assertTrue(tf.reduce_all(tf.equal(prediction, first_prediction)))


if __name__ == "__main__":
    unittest.main() 