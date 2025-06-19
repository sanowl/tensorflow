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
"""TensorFlow Production MLOps Toolkit.

This module provides comprehensive MLOps capabilities for production deployment,
monitoring, testing, and management of TensorFlow models.

Key Features:
- A/B Testing Framework
- Canary Deployments
- Model Monitoring & Drift Detection
- Automated Rollbacks
- Model Registry & Versioning
- Production Debugging Tools
"""

from tensorflow.python.production.deployment.auto_scaling import AutoScaling
from tensorflow.python.production.deployment.registry import ModelRegistry
from tensorflow.python.production.deployment.serving import deploy

from tensorflow.python.production.testing.ab_testing import ABTest
from tensorflow.python.production.testing.canary import CanaryDeployment
from tensorflow.python.production.testing.shadow_testing import ShadowTest

from tensorflow.python.production.monitoring.metrics import ModelMonitor
from tensorflow.python.production.monitoring.drift_detection import DriftDetector
from tensorflow.python.production.monitoring.alerting import AlertManager

from tensorflow.python.production.safety.circuit_breaker import SafetyNet
from tensorflow.python.production.safety.rollback import AutoRollback
from tensorflow.python.production.safety.validation import ModelValidator

from tensorflow.python.production.explainability.debugger import ModelDebugger

# Main API exports
__all__ = [
    # Deployment
    'deploy',
    'AutoScaling', 
    'ModelRegistry',
    
    # Testing
    'ABTest',
    'CanaryDeployment', 
    'ShadowTest',
    
    # Monitoring
    'ModelMonitor',
    'DriftDetector',
    'AlertManager',
    
    # Safety
    'SafetyNet',
    'AutoRollback',
    'ModelValidator',
    
    # Debugging
    'ModelDebugger',
] 