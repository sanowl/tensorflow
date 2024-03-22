# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# LINT.IfChange
"""TensorFlow root package"""

import sys as _sys
import importlib as _importlib
import types as _types

from tensorflow.python.util import module_wrapper


class _LazyLoader(_types.ModuleType):
  """Lazily import a module so that we can forward it."""

  def __init__(self, local_name, parent_module_globals, name):
    self._local_name = local_name
    self._parent_module_globals = parent_module_globals
    super(_LazyLoader, self).__init__(name)

  def _load(self):
    """Import the target module and insert it into the parent's namespace."""
    module = _importlib.import_module(self.__name__)
    self._parent_module_globals[self._local_name] = module
    self.__dict__.update(module.__dict__)
    return module

  def __getattr__(self, item):
    module = self._load()
    return getattr(module, item)

  def __dir__(self):
    module = self._load()
    return dir(module)

  def __reduce__(self):
    return __import__, (self.__name__,)


from tensorflow_core import _LazyLoader, module_wrapper

def _forward_module(old_name):
  parts = old_name.split(".")
  parts[0] = parts[0] + "_core"
  local_name = parts[-1]
  existing_name = ".".join(parts)
  _module = _LazyLoader(local_name, globals(), existing_name)
  return _sys.modules.setdefault(old_name, _module)


_top_level_modules = [
    "tensorflow._api",
    "tensorflow.python",
    "tensorflow.tools",
    "tensorflow.core",
    "tensorflow.compiler",
    "tensorflow.lite",
    "tensorflow.keras",
    "tensorflow.compat",
    "tensorflow.summary",  # tensorboard
    "tensorflow.examples",
]

for module_name in _top_level_modules:
  _forward_module(module_name)

_major_api_version = 1

if not isinstance(_sys.modules[__name__], module_wrapper.TFModuleWrapper):
  _sys.modules[__name__] = module_wrapper.TFModuleWrapper(
      _sys.modules[__name__], "")