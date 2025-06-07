# Copyright 2025 Google LLC
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

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from .base_agent_config import BaseAgentConfig
from .common_config import JavaStaticObjectConfig


class GraphConfig(BaseModel):
  """Configuration for LangGraph graph."""

  py_qual_name: Optional[str] = None
  """The fully qualified name for CompiledGraph in python."""

  java_graph: Optional[JavaStaticObjectConfig] = None
  """The Java Static Object for CompiledGraph."""


class LangGraphAgentConfig(BaseAgentConfig):
  """Configuration for LangGraph agent."""

  graph: GraphConfig
  """The CompiledGraph for LangGraph agent."""

  instruction: Optional[str] = None
  """LangGraphAgent.instruction."""
