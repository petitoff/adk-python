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

from typing import Literal
from typing import Optional

from pydantic import BaseModel

from .base_agent_config import BaseAgentConfig
from .common_config import CallbackConfig
from .common_config import JavaStaticObjectConfig
from .common_config import PyInlineCodeConfig


class ToolConfig(BaseModel):
  """Config for a tool in LlmAgent."""

  name: Optional[str]
  """The name of ADK builtin tool.

  This field only allows tools defined in ADK tools package.
  - In python, it's google.adk.tools
  - In Java, it's com.google.adk.tool

  Example: google_search, load_memory, etc.
  """

  py_qual_name: Optional[str]
  """The fully qualified name of a ToolUnion object in python.

  This is used to reference a Tool defined in users' python library.

  If set, this field will override `name` field in ADK Python runtime

  The format is package.to.module.func_name. The expected type should be a
  ToolUnion, which will be added to LlmAgent.tools directly.

  Example usage in yaml file:

  ```
  - tools:
    - py_qual_name: my_library.my_tools.tool_name
    - py_qual_name: my_library.my_tools.toolset_name
  ```
  """

  py_inline: Optional[PyInlineCodeConfig] = None
  """LlmAgent.tools. Optional. Inline python code.

  If set, this field will override `name` field in ADK Python runtime
  """

  java_tool: Optional[JavaStaticObjectConfig] = None
  """The static Java object for the tool.

  If set, this field will override `name` field in ADK Java runtime.

  Sample usage in yaml:

  ```
  - tools:
    - java_tool:
        static_field_name: blockBadWordTool
        class_name: com.acme.SecurityTools
  ```
  """


class LlmAgentConfig(BaseAgentConfig):
  """Config for LlmAgent."""

  model: Optional[str] = None
  """LlmAgent.model. Optional.

  When not set, using the same model as the parent model.
  """

  instruction: str
  """LlmAgent.instruction. Required."""

  tools: Optional[list[ToolConfig]] = None
  """LlmAgent.tools. Optional."""

  before_model_callbacks: Optional[list[CallbackConfig]] = None
  after_model_callbacks: Optional[list[CallbackConfig]] = None
  before_tool_callbacks: Optional[list[CallbackConfig]] = None
  after_tool_callbacks: Optional[list[CallbackConfig]] = None

  disallow_transfer_to_parent: bool = False
  """LlmAgent.disallow_transfer_to_parent. Optional."""

  disallow_transfer_to_peers: bool = False
  """LlmAgent.disallow_transfer_to_peers. Optional."""

  include_contents: Literal['default', 'none'] = 'default'
  """LlmAgent.include_contents. Optional."""

  output_key: Optional[str] = None
  """LlmAgent.output_key. Optional."""
