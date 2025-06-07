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

from typing import List
from typing import Optional

from pydantic import BaseModel

from .common_config import CallbackConfig
from .common_config import JavaStaticObjectConfig


class SubAgentConfig(BaseModel):
  """The config for a sub-agent."""

  config_path: Optional[str] = None
  """The config path of the sub-agent defined via agent config."""

  py_agent: Optional[str] = None
  """The fully qualified name of the sub-agent object defined via python code.

  When this exists, it override the config_path.

  Example: `custom_agents.my_toy_agent` - an LlmAgent instance.
  """

  java_agent: Optional[JavaStaticObjectConfig] = None
  """The sub-agent defined as a static class instance in Java.

  When this exists, it override the config_path.

  Example: to reference the myAgent object in below Java container class:

  ```
  package com.acme.agent

  class MyAgent {
    public static final LlmAgent myAgent = LlmAgent.builder()
      .name("search_assistant")
      .model("gemini-2.0-flash")
      .instruction("You are a helpful assistant.")
      .build();
  }
  ```

  The yaml config should be

  ```
  sub_agents:
    - java_agent:
        name: myAgent
        class_qual_name: com.acme.agent.MyAgent
  ```
  """


class BaseAgentConfig(BaseModel):
  """The config for a BaseAgent."""

  agent_class: str
  """ADK builtin Agent type or a fully qualified agent class.

  ADK builtin agents (in google.adk.agents package) can be abbreviated.

  Allowed abbreviations: LlmAgent, SequentialAgent, LoopAgent, ParallelAgent.

  For custom agent, use the fully qualified name, e.g.
  `my_library.my_agents.MyAgent`.

  The custom agent class should also provide `from_config(config: dict)` class
  method to load from config.
  """

  name: str
  """BaseAgent.name. Required."""

  description: str = ""
  """BaseAgent.description. Optional."""

  before_agent_callbacks: Optional[List[CallbackConfig]] = None
  """The callbacks to be invoked before the agent run.

  Example usage:

  Below is a sample of two callbacks in yaml:

  ```
  before_agent_callbacks:
    - py_qual_name: security_callbacks.before_agent_callback
      java_callback:
        name: beforeAgentCallback
        class_qual_name: com.acme.security.Callbacks
    - py_inline: # a inline defined before_agent_callback for logging in python
        name: log_before_agent
        code: |
          def log_before_agent(callback_context):
            ...
      java_callback: # a before_agent_callback for logging in Java.
        name: beforeAgentCallback
        class_qual_name: com.acme.agentlog.Callbacks
  ```
  """

  after_agent_callbacks: Optional[List[CallbackConfig]] = None
  """The callbacks to be invoked after the agent run."""

  sub_agents: Optional[List[SubAgentConfig]] = None
  """The sub_agents of this agent.

  Example usage:

  Below is a sample with two sub-agents in yaml.
  - The first agent is defined via config.
  - The second agent is implemented Python and Java respectively.

  ```
  sub_agents:
    - config_path: search_agent.yaml # No.1 sub-agent defined via config
    - py_agent: security.policy_agent # No.2 sub-agent defined in python.
      java_agent: # No.2 sub-agent defined in Java.
        name: policyAgent
        class_qual_name: com.acme.SecurityAgents
  ```
  """
