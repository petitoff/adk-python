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


class PyInlineCodeConfig(BaseModel):
  """Config for inline python function.

  E.g. when referencing `my_func` in below code:

  ```python
  import ....

  def _helper_func(...):
    ...

  def my_func(...):
    ...
  ```

  The corresponding yaml config should be:

  ```
  name: my_func
  code: |
    import ....

    def _helper_func(...):
      ...

    def my_func(...):
      ...
    ```
  """

  name: str
  """The variable name from the code."""

  code: str
  """The self-contained python code for the function."""


class JavaStaticObjectConfig(BaseModel):
  """Config to reference a static object in Java.

  E.g. when referencing the `myAgent` in the following Java code:

  ```java
  package com.acme.agent

  class MyAgent {
    public static final LlmAgent myAgent = LlmAgent.builder()
      .name("search_assistant")
      .model("gemini-2.0-flash")
      .instruction("You are a helpful assistant.")
      .build();
  }
  ```

  The correspoding yaml config should be:

  ```
  name: myAgent
  class_qual_name: com.acme.agent.MyClass
  ```
  """

  name: str
  """The static variable or method name in the container class."""

  class_qual_name: str
  """The fully qualified name of the container class."""


class CallbackConfig(BaseModel):
  """The config for a callback function."""

  py_qual_name: Optional[str] = None
  """The fully qualified name of the callback function in python."""

  py_inline: Optional[PyInlineCodeConfig] = None
  """Defines the callback function via inline python code.

  When this exists, it override the py_qual_name.
  """

  java_callback: Optional[JavaStaticObjectConfig] = None
  """The callback function defined as a static class instance in Java."""
