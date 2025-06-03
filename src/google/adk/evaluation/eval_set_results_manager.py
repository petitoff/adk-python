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

from abc import ABC
from abc import abstractmethod
import time

from .eval_result import EvalCaseResult
from .eval_result import EvalSetResult


def _sanitize_eval_set_result_name(eval_set_result_name: str) -> str:
  """Sanitizes the eval set result name."""
  return eval_set_result_name.replace("/", "_")


class EvalSetResultsManager(ABC):
  """An interface to manage Eval Set Results."""

  @abstractmethod
  def save_eval_set_result(
      self,
      app_name: str,
      eval_set_id: str,
      eval_case_results: list[EvalCaseResult],
  ) -> None:
    """Creates and saves a new EvalSetResult given eval_case_results."""
    raise NotImplementedError()

  @abstractmethod
  def get_eval_set_result(
      self, app_name: str, eval_set_result_id: str
  ) -> EvalSetResult:
    """Returns an EvalSetResult identified by app_name and eval_set_result_id."""
    raise NotImplementedError()

  @abstractmethod
  def list_eval_set_results(self, app_name: str) -> list[str]:
    """Returns the eval result ids that belong to the given app_name."""
    raise NotImplementedError()

  def create_eval_set_result(
      self,
      app_name: str,
      eval_set_id: str,
      eval_case_results: list[EvalCaseResult],
  ) -> EvalSetResult:
    """Creates a new EvalSetResult given eval_case_results."""
    timestamp = time.time()
    eval_set_result_id = f"{app_name}_{eval_set_id}_{timestamp}"
    eval_set_result_name = _sanitize_eval_set_result_name(eval_set_result_id)
    eval_set_result = EvalSetResult(
        eval_set_result_id=eval_set_result_id,
        eval_set_result_name=eval_set_result_name,
        eval_set_id=eval_set_id,
        eval_case_results=eval_case_results,
        creation_timestamp=timestamp,
    )
    return eval_set_result
