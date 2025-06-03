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
import logging
from typing import Optional

from ..errors.not_found_error import NotFoundError
from .eval_case import EvalCase
from .eval_set import EvalSet

logger = logging.getLogger("google_adk." + __name__)


class EvalSetsManager(ABC):
  """An interface to manage an Eval Sets."""

  @abstractmethod
  def get_eval_set(self, app_name: str, eval_set_id: str) -> Optional[EvalSet]:
    """Returns an EvalSet identified by an app_name and eval_set_id."""

  @abstractmethod
  def create_eval_set(self, app_name: str, eval_set_id: str):
    """Creates an empty EvalSet given the app_name and eval_set_id."""

  @abstractmethod
  def list_eval_sets(self, app_name: str) -> list[str]:
    """Returns a list of EvalSets that belong to the given app_name."""

  @abstractmethod
  def add_eval_case(self, app_name: str, eval_set_id: str, eval_case: EvalCase):
    """Adds the given EvalCase to an existing EvalSet identified by app_name and eval_set_id.

    Raises:
      NotFoundError: If the eval set is not found.
    """

  @abstractmethod
  def update_eval_case(
      self, app_name: str, eval_set_id: str, updated_eval_case: EvalCase
  ):
    """Updates an existing EvalCase give the app_name and eval_set_id.

    Raises:
      NotFoundError: If the eval set or the eval case is not found.
    """

  @abstractmethod
  def delete_eval_case(
      self, app_name: str, eval_set_id: str, eval_case_id: str
  ):
    """Deletes the given EvalCase identified by app_name, eval_set_id and eval_case_id.

    Raises:
      NotFoundError: If the eval set or the eval case to delete is not found.
    """

  def get_eval_case(
      self, app_name: str, eval_set_id: str, eval_case_id: str
  ) -> Optional[EvalCase]:
    """Returns an EvalCase if found, otherwise None."""
    eval_set = self.get_eval_set(app_name, eval_set_id)

    if not eval_set:
      return None

    eval_case_to_find = None

    # Look up the eval case by eval_case_id
    for eval_case in eval_set.eval_cases:
      if eval_case.eval_id == eval_case_id:
        eval_case_to_find = eval_case
        break

    return eval_case_to_find

  def _add_eval_case_to_eval_set(
      self, app_name: str, eval_set_id: str, eval_case: EvalCase
  ) -> EvalSet:
    """Adds an eval case to an eval set and returns the updated eval set.

    Returns:
      The updated eval set with the added eval case.

    Raises:
      NotFoundError: If the eval set is not found.
      ValueError: If the eval case already exists in the eval set.
    """
    eval_set = self.get_eval_set(app_name, eval_set_id)
    if not eval_set:
      raise NotFoundError(f"Eval set `{eval_set_id}` not found.")
    eval_case_id = eval_case.eval_id

    if [x for x in eval_set.eval_cases if x.eval_id == eval_case_id]:
      raise ValueError(
          f"Eval id `{eval_case_id}` already exists in `{eval_set_id}`"
          " eval set.",
      )

    eval_set.eval_cases.append(eval_case)
    return eval_set

  def _update_eval_case_in_eval_set(
      self, app_name: str, eval_set_id: str, updated_eval_case: EvalCase
  ) -> EvalSet:
    """Updates an eval case in an eval set and returns the updated eval set.

    Returns:
      The updated eval set with the updated eval case.

    Raises:
      NotFoundError: If the eval set or the eval case to delete is not found.
    """
    eval_set = self.get_eval_set(app_name, eval_set_id)
    if not eval_set:
      raise NotFoundError(f"Eval set `{eval_set_id}` not found.")

    # Find the eval case to be updated.
    eval_case_id = updated_eval_case.eval_id
    eval_case_to_update = self.get_eval_case(
        app_name, eval_set_id, eval_case_id
    )

    if not eval_case_to_update:
      raise NotFoundError(
          f"Eval case `{eval_case_id}` not found in eval set `{eval_set_id}`."
      )

    # Remove the existing eval case and add the updated eval case.
    eval_set.eval_cases.remove(eval_case_to_update)
    eval_set.eval_cases.append(updated_eval_case)
    return eval_set

  def _delete_eval_case_from_eval_set(
      self, app_name: str, eval_set_id: str, eval_case_id: str
  ) -> EvalSet:
    """Deletes an eval case from an eval set and returns the updated eval set.

    Returns:
      The updated eval set with eval case removed.

    Raises:
      NotFoundError: If the eval set or the eval case to delete is not found.
    """
    eval_set = self.get_eval_set(app_name, eval_set_id)
    if not eval_set:
      raise NotFoundError(f"Eval set `{eval_set_id}` not found.")

    # Find the eval case to be deleted.
    eval_case_to_delete = self.get_eval_case(
        app_name, eval_set_id, eval_case_id
    )

    if not eval_case_to_delete:
      raise NotFoundError(
          f"Eval case `{eval_case_id}` not found in eval set `{eval_set_id}`."
      )

    # Remove the existing eval case.
    logger.info(
        "EvalCase`%s` was found in the eval set. It will be removed "
        "permanently.",
        eval_case_id,
    )
    eval_set.eval_cases.remove(eval_case_to_delete)
    return eval_set
