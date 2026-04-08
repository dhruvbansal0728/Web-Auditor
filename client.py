# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Web Auditor Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from models import WebAuditorAction, WebAuditorObservation


class WebAuditorEnv(
    EnvClient[WebAuditorAction, WebAuditorObservation, State]
):
    """
    Client for the Web Auditor Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with WebAuditorEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.echoed_message)
        ...
        ...     result = client.step(WebAuditorAction(message="Hello!"))
        ...     print(result.observation.echoed_message)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = WebAuditorEnv.from_docker_image("web_auditor-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(WebAuditorAction(message="Test"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: WebAuditorAction) -> Dict:
        """
        Convert WebAuditorAction to JSON payload for step message.
        """
        return {
            "command": action.command,
        }

    def _parse_result(self, payload: Dict) -> StepResult[WebAuditorObservation]:
        """
        Parse server response into StepResult[WebAuditorObservation].
        """
        obs_data = payload.get("observation", {})
        observation = WebAuditorObservation(
            output=obs_data.get("output", ""),
            current_directory_structure=obs_data.get("current_directory_structure", ""),
            file_content=obs_data.get("file_content"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
