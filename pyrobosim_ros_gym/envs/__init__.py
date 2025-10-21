#!/usr/bin/env python3

# Copyright (c) 2025, Sebastian Castro, Christian Henkel
# All rights reserved.

# This source code is licensed under the BSD 3-Clause License.
# See the LICENSE file in the project root for license information.

from typing import Any, Callable, List, Dict

import rclpy
from rclpy.executors import Executor

from .banana import BananaEnv
from .greenhouse import GreenhouseEnv
from .pyrobosim_ros_env import PyRoboSimRosEnv


ENV_CLASS_FROM_NAME: Dict[str, type[PyRoboSimRosEnv]] = {
    "Banana": BananaEnv,
    "Greenhouse": GreenhouseEnv,
}


def available_envs_w_subtype() -> List[str]:
    """Return a list of environment types including subtypes."""
    envs: List[str] = []
    for name, env_class in ENV_CLASS_FROM_NAME.items():
        for sub_type in env_class.sub_types:
            envs.append("".join((name, sub_type.name)))
    return envs


def get_env_class_and_subtype_from_name(req_name: str):
    """Return the class of a chosen environment name (ignoring `sub_type`s)."""
    for name, env_class in ENV_CLASS_FROM_NAME.items():
        if req_name.startswith(name):
            sub_type_str = req_name.replace(name, "")
            for st in env_class.sub_types:
                if st.name == sub_type_str:
                    sub_type = st
                    return env_class, sub_type
    raise RuntimeError(f"No environment found for {req_name}.")


def get_env_by_name(
    env_name: str,
    node: rclpy.node.Node,
    max_steps_per_episode: int,
    realtime: bool,
    discrete_actions: bool,
    reward_fn: Callable[..., Any],
    executor: Executor | None = None,
) -> PyRoboSimRosEnv:
    """
    Instantiate an environment class for a given type and `sub_type`.

    :param env_name: Name of environment, with subtype, e.g. BananaPick.
    :param node: Node instance needed for ROS communication.
    :param max_steps_per_episode: Limit the steps (when to end the episode).
    :param realtime: Whether actions take time.
    :param discrete_actions: Choose discrete actions (needed for DQN).
    :param reward_fn: The function used to compute the reward at each step.
    :param executor: Optional ROS executor. It must be already spinning!
    """
    base_class, sub_type = get_env_class_and_subtype_from_name(env_name)
    return base_class(
        sub_type,
        node,
        max_steps_per_episode,
        realtime,
        discrete_actions,
        reward_fn,
        executor=executor,
    )
