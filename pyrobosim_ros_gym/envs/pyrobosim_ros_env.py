#!/usr/bin/env python3

# Copyright (c) 2025, Sebastian Castro, Christian Henkel
# All rights reserved.

# This source code is licensed under the BSD 3-Clause License.
# See the LICENSE file in the project root for license information.

import time
from enum import Enum
from functools import partial

import gymnasium as gym
import rclpy
from rclpy.action import ActionClient

from pyrobosim_msgs.action import ExecuteTaskAction
from pyrobosim_msgs.srv import (
    RequestWorldInfo,
    RequestWorldState,
    ResetWorld,
    SetLocationState,
)


class PyRoboSimRosEnv(gym.Env):
    """Gym environment wrapping around the PyRoboSim ROS Interface."""

    sub_types = Enum("sub_types", "DEFINE_IN_SUBCLASS")

    def __init__(
        self,
        node,
        reward_fn,
        reset_validation_fn=None,
        max_steps_per_episode=50,
        realtime=True,
        discrete_actions=True,
        executor=None,
    ):
        """
        Instantiates a PyRoboSim ROS environment.

        :param node: The ROS node to use for creating clients.
        :param reward_fn: Function that calculates the reward (and possibly other outputs).
        :param reset_validation_fn: Function that calculates whether a reset is valid.
            If None (default), all resets are valid.
        :param max_steps_per_episode: Maximum number of steps before truncating an episode.
            If -1, there is no limit to number of steps.
        :param realtime: If True, commands PyRoboSim to run actions in real time.
            If False, actions run as quickly as possible for faster training.
        :param discrete_actions: If True, uses discrete actions, else uses continuous.
        :param executor: Optional ROS executor. It must be already spinning!
        """
        super().__init__()
        self.node = node
        self.executor = executor
        if self.executor is not None:
            self.executor.add_node(self.node)
            self.executor.wake()

        self.realtime = realtime
        self.max_steps_per_episode = max_steps_per_episode
        self.discrete_actions = discrete_actions

        if reward_fn is None:
            self.reward_fn = lambda _: 0.0
        else:
            self.reward_fn = partial(reward_fn, self)

        if reset_validation_fn is None:
            self.reset_validation_fn = lambda: True
        else:
            self.reset_validation_fn = lambda: reset_validation_fn(self)

        self.step_number = 0
        self.previous_location = None
        self.previous_action_type = None

        self.request_info_client = node.create_client(
            RequestWorldInfo, "/request_world_info"
        )
        self.request_state_client = node.create_client(
            RequestWorldState, "/request_world_state"
        )
        self.execute_action_client = ActionClient(
            node, ExecuteTaskAction, "/execute_action"
        )
        self.reset_world_client = node.create_client(ResetWorld, "reset_world")
        self.set_location_state_client = node.create_client(
            SetLocationState, "set_location_state"
        )

        self.request_info_client.wait_for_service()
        self.request_state_client.wait_for_service()
        self.execute_action_client.wait_for_server()
        self.reset_world_client.wait_for_service()
        self.set_location_state_client.wait_for_service()

        future = self.request_info_client.call_async(RequestWorldInfo.Request())
        self._spin_future(future)
        self.world_info = future.result().info

        future = self.request_state_client.call_async(RequestWorldState.Request())
        self._spin_future(future)
        self.world_state = future.result().state

        self.all_locations = []
        for loc in self.world_state.locations:
            self.all_locations.extend(loc.spawns)
        self.num_locations = sum(len(loc.spawns) for loc in self.world_state.locations)
        self.loc_to_idx = {loc: idx for idx, loc in enumerate(self.all_locations)}
        print(f"{self.all_locations=}")

        self.action_space = self._action_space()
        print(f"{self.action_space=}")

    def _spin_future(self, future):
        if self.executor is None:
            rclpy.spin_until_future_complete(self.node, future)
        else:
            while not future.done():
                time.sleep(0.1)

    def _action_space(self):
        raise NotImplementedError("implement in sub-class")

    def initialize(self):
        """Resets helper variables for deployment without doing a full reset."""
        raise NotImplementedError("implement in sub-class")

    def step(self, action):
        raise NotImplementedError("implement in sub-class")

    def reset(self, seed=None, options=None):
        """Resets the environment with a specified seed and options."""
        print(f"Resetting environment with {seed=}")
        super().reset(seed=seed)
