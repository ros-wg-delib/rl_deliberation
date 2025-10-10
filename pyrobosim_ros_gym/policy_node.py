#!/usr/bin/env python3

# Copyright (c) 2025, Sebastian Castro, Christian Henkel
# All rights reserved.

# This source code is licensed under the BSD 3-Clause License.
# See the LICENSE file in the project root for license information.

"""Serves a policy as a ROS node with an action server for deployment."""

import argparse
import time

from gymnasium.spaces import Discrete
import rclpy
from rclpy.action import ActionServer, CancelResponse
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node

from rl_interfaces.action import ExecutePolicy  # type: ignore[attr-defined]

from pyrobosim_ros_gym.envs import get_env_by_name
from pyrobosim_ros_gym.policies import model_and_env_type_from_path


class PolicyServerNode(Node):
    def __init__(self, model: str, executor):
        super().__init__("policy_node")

        # Load the model and environment
        self.model, env_type = model_and_env_type_from_path(model)
        self.env = get_env_by_name(
            env_type,
            self,
            executor=executor,
            max_steps_per_episode=-1,
            realtime=True,
            discrete_actions=isinstance(self.model.action_space, Discrete),
        )

        self.action_server = ActionServer(
            self,
            ExecutePolicy,
            "/execute_policy",
            execute_callback=self.execute_policy,
            cancel_callback=self.cancel_policy,
        )

        self.get_logger().info(f"Started policy node with model '{model}'.")

    def cancel_policy(self, goal_handle):
        self.get_logger().info("Canceling policy execution...")
        return CancelResponse.ACCEPT

    async def execute_policy(self, goal_handle):
        self.get_logger().info("Starting policy execution...")
        result = ExecutePolicy.Result()

        self.env.initialize()  # Resets helper variables
        obs = self.env._get_obs()
        cumulative_reward = 0.0
        while True:
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info("Policy execution canceled")
                return result

            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.env.step(action)
            num_steps = self.env.step_number
            cumulative_reward += reward
            self.get_logger().info(
                f"Step {num_steps}: cumulative reward = {cumulative_reward}"
            )

            goal_handle.publish_feedback(
                ExecutePolicy.Feedback(
                    num_steps=num_steps, cumulative_reward=cumulative_reward
                )
            )

            if terminated or truncated:
                break

            time.sleep(0.1)  # Small sleep between actions

        goal_handle.succeed()
        result.success = info["success"]
        result.num_steps = num_steps
        result.cumulative_reward = cumulative_reward
        self.get_logger().info(f"Policy completed in {num_steps} steps.")
        self.get_logger().info(
            f"success: {result.success}, cumulative reward: {cumulative_reward}"
        )
        return result


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, help="The name of the model to serve."
    )
    cli_args = parser.parse_args()

    rclpy.init(args=args)
    try:
        executor = MultiThreadedExecutor(num_threads=2)
        import threading

        threading.Thread(target=executor.spin).start()
        policy_server = PolicyServerNode(cli_args.model, executor)
        while True:
            time.sleep(0.5)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass


if __name__ == "__main__":
    main()
