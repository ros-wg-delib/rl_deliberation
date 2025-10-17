#!/usr/bin/env python3

# Copyright (c) 2025, Sebastian Castro, Christian Henkel
# All rights reserved.

# This source code is licensed under the BSD 3-Clause License.
# See the LICENSE file in the project root for license information.

"""Evaluates a trained RL policy."""

import argparse
from typing import Dict, List, Union

import rclpy
from gymnasium.spaces import Discrete
from rclpy.node import Node
from stable_baselines3.common.base_class import BaseAlgorithm

from pyrobosim_ros_gym.envs import available_envs_w_subtype, get_env_by_name
from pyrobosim_ros_gym.policies import model_and_env_type_from_path


class ManualPolicy:
    """A policy that allows manual control of the robot."""

    def __init__(self, action_space):
        print("Welcome. You are the agent now!")
        self.action_space = action_space

    def predict(self, observation, deterministic):
        # print(f"Observation: {observation}")
        # print(f"Action space: {self.action_space}")
        possible_actions = list(range(self.action_space.n))
        while True:
            try:
                action = int(input(f"Enter action from {possible_actions}: "))
                if action in possible_actions:
                    return action, None
                else:
                    raise RuntimeError(f"Action {action} not in {possible_actions}.")
            except ValueError:
                raise RuntimeError("Invalid input, please enter an integer.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="The name of the model to evaluate.")
    parser.add_argument(
        "--manual-env",
        type=str,
        help="Use manual control for the environment.",
        choices=available_envs_w_subtype(),
    )
    parser.add_argument(
        "--num-episodes",
        default=3,
        type=int,
        help="The number of episodes to evaluate.",
    )
    parser.add_argument("--seed", default=42, type=int, help="The RNG seed to use.")
    parser.add_argument(
        "--realtime", action="store_true", help="If true, slows down to real time."
    )
    args = parser.parse_args()

    rclpy.init()
    node = Node("pyrobosim_ros_env")

    assert (args.manual_env is not None) ^ (
        args.model is not None
    ), "Exactly one of --manual-env-control or --model must be set."

    if args.manual_env is not None:
        env = get_env_by_name(
            args.manual_env,
            node,
            max_steps_per_episode=100,
            realtime=True,
            discrete_actions=True,
        )
        model = ManualPolicy(
            env.action_space
        )  # type: Union[ManualPolicy, BaseAlgorithm]
        args.num_episodes = 1  # Only one episode for manual control
        print("warning: Manual control enabled, only one episode will be run.")
    else:
        # Load the model and environment
        model, env_type = model_and_env_type_from_path(args.model)
        env = get_env_by_name(
            env_type,
            node,
            max_steps_per_episode=15,
            realtime=True,
            discrete_actions=isinstance(model.action_space, Discrete),
        )

    # Evaluate the model for some steps
    num_successful_episodes = 0
    reward_per_episode = [0.0 for _ in range(args.num_episodes)]
    custom_metrics_store: Dict[str, List[float]] = {}
    custom_metrics_episode_mean: Dict[str, float] = {}
    custom_metrics_per_episode: Dict[str, List[float]] = {}
    for i_e in range(args.num_episodes):
        print(f">>> Starting episode {i_e+1}/{args.num_episodes}")
        obs, _ = env.reset(seed=i_e + args.seed)
        terminated = False
        truncated = False
        i_step = 0
        while not (terminated or truncated):
            print(f"{obs=}")
            action, _ = model.predict(obs, deterministic=True)
            print(f"{action=}")
            obs, reward, terminated, truncated, info = env.step(action)
            custom_metrics = info.get("metrics", {})

            print(f"{reward=}")
            reward_per_episode[i_e] += reward

            print(f"{custom_metrics=}")
            for k, v in custom_metrics.items():
                if k not in custom_metrics_store:
                    custom_metrics_store[k] = []
                custom_metrics_store[k].append(v)

            print(f"{terminated=}")
            print(f"{truncated=}")
            print("." * 10)
            if terminated or truncated:
                success = info["success"]
                if success:
                    num_successful_episodes += 1
                print(f"<<< Episode {i_e+1} finished with {success=}.")
                print(f"Total reward: {reward_per_episode[i_e]}")
                for k, v in custom_metrics_store.items():
                    mean_metric = sum(v) / len(v) if len(v) > 0 else 0.0
                    custom_metrics_episode_mean[k] = mean_metric
                    if k not in custom_metrics_per_episode:
                        custom_metrics_per_episode[k] = []
                    custom_metrics_per_episode[k].append(mean_metric)
                    print(f"Mean {k}: {mean_metric}")
                print("=" * 20)
                break

    print("Summary:")
    success_percent = 100.0 * num_successful_episodes / args.num_episodes
    print(
        f"Successful episodes: {num_successful_episodes} / {args.num_episodes} "
        f"({success_percent:.2f}%)"
    )
    print(f"Reward over {args.num_episodes} episodes:")
    print(f" Mean: {sum(reward_per_episode)/args.num_episodes}")
    print(f" Min: {min(reward_per_episode)}")
    print(f" Max: {max(reward_per_episode)}")
    for k, v in custom_metrics_per_episode.items():
        print(f"Custom metric '{k}' over {args.num_episodes} episodes:")
        print(f" Mean: {sum(v)/args.num_episodes}")
        print(f" Min: {min(v)}")
        print(f" Max: {max(v)}")

    rclpy.shutdown()
