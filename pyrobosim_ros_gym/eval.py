#!/usr/bin/env python3

# Copyright (c) 2025, Sebastian Castro, Christian Henkel
# All rights reserved.

# This source code is licensed under the BSD 3-Clause License.
# See the LICENSE file in the project root for license information.

"""Evaluates a trained RL policy."""

import argparse
import os
from typing import Dict, List

import rclpy
from rclpy.node import Node
from stable_baselines3 import DQN, PPO, SAC, A2C
from stable_baselines3.common.base_class import BaseAlgorithm

from pyrobosim_ros_gym.envs import get_env_by_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, help="The name of the model to evaluate."
    )
    parser.add_argument(
        "--discrete-actions",
        action="store_true",
        help="If true, uses discrete action space. Otherwise, uses continuous action space.",
    )
    parser.add_argument(
        "--num-episodes",
        default=3,
        type=int,
        help="The number of episodes to evaluate.",
    )
    parser.add_argument("--seed", default=42, type=int, help="The RNG seed to use.")
    args = parser.parse_args()

    assert os.path.isfile(args.model), f"Model {args.model} must be a valid file."
    model_fname = os.path.basename(args.model)
    model_name_parts = model_fname.split("_")
    assert (
        len(model_name_parts) >= 2
    ), f"Model name {model_fname} must be of the form <env>_<model>[_<otherinfo>].pt"
    env_type = model_name_parts[0]
    model_type = model_name_parts[1]

    # Create the environment
    rclpy.init()
    node = Node("pyrobosim_ros_env")

    env = get_env_by_name(
        env_type,
        node,
        max_steps_per_episode=15,
        realtime=True,
        discrete_actions=args.discrete_actions,
    )
    env.reset()

    # Load a model
    if model_type == "DQN":
        model: BaseAlgorithm = DQN.load(args.model, env=None)
    elif model_type == "PPO":
        model = PPO.load(args.model, env=None)
    elif model_type == "SAC":
        model = SAC.load(args.model, env=None)
    elif model_type == "A2C":
        model = A2C.load(args.model, env=None)
    else:
        raise RuntimeError(f"Invalid model type: {model_type}")

    # Evaluate it for some steps
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
            custom_metrics = env.eval()

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
                print(f"<<< Episode {i_e+1} finished.")
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
