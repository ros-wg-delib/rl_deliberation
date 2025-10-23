#!/usr/bin/env python3

# Copyright (c) 2025, Sebastian Castro, Christian Henkel
# All rights reserved.

# This source code is licensed under the BSD 3-Clause License.
# See the LICENSE file in the project root for license information.

"""Evaluates a trained RL policy."""

import argparse

import rclpy
from gymnasium.spaces import Discrete
from rclpy.node import Node
from stable_baselines3.common.base_class import BaseAlgorithm

from pyrobosim_ros_gym import get_config
from pyrobosim_ros_gym.envs import available_envs_w_subtype, get_env_by_name
from pyrobosim_ros_gym.policies import ManualPolicy, model_and_env_type_from_path

MANUAL_STR = "manual"


def get_args() -> argparse.Namespace:
    """Helper function to parse the command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        help=f"The path of the model to evaluate. Can be '{MANUAL_STR}' for manual control.",
    )
    parser.add_argument(
        "--env",
        type=str,
        help=f"The name of the environment to use if '--model {MANUAL_STR}' is selected.",
        choices=available_envs_w_subtype(),
    )

    parser.add_argument(
        "--config",
        help="Path to the configuration YAML file.",
        required=True,
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

    # Ensure '--env' is provided if '--model' is 'manual'
    if args.model == MANUAL_STR and not args.env:
        parser.error(f"--env must be specified when --model is '{MANUAL_STR}'.")
    if args.env and args.model is None:
        print("--env is specified but --model is not. Defaulting to manual control.")
        args.model = MANUAL_STR
    return args


if __name__ == "__main__":
    args = get_args()
    config = get_config(args.config)

    rclpy.init()
    node = Node("pyrobosim_ros_env")

    # Load the model and environment
    model: BaseAlgorithm | ManualPolicy
    if args.model == MANUAL_STR:
        env = get_env_by_name(
            args.env,
            node,
            max_steps_per_episode=15,
            realtime=True,
            discrete_actions=True,
            reward_fn=config["training"].get("reward_fn"),
        )
        model = ManualPolicy(env.action_space)
    else:
        model, env_type = model_and_env_type_from_path(args.model)
        env = get_env_by_name(
            env_type,
            node,
            max_steps_per_episode=15,
            realtime=args.realtime,
            discrete_actions=isinstance(model.action_space, Discrete),
            reward_fn=config["training"].get("reward_fn"),
        )

    # Evaluate the model for some steps
    num_successful_episodes = 0
    reward_per_episode = [0.0 for _ in range(args.num_episodes)]
    custom_metrics_store: dict[str, list[float]] = {}
    custom_metrics_episode_mean: dict[str, float] = {}
    custom_metrics_per_episode: dict[str, list[float]] = {}
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
