#!/usr/bin/env python3

# Copyright (c) 2025, Sebastian Castro, Christian Henkel
# All rights reserved.

# This source code is licensed under the BSD 3-Clause License.
# See the LICENSE file in the project root for license information.

"""Trains an RL policy."""

import argparse
from datetime import datetime

import rclpy
from rclpy.node import Node
from stable_baselines3 import DQN, PPO, SAC, A2C
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.base_class import BaseAlgorithm

from pyrobosim_ros_gym import get_config
from pyrobosim_ros_gym.envs import get_env_by_name, available_envs_w_subtype


def get_args() -> argparse.Namespace:
    """Helper function to parse the command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        choices=available_envs_w_subtype(),
        help="The environment to use.",
        required=True,
    )
    parser.add_argument(
        "--config",
        help="Path to the configuration YAML file.",
        required=True,
    )
    parser.add_argument(
        "--algorithm",
        default="DQN",
        choices=["DQN", "PPO", "SAC", "A2C"],
        help="The algorithm with which to train a model.",
    )
    parser.add_argument(
        "--discrete-actions",
        action="store_true",
        help="If true, uses discrete action space. Otherwise, uses continuous action space.",
    )
    parser.add_argument("--seed", default=42, type=int, help="The RNG seed to use.")
    parser.add_argument(
        "--realtime", action="store_true", help="If true, slows down to real time."
    )
    parser.add_argument(
        "--log",
        default=True,
        action="store_true",
        help="If true, logs data to Tensorboard.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    config = get_config(args.config)

    # Create the environment
    rclpy.init()
    node = Node("pyrobosim_ros_env")
    env = get_env_by_name(
        args.env,
        node,
        max_steps_per_episode=25,
        realtime=args.realtime,
        discrete_actions=args.discrete_actions,
        reward_fn=config["training"].get("reward_fn"),
    )

    # Train a model
    log_path = "train_logs" if args.log else None
    if args.algorithm == "DQN":
        dqn_config = config.get("training", {}).get("DQN", {})
        if "policy_kwargs" in dqn_config:
            policy_kwargs = dqn_config["policy_kwargs"]
            del dqn_config["policy_kwargs"]
        model: BaseAlgorithm = DQN(
            "MlpPolicy",
            env=env,
            seed=args.seed,
            tensorboard_log=log_path,
            **dqn_config,
        )
    elif args.algorithm == "PPO":
        ppo_config = config.get("training", {}).get("PPO", {})
        if "policy_kwargs" in ppo_config:
            policy_kwargs = ppo_config["policy_kwargs"]
            del ppo_config["policy_kwargs"]
        model = PPO(
            "MlpPolicy",
            env=env,
            seed=args.seed,
            tensorboard_log=log_path,
            **ppo_config,
        )
    elif args.algorithm == "SAC":
        sac_config = config.get("training", {}).get("SAC", {})
        if "policy_kwargs" in sac_config:
            policy_kwargs = sac_config["policy_kwargs"]
            del sac_config["policy_kwargs"]
        model = SAC(
            "MlpPolicy",
            env=env,
            seed=args.seed,
            tensorboard_log=log_path,
            **sac_config,
        )
    elif args.algorithm == "A2C":
        a2c_config = config.get("training", {}).get("A2C", {})
        if "policy_kwargs" in a2c_config:
            policy_kwargs = a2c_config["policy_kwargs"]
            del a2c_config["policy_kwargs"]
        model = A2C(
            "MlpPolicy",
            env=env,
            seed=args.seed,
            tensorboard_log=log_path,
            **a2c_config,
        )
    else:
        raise RuntimeError(f"Invalid algorithm type: {args.algorithm}")
    print(f"\nTraining with {args.algorithm}...\n")

    # Train the model until it exceeds a specified reward threshold in evals.
    training_config = config["training"]
    callback_on_best = StopTrainingOnRewardThreshold(
        reward_threshold=training_config["eval"]["reward_threshold"],
        verbose=1,
    )
    eval_callback = EvalCallback(
        env,
        callback_on_new_best=callback_on_best,
        verbose=1,
        eval_freq=training_config["eval"]["eval_freq"],
        n_eval_episodes=training_config["eval"]["n_eval_episodes"],
    )

    date_str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    log_name = f"{args.env}_{args.algorithm}_seed{args.seed}_{date_str}"
    model.learn(
        total_timesteps=training_config["max_training_steps"],
        progress_bar=True,
        tb_log_name=log_name,
        log_interval=1,
        callback=eval_callback,
    )

    # Save the trained model
    model_name = f"{log_name}.pt"
    model.save(model_name)
    print(f"\nSaved model to {model_name}\n")

    rclpy.shutdown()
