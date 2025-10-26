#!/usr/bin/env python3

# Copyright (c) 2025, Sebastian Castro, Christian Henkel
# All rights reserved.

# This source code is licensed under the BSD 3-Clause License.
# See the LICENSE file in the project root for license information.

import os

from gymnasium import Space
from stable_baselines3 import DQN, PPO, SAC, A2C
from stable_baselines3.common.base_class import BaseAlgorithm


AVAILABLE_POLICIES = {alg.__name__: alg for alg in (DQN, PPO, SAC, A2C)}


class ManualPolicy:
    """A policy that allows manual keyboard control of the robot."""

    def __init__(self, action_space: Space):
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
                    print(f"Action {action} not in {possible_actions}.")
            except ValueError:
                print("Invalid input, please enter an integer.")


def model_and_env_type_from_path(model_path: str) -> tuple[BaseAlgorithm, str]:
    """
    Loads a model and its corresponding environment type from its file path.

    The models are of the form <env>_<model>[_<otherinfo>].pt.
    For example, path/to/model/GreenhousePlain_DQN_seed42_2025_10_18_18_02_21.pt.
    """
    # Validate the file path to be of the right form
    assert os.path.isfile(model_path), f"Model {model_path} must be a valid file."
    model_fname = os.path.basename(model_path)
    model_name_parts = model_fname.split("_")
    assert (
        len(model_name_parts) >= 2
    ), f"Model name {model_fname} must be of the form <env>_<model>[_<otherinfo>].pt"
    env_type = model_name_parts[0]
    algorithm = model_name_parts[1]

    # Load the model
    if algorithm in AVAILABLE_POLICIES:
        model_class = AVAILABLE_POLICIES[algorithm]
        model = model_class.load(model_path)
    else:
        raise RuntimeError(f"Invalid algorithm type: {algorithm}")

    return (model, env_type)
