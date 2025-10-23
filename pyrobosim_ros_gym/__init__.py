#!/usr/bin/env python3

# Copyright (c) 2025, Sebastian Castro, Christian Henkel
# All rights reserved.

# This source code is licensed under the BSD 3-Clause License.
# See the LICENSE file in the project root for license information.

import importlib
import os
from typing import Any

import yaml


def get_config(config_path: str) -> dict[str, Any]:
    """Helper function to parse the configuration YAML file."""
    if not os.path.isabs(config_path):
        default_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "config"
        )
        config_path = os.path.join(default_path, config_path)
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Handle special case of reward function
    training_args = config.get("training", {})
    if "reward_fn" in training_args:
        module_name, function_name = training_args["reward_fn"].rsplit(".", 1)
        module = importlib.import_module(module_name)
        training_args["reward_fn"] = getattr(module, function_name)

    # Handle special case of policy_kwargs activation function needing to be a class instance.
    for subtype in training_args:
        subtype_config = training_args[subtype]
        if not isinstance(subtype_config, dict):
            continue
        policy_kwargs = subtype_config.get("policy_kwargs", {})
        if "activation_fn" in policy_kwargs:
            module_name, class_name = policy_kwargs["activation_fn"].rsplit(".", 1)
            module = importlib.import_module(module_name)
            policy_kwargs["activation_fn"] = getattr(module, class_name)

    return config
