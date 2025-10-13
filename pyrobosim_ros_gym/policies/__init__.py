import os

from stable_baselines3 import DQN, PPO, SAC, A2C
from stable_baselines3.common.base_class import BaseAlgorithm


def model_and_env_type_from_path(model_path: str) -> tuple[BaseAlgorithm, str]:
    # Validate the file path to be of the right form
    assert os.path.isfile(model_path), f"Model {model_path} must be a valid file."
    model_fname = os.path.basename(model_path)
    model_name_parts = model_fname.split("_")
    assert (
        len(model_name_parts) >= 2
    ), f"Model name {model_fname} must be of the form <env>_<model>[_<otherinfo>].pt"
    env_type = model_name_parts[0]
    model_type = model_name_parts[1]

    # Load the model
    if model_type == "DQN":
        model: BaseAlgorithm = DQN.load(model_path, env=None)
    elif model_type == "PPO":
        model = PPO.load(model_path, env=None)
    elif model_type == "SAC":
        model = SAC.load(model_path, env=None)
    elif model_type == "A2C":
        model = A2C.load(model_path, env=None)
    else:
        raise RuntimeError(f"Invalid model type: {model_type}")

    return (model, env_type)
