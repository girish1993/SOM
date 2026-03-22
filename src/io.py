import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import yaml


def parse_config(file_path: str) -> Dict:
    """Parse a yaml file that contains the run config

    Args:
        file_path (str): path to the config file

    Raises:
        FileNotFoundError: If file not found in the said path

    Returns:
        Dict: parsed yaml config
    """
    try:
        with open(file_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError("e")


def save_weights(weights: np.ndarray, dir_path: str, run_id: int, iter: int):
    """Method to write the updated weights as an image under the directory

    Args:
        weights (np.ndarray): The updated weights
        dir_path (str): directory path to save the images under
        run_id (int): id that represents the run number in case of multiple run configs
        iter (int): number of iterations to distinguish between runs
    """
    os.makedirs(dir_path, exist_ok=True)
    target_file_name = f"run_{run_id}_{iter}.png"

    complete_path = os.path.join(dir_path, target_file_name)

    plt.imsave(complete_path, weights)
