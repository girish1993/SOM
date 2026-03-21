import yaml
from typing import Dict
import numpy as np
import os
import matplotlib.pyplot as plt


def parse_config(file_path: str) -> Dict:
    try:
        with open(file_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError("e")


def save_weights(weights: np.ndarray, dir_path: str, run_id: int, iter: int):
    os.makedirs(dir_path, exist_ok=True)
    target_file_name = f"run_{run_id}_{iter}.png"

    complete_path = os.path.join(dir_path, target_file_name)

    plt.imsave(complete_path, weights)
