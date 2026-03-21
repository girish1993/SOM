import os

import numpy as np

from src.io import parse_config, save_weights
from src.som import SOM

if __name__ == "__main__":
    config_file_name = "config.yaml"
    mapped_img_dir_name = "mapped_imgs"

    mapped_img_dir_path = os.path.join(os.getcwd(), mapped_img_dir_name)
    config_file_path = os.path.join(os.getcwd(), config_file_name)
    run_config = parse_config(file_path=config_file_path)

    input_data = np.random.random((10, 3))
    for i, config in enumerate(run_config.get("run_config")):
        som = SOM(**config)
        som.fit(input_data=input_data)
        save_weights(
            weights=som.weights,
            dir_path=mapped_img_dir_path,
            run_id=i + 1,
            iter=config.get("num_iterations"),
        )
