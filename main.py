import logging
import os

import numpy as np

from src.io import parse_config, save_weights
from src.som import SOM

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    CONFIG_FILE_NAME = "config.yaml"
    MAPPED_IMG_DIR_NAME = "mapped_imgs"

    mapped_img_dir_path = os.path.join(os.getcwd(), MAPPED_IMG_DIR_NAME)
    config_file_path = os.path.join(os.getcwd(), CONFIG_FILE_NAME)
    run_config = parse_config(file_path=config_file_path)

    input_data = np.random.random((10, 3))
    for i, config in enumerate(run_config.get("run_config")):
        logger.info(
            f"Starting run {i + 1} with grid size ({config.get('grid_width')}, {config.get('grid_height')}) with {config.get('num_iterations')} iterations"
        )
        som = SOM(**config)
        som.fit(input_data=input_data)
        save_weights(
            weights=som.weights,
            dir_path=mapped_img_dir_path,
            run_id=i + 1,
            iter=config.get("num_iterations"),
        )
