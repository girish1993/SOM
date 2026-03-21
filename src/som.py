import logging
from typing import Optional, Tuple

import numpy as np

from src.helper import timer

# Create a module-level logger. The name is automatically set to the module name.
logger = logging.getLogger(__name__)


class SOM:
    def __init__(
        self,
        grid_width: int,
        grid_height: int,
        num_features: int,
        num_iterations: int = 1000,
        init_learning_rate: Optional[float] = 0.1,
        init_radius: Optional[float] = None,
        random_seed: Optional[int] = 42,
    ):
        self.width = grid_width
        self.height = grid_height
        self.num_features = num_features
        self.init_learning_rate = init_learning_rate
        self.num_iterations = num_iterations

        self.init_radius = (
            init_radius if init_radius else max(grid_width, grid_height) / 2
        )

        rng = np.random.default_rng(random_seed)
        self.weights: np.ndarray = rng.random((grid_width, grid_height, num_features))
        self.time_constant: float = num_iterations / np.log(self.init_radius)
        self.x_cords, self.y_cords = np.meshgrid(
            np.arange(self.width), np.arange(self.height)
        )

    def get_weights(self):
        return self.weights

    def _get_decayed_params(self, iter_num: int) -> Tuple[float, float]:
        updated_radius = self.init_radius * np.exp(-iter_num / self.time_constant)
        updated_learning_rate = self.init_learning_rate * np.exp(
            -iter_num / self.time_constant
        )
        return updated_radius, updated_learning_rate

    def _find_bmu(self, sample: np.ndarray) -> Tuple[int, int]:
        distances = np.sum((self.weights - sample) ** 2, axis=2)
        bmu_index = np.argmin(distances)
        return np.unravel_index(bmu_index, (self.width, self.height))

    def _find_neighbourhood(self, bmu_x: int, bmu_y: int, radius: float):
        squared_distance = (self.x_cords - bmu_x) ** 2 + (self.y_cords - bmu_y) ** 2
        return np.exp(-squared_distance / (2 * radius**2))

    def _update_weights(
        self, neighbourhood: np.ndarray, learning_rate: float, sample: np.ndarray
    ) -> None:
        self.weights += (
            learning_rate * neighbourhood[:, :, np.newaxis] * (sample - self.weights)
        )

    @timer
    def fit(self, input_data: np.ndarray):
        for t in range(self.num_iterations):
            radius_t, lr_t = self._get_decayed_params(iter_num=t)
            for sample in input_data:
                bmu_x, bmu_y = self._find_bmu(sample=sample)
                neighbourhood = self._find_neighbourhood(
                    bmu_x=bmu_x, bmu_y=bmu_y, radius=radius_t
                )
                self._update_weights(neighbourhood, lr_t, sample)
