import logging
from typing import Optional, Tuple

import numpy as np

from src.helper import timer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class SOM:
    """Represents SOM and its behaviours.

    This class implements methods similar to scikit-learns fit() method. transform() and fit_transform()
    are not implemented to maintain simplicity.
    """

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

        # setting a seed to ensure reproducibility between runs.
        rng = np.random.default_rng(random_seed)
        self.weights: np.ndarray = rng.random((grid_width, grid_height, num_features))
        self.time_constant: float = num_iterations / np.log(self.init_radius)
        self.x_cords, self.y_cords = np.meshgrid(
            np.arange(self.width), np.arange(self.height)
        )

    def get_weights(self) -> np.ndarray:
        """Get weights of the instance.

        Returns:
            np.ndarray: The weights of the instance
        """
        return self.weights

    def _get_decayed_params(self, iter_num: int) -> Tuple[float, float]:
        """Computes the exponentially decayed neighbourhood radius and learning rate for a given iteration

        Args:
            iter_num (int): The current iteration number

        Returns:
            Tuple[float, float]:
                - updated_radius : the decayed neighbourhood radius
                - learnning_rate: the decayed learning rate
        """
        updated_radius = self.init_radius * np.exp(-iter_num / self.time_constant)
        updated_learning_rate = self.init_learning_rate * np.exp(
            -iter_num / self.time_constant
        )
        return updated_radius, updated_learning_rate

    def _find_bmu(self, sample: np.ndarray) -> Tuple[int, int]:
        """Computes the best matching unit(BMU) for a given input sample vector

        - calculate the eucledian distance between the input sample vector and the weights
        - find the index of the weights vector that is closest to the input sample vector
        - Get the x,y coord of the closest matching node

        Args:
            sample (np.ndarray): The sample input vector to compute the closest node against.

        Returns:
            Tuple[int, int]: (x, y) coordinates of the closest matching node in the given grid.
        """
        distances = np.sum((self.weights - sample) ** 2, axis=2)
        bmu_index = np.argmin(distances)
        return np.unravel_index(bmu_index, (self.width, self.height))

    def _find_neighbourhood(self, bmu_x: int, bmu_y: int, radius: float):
        """Compute the neighborhood influence of the BMU over the grid.

        Args:
            bmu_x (int): BMU x-coordinate.
            bmu_y (int): BMU y-coordinate.
            radius (float): Neighborhood radius (sigma).

        Returns:
             np.ndarray: 2D array of neighborhood influence values.
        """
        squared_distance = (self.x_cords - bmu_x) ** 2 + (self.y_cords - bmu_y) ** 2
        return np.exp(-squared_distance / (2 * radius**2))

    def _update_weights(
        self, neighbourhood: np.ndarray, learning_rate: float, sample: np.ndarray
    ) -> None:
        """
        Update SOM weights using the BMU neighbourhood influence.

        Args:
            neighbourhood (np.ndarray): 2D array of neighbourhood influence values.
            learning_rate (float): Current learning rate.
            sample (np.ndarray): Input sample vector.

        Notes:
            - The neighbourhood matrix is expanded to (w, h, 1) to broadcast across

        """
        self.weights += (
            learning_rate * neighbourhood[:, :, np.newaxis] * (sample - self.weights)
        )

    @timer
    def fit(self, input_data: np.ndarray):
        """
        Train the Self-Organizing Map (SOM) on the given input data.

        Iteratively updates the weight grid by finding the Best Matching Unit (BMU)
        for each sample and adjusting neighbouring neurons using a decaying
        learning rate and radius.

        Args:
            input_data (np.ndarray): Training data of shape (n_samples, input_dim).

        Notes:
            - Learning rate and neighbourhood radius decay over time.
            - Logs training progress at regular intervals.
        """
        if input_data.ndim != 2:
            raise ValueError(f"input data must be 2-D, got {input_data.shape}")

        for t in range(self.num_iterations):
            radius_t, lr_t = self._get_decayed_params(iter_num=t)

            if t % 50 == 0:
                logger.info(
                    f"Iteration = {t} / learning_rate = {lr_t}  / radius= {radius_t}"
                )
            for sample in input_data:
                bmu_x, bmu_y = self._find_bmu(sample=sample)
                neighbourhood = self._find_neighbourhood(
                    bmu_x=bmu_x, bmu_y=bmu_y, radius=radius_t
                )
                self._update_weights(neighbourhood, lr_t, sample)
