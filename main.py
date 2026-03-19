from src.som import SOM
import numpy as np

if __name__ == "__main__":
    som = SOM(width=100, height=100, num_features=3)
    input_data = np.random.random((10, 3))
    som.fit(input_data=input_data)
    print(som.get_weights())
