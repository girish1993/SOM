### Self-Organizing Map (SOM)

This project implements a Self-Organizing Map (SOM), an unsupervised learning algorithm used for clustering and dimensionality reduction. The model maps high-dimensional input data onto a 2D grid while preserving the structure of the data.

During training, each input sample is matched to its Best Matching Unit (BMU), and the BMU along with its neighbouring neurons are updated using a Gaussian neighbourhood function. Both the learning rate and neighbourhood radius decay over time, allowing the model to transition from coarse global organization to fine-grained local tuning.

The implementation is vectorized for efficiency, supports configurable hyperparameters, and is structured for extensibility and reuse.