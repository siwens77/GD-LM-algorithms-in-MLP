<h1 align="center">Gradient Descent and Levenberg-Marquardt in Pytorch</h1>

Two MLP learning models created in PyTorch using materials provided by my professor. They both classify 2D points into classes and are visualized using `matplotlib` by loss history and model prediction. On the visualization, the points are a representation of the datasets and their color is the correct class. The background is my model's prediction. Plots can be seen in [plots-directory](./plots/). Almost all the coding was done by hand, but I used an LLM to help me visualize my results.

**Gradient-Descent**

* Optimizer: Stochastic Gradient Descent with a learning rate of 0.01 and momentum of 0.9
* Output: Plots the training and validating loss history (Mean Squared Error) curve to show convergence over the epochs and classification

**Levenberg-Marquardt**

* Optimizer: Levenberg-Marquardt algorithm (utilizing the torch_levenberg_marquardt library)
* Output: Generates a 2D mesh grid plot showing the complex decision boundaries learned by the network.
