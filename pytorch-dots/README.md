# PyTorch dots

This is a directory with some PyTorch classification models for "dots" datasets, for example circles and moons/spirals datasets.

## Example

We have a dataset of circles, we generated it with SciKit-Learn's `make_circles` function from `sklearn.datasets`.

![Dataset](./plots/circles-dataset.png)

Before training we can plot the decision boundary using our helper function `plot_decision_boundary`.

![Untrained model](./plots/circles-model-untrained.png)

What we got is a randomly generated model's predictions. Basically guessing. To get our desired outcome we can train it to classify the dots. After training we get this:

![Trained model](./plots/circles-model-trained.png)

Model is very lightweight. Only 2 linear layers and 1 ReLU layer. Trained with only 500 epochs.

## Tech stack

- **PyTorch** - Main neural network tool
- **SciKit-Learn** - Generating datasets
- **Matplotlib** - Visualizing data