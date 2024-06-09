# PyTorch dots

This directory contains some PyTorch classification models for "dots" datasets, such as circles and moons/spirals datasets.

## Example

We have a dataset of circles, generated with SciKit-Learn's make_circles function from sklearn.datasets.

![Dataset](./plots/circles-dataset.png)

Before training, we can plot the decision boundary using our helper function plot_decision_boundary.

![Untrained model](./plots/circles-model-untrained.png)

What we got is a randomly generated model's predictions. Basically guessing. To get our desired outcome, we can train it to classify the dots. After training, we get this:

![Trained model](./plots/circles-model-trained.png)

The model is very lightweight, with only 2 linear layers and 1 ReLU layer. Trained with only 500 epochs.

## Overfitting example

`moons-overfit.ipynb` contains an example of overfitting a model. I used a larger and more flexible neural network, trained it for much longer and adjusted learning rate to minimize loss on the training data. What we got is a bit unusual decision boundary plot, with sharp lines and curves.

![Overfitted model](./plots/moons-model-overfit.png)

The accuracy on the training data is *99.50%* which is very good. Although the accuracy on the testing data is *98.00%*. The accuracy on the testing data is not that bad, but consider that the dataset is very simple. Avoiding overfitting we could have gotten much higher testing accuracy.

This is what a good, trained model's decision boundary should have looked like:

![Trained model](./plots/moons-model-trained.png)

It has much smoother lines, much lighter neural network, with less layers and flexibility. Even though the accuracy on the training dataset is *98.62%*, we still manage to get *99.00%* accuracy on the testing dataset.

## Multi-class classification example

This one is fun. We got us here some blobs. Clusters of colored dots:

![Blobs dataset](./plots/blobs-dataset.png)

Now, we can fit this data to a linear model. But it will struggle a bit since we have 5 classes: purple, blue, green, yellow and red. This is what we get when fitting this data to a linear model:

![Linear blobs model](./plots/blobs-model-linear.png)

Kinda messy. Although it was trained with 1000 epochs. But trust me, it doesn't get any better with more epochs. Anyway... I made a non-linear model. I trained it for 5000 epochs and got this:

![Non-linear blobs model](./plots/blobs-model-non-linear.png)

And this is slightly better. Although considering the complexity of this non-linear model, I would say its not worthit to use such model for this task, stick to linear, simple models for simple tasks.

## Tech stack

- **PyTorch** - Main neural network tool
- **SciKit-Learn** - Generating datasets
- **Matplotlib** - Visualizing data