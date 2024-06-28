# TensorFlow dots

This directory contains some TensorFlow classification models for "dots" datasets, such as circles and moons/spirals datasets.

## Dots model

We got us here some blobs again. Clusters of colored dots:

![Blobs dataset](./plots/blobs-dataset.png)

We just build a simple model, using `tf.keras.Sequential`. The model has 3 layers. Input layer with 2 neurons, hidden layer with 10 neurons, and an output layer with 5 neurons, corresponding with each class.

I used the `tf.keras.optimizers.Adam` optimizer and `tf.keras.losses.categorical_crossentropy` loss function, and trained it for 50 epochs. The resulting decision boundary plot looks like this:

![Trained model](./plots/blobs-model-linear.png)

Which means that it did well (:

## Circles model

This is an classic example of binary classification, which is very cool. Our dataset consists of a lot of dots, colored in red and blue, arranged in a circular shape.

![Circles dataset](./plots/circles-dataset.png)

I have built a TensorFlow model to classify the dots color based on it's position. Since our dataset is non-linear, I used the ReLU activation function in the hidden layer of the neural network, with 10 neurons. I have plotted the decision boundary and all the training statistics on one graph:

![Circles model stats](./plots/circles-model-all.png)

As you can see by the graphs on the right. The loss value goes down over the iterations, and accuracy goes up with the training time. During training, on last epoch TensorFlow wrote `accuracy: 0.9896 - loss: 0.2139` which means our model performed at `98.96%` accuracy on the training data. Which is good!

## Tech stack

- **TensorFlow** - Main neural network tool
- **SciKit-Learn** - Generating datasets
- **Matplotlib** - Visualizing dataDecision