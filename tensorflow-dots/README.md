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

## Moons model

This is an another binary classification problem. Here we have, once again, a non-linear dataset from SciKit-learn's toy datasets. A moons dataset, or as I like to call it, spirals! The graph looks like this:

![Moons dataset](./plots/moons-dataset.png)

I built a simple TensorFlow sequential model which consists of four layers. The hidden layers both have 10 neurons, so 20 hidden neurons total, the first hidden layers uses the SoftPlus activation function, and the second hidden layer uses the leaky ReLU activation function. Finally we have the Sigmoid activation function at the output. The training statistics and the decision boundary plots looks like this:

![Moons model stats](./plots/moons-model-all.png)

The model was trained on only 50 epochs and it is very lightweight. The results are as expected, not perfect, but good enough.

## Spirals model

This is a multiclass classification problem. A more complex version of the moons dataset, I call it **the dataway galaxy**. And the plot looks like this:

![Spirals dataset](./plots/spirals-dataset.png)

I have built a non-linear model with TensorFlow, which consists of 5 layers. It has 3 hidden layers with 10 neurons each and SoftMax, SoftPlus and leaky ReLU activation functions respectively. I trained the model for 100 iterations and the results were pretty good.

![Spirals model stats](./plots/spirals-model-all.png)

Although it still makes a lot of mistakes. It has succeeded to fit the spiral shaped pattern. Which is also good!

## Tech stack

- **TensorFlow** - Main neural network tool
- **SciKit-Learn** - Generating datasets
- **Matplotlib** - Visualizing dataDecision