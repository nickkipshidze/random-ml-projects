# PyTorch vision

Computer vision solutions with PyTorch. Using multi-layer perceptrons, convolutional neural networks and their different kinds of architectures.

This directory, to me, is a very interesting one. Which is the reason why I put so many comments and headers in my notebooks. I hope they are clear and communicative.

## Multi-layer perceptron

I've went ahead and built a multi-layer perceptron model, which basically means a linear model, to classify handwritten digits from the infamous MNIST dataset. Here's a sample from the dataset:

![MNIST sample](./plots/mnist-sample.png)

The model I build has 2 linear layers, an input layer of 784 neurons, and an output layer of 10 neurons, each for their own class, which is a number, we are classifying numbers, from 0 to 9. I've also made a pretty visualization too using this very interesting and useful website called [NN-SVG](https://alexlenail.me/NN-SVG/index.html).

![Visual MLP](./plots/mlp-graph-mnist.png)

I've truncated the input layer, so that you don't have to see all of the 784 neurons. No need to thank me. The trained model can classify a handwritten number with *93.75%* accuracy. Which is very good for a linear neural network.

## Non-linear neural network

I have built a similar neural network to the multi-layer perceptron to classify handwritten digits from the MNIST dataset, but I also threw in some ReLU activation functions. Although the result is disappointing. Obviously this dataset doesn't require nonlinearity, because of that, the non-linear model performs just as accurately as the linear mode. With **93.75%** accuracy.

I also made this simple visual representation of the neural network.

![Visual non-linear](./plots/non-linear-graph-mnist.png)

## Fashion MNIST model

Fashion MNIST is just like the handwritten MNIST numbers dataset, but instead of numbers, we got clothes. Here's a sample from the dataset:

![Fashion MNIST sample](./plots/fashion-sample.png)

As you can see, there is a lot of clothes. It would be very hard and demanding to use a linear MLP for this task. So instead I choose to use a CNN. A convolutional neural network. I used TinyVGG architecture as a reference and built a model that can classify with *88.43%* accuracy. Which is not that bad.

I also made a confusion matrix plot for the model and here's how it is:

![Fashion MNIST model confusion matrix](./plots/fashion-model-confmat.png)

Class names on the x-axis are overlapping... but blame SciKit-learn for that, I used their helper function.

## Tech stack

- **PyTorch** - Main neural network tool
- **SciKit-Learn** - Helper functions
- **Matplotlib** - Visualizing data