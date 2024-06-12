# SciKit-Learn toy

This directory contains some SciKit-Learn estimators for [toy datasets](https://scikit-learn.org/stable/datasets/toy_dataset.html), such as [iris plants](https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-plants-dataset) and [diabetes dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-plants-dataset).

## Example

We have a iris plants dataset, from `sklearn.datasets` module. Since the dataset is so simple and the `RandomForestClassifier` estimator from `sklearn.ensemble` is very flexible, we get the perfect model.

By plotting the [ROC curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html) we can see that we get the perfect results for every class. Each class has its own plot, although they are all overlapping.

![ROC curve perfect](./plots/iris-model-roc-perfect.png)

We can also plot the confusion matrix, here we can also see the perfect results. No errors at all.

![Confusion matrix perfect](./plots/iris-model-confmat-perfect.png)

## Handwritten digits classification

Here's a classic ML exercise. Build a model that can classify images of handwritten digits. The model should be able to recognise monochrome pixel values which make up a handwritten number. Here's a sample from SciKit-Learn's digits dataset:

![Digits dataset sample](./plots/digits-dataset-sample.png)

I used Matplotlib to plot the pixel values to a grid, so that we can visualize the data. What we get is a rough representation of a handwritten digit. It looks like that because its an 8 by 8 image, that means it has only 64 pixels. There's not a lot you can do with just 64 pixels.

Regardless, our trained model was able to classify the images with *94.00%* accuracy, and also get a pretty good looking confusion matrix:

![Digits model confusion matrix](./plots/digits-model-confmat.png)

What this confusion matrix is show, is that our model is predicting the true label almost every time. That is why we can see a sort of diagonal line.