# SciKit-Learn toy

This directory contains some SciKit-Learn estimators for [toy datasets](https://scikit-learn.org/stable/datasets/toy_dataset.html), such as [iris plants](https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-plants-dataset) and [diabetes dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-plants-dataset).

## Example

We have a iris plants dataset, from `sklearn.datasets` module. Since the dataset is so simple and the `RandomForestClassifier` estimator from `sklearn.ensemble` is very flexible, we get the perfect model.

By plotting the [ROC curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html) we can see that we get the perfect results for every class. Each class has its own plot, although they are all overlapping.

![ROC curve perfect](./plots/iris-model-roc-perfect.png)

We can also plot the confusion matrix, here we can also see the perfect results. No errors at all.

![Confusion matrix perfect](./plots/iris-model-confmat-perfect.png)