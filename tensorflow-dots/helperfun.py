# Helper functions

import torch
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def plot_decision_boundary(model: tf.keras.Sequential, X: np.ndarray, y: np.ndarray):    
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    X_to_pred_on = np.column_stack((xx.ravel(), yy.ravel()))
    y_logits = torch.from_numpy(model.predict(X_to_pred_on))

    y_pred = y_logits.argmax(dim=1)

    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.style.use("seaborn-v0_8-darkgrid")
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())