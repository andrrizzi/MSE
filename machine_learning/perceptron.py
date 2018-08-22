#!/usr/bin/env python

"""
A simple implementation of a perceptron in Python.

"""

import copy

import numpy as np


class Perceptron:
    """A perceptron classifier.

    Parameters
    ----------
    weights : array-like, optional
        The initial weights. If not specified, it will be set to the null
        vector of the same dimensionality of the first learned input vector.
    bias : float or None, optional
        The initial bias. If set to None, the bias will not be trained and
        the decision boundary will pass through the origin. If not specified,
        it will be set to 0.0.

    """

    def __init__(self, weights=None, bias=0.0):
        self.bias = bias
        self.weights = weights

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, new_weights):
        # Keep weights as a numpy array.
        if new_weights is None:
            self._weights = new_weights
        else:
            self._weights = np.array(new_weights)

    @property
    def bias(self):
        if self._bias is None:
            return 0.0
        return self._bias

    @bias.setter
    def bias(self, new_bias):
        self._bias = new_bias

    def classify(self, data_point):
        """Classify the data point.

        Parameters
        ----------
        data_point: array-like
            The input vector of the perceptron.

        Returns
        -------
        label : int
            The predicted class. Either 1, -1, or 0 if the point lies
            on the decision boundary.

        """
        # Convert data_point to a numpy array.
        data_point = np.array(data_point)
        label = np.sign(np.dot(data_point, self.weights) + self.bias)
        return int(label)

    def learn_sample(self, data_point, label):
        """Update decision boundary with this input-label pair.

        This has no effect if the point is already correctly classified.

        Parameters
        ----------
        data_point : list
            The input vector.
        label : int
            The class. Either 1 or -1.

        """
        # Initialize weights and bias if not done in the constructor.
        if self.weights is None:
            self.weights = [0.0 for _ in range(len(data_point))]

        # Convert input vector into a numpy array.
        data_point = np.array(data_point)

        # Check if we're currently misclassifying this sample.
        predicted_label = self.classify(data_point)
        if predicted_label == label:
            return

        # Perceptron learning algorithm.
        self.weights = self.weights + label*data_point
        if self._bias is not None:
            self.bias = self.bias + label

    def train(self, data_points, labels):
        """Iteratively learn the decision boundary from the samples until convergence.

        Parameters
        ----------
        data_points

        """
        # Iterate until convergence.
        old_bias = None
        old_weights = None
        while self.bias != old_bias or np.any(self.weights != old_weights):
            # Update old decision boundary before update.
            old_bias = self.bias
            old_weights = copy.deepcopy(self.weights)
            # One cycle learning from all data-label pairs.
            for data_point, label in zip(data_points, labels):
                self.learn_sample(data_point, label)


if __name__ == '__main__':
    # Example data.
    data_points = [
        [-1, 1],
        [0, -1],
        [10, 1]
    ]
    labels = [1, -1, 1]

    # Train perceptron
    perceptron = Perceptron()
    perceptron.train(data_points, labels)
    print('weights:', perceptron.weights)
    print('bias:', perceptron.bias)
