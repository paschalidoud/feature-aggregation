#!/usr/bin/env python
"""This module contains function and classes relevant with the computation of
a bag of words model. At start we suppose that suitable descriptors for our
dataset are already extracted. Subsequently we procceed to the second step of
quantization, in this step we use a clustering algorithm such as Kmeas to
create our visual vocabulary. At the end in the final step we represent all our
features according to the previously caclulated vocabulary.
"""

import numpy as np
from sklearn.cluster import KMeans


class Encoding:
    """This class is responsible for computing a Bag of Words model"""

    def __init__(self, n_codewords, iterations, clusterer=KMeans):
        """Initialize the class instance.

        Parameters:
        -----------
        n_codewords: int
                     The number of clusters to be created. Each cluster's
                     centroid corresponds to a codeword.
        iterations:  int
                     The maximum number of iterations performed by the
                     clusterer.
        clusterer:   callable
                     A callable that when given the number of clusters it
                     returns a clusterer that implements the fit and predict
                     method.
        """
        self.n_codewords = n_codewords
        self.iterations = iterations

        self._clusterer = clusterer(
            n_clusters=n_codewords,
            max_iter=iterations
        )

    @property
    def centroids(self):
        """The centroids of the encoding"""
        return self._clusterer.cluster_centers_.copy()

    @centroids.setter
    def centroids(self, centroids):
        self._clusterer.cluster_centers_ = centroids.copy()

    def fit(self, data):
        """Build a visual dictionary for the Bag of Words model.

        Apply a clustering algorithm to the data, the default option is Kmeans,
        in order to create a suitable visual vocabulary. If Kmeans is chosen,
        every centroid corresponds to a visual codeword of our vocabulary

        Parameters:
        -----------
        data: array_like
              Data of datapoints used to create visual vocabulary.
        """
        # Compute clustering
        self._clusterer.fit(data)

    def encode(self, data, density):
        """Encode a list of data using the learnt Bag of Words model

        Parameters:
        -----------
        data: array_like
              List of data points that will be encoded using the already
              computed Bag of Words model
        """
        # If there are no features for a specific video return a zero array
        if len(data) == 0:
            return np.zeros(self.n_codewords)

        # Represent each datapoint as histogram. When n_codewords is sequence
        # bins arguement corresponds to bin edges, this is the reason why we
        # add 1. Moreover we subtract 0.5 so that each bin's label is in the
        # middle of it's corresponding bin.
        hist, edges = np.histogram(
            self._clusterer.predict(data),
            bins=np.arange(self.n_codewords + 1) - .5,
            density=density
        )
        return hist
