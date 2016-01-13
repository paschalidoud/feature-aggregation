#!/usr/bin/env python
"""This module contains function and classes relevant with the computation of
a bag of words model. At start we suppose that suitable descriptors for our
dataset are already extracted. Subsequently we procceed to the second step of
quantization, in this step we use a clustering algorithm such as Kmeas to
create our visual vocabulary. At the end in the final step we represent all our
features according to the previously caclulated vocabulary.
"""

import numpy as np
from sklearn import cluster
from sklearn.base import BaseEstimator
from sklearn.metrics import pairwise_distances_argmin_min

from common.functional import partial, ___


def _mini_batch_kmeans_factory(n_clusters=None, max_iter=None):
    return cluster.MiniBatchKMeans(
        n_clusters=n_clusters,
        max_iter=max_iter,
        n_init=1,
        batch_size=max(100, 5*n_clusters),
    )


class Encoding(BaseEstimator):
    """This class is responsible for computing a Bag of Words model"""

    _clusterers = {
        "kmeans": partial(cluster.KMeans, n_init=1),
        "fast_kmeans": _mini_batch_kmeans_factory
    }

    def __init__(self, n_codewords, iterations, clusterer="fast_kmeans"):
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
        self.clusterer = clusterer

        self._clusterer = self._clusterers[clusterer](
            n_clusters=n_codewords,
            max_iter=iterations
        )

    @property
    def centroids(self):
        """The centroids of the encoding"""
        return self._clusterer.cluster_centers_.copy()

    @centroids.setter
    def centroids(self, _centroids):
        self._clusterer.cluster_centers_ = _centroids.copy()

    def fit(self, data, y=None):
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

        return self

    def partial_fit(self, data, y=None):
        """Partially learn the visual dictionary from the provided data.

        This allows us to learn from data that do not fit in memory.

        Parameters:
        -----------
        data: array_like
              The batch to partially learn the vocabulary from
        """
        self._clusterer.partial_fit(data)

        return self

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

    def transform(self, data, y=None):
        """Transforms the data according to the computed model. This function
        basically creates a sequence of symbols from the computed data

        Parameters:
        -----------
        data: array_like
              List of data points that will be encoded using the already
              computed Bag of Words model
        """
        # If there are no features for a specific video return a zero array
        if len(data) == 0:
            return np.array([])

        return self._clusterer.predict(data)

    def inertia(self, data):
        """Return the value of the KMeans objective function on the provided
        data"""
        centroids = self.centroids

        return pairwise_distances_argmin_min(
            data,
            centroids,
            metric='sqeuclidean'
        )[1].sum()

    def score(self, data, y=None):
        """Return the negative inertia so that the best score is the max
        score"""
        return -self.inertia(data)
