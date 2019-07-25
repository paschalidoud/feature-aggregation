"""Aggregate local features using Locality-constrained Linear Coding"""

import numpy as np
from sklearn import cluster
from sklearn.metrics import pairwise_distances

from .base import BaseAggregator


class LLC(BaseAggregator):
    """Compute a Locality-constrained Linear Coding and aggregate local
    features with it.
    
    Parameters
    ----------
    n_codewords : int
                  The codebook size aka the number of clusters
    dimension_ordering : {'th', 'tf'}
                         Changes how n-dimensional arrays are reshaped to form
                         simple local feature matrices. 'th' ordering means the
                         local feature dimension is the second dimension and
                         'tf' means it is the last dimension.
    """
    def __init__(self, n_codewords, neighbors=5, beta=1e-4, dimension_ordering="tf"):
        self.n_codewords = n_codewords
        self.neighbors = neighbors
        self.beta = beta
        self._clusterer = cluster.MiniBatchKMeans(
            n_clusters=self.n_codewords,
            n_init=1,
            compute_labels=False
        )

        super(self.__class__, self).__init__(dimension_ordering)

    def fit(self, X, y=None):
        """Build the codebook for the LLC model.

        Apply the clustering algorithm to the data and use the cluster centers
        as codewords for the codebook.

        Parameters:
        -----------
        X : array_like or list
            The local features to train on. They must be either nd arrays or
            a list of nd arrays.
        """
        X, _ = self._reshape_local_features(X)

        self._clusterer.fit(X)

        return self

    def partial_fit(self, X, y=None):
        """Partially learn the codebook from the provided data.

        Run a single iteration of the minibatch KMeans on the provided data.

        Parameters:
        -----------
        X : array_like or list
            The local features to train on. They must be either nd arrays or
            a list of nd arrays.
        """
        X, _ = self._reshape_local_features(X)
        self._clusterer.partial_fit(X)

        return self

    def transform(self, X):
        """Compute the LLC representation of the provided data.

        Parameters
        ----------
        X : array_like or list
            The local features to aggregate. They must be either nd arrays or
            a list of nd arrays. In case of a list each item is aggregated
            separately.
        """
        # Get the local features and the number of local features per document
        X, lengths = self._reshape_local_features(X)

        # Preprocess the lengths list into indexes in the local feature array
        starts = np.cumsum([0] + lengths).astype(int)
        ends = np.cumsum(lengths).astype(int)

        # Calculate the nearest neighbors
        centroids = self._clusterer.cluster_centers_
        distances = pairwise_distances(X, centroids)
        K = self.neighbors
        neighbors = np.argpartition(distances, K)[:, :K]

        # Compute the llc representation
        llc = np.zeros((len(lengths), self.n_codewords))
        L2 = self.beta * np.eye(X.shape[1])
        for i, (s, e) in enumerate(zip(starts, ends)):
            for j in range(s, e):
                # a = argmin_{1^T a = 1} ||x - Ca||_2^2 + \beta ||a||_2^2
                C = centroids[neighbors[j]]
                a = C.dot(np.linalg.inv(C.T.dot(C) + L2)).dot(X[j])
                llc[i, neighbors[j]] = np.maximum(
                    llc[i, neighbors[j]],
                    a / a.sum()
                )

        return llc
