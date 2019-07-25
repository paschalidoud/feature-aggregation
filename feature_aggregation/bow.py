"""Quantize local features and aggregate them in a Bag Of Words manner"""

import numpy as np
from sklearn import cluster

from .base import BaseAggregator


class BagOfWords(BaseAggregator):
    """Compute a Bag of Words model and aggregate local features with it.
    
    Train a MiniBatchKMeans on the data and then use the centroids as a
    codebook to encode any set of local features.

    Parameters
    ----------
    n_codewords : int
                  The codebook size aka the number of clusters
    l1_norm : boolean
              Whether to normalize the transformed data or not
    dimension_ordering : {'th', 'tf'}
                         Changes how n-dimensional arrays are reshaped to form
                         simple local feature matrices. 'th' ordering means the
                         local feature dimension is the second dimension and
                         'tf' means it is the last dimension.
    """
    def __init__(self, n_codewords, l1_norm=True, dimension_ordering="tf"):
        self.n_codewords = n_codewords
        self.l1_norm = l1_norm
        self._clusterer = cluster.MiniBatchKMeans(
            n_clusters=self.n_codewords,
            n_init=1,
            compute_labels=False
        )

        super(self.__class__, self).__init__(dimension_ordering)

    @property
    def centroids(self):
        """The centroids of the encoding"""
        return self._clusterer.cluster_centers_.copy()

    @centroids.setter
    def centroids(self, _centroids):
        self._clusterer.cluster_centers_ = _centroids.copy()

    def fit(self, X, y=None):
        """Build the codebook for the Bag of Words model.

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
        """Compute the Bag of Words representation of the provided data.
        
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

        # Transform and aggregate the local features
        words = self._clusterer.predict(X)
        bow = np.vstack([
            np.histogram(
                words[s:e],
                bins=np.arange(self.n_codewords + 1) - 0.5,
                density=False
            )[0]
            for s, e in zip(starts, ends)
        ])

        if self.l1_norm:
            bow = bow.astype(float) / bow.sum(axis=1).reshape(-1, 1)

        return bow

    def inertia(self, X):
        """Return the value of the KMeans objective function on the provided
        data.

        X : array_like or list
            The local features to train on. They must be either nd arrays or
            a list of nd arrays.
        """
        X, _ = self._reshape_local_features(X)

        return -self._clusterer.score(X)

    def score(self, X, y=None):
        """Return the negative inertia so that the best score is the max
        score.

        X : array_like or list
            The local features to train on. They must be either nd arrays or
            a list of nd arrays.
        """
        return -self.inertia(X)
