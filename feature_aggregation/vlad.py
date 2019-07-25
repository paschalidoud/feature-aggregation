"""Quantize local features and aggregate them using the Vector of Locally
Aggregated Descriptors (VLAD) encoding"""

import numpy as np
from sklearn import cluster
from sklearn.metrics import pairwise_distances

from .base import BaseAggregator


class Vlad(BaseAggregator):
    """Compute a VLAD model and aggregate local features with it.

    Parameters
    ----------
    n_codewords: int
                 The codebook size aka the number of clusters.
    inner_batch: int
                 The batch size used to compute the differences between
                 the feature descriptors and the centroids.
    normalization: int
                   A bitmask of possible normalizations
    dimension_ordering : {'th', 'tf'}
                         Changes how n-dimensional arrays are reshaped to form
                         simple local feature matrices. 'th' ordering means the
                         local feature dimension is the second dimension and
                         'tf' means it is the last dimension.
    """


    POWER_NORMALIZATION = 1
    L2_NORMALIZATION = 2

    def __init__(self, n_codewords, normalization=3, inner_batch=128,
                 dimension_ordering="tf"):
        self.n_codewords = n_codewords
        self.inner_batch = inner_batch
        self.normalization = normalization

        self._clusterer = cluster.MiniBatchKMeans(
            n_clusters=self.n_codewords,
            n_init=1,
            compute_labels=False
        )

        super(self.__class__, self).__init__(dimension_ordering)

    def fit(self, X, y=None):
        """Build the codebook for the VLAD model using KMeans.

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

        words = self._clusterer.predict(X)
        dims = len(X[0])

        vlad = np.zeros((len(lengths), dims*self.n_codewords))
        v = np.zeros((self.inner_batch, self.n_codewords, dims))
        for i, (s, e) in enumerate(zip(starts, ends)):
            for j in range(s, e, self.inner_batch):
                ee = min(j+self.inner_batch, e)

                v.fill(0)
                v[range(ee-j), words[j:ee]] = \
                    X[j:ee] - self._clusterer.cluster_centers_[words[j:ee]]
                vlad[i] += v[:ee-j].sum(axis=0).ravel()
            vlad[i] /= lengths[i]
        
        # Check if we should be normalizing the power
        if self.normalization & self.POWER_NORMALIZATION:
            vlad = np.sqrt(np.abs(vlad))*np.sign(vlad)

        # Check if we should be performing L2 normalization
        if self.normalization & self.L2_NORMALIZATION:
            vlad /= np.sqrt(np.sum(vlad**2, axis=1)).reshape(-1, 1)

        return vlad
