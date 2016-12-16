
import numpy as np
from sklearn.base import BaseEstimator


class BaseAggregator(BaseEstimator):
    """Implement any functions that can be shared among all feature
    aggregation methods."""

    def __init__(self, dimension_ordering="tf"):
        self.dimension_ordering = dimension_ordering

    def _reshape_local_features(self, X):
        """Reshape a n-dimensional array into a 2d array of local features.

        Account for the case that X is a list because not all samples have
        the same number of local features.
        """
        if len(X) == 0:
            raise ValueError("X cannot be empty")

        dims = len(X[0]) if self.dimension_ordering == "th" else X[0].shape[-1]
        if isinstance(X, list):
            arrays = [
                x.T.reshape(-1, dims)
                if self.dimension_ordering == "th"
                else x.reshape(-1, dims)
                for x in X
            ]
            lengths = [len(x) for x in arrays]
            X = np.vstack(arrays)
        else:
            if self.dimension_ordering == "th":
                X = X.transpose(*([0] + range(2, len(X.shape)) + [1]))
            lengths = [int(np.prod(X.shape[1:-1]))]*X.shape[0]
            X = X.reshape(-1, dims)

        return X, lengths
