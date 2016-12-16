"""Aggregate local features using Fisher Vectors with a GMM as the
probabilistic model"""

import math
import numpy as np
from sklearn.mixture import GaussianMixture

from base import BaseAggregator


class FisherVectors(BaseAggregator):
    """Aggregate local features using Fisher Vector encoding with a GMM.

    Train a GMM on some local features and then extract the normalized
    derivative

    Parameters
    ----------
    n_gaussians : int
                  The number of gaussians to be used for the fisher vector
                  encoding
    max_iter : int
               The maximum number of EM iterations
    normalization : int
                    A bitmask of POWER_NORMALIZATION and L2_NORMALIZATION
    dimension_ordering : {'th', 'tf'}
                         Changes how n-dimensional arrays are reshaped to form
                         simple local feature matrices. 'th' ordering means the
                         local feature dimension is the second dimension and
                         'tf' means it is the last dimension.
    """

    POWER_NORMALIZATION = 1
    L2_NORMALIZATION = 2

    def __init__(self, n_gaussians, max_iter=100, normalization=3,
                 dimension_ordering="tf"):
        self.n_gaussians = n_gaussians
        self.max_iter = max_iter
        self.normalization = normalization

        super(self.__class__, self).__init__(dimension_ordering)

        # initialize the rest of the  attributes of the class for any use
        # (mainly because we want to be able to check if fit has been called
        # before on this instance)
        self.weights = None
        self.means = None
        self.covariances = None
        self.inverted_covariances = None
        self.inverted_covariances_sqrt = None
        self.inverted_covariances_3rd_power = None
        self.normalization_factor = None

    def __getstate__(self):
        """Return the data that should be pickled in order to save the fisher
        encoder after it is trained.

        This way allows us to control what is actually saved to disk and to
        recreate whatever cannot be saved like the probability density
        functions. Moreover we can choose if we want to trade between storage
        space and initialization time (currently maximum space is used).
        """
        # we could be simply grabing self.__dict__ removing pdfs and returning
        # it but I believe this is more explicit
        return {
            "n_gaussians": self.n_gaussians,
            "max_iter": self.max_iter,
            "normalization": self.normalization,
            "dimension_ordering": self.dimension_ordering,

            "weights": self.weights,
            "means": self.means,
            "covariances": self.covariances,
            "inverted_covariances": self.inverted_covariances,
            "normalization_factor": self.normalization_factor
        }

    def __setstate__(self, state):
        """Restore the class's state after unpickling.

        Parameters
        ----------
        state: dictionary
               The unpickled data that were returned by __getstate__
        """
        self.n_gaussians = state["n_gaussians"]
        self.max_iter = state["max_iter"]
        self.normalization = state["normalization"]
        self.dimension_ordering = state["dimension_ordering"]

        self.weights = state["weights"]
        self.means = state["means"]
        self.covariances = state["covariances"]
        self.inverted_covariances = state["inverted_covariances"]
        self.normalization_factor = state["normalization_factor"]

    def fit(self, X, y=None):
        """Learn a fisher vector encoding.

        Fit a gaussian mixture model to the data using n_gaussians with
        diagonal covariance matrices.

        Parameters
        ----------
        X : array_like or list
            The local features to train on. They must be either nd arrays or
            a list of nd arrays.
        """
        X, _ = self._reshape_local_features(X)

        # consider changing the initialization parameters
        gmm = GaussianMixture(
            n_components=self.n_gaussians,
            max_iter=self.max_iter,
            covariance_type='diag'
        )
        gmm.fit(X)

        # save the results of the gmm
        self.weights = gmm.weights_
        self.means = gmm.means_
        self.covariances = gmm.covariances_

        # precompute some values for encoding
        D = X[0].size
        self.inverted_covariances = (1./self.covariances)
        self.normalization_factor = np.hstack([
            np.repeat(1.0/np.sqrt(self.weights), D),
            np.repeat(1.0/np.sqrt(2*self.weights), D)
        ])

        return self

    def transform(self, X):
        """Compute the fisher vector implementation of the provided data.

        Parameters
        ----------
        X : array_like or list
            The local features to aggregate. They must be either nd arrays or
            a list of nd arrays. In case of a list each item is aggregated
            separately.
        """
        # Check if the GMM is fitted
        if self.weights is None:
            raise RuntimeError(
                "GMM model not found. Have you called fit(data) first?"
            )

        # Get the local features and the number of local features per document
        X, lengths = self._reshape_local_features(X)

        # Allocate the memory necessary for the encoded data
        fv = np.zeros((len(lengths), self.normalization_factor.shape[0]))

        # Do a naive double loop for now
        s, e = 0, 0
        for i, l in enumerate(lengths):
            s, e = e, e+l
            for j in xrange(s, e):
                self._encode_single(X[j], fv[i])


        # normalize the vectors
        fv *= 1.0/np.array(lengths).reshape(-1, 1)
        fv *= self.normalization_factor.reshape(1, -1)

        # check if we should be normalizing the power
        if self.normalization & self.POWER_NORMALIZATION:
            fv = np.sqrt(np.abs(fv))*np.sign(fv)

        # check if we should be performing L2 normalization
        if self.normalization & self.L2_NORMALIZATION:
            fv /= np.sqrt(np.einsum("...j,...j", fv, fv)).reshape(-1, 1)

        return fv

    def _encode_single(self, x, fisher):
        """Compute the grad with respect to the parameters of the model for the
        vector x and add it to fisher.

        see "Improving the Fisher Kernel for Large-Scale Image Classification"
        by Perronnin et al. for the equations

        Parameters
        ----------
        x: array
           The feature vector to be encoded with fisher encoding
        fisher : array
                 A target array to accumulate the fisher vector in
        """
        # number of gaussians
        N = self.n_gaussians

        # number of dimensions
        D = self.means.shape[1]

        # calculate the probabilities that x was created by each gaussian
        # distribution keeping some intermediate computations as well
        diff = x.reshape(1, -1) - self.means
        diff_over_cov = diff * self.inverted_covariances
        dist = -0.5 * (diff_over_cov * diff).sum(axis=1)
        q = self._softmax(dist).reshape(-1, 1)

        # Check if the probability vector q contains inf or nan
        if np.any(np.isinf(q)) or np.any(np.isnan(q)):
            return

        # derivative with respect to the means
        fisher[:N*D] += (q * diff_over_cov).ravel()
        # derivative with respect to the variances
        fisher[N*D:] += (q * (diff_over_cov ** 2 - 1)).ravel()

    @staticmethod
    def _softmax(x):
        a = np.exp(x - x.max())
        return a / a.sum()
