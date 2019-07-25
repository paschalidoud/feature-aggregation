"""Aggregate local features using Fisher Vectors with a GMM as the
probabilistic model"""

from joblib import Parallel, delayed
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

from .base import BaseAggregator


def _transform_batch(x, means, inv_covariances, inv_sqrt_covariances):
    """Compute the grad with respect to the parameters of the model for the
    each vector in the matrix x and return the sum.

    see "Improving the Fisher Kernel for Large-Scale Image Classification"
    by Perronnin et al. for the equations

    Parameters
    ----------
    x: array
       The feature matrix to be encoded with fisher encoding
    means: array
           The GMM means
    inverted_covariances: array
                          The inverse diagonal covariance matrix

    Return
    ------
    vector The fisher vector for the passed in local features
    """
    # number of gaussians
    N, D = means.shape

    # number of dimensions
    M, D = x.shape

    # calculate the probabilities that each x was created by each gaussian
    # distribution keeping some intermediate computations as well
    diff = x.reshape(-1, D, 1) - means.T.reshape(1, D, N)
    diff = diff.transpose(0, 2, 1)
    q = -0.5 * (diff * inv_covariances.reshape(1, N, D) * diff).sum(axis=-1)
    q = np.exp(q - q.max(axis=1, keepdims=True))
    q /= q.sum(axis=1, keepdims=True)

    # Finally compute the unnormalized FV and return it
    diff_over_cov = diff * inv_sqrt_covariances.reshape(1, N, D)
    return np.hstack([
        (q.reshape(M, N, 1) * diff_over_cov).sum(axis=0),
        (q.reshape(M, N, 1) * (diff_over_cov**2 - 1)).sum(axis=0)
    ]).ravel()


class FisherVectors(BaseAggregator):
    """Aggregate local features using Fisher Vector encoding with a GMM.

    Train a GMM on some local features and then extract the normalized
    derivative

    Parameters
    ----------
    n_gaussians : int
                  The number of gaussians to be used for the fisher vector
                  encoding
    n_pca_components : float
                     Control the number of PCA components we will use to
                     reduce the dimensionality of our data. The valid range
                     for this parameter is (0, 1), whith 1 being used to denote
                     that the PCA components are equal to the number of feature's
                     dimension
    max_iter : int
               The maximum number of EM iterations
    normalization : int
                    A bitmask of POWER_NORMALIZATION and L2_NORMALIZATION
    dimension_ordering : {'th', 'tf'}
                         Changes how n-dimensional arrays are reshaped to form
                         simple local feature matrices. 'th' ordering means the
                         local feature dimension is the second dimension and
                         'tf' means it is the last dimension.
    inner_batch : int
                  Compute the fisher vector of 'inner_batch' vectors together.
                  It controls a trade off between speed and memory.
    n_jobs : int
            The threads to use for the transform
    verbose : int
              Controls the verbosity of the GMM
    """

    POWER_NORMALIZATION = 1
    L2_NORMALIZATION = 2

    def __init__(self, n_gaussians, n_pca_components=0.8, max_iter=100, 
                 normalization=3, dimension_ordering="tf", inner_batch=64,
                 n_jobs=-1, verbose=0):
        self.n_gaussians = n_gaussians
        self.max_iter = max_iter
        self.normalization = normalization
        self.inner_batch = inner_batch
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.n_pca_components = n_pca_components

        super(self.__class__, self).__init__(dimension_ordering)

        # initialize the rest of the  attributes of the class for any use
        # (mainly because we want to be able to check if fit has been called
        # before on this instance)
        self.pca_model = None
        self.weights = None
        self.means = None
        self.covariances = None
        self.inverted_covariances = None
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
            "n_pca_components": self.n_pca_components,
            "max_iter": self.max_iter,
            "normalization": self.normalization,
            "dimension_ordering": self.dimension_ordering,
            "inner_batch": self.inner_batch,
            "n_jobs": self.n_jobs,
            "verbose": self.verbose,

            "pca_model": self.pca_model,

            "weights": self.weights,
            "means": self.means,
            "covariances": self.covariances,
            "inverted_covariances": self.inverted_covariances,
            "inverted_sqrt_covariances": self.inverted_sqrt_covariances,
            "normalization_factor": self.normalization_factor
        }

    def __setstate__(self, state):
        """Restore the class's state after unpickling.

        Parameters
        ----------
        state: dictionary
               The unpickled data that were returned by __getstate__
        """
        # A temporary instance for accessing the default values
        t = FisherVectors(0)

        # Load from state
        self.n_gaussians = state["n_gaussians"]
        self.n_pca_components = state["n_pca_components"]
        self.max_iter = state.get("max_iter", t.max_iter)
        self.normalization = state.get("normalization", t.normalization)
        self.dimension_ordering = \
            state.get("dimension_ordering", t.dimension_ordering)
        self.inner_batch = state.get("inner_batch", t.inner_batch)
        self.n_jobs = state.get("n_jobs", t.n_jobs)
        self.verbose = state.get("verbose", t.verbose)

        self.pca_model = state.get("pca_model", t.pca_model)

        self.weights = state.get("weights", t.weights)
        self.means = state.get("means", t.means)
        self.covariances = state.get("covariances", t.covariances)
        self.inverted_covariances = \
            state.get("inverted_covariances", t.inverted_covariances)
        self.inverted_sqrt_covariances= \
            state.get("inverted_sqrt_covariances", t.inverted_sqrt_covariances)
        self.normalization_factor = \
            state.get("normalization_factor", t.normalization_factor)

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
        
        if self.n_pca_components != 1:
            # train PCA
            self.pca_model = PCA(n_components=int(X.shape[-1]*self.n_pca_components))
            self.pca_model.fit(X)

            # apply PCA and reduce dimensionality
            X = self.pca_model.transform(X)

        # consider changing the initialization parameters
        gmm = GaussianMixture(
            n_components=self.n_gaussians,
            max_iter=self.max_iter,
            covariance_type='diag',
            verbose=self.verbose
        )
        gmm.fit(X)

        # save the results of the gmm
        self.weights = gmm.weights_
        self.means = gmm.means_
        self.covariances = gmm.covariances_

        # precompute some values for encoding
        D = X[0].size
        self.inverted_covariances = (1./self.covariances)
        self.inverted_sqrt_covariances = np.sqrt(1./self.covariances)
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

        if self.n_pca_components != 1:
            # Apply PCA and reduce dimensionality
            X = self.pca_model.transform(X)

        # Allocate the memory necessary for the encoded data
        fv = np.zeros((len(lengths), self.normalization_factor.shape[0]))

        # Do a naive double loop for now
        s, e = 0, 0
        for i, l in enumerate(lengths):
            s, e = e, e+l
            fv[i] = sum(
                Parallel(n_jobs=self.n_jobs, backend="threading")(
                    delayed(_transform_batch)(
                        X[j:min(e, j+self.inner_batch)],
                        self.means,
                        self.inverted_covariances,
                        self.inverted_sqrt_covariances
                    )
                    for j in range(s, e, self.inner_batch)
                )
            )

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
