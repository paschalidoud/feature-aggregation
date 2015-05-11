"""This module will contain functions and classes that allow the computation of
a fisher vector encoding from a dataset and usage of an already computed
encoding."""

import math
import numpy as np
from sklearn.mixture import GMM


class Encoding:
    """This class is responsible for computing the fisher vector encoding.

    It is picklable.

    The proposed way of using it is to create an instance during training and
    pass it the data to cluster them using the GMM method. Then pickle the
    instance. When encoding is needed unpickle the instance and use encode.

    Example
    -------
    >>> import numpy as np
    >>> from fisher import Encoding as FisherEncoding
    >>> data = np.random.random(10000).reshape(-1, 10)
    >>> encoding = FisherEncoding(2, 100)
    >>> encoding.fit(data)
    >>> fv = encoding.encode_single(np.random.random(10))
    >>> assert len(fv) == 2+2*10+2*10
    >>> fv = encoding.encode([np.random.random(10) for _ in xrange(100)])
    >>> assert len(fv) == 2+2*10+2*10
    >>> encoding.normalization |= FisherEncoding.L2_NORMALIZATION
    >>> fv = encoding.encode([np.random.random(10) for _ in xrange(100)])
    >>> assert len(fv) == 2+2*10+2*10
    >>> encoding.normalization |= FisherEncoding.POWER_NORMALIZATION
    >>> fv = encoding.encode([np.random.random(10) for _ in xrange(100)])
    >>> assert len(fv) == 2+2*10+2*10
    >>>
    """

    POWER_NORMALIZATION = 1
    L2_NORMALIZATION = 2

    def __init__(self, N, iterations, normalization=0):
        """Initialize the class instance.

        Parameters
        ----------
        N:             int
                       The number of gaussians to be used for the fisher vector
                       encoding
        iterations:    int
                       The maximum number of iterations to perform fitting the
                       GMM
        normalization: ['sum', 'closed-form']
                       A way to approximate the diagonal of the fisher
                       information matrix in order to normalize the fisher
                       vectors so that they can be used with linear classifiers
        """
        self.N = N
        self.iterations = iterations
        self.normalization = normalization

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
            "N": self.N,
            "iterations": self.iterations,
            "normalization": self.normalization,

            "weights": self.weights,
            "means": self.means,
            "covariances": self.covariances,
            "inverted_covariances": self.inverted_covariances,
            "inverted_covariances_sqrt": self.inverted_covariances_sqrt,
            "inverted_covariances_3rd_power":
                self.inverted_covariances_3rd_power,
            "normalization_factor": self.normalization_factor
        }

    def __setstate__(self, state):
        """Restore the class's state after unpickling.

        Parameters
        ----------
        state: dictionary
               The unpickled data that were returned by __getstate__
        """
        self.N = state["N"]
        self.iterations = state["iterations"]
        self.normalization = state["normalization"]

        self.weights = state["weights"]
        self.means = state["means"]
        self.covariances = state["covariances"]
        self.inverted_covariances = state["inverted_covariances"]
        self.inverted_covariances_sqrt = state["inverted_covariances_sqrt"]
        self.inverted_covariances_3rd_power = state[
            "inverted_covariances_3rd_power"
        ]
        self.normalization_factor = state["normalization_factor"]

        # re-calculate the probability density functions
        self._calculate_pdfs()

    def fit(self, data):
        """Learn a fisher vector encoding.

        Fit a gaussian mixture model to the `data` using `N` gaussians with
        diagonal covariance matrices.

        Parameters
        ----------
        data: array_like, shape(n, n_features)
              List of data points that will be passed to the GMM for fitting.
        """

        # consider changing the initialization parameters
        gmm = GMM(
            n_components=self.N,
            n_iter=self.iterations,
            covariance_type='diag'
        )
        gmm.fit(data)

        # save the results of the gmm
        self.weights = gmm.weights_
        self.means = gmm.means_
        self.covariances = gmm.covars_

        # precompute stats for encoding
        # we might be unnecessarily spending some memory here to speed up some
        # computations, in any way that memory should not  exceed a couple of
        # megabytes
        self._calculate_pdfs()
        self.inverted_covariances = (1./self.covariances)
        self.inverted_covariances_sqrt = np.sqrt(self.inverted_covariances)
        self.inverted_covariances_3rd_power = self.inverted_covariances_sqrt**3

        # calculate the fisher information matrix diagonal
        N = self.N
        D = data[0].size
        self.normalization_factor = np.zeros(
            N +    # weights
            N*D +  # means
            N*D    # diagonal covariance
        )
        self.normalization_factor[:N] = 1/self.weights
        for i in xrange(self.N):
            self.normalization_factor[N+i*D:N+(i+1)*D] = (
                self.weights[i]/self.inverted_covariances[i]
            )
        for i in xrange(self.N):
            self.normalization_factor[N+N*D+i*D:N+N*D+(i+1)*D] = (
                2*self.weights[i]/self.inverted_covariances[i]
            )
        self.normalization_factor = 1/np.sqrt(self.normalization_factor)

    def encode(self, data):
        """Encode a list of data using the the learnt fisher vector encoding.

        The call to encode should result to one single vector assuming that the
        rows of the data are independent and belong to the same instance.

        Parameters
        ----------
        data: array_like, shape(n, n_features)
              List of data points that will be encoded using the already fit
              GMM.
        """
        # assume independent components in data
        fisher_vector = 0
        count = 0
        for x in data:
            fisher_vector += self.encode_single(x)
            count += 1

        # if there is no features for a specific video so the count is 0 return
        # a zero vector
        if count == 0:
            # the normalization factor is the same size as the fisher vector
            return np.zeros(self.normalization_factor.shape)

        # normalize the vector
        fisher_vector *= (1./count)*self.normalization_factor

        # check if we should be normalizing the power
        if self.normalization & self.POWER_NORMALIZATION:
            fisher_vector = np.sqrt(np.abs(fisher_vector))*np.sign(fisher_vector)

        # check if we should be performing L2 normalization
        if self.normalization & self.L2_NORMALIZATION:
            fisher_vector /= np.sqrt(np.dot(fisher_vector, fisher_vector))

        return fisher_vector

    def encode_single(self, x):
        """Compute the grad with respect to the parameters of the model for the
        vector x.

        The gradient is computed according to the following equations (in
        latex):

        \theta = \{ w_i, \mu_i, \Sigma_i\}

        P_{\theta}(x) = \sum_{i=1}^{N} w_i N_{\mu_i \Sigma_i}(x)

        \frac{\partial log(P)}{\partial w_j} =
            \frac{N_{\mu_j \Sigma_j}(x)}{P(x)}

        \frac{\partial log(P)}{\partial \mu_j^d} =
            \frac{w_j N_{\mu_j \Sigma_j}(x)}{P(x)}
            \frac{x^d-\mu_j^d}{\Sigma_j^d}

        \frac{\partial log(P)}{\partial \sqrt{\Sigma_j^d}} =
            \frac{w_j N_{\mu_j \Sigma_j}(x)}{P(x)}
            [
            \frac{(x^d-\mu_j^d)^2}{(\Sigma_j^d)^\frac{3}{2}} -
            \frac{1}{\sqrt{\Sigma_j^d}}
            ]

        Parameters
        ----------
        x: array_like, shape(n_features)
           The feature vector to be encoded with fisher encoding
        """
        # The following check is for the class to be a bit more user friendly
        # and provide a meaningful error message in case it is used
        # incorrectly. The performance hit is negligible since it will probably
        # be nanoseconds per call.
        if self.weights is None:
            raise Exception(
                "GMM model not found. Have you called fit(data) first?"
            )

        # number of gaussians
        N = self.N

        # number of dimensions
        D = self.means.shape[1]

        # Evaluate if we are overusing memory by creating a new vector with
        # np.zeros here
        fisher = np.zeros(
            N +    # weights
            N*D +  # means
            N*D    # diagonal covariance
        )

        # calculate the probabilities that x was created by each gaussian
        # distribution
        probs = np.zeros(N)
        for i, pdf in enumerate(self.pdfs):
            probs[i] = pdf(x)

        # weight those probs according to the weights for each of the
        # distributions
        weighted_probs = self.weights*probs

        # the normalization factor that can be seen in the equations in the
        # method's comment as 1/P(x)
        Z = 1./np.sum(weighted_probs)

        # first derivative with respect to the weights
        # this doesn't account for the constraint that the sum of the weights
        # should always equal to 1 evaluate if we need to change it as it is in
        # the paper
        fisher[0:N] = Z*probs

        # second derivative with respect to the means
        for i in xrange(N):
            fisher[N+i*D:N+(i+1)*D] = (
                Z * weighted_probs[i] * (x - self.means[i]) *
                self.inverted_covariances[i]
            )

        # third derivative with respect to the covariance
        for i in xrange(N):
            fisher[N+N*D+i*D:N+N*D+(i+1)*D] = (
                Z * weighted_probs[i] * (
                    self.inverted_covariances_3rd_power[i]*(
                        (x - self.means[i])**2
                    ) -
                    self.inverted_covariances_sqrt[1]
                )
            )

        # finally we have our fisher vector
        return fisher

    def _calculate_pdfs(self):
        """Calculate the probability density functions for the means and
        covariances that have been calculated.

        Not meant to be used out of the module.
        """
        self.pdfs = [
            self._multivariate_normal_pdf(self.means[i], self.covariances[i])
            for i in xrange(self.N)
        ]

    @staticmethod
    def _multivariate_normal_pdf(mu, sigma):
        """Return the pdf of a multivariate gaussian distribution with diagonal
        covariance matrix.

        This function is not meant to be used out of the module (hence the
        initial underscore).

        Parameters
        ----------
        mu:    array_like
               The mean of the normal distribution
        sigma: array_like
               The covariance matrix diagonal
        """
        sigma_inv = -1./(2*sigma)
        a = (2*np.pi)**(-mu.shape[0]/2)*math.sqrt(np.sum(sigma))

        def pdf(x):
            err = x - mu
            return a*math.exp(
                np.dot(np.multiply(err, sigma_inv), err)
            )

        return pdf
