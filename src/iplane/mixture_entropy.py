'''Calculate entropy for a mixture of distributions. Currently, it supports Gaussian Mixtures.'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.mixture import GaussianMixture  # type: ignore
from typing import List, Optional # type: ignore
import collections
from scipy.stats import norm # type: ignore


class MixtureEntropy(object):
    """Calculate entropy for a mixture of distributions."""

    def __init__(self, n_components:int = 2, random_state:int = 42,
            means:Optional[np.ndarray] = None,
            covariances:Optional[np.ndarray] = None,
            weights:Optional[np.ndarray] = None,
            num_sample:Optional[float]=None):
        self.n_components = n_components
        self.random_state = random_state
        # Use k-means clustering to initialize the Gaussian Mixture Model
        self.gmm = GaussianMixture(n_components=n_components, random_state=random_state)
        if means is not None:
            if (covariances is None) or (weights is None):
                raise ValueError("Means, covariances, and weights must be provided together.")
            std = np.std([means.ndim, covariances.ndim, weights.ndim])
            if std > 0:
                raise ValueError("Means, covariances, and weights must be 1D arrays.")
        # Calculated
        self._num_sample = num_sample
        self._mean_arr:np.ndarray = means  # type: ignore
        self._covariance_arr:np.ndarray = covariances # type: ignore
        self._weight_arr:np.ndarray = weights # type: ignore
        self.Hx = np.nan
        self.pdf_arr = np.array([])
        self.variante_arr = np.array([])
        self.dx = np.nan

    @property
    def num_samples(self)-> float:
        """Returns the means of the Gaussian Mixture Model."""
        if self._num_sample is None:
                raise ValueError("GMM has not been fitted yet. Call fit() first.")
        return self._num_sample
    
    @property
    def mean_arr(self)-> np.ndarray:
        """Returns the means of the Gaussian Mixture Model."""
        if self._mean_arr is None:
            if not hasattr(self.gmm, 'means_'):
                self._mean_arr = self.gmm.means_.flatten() # type: ignore
            else:
                raise ValueError("GMM has not been fitted yet. Call fit() first.")
        return self._mean_arr
    
    @property
    def covariance_arr(self)-> np.ndarray:
        """Returns the means of the Gaussian Mixture Model."""
        if not hasattr(self.gmm, 'covariances_'):
            self._covariance_arr = self.gmm.covariances_.flatten() # type: ignore
        else:
            raise ValueError("GMM has not been fitted yet. Call fit() first.")
        return self._covariance_arr
    
    @property
    def weight_arr(self)-> np.ndarray:
        """Returns the means of the Gaussian Mixture Model."""
        if not hasattr(self.gmm, 'covariances_'):
            self._weight_arr = self.gmm.weights_.flatten() # type: ignore
        else:
            raise ValueError("GMM has not been fitted yet. Call fit() first.")
        return self._weight_arr

    @classmethod
    def generateMixture(cls, num_samples:List[int], means:List[float], 
            stds:List[float], num_dim=2, noise:float=0.8)->np.ndarray:
        """
        Generates synthetic data for a multidimensional Gaussian Mixture Model.

        Args:
            num_sample (int): number of samples in the n-th mixture
            means (float): mean of the n-th mixture
            std (float): standard deviation of the n-th mixture
            num_dim (int): number of dimensions of the data
            noise (float): noise level to add to the data
        Returns:
            np.array (num_sample, 1), int. total count is = sum(num_samples)
        """
        results = [np.random.normal(m, s, n) for n, m, s in zip(num_samples, means, stds)]
        merged_arr = np.concatenate(results)
        data_arr = np.random.permutation(merged_arr).reshape(-1, 1)
        # Add the other dimensions
        arrs = [data_arr]
        num_total = sum(num_samples)
        for _ in range(num_dim - 1):
            noise_arr = np.random.normal(0, noise, num_total)
            arrs.append(arrs[0] + noise_arr)
        if num_dim > 1:
            arr = np.array(arrs)
            arr = np.reshape(arr, num_sample, num_dim)
        else:
            arr = data_arr
        import pdb; pdb.set_trace()
        return arr
    
    def calculateEntropy(self)->None:
        """
        Uses the normal distribut to calculate the expected entropy of a guassian mixture.

        Args:
            means: distribution of the means
            stds: variances of the distributions
            weights: weight of component

        Calculates
            self.Hx
            self.pdf_arr
            self.dx

        """
        NUM_POINT =1000
        EXTREME_SIGMA = 4
        #
        # Calculate the range for the x-variate
        std_arr = (self.covariance_arr**0.5).diagonal()
        lower_bound = min([m - EXTREME_SIGMA*s for m, s in zip(self.mean_arr, std_arr)])
        upper_bound = max([m + EXTREME_SIGMA*s for m, s in zip(self.mean_arr, std_arr)])
        variate_arr = np.linspace(lower_bound, upper_bound, NUM_POINT)
        self.dx = np.mean(np.diff(variate_arr))
        # Calculate the PDFs
        self.pdf_arr = np.zeros(NUM_POINT)
        for mean, std, weight in zip(self.mean_arr, std_arr, self.weight_arr):
            self.pdf_arr += weight*norm.pdf(variate_arr, loc=mean, scale=std)
        # Entropy
        self.Hx = -sum(self.pdf_arr*np.log2(self.pdf_arr))*self.dx
    
    def calculateMixture1d(self, sample_arr)->None:
        """
        Calculates a guassian mixture distribution for 1d sample.

        Args:
            sample_arr: array of one dimensional variates

        Returns:
            Mixture:
                gmm: GausianMixture gitted object
                Hx: differential entropy
                variate_arr: values on x-axis
                pdf_arr: density
                dx (float): change in x values in calculation
        """
        self.gmm.fit(sample_arr)
        self._num_sample = len(sample_arr)
        self.calculateEntropy()

class TestMixtureEntropy(unittest.TestCase):
    """Test class for MixtureEntropy."""

    def setUp(self):
        """Set up the test case."""
        num_sample = np.array([100, 200])
        means = np.array([0, 10])
        stds = np.array([1, 2])
        self.mixture_entropy = MixtureEntropy(
            n_components=2,
            means=means,
            covariances=stds**2,
            weights=self.weights
        )

    def testGenerateMixture(self):
        """Test the generateMixture method."""
        arr = MixtureEntropy.generateMixture(num_sample, means, stds)
        self.assertEqual(arr.shape[0], sum(num_sample))
        self.assertEqual(arr.shape[1], 1)
        self.assertTrue(np.all(arr >= -10))
        self.assertTrue(np.all(arr <= 20))
num_point = 1000
sigmas = [1, 2, 4]
num_component = len(sigmas)
sample_arr = generateMixture([10*num_point]*num_component, 10*np.array(range(num_component)), sigmas)
result = calculateMixture1d(sample_arr)
assert(isinstance(result.gmm, GaussianMixture))
print("OK!")


# Tests
if __name__ == "__main__":
    precision = 2
    num_sample1 = 3
    small_mean = 0.1
    arr = MixtureEntropy.generateMixture([num_sample1, 5],
            [small_mean, 100], [.01, 10], num_dim=1)
    assert(np.ndim(arr) == 2)
    assert(np.sum(arr < 10*small_mean)) == num_sample1
    print("OK!")
    # Mxiture entropy
    #   Guassian
    for std in [1, 2, 4, 8]:
        actual = 0.5*np.log2(2*np.pi*np.e*std**2)
        estimated = calculateGaussianMixtureEntropy([0], [std], [1]).Hx
        #print(actual, estimated)
        assert(np.abs(actual - estimated) < 0.01)
    #   Change in mean
    estimate1 = calculateGaussianMixtureEntropy([0, 1], [4, 4], [0.5, 0.5]).Hx
    estimate2 = calculateGaussianMixtureEntropy([0, 4], [4, 4], [0.5, 0.5]).Hx
    #print(estimate1, estimate2)
    assert(estimate1 < estimate2)
    #   Change in std
    estimate1 = calculateGaussianMixtureEntropy([0, 1], [4, 4], [0.5, 0.5]).Hx
    estimate2 = calculateGaussianMixtureEntropy([0, 1], [4, 16], [0.5, 0.5]).Hx
    #print(estimate1, estimate2)
    assert(estimate1 < estimate2)
    print("OK!")