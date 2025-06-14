'''Calculate entropy for a mixture of distributions. Currently, it supports Gaussian Mixtures.'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.mixture import GaussianMixture  # type: ignore
from typing import List, Optional # type: ignore
import collections
from scipy.stats import norm # type: ignore

NULL_ARR = np.array([])  # type: ignore


class MixtureEntropy(object):
    """Calculate entropy for a mixture of distributions."""

    def __init__(self,
            n_components:int = 2,
            random_state:int = 42,
            num_dim:int = 1,
            means:np.ndarray=NULL_ARR,
            covariances:np.ndarray=NULL_ARR,
            ):
        """
        Initializes the MixtureEntropy object.
        Args:
            n_components (int): number of components in the mixture model.
            random_state (int): random state for reproducibility.
            num_dim (int): number of dimensions of the data.
            means (np.ndarray, optional): means of the Gaussian components.
            covariances (np.ndarray, optional): covariances of the Gaussian components.
            weights (np.ndarray, optional): weights of the Gaussian components.
        """
        self.n_components = n_components
        self.random_state = random_state
        self.num_dim = num_dim
        # Use k-means clustering to initialize the Gaussian Mixture Model
        self.gmm = GaussianMixture(n_components=n_components, random_state=random_state)
        if means is not None:
            if means.ndim != 1 or covariances.ndim != 2:
                import pdb; pdb.set_trace()
                raise ValueError("Means must be 1D arrays; Covariances must be 2D arrays.")
        # Calculated
        self._mean_arr = means
        self._covariance_arr = covariances
        self._weight_arr = np.array([])
        self.Hx = np.nan
        self.pdf_arr = np.array([])
        self.variante_arr = np.array([])
        self.dx = np.nan

    ############## PROPERTIES ##############
    @property
    def mean_arr(self)-> np.ndarray:
        """Returns the means of the Gaussian Mixture Model."""
        if self._mean_arr is None:
            if hasattr(self.gmm, 'means_'):
                self._mean_arr = self.gmm.means_.flatten() # type: ignore
            else:
                if self._covariance_arr is None:
                    raise ValueError("GMM has not been fitted yet. Call fit() first.")
        return self._mean_arr
    
    @property
    def covariance_arr(self)-> np.ndarray:
        """Returns the means of the Gaussian Mixture Model."""
        if hasattr(self.gmm, 'covariances_'):
            self._covariance_arr = self.gmm.covariances_.flatten() # type: ignore
        else:
            if self._covariance_arr is None:
                raise ValueError("GMM has not been fitted yet. Call fit() first.")
        return self._covariance_arr
    
    @property
    def std_arr(self)-> np.ndarray:
        """Returns the standard deviations of the Gaussian Mixture Model."""
        return np.diagonal(self.covariance_arr**0.5)
    
    @property
    def weight_arr(self)-> np.ndarray:
        """Returns the means of the Gaussian Mixture Model."""
        if hasattr(self.gmm, 'covariances_'):
            self._weight_arr = self.gmm.weights_.flatten() # type: ignore
        else:
            if self._covariance_arr is None:
                raise ValueError("GMM has not been fitted yet. Call fit() first.")
        return self._weight_arr
    
    ############## METHODS ##############
    def generateMixture(self, num_samples:List[int], shift:float=0.8)->np.ndarray:
        """
        Generates synthetic data for a multidimensional Gaussian Mixture Model.
        Uses current values of means, stds, weights, and num_samples.
        Additional dimensions are the first dimension plus a noise term.

        Args:
            shifht: amount by which the mean is shifted for each successive dimension.

        Returns:
            np.array (num_sample, 1), int. total count is = sum(num_samples)
        """
        # Calculate samples based on the Guassian mixure parameters.
        results = [np.random.normal(m, s, n) for m, s, n in zip(self.mean_arr, self.std_arr, num_samples)]
        merged_arr = np.concatenate(results)
        data_arr = np.random.permutation(merged_arr)
        # Include the other dimensions
        arrs = [data_arr]
        num_total = np.sum(num_samples)
        for idx in range(1, self.num_dim):
            arrs.append(arrs[0] + idx*shift)
        if self.num_dim > 1:
            arr = np.array(arrs)
            arr = arr.T  # Transpose to get the correct shape
        else:
            # Ensure the array is 2D
            arr = data_arr.reshape(-1, 1)
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
        self.calculateEntropy()

    @staticmethod
    def makeConvariance(stds:List[float])->np.ndarray:
        """
        Creates a covariance matrix for the Gaussian Mixture Model.

        Args:
            stds (List[float]): standard deviations of the Gaussian components.

        Returns:
            np.ndarray: covariance matrix.
        """
        if len(stds) == 1:
            return np.array([[stds[0]**2]])
        else:
            return np.diag(np.array(stds)**2)