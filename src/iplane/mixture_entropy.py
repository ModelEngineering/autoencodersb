'''Calculate entropy for a mixture of distributions. Currently, it supports Gaussian Mixtures.'''

import collections
import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.mixture import GaussianMixture  # type: ignore
from scipy.stats import norm # type: ignore
from typing import List, Optional, Union # type: ignore

NULL_ARR = np.array([])  # type: ignore

# Define a named tuple for the density
#   variate_arr: values for the variate has the same number of dimensions as the distribution
#   pdf_arr: dimensioned as the variate_arr
#   dx_arr: change in x values in each dimension
EntropyResult = collections.namedtuple('EntropyResult', ['variate_arr', 'pdf_arr', 'dx_arr', 'Hx'])  # type: ignore

class MixtureEntropy(object):
    """Calculate entropy for a mixture of distributions."""

    def __init__(self,
            n_components:int = 2,
            random_state:int = 42,
            ):
        """
        Initializes the MixtureEntropy object.
        Args:
            n_components (int): number of components in the mixture model.
            random_state (int): random state for reproducibility.
        """
        self.n_components = n_components
        self.random_state = random_state
        # Use k-means clustering to initialize the Gaussian Mixture Model
        self.gmm = GaussianMixture(n_components=n_components, random_state=random_state)
        # Calculated
        self.Hx = np.nan
        self.pdf_arr = np.array([])
        self.variate_arr = np.array([])
        self.dx_arr = np.array([])

    ############## PROPERTIES ##############
    @property
    def mean_arr(self)-> np.ndarray:
        if hasattr(self.gmm, 'means_'):
            return self.gmm.means_ # type: ignore
        else:
            raise ValueError("GMM has not been fitted yet. Call fit() first.")
    
    @property
    def covariance_arr(self)-> np.ndarray:
        """Returns the means of the Gaussian Mixture Model."""
        if hasattr(self.gmm, 'means_'):
            return self.gmm.covariances_ # type: ignore
        else:
            raise ValueError("GMM has not been fitted yet. Call fit() first.")
    
    @property
    def std_arr(self)-> np.ndarray:
        """Returns the standard deviations of the Gaussian Mixture Model."""
        return np.diagonal(self.covariance_arr**0.5)
    
    @property
    def num_dim(self)-> int:
        """Returns the number of dimensions in the Gaussian Mixture Model."""
        if hasattr(self.gmm, 'means_'):
            return self.mean_arr.shape[1]
        else:
            raise ValueError("GMM has not been fitted yet. Call fit() first.")
    
    @property
    def weight_arr(self)-> np.ndarray:
        """Returns the means of the Gaussian Mixture Model."""
        if hasattr(self.gmm, 'means_'):
            return self.gmm.weights_.flatten() # type: ignore
        else:
            raise ValueError("GMM has not been fitted yet. Call fit() first.")
    
    ############## METHODS ##############
    @classmethod
    def generateMixture(cls,
            sample_arr:np.ndarray,
            mean_arr:np.ndarray,
            covariance_arr:np.ndarray,
            )-> np.ndarray:
        """
        Generates synthetic data for a multidimensional Gaussian Mixture Model.
            Each Gaussian component is defined by its mean and covariance marix.
            Components are indexed by the first array index
            Dimensions are indexed by the second array index for mean.
            For covariance, there is a third index for the covariance matrix.

        Args:
            sample_arr (np.ndarray): Number of samples to generate for each component.
            mean_arr (np.ndarray): Mean of each Gaussian component.
            covariance_arr (np.ndarray): Covariance matrix for each Gaussian component.

        Returns:
            np.array (num_sample, 1), int. total count is = sum(num_samples)
        """
        if np.std([np.shape(covariance_arr)[0], len(sample_arr), len(mean_arr)]) != 0:
            raise ValueError("Covariance, sample, and mean arrays must have the same number of components.")
        num_component = len(sample_arr)
        # Calculate samples based on the Guassian mixure parameters.
        arrs = []
        if (mean_arr.ndim == 1) or (np.shape(mean_arr)[1] == 1):
            std_arr = np.sqrt(covariance_arr)
            for m, s, n in zip(mean_arr, std_arr, sample_arr):
                result_arr = np.random.normal(loc=m, scale=s, size=n)
                arrs.append(result_arr)
        else:
            for idx in range(num_component):
                result_arr = np.random.multivariate_normal(mean_arr[idx, :], covariance_arr[idx, :, :],  sample_arr[idx])
                arrs.append(result_arr)
        merged_arr = np.concatenate(arrs)
        merged_arr = np.random.permutation(merged_arr)
        return merged_arr
    
    def calculateEntropy(self):
        """Calculates the entropy of the Gaussian Mixture Model.
        Updates self.Hx and distribution information

        Raises:
            ValueError: No fitted model
        """
        if not hasattr(self.gmm, 'means_'):
            raise ValueError("GMM has not been fitted yet. Call fit() first.")
        # Calculate the density
        entropy_result = self.calculateMixtureEntropy(
            mean_arr=self.gmm.means_,  # type: ignore
            covariance_arr=self.gmm.covariances_,  # type: ignore
            weight_arr=self.gmm.weights_,  # type: ignore
        )
        # Update the properties
        self.variante_arr = entropy_result.variate_arr
        self.pdf_arr = entropy_result.pdf_arr
        self.dx_arr = entropy_result.dx_arr
        self.Hx = entropy_result.Hx

    @staticmethod
    def calculateMixtureEntropy(
            mean_arr:np.ndarray,
            covariance_arr:np.ndarray,
            weight_arr:np.ndarray,
            max_num_sample:int = int(1e7),
            )-> EntropyResult:
        """
        Calculates the probability density function (PDF) for a multi-dimensional Gaussian mixture model
        and calculates its differential entropy.

        Args:
            num_sample (int): Number of samples to generate for each dimension
            mean_arr (np.ndarray): N X D Mean of each Gaussian component (N) for each dimension (D).
            covariance_arr (np.ndarray): N X D X D Covariance matrix for each Gaussian component (N) for the dimensions (D).
                            (If only one dimension for each component, then N X 1)
            weight_arr (Optional[np.ndarray]): N Weights for each Gaussian component (N).
            max_num_sample (int): Maximum number of samples to generate for each dimension. 

        Returns:
            Density
        """
        # Checks
        if not np.isclose(sum(weight_arr), 1):
            raise ValueError("Weights must sum to 1.")
        if mean_arr.ndim != 2:
            raise ValueError("Mean array must be 2-dimensional.")
        if covariance_arr.ndim < 2:
            raise ValueError("Covariance array must be at least 2-dimensional.")
        #
        dims = [mean_arr.shape[1], covariance_arr.shape[1]]
        if mean_arr.ndim == 1:
            mean_arr = np.reshape(mean_arr, (dims[0], -1))  # Ensure mean_arr is 2D
        if not all([d == dims[0] for d in dims]):
            raise ValueError("Mean and covariance arrays must have the same number of components.")
        #
        num_dim = dims[0]
        n_component = mean_arr.shape[0]
        # Calculate the number of samples for each dimension
        num_sample = int(max_num_sample**(1/num_dim))
        if num_sample < 8:
            raise ValueError("Number of samples per dimension must be at least 8. Increase max_num_sample.")
        STD_MAX = 4
        # Caclulate the coordinates for each dimension
        linspaces:list = []
        dxs = []
        for dim_idx in range(num_dim):
            if num_dim == 1:
                std_arr = np.sqrt(covariance_arr)
            else:
                std_arr = np.sqrt(covariance_arr[:, dim_idx, dim_idx])
            std_arr = std_arr.flatten()
            min_val = min(mean_arr[:, dim_idx] - STD_MAX * std_arr)
            max_val = max(mean_arr[:, dim_idx] + STD_MAX * std_arr)
            linspaces.append(np.linspace(min_val, max_val, num_sample))
            dx = np.mean(np.diff(linspaces[dim_idx]))
            dxs.append(dx)
        #variate_arrs = np.meshgrid(*linspaces, indexing='xy')  # Create a grid of variates
        variate_arr = np.array(list(itertools.product(*linspaces)))  # Create a grid of variates
        dx_arr = np.array(dxs)
        # Calculate the densities at each variate value
        pdfs:list = []
        pdf = 0
        for i_component in range(n_component):
            means = mean_arr[i_component, :]
            if num_dim == 1:
                covariance = covariance_arr[i_component]
            else:
                covariance = covariance_arr[i_component, :, :]
            weight = weight_arr[i_component]
            mvn = multivariate_normal(mean=means, cov=covariance)   # type: ignore
            pdfs.append(weight*mvn.pdf(variate_arr))
        pdf_arr = np.sum(pdfs, axis=0)  # Sum the PDFs of all components
        # Calculate entropy
        Hx = -np.sum(pdf_arr * np.log2(pdf_arr + 1e-10)) * np.prod(dxs)  # Add small value to avoid log(0)
        return EntropyResult(variate_arr=variate_arr, pdf_arr=pdf_arr, dx_arr=dx_arr, Hx=Hx)
    
    @staticmethod
    def calculateUnivariateGaussianEntropy(variance:Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculates the entropy of one or more univariate Gaussian distributions

        Args:
            variance (float or array of floats): Variance of the Gaussian distribution.

        Returns:
            float (array of floats if input is array of floats): Entropy of the Gaussian distribution.
        """
        if np.any(variance <= 0):
            raise ValueError("Variance must be positive.")
        return 0.5 * np.log2(2 * np.pi * np.e * variance)

    @staticmethod
    def calculateMultivariateGaussianEntropy(covariance_arr:np.ndarray) -> float:
        """
        Calculates the entropy of a multivariate Gaussian distribution

        Args:
            covariance_arr (np.ndarray): Covariance matrix of the Gaussian distribution.

        Returns:
            float: Entropy of the Gaussian distribution.
        """
        det = np.linalg.det(covariance_arr)
        num_dim = covariance_arr.shape[0]
        if np.isclose(det, 0):
            raise ValueError("Covariance matrix must be non-singular.")
        return 0.5 * np.log2(((2 * np.pi * np.e)**num_dim) * det)
    
    def fit(self, sample_arr)->None:
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
        if sample_arr.ndim == 1:
            sample_arr = sample_arr.reshape(-1, 1)
        self.gmm.fit(sample_arr)

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