'''Random mixture of Gaussian distribution.'''
import iplane.constants as cn  # type: ignore
from iplane.random_continuous import RandomContinuous  # type: ignore
from iplane.random_mixture_collection import PCollectionMixture, DCollectionMixture  # type: ignore

import pandas as pd  # type: ignore
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal # type: ignore
from sklearn.mixture import GaussianMixture  # type: ignore
from typing import Optional, cast
import warnings


class RandomMixture(RandomContinuous):
    """Handles Gaussian Mixture Models."""

    def __init__(self,
            pcollection:Optional[PCollectionMixture]=None,
            dcollection:Optional[DCollectionMixture]=None,
            num_component:int = 2,
            random_state:int = 42,
            **kwargs
            ):
        """
        Initializes the MixtureEntropy object.
        Args:
            pcollection (PCollectionMixture): collection of parameters for the Gaussian mixture model.
            dcollection (DCollectionMixture): collection of distribution properties.
            total_num_sample (int): total number of samples to generate.
            num_component (int): number of components in the mixture model.
            random_state (int): random state for reproducibility.
        """
        super().__init__(pcollection=pcollection, dcollection=dcollection, **kwargs)
        self.num_component = num_component
        self.random_state = random_state
        # Use k-means clustering to initialize the Gaussian Mixture Model
        self.gmm = GaussianMixture(n_components=num_component, random_state=random_state)

    def generateSample(self, pcollection:PCollectionMixture, num_sample:int) -> np.ndarray:
        """
        Generates synthetic data for a multidimensional Gaussian Mixture Model.
            Each Gaussian component is defined by its mean and covariance marix.
            Components are indexed by the first array index
            Dimensions are indexed by the second array index for mean.

        Args: N is number of components, D is number of dimensions, K is number of categories.
            sample_arr (np.ndarray N): Number of samples to generate for each component.
            mean_arr (np.ndarray N X D): Mean of each Gaussian component.
            covariance_arr (np.ndarray N X D X D): Covariance matrix for each Gaussian component.
                    if D = 1, then covariance_arr is N X 1

        Returns:
            np.array (num_sample, 1), int. total count is = sum(num_samples)
        """
        parameter = cast(PCollectionMixture, pcollection)
        mean_arr, covariance_arr, weight_arr = pcollection.getAll()
        num_component = parameter.getShape().num_component
        # Calculate samples based on the Guassian mixure parameters.
        sample_arr = np.array([int(w*num_sample) for w in weight_arr])
        arrs = []
        #if (mean_arr.ndim == 1) or (np.shape(mean_arr)[1] == 1):
        for i_component in range(num_component):
            result_arr = np.random.multivariate_normal(mean_arr[i_component, :], covariance_arr[i_component, :, :],
                    sample_arr[i_component])
            arrs.append(result_arr)
            """ for idx in range(num_component):
                result_arr = np.random.multivariate_normal(mean_arr[idx, :], covariance_arr[idx, :, :],  sample_arr[idx])
                arrs.append(result_arr) """
        merged_arr = np.concatenate(arrs)
        merged_arr = np.random.permutation(merged_arr)
        return merged_arr
    
    def predict(self, *args, **predict_kwargs) -> np.ndarray:
        """
        Predicts the probability density function (PDF) for a given array of variates using the Gaussian Mixture Model.

        Args:
            args[0]: variate_arr (np.ndarray): Single variate array of shape (1, num_dimension)
            predict_kwargs (dict): Additional keyword arguments for prediction, not used in this method.
                pcollection (PCollectionMixture): PCollectionMixture with the parameters of the distribution.

        Returns:
            np.ndarray: Array of predicted PDF values for each variate.

        Notes:
            dcollection is included for compatibility with the base class, but not used in this method.
        """
        small_variate_arr = args[0]
        # Error checking
        if not "pcollection" in predict_kwargs:
            raise ValueError("pcollection must be provided.")
        else:
            pcollection = cast(PCollectionMixture, predict_kwargs["pcollection"])
        # Initializations
        mean_arr, covariance_arr, weight_arr = pcollection.getAll()
        num_component = pcollection.getShape().num_component
        # Calculation
        densities:list = []
        for i_component in range(num_component):
            weight = weight_arr[i_component]
            covariance = covariance_arr[i_component, :, :]
            mean = mean_arr[i_component, :]
            mvn = multivariate_normal(mean=mean, cov=covariance)   # type: ignore
            density = mvn.pdf(small_variate_arr)
            densities.append(weight*density)
        density_arr = np.sum(densities, axis=0)  # Sum the PDFs of all components
        return density_arr

    def calculateEntropy(self, collection:PCollectionMixture) -> float:
        """
        Analytic calculation of entropy of a multivariate Gaussian distribution. If there are multiple components,
        it estimates the first.

        Args:
            covariance_arr (np.ndarray): Covariance matrix of the Gaussian distribution.

        Returns:
            float: Entropy of the Gaussian distribution.
        """
        num_dimension = collection.getShape().num_dimension
        constant_term = (2 * np.pi * np.e)**num_dimension
        ##
        def calcHx(cov):
            # Calculates entropy for a single Multivariate Gaussian distribution.
            det = np.linalg.det(cov)
            if np.isclose(det, 0):
                raise ValueError("Covariance matrix must be non-singular.")
            return 0.5 * np.log2(constant_term * det)
        ##
        covariance_arr = collection.get(cn.PC_COVARIANCE_ARR)
        weight_arr = collection.get(cn.PC_WEIGHT_ARR)
        if covariance_arr is None:
            raise ValueError("Covariance array must be provided.")
        if len(covariance_arr) > 1:
            warnings.warn("Multiple components detected in a 1D Gaussian mixture. Using sum of component entropy.")
        result = 0.0
        for cov, weight in zip(covariance_arr, weight_arr):
            result += weight*calcHx(cov)
        offset = -np.sum([w*np.log2(w) for w in weight_arr])  # log2 of the weights
        result += offset  # Add the log2 of the weights to the entropy
        if isinstance(result, np.ndarray):
            result = result.item()
        return result

    def makePCollection(self, sample_arr:np.ndarray)->PCollectionMixture:
        """
        Estimates the parameters of a Gaussian Mixture Model from the sample array.

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
        #
        self.pcollection = PCollectionMixture(
            mean_arr=cast(np.ndarray, self.gmm.means_),
            covariance_arr=cast(np.ndarray, self.gmm.covariances_),
            weight_arr=np.array(self.gmm.weights_, dtype=np.float64),
        )
        return self.pcollection
    
    def makeDCollection(self, pcollection:PCollectionMixture, variate_arr:Optional[np.ndarray]=None,
            dx_arr:Optional[np.ndarray]=None) -> DCollectionMixture:
        """
        Calculates the probability density function (PDF) for a multi-dimensional Gaussian mixture model
        and calculates its differential entropy.

        Args:
            pcollection (PCollectionMixture): The collection of parameters for the Gaussian mixture model.
            variate_arr (Optional[np.ndarray]): Optional array of variates to evaluate the PDF.
            num_sample (int): The number of samples to generate for each dimension.

        Returns:
            DistributionCollectionMGaussian: The distribution object containing the variate array, PDF array, dx array, and entropy.
        """
        # Initializations
        mean_arr, covariance_arr, weight_arr = pcollection.getAll()
        shape = pcollection.getShape()
        num_component, num_dimension = shape.num_component, shape.num_dimension
        # Construct the variate array
        if variate_arr is None:
            num_variate_sample = self.num_variate_sample
            std_arrs = np.array([np.sqrt(np.diagonal(covariance_arr[n, :, :])) for n in range(num_component)])
            std_arrs = np.reshape(std_arrs, (num_component, num_dimension))
            min_arr = mean_arr - 0.5*self.axis_length_std * std_arrs  # Calculate the minimum point for each dimension
            max_arr = mean_arr + 0.5*self.axis_length_std * std_arrs  # Calculate the minimum point for each dimension
            min_point = np.min(min_arr, axis=0)  # Minimum point across all 
            max_point = np.max(max_arr, axis=0)  # Maximum point across all 
            variate_arr, dx_arr = self.makeVariate(min_point, max_point, num_variate_sample=num_variate_sample)
        num_variate_sample = variate_arr.shape[0]
        # Calculate the densities at each variate value
        density_arr = self.predict(variate_arr, dx_arr, pcollection=pcollection)
        # Calculate entropy
        entropy = self.makeEntropy(density_arr=density_arr, dx_arr=cast(np.ndarray, dx_arr))
        # Construct the DCollectionMixture object
        self.dcollection = DCollectionMixture(
            variate_arr=variate_arr,
            density_arr=density_arr,
            dx_arr=dx_arr,
            entropy=entropy,
        )
        return self.dcollection