'''Random mixture of Gaussian distribution.'''
import iplane.constants as cn  # type: ignore
from iplane.random import Random, PCollection, DCollection
from iplane.random_mixture_collection import PCollectionMixture, DCollectionMixture  # type: ignore

import itertools
import pandas as pd  # type: ignore
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal # type: ignore
from sklearn.mixture import GaussianMixture  # type: ignore
from typing import Union, Optional, cast
import warnings


class RandomMixture(Random):
    """Handles Gaussian Mixture Models."""

    def __init__(self,
            pcollection:Optional[PCollectionMixture]=None,
            dcollection:Optional[DCollectionMixture]=None,
            num_component:int = 2,
            random_state:int = 42,
            max_num_sample:int = cn.MAX_NUM_SAMPLE,
            ):
        """
        Initializes the MixtureEntropy object.
        Args:
            num_component (int): number of components in the mixture model.
            random_state (int): random state for reproducibility.
        """
        super().__init__(pcollection, dcollection)
        self.max_num_sample = max_num_sample
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
    
    def makeDCollection(self, pcollection:PCollectionMixture) -> DCollectionMixture:
        """
        Calculates the probability density function (PDF) for a multi-dimensional Gaussian mixture model
        and calculates its differential entropy.

        Args:
            parameter (PCollectionMixture): The parameter collection for the distribution.

        Returns:
            DistributionCollectionMGaussian: The distribution object containing the variate array, PDF array, dx array, and entropy.
        """
        # Checks
        shape = pcollection.getShape()
        num_component, num_dimension = shape.num_component, shape.num_dimension
        mean_arr, covariance_arr, weight_arr = pcollection.getAll()
        # Calculate the number of samples for each dimension
        num_sample = int(self.max_num_sample**(1/num_dimension))
        if num_sample < 8:
            msg = "Number of samples per dimension must be at least 8."
            msg += f"\n  Increase max_num_sample so that 8**num_dimesion <= max_num_sample,"
            msg += f"\n  Currently:"
            msg += f"\n    num_sample={num_sample}"
            msg += f"\n    max_num_sample={self.max_num_sample}"
            msg += f"\n    num_dimension={num_dimension})."
            raise ValueError(msg)
        STD_MAX = 4
        # Caclulate the coordinates for each dimension
        linspaces:list = []
        dxs = []
        for i_dim in range(num_dimension):
            std = np.sqrt(covariance_arr[:, i_dim, i_dim])  # Get the standard deviation for each component in the current dimension
            min_val = min(mean_arr[:, i_dim]) - STD_MAX * std
            max_val = max(mean_arr[:, i_dim]) + STD_MAX * std
            std = std.flatten()
            linspace_arr = np.linspace(min_val, max_val, num=num_sample).flatten()
            linspaces.append(linspace_arr)
            dx = np.mean(np.diff(linspace_arr))  # Calculate the average change in x values
            dxs.append(dx)
        variate_arr = np.array(list(itertools.product(*linspaces)))  # Create a grid of variates
        variate_arr = variate_arr.reshape(-1, num_dimension)  # Reshape to (num_sample, num_dimension)
        dx_arr = np.array(dxs)
        # Calculate the densities at each variate value
        densities:list = []
        num_sample = variate_arr.shape[0]
        for i_component in range(num_component):
            weight = weight_arr[i_component]
            covariance = covariance_arr[i_component, :, :]
            mean = mean_arr[i_component, :]
            mvn = multivariate_normal(mean=mean, cov=covariance)   # type: ignore
            result_arr = mvn.pdf(variate_arr)
            densities.append(weight*result_arr)
        density_arr = np.sum(densities, axis=0)  # Sum the PDFs of all components
        # Calculate entropy
        Hx = -np.sum(density_arr * np.log2(density_arr + 1e-10)) * np.prod(dxs)  # Add small value to avoid log(0)
        dcollection_dct:dict = {}
        dcollection_dct[cn.DC_ENTROPY] = Hx
        dcollection_dct[cn.DC_DENSITY_ARR] = density_arr
        dcollection_dct[cn.DC_VARIATE_ARR] = variate_arr
        dcollection_dct[cn.DC_DX_ARR] =  dx_arr
        self.dcollection = DCollectionMixture(**dcollection_dct)
        return self.dcollection

    def calculateEntropy(self, pcollection:PCollectionMixture) -> float:
        """
        Calculates the entropy of a multivariate Gaussian distribution. If there are multiple components,
        it estimates the first.

        Args:
            covariance_arr (np.ndarray): Covariance matrix of the Gaussian distribution.

        Returns:
            float: Entropy of the Gaussian distribution.
        """
        num_dimension = pcollection.getShape().num_dimension
        constant_term = (2 * np.pi * np.e)**num_dimension
        ##
        def calcHx(cov):
            # Calculates entropy for a single Multivariate Gaussian distribution.
            det = np.linalg.det(cov)
            if np.isclose(det, 0):
                raise ValueError("Covariance matrix must be non-singular.")
            return 0.5 * np.log2(constant_term * det)
        ##
        covariance_arr = pcollection.get(cn.PC_COVARIANCE_ARR)
        weight_arr = pcollection.get(cn.PC_WEIGHT_ARR)
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

    def estimatePCollection(self, sample_arr:np.ndarray)->PCollectionMixture:
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
        dct = {
            cn.PC_MEAN_ARR: self.gmm.means_, 
            cn.PC_COVARIANCE_ARR: self.gmm.covariances_,
            cn.PC_WEIGHT_ARR: np.array(self.gmm.weights_, dtype=np.float64),}
        self.pcollection = PCollectionMixture(**dct)
        return self.pcollection