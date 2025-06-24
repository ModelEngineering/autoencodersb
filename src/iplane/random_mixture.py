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
from typing import Any, List, Tuple, Optional, Dict, cast


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
        num_component, num_dimension = parameter.getComponentAndDimension()
        # Calculate samples based on the Guassian mixure parameters.
        sample_arr = cast(np.ndarray, int(num_sample * weight_arr))
        arrs = []
        if (mean_arr.ndim == 1) or (np.shape(mean_arr)[1] == 1):
            std_arr = np.sqrt(covariance_arr)
            for idx in range(num_component):
                result_arr = np.random.normal(loc=mean_arr[idx], scale=std_arr[idx], size=sample_arr[idx])
                arrs.append(result_arr)
        else:
            for idx in range(num_component):
                result_arr = np.random.multivariate_normal(mean_arr[idx, :], covariance_arr[idx, :, :],  sample_arr[idx])
                arrs.append(result_arr)
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
        num_component, num_dimension = pcollection.getComponentAndDimension()
        mean_arr, covariance_arr, weight_arr = pcollection.getAll()
        # Calculate the number of samples for each dimension
        num_sample = int(self.max_num_sample**(1/num_dimension))
        if num_sample < 8:
            raise ValueError("Number of samples per dimension must be at least 8. Increase max_num_sample.")
        STD_MAX = 4
        # Caclulate the coordinates for each dimension
        linspaces:list = []
        dxs = []
        for dim_idx in range(num_dimension):
            if num_dimension == 1:
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
        for i_component in range(num_component):
            means = mean_arr[i_component, :]
            if num_dimension == 1:
                covariance = covariance_arr[i_component]
            else:
                covariance = covariance_arr[i_component, :, :]
            weight = weight_arr[i_component]
            mvn = multivariate_normal(mean=means, cov=covariance)   # type: ignore
            pdfs.append(weight*mvn.pdf(variate_arr))
        pdf_arr = np.sum(pdfs, axis=0)  # Sum the PDFs of all components
        # Calculate entropy
        Hx = -np.sum(pdf_arr * np.log2(pdf_arr + 1e-10)) * np.prod(dxs)  # Add small value to avoid log(0)
        parameter_dct:dict = {}
        parameter_dct[cn.DC_ENTROPY] = Hx
        parameter_dct[cn.DC_DENSITY_ARR] = pdf_arr
        parameter_dct[cn.DC_VARIATE_ARR] = variate_arr
        parameter_dct[cn.DC_DX_ARR] =  dx_arr
        self.dcollection = DCollectionMixture(collection_dct=parameter_dct)
        return self.dcollection

    def calculateEntropy(self, pcollection:PCollectionMixture) -> float:
        """
        Calculates the entropy of a multivariate Gaussian distribution

        Args:
            covariance_arr (np.ndarray): Covariance matrix of the Gaussian distribution.

        Returns:
            float: Entropy of the Gaussian distribution.
        """
        covariance_arr = pcollection.get(cn.PC_COVARIANCE_ARR)
        if covariance_arr is None:
            raise ValueError("Covariance array must be provided.")
        #
        num_dim = covariance_arr.shape[0]
        # Calculate determinant
        if num_dim == 1:
            det = covariance_arr[0]
        else:
            det = np.linalg.det(covariance_arr)
        # Calculate entropy
        if np.isclose(det, 0):
            raise ValueError("Covariance matrix must be non-singular.")
        return 0.5 * np.log2(((2 * np.pi * np.e)**num_dim) * det)

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
            cn.PC_WEIGHT_ARR: self.gmm.weights_}
        self.pcollection = PCollectionMixture(**dct)
        return self.pcollection