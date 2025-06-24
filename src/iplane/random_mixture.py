'''Random mixture of Gaussian distribution.'''
import iplane.constants as cn  # type: ignore
from iplane.random import Random, PCollection, DCollection

import collections
import itertools
import pandas as pd  # type: ignore
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.mixture import GaussianMixture  # type: ignore
from scipy.stats import norm # type: ignore
from typing import Any, List, Tuple, Optional, Union, Dict, cast, overload

MEAN_ARR = 'mean_arr'
COVARIANCE_ARR = 'covariance_arr'
WEIGHT_ARR = 'weight_arr'
PARAMETER_NAMES = [MEAN_ARR, COVARIANCE_ARR, WEIGHT_ARR]
VARIATE_ARR = 'variate_arr'
DENSITY_ARR = 'density_arr'
VARIATE_ARR = 'continuous_variate_arr'
DISCRETE_VARIATE_ARR = 'discrete_variate_arr'
DX_ARR = 'dx_arr'
DISTRIBUTION_NAMES = [cn.ENTROPY, VARIATE_ARR, DENSITY_ARR, VARIATE_ARR, DISCRETE_VARIATE_ARR, DX_ARR]


# Define a named tuple for the density
#   variate_arr: values for the variate has the same number of dimensions as the distribution
#   pdf_arr: dimensioned as the variate_arr
#   dx_arr: change in x values in each dimension
EntropyResult = collections.namedtuple('EntropyResult', ['variate_arr', 'pdf_arr', 'dx_arr', 'Hx'])  # type: ignore


################################################
class PCollectionMixture(PCollection):
    # Parameter collection for mixture of Gaussian distributions.
    # Instance variables:
    #   collection_names: list of all names of parameters
    #   dct: dictionary of subset name-value pairs

    def __init__(self, parameter_dct:Optional[Dict[str, Any]]=None)->None:
        """
        Args:
            parameter_dct (Optional[Dict[str, Any]], optional): parameter name-value pairs.
        """
        super().__init__(PARAMETER_NAMES, parameter_dct)

    def isAnyNull(self) -> bool:
        """Check if any parameter value is None."""
        for key in PARAMETER_NAMES:
            if key not in self.dct or self.dct[key] is None:
                return True
        return False
    
    def getComponentAndDimension(self) -> Tuple[int, int]:
        """
        Finds the number of components and dimensions for the gaussian distribution.

        Returns:
            Tuple[int, int]: Number of components and number of dimensions for the multivariate distribution.
        """
        if self.isAnyNull():
            raise ValueError("Parameter dictionary must contain mean_arr, covariance_arr, and weight_arr.")
        mean_arr, covariance_arr, weight_arr = cast(np.ndarray, self.dct.get(MEAN_ARR)),   \
                cast(np.ndarray, self.dct.get(COVARIANCE_ARR)), cast(np.ndarray, self.dct.get(WEIGHT_ARR))
        # Consistent number of components and dimensions
        if weight_arr.ndim != 1:
            raise ValueError("Weight must have 1 dimension.")
        is_correct_shape = False
        if mean_arr.ndim == 2 and covariance_arr.ndim == 3:
            is_correct_shape = True
        elif mean_arr.ndim == 1 and covariance_arr.ndim == 1:
            is_correct_shape = True
        if not is_correct_shape:
            raise ValueError("Either mean_arr must be 2D and covariance_arr must be 3D, or both must be 1D.")
        num_component = mean_arr.shape[0]
        num_dimension = mean_arr.shape[1] if mean_arr.ndim == 2 else 1
        if not np.all([mean_arr.shape[0], covariance_arr.shape[0], weight_arr.shape[0]] == [num_component] * 3):
            raise ValueError("Mean, covariance, and weight arrays must have the same number of components.")
        #
        return num_component, num_dimension

    def __eq__(self, other:Any) -> bool:
        """Check if two ParameterMGaussian objects are equal."""
        if not isinstance(other, PCollectionMixture):
            return False
        # Check if all expected parameters are present and equal
        for key in PARAMETER_NAMES:
            if key not in self.dct or key not in other.dct:
                return False
            if np.all(self.dct[key]  != other.dct[key]):
                return False
            if not np.allclose(self.dct[key].flatten(), other.dct[key].flatten()):
                return False
        return True
    
    def select(self, dimensions:List[int]) -> 'PCollectionMixture':
        """
        Selects a subset of the parameters based on the provided indices.

        Args:
            indices (List[int]): List of indices to select from the parameter collection.

        Returns:
            PCollectionMixture: A new PCollectionMixture object with the selected parameters.
        """
        # Check if the dimensions are valid
        self.isValid()
        num_component, num_dimension = self.getComponentAndDimension()
        if max(dimensions) >= num_dimension:
            raise ValueError(f"Dimensions must be less than {num_dimension}. Provided dimensions: {dimensions}")
        # Create the new PCollectionMixture with selected dimensions
        indices = np.array(dimensions, dtype=int)
        dct = dict(self.dct)
        dct[MEAN_ARR] = dct[MEAN_ARR][:, indices] if dct[MEAN_ARR].ndim == 2 else dct[MEAN_ARR][indices]
        dct[COVARIANCE_ARR] = dct[COVARIANCE_ARR][indices, :] if dct[COVARIANCE_ARR].ndim == 2 else dct[COVARIANCE_ARR][indices]
        if len(dimensions) == 1:
            dct[MEAN_ARR] = dct[MEAN_ARR].reshape(num_component)
            dct[COVARIANCE_ARR] = dct[COVARIANCE_ARR].reshape(num_component)
        return PCollectionMixture(dct)
    
    def isValid(self) -> bool:
        return True


################################################
class DCollectionMixture(DCollection):
    # Distribution collection for mixture of Gaussian distributions.

    def __init__(self, collection_dct:Optional[Dict[str, Any]]=None)->None:
        super().__init__(DISTRIBUTION_NAMES, collection_dct)
        # Initialize the properties
        self.entropy = self.dct.get(cn.ENTROPY, None)
        self.density_arr = self.dct.get(DENSITY_ARR, None)
        self.variate_arr = self.dct.get(VARIATE_ARR, None)
        self.dx_arr = self.dct.get(DX_ARR, None)
        # Consistency checks
    
    def isValid(self) -> bool:
        if self.variate_arr is None:
            return False
        if self.density_arr is None:
            return False
        if self.dx_arr is None:
            return False
        num_component = self.variate_arr.shape[0]
        if self.variate_arr.ndim == 1:
            num_dimension = 1
        elif self.variate_arr.ndim == 2:
            num_dimension = self.variate_arr.shape[1]
        else:
            raise ValueError("Variate array must be 1D or 2D.")
        #
        if num_dimension == 1:
            if self.density_arr.ndim != 1:
                return False
            if self.dx_arr.ndim != 1:
                return False
        elif num_dimension > 1:
            if self.density_arr.ndim != 2:
                return False
            if self.dx_arr.ndim != 1:
                return False
            if self.dx_arr.shape[0] != num_dimension:
                return False
            if self.density_arr.shape[0] != num_dimension:
                return False
        return True


################################################
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
        mean_arr, covariance_arr, weight_arr = (cast(np.ndarray, parameter.dct.get(MEAN_ARR, None)),
                                                cast(np.ndarray, parameter.dct.get(COVARIANCE_ARR, None)),
                                                cast(np.ndarray, parameter.dct.get(WEIGHT_ARR, None)))
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
        mean_arr = cast(np.ndarray, pcollection.dct.get(MEAN_ARR, None))
        covariance_arr = cast(np.ndarray, pcollection.dct.get(COVARIANCE_ARR,None))
        weight_arr = cast(np.ndarray, pcollection.dct.get(WEIGHT_ARR, None))
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
        parameter_dct[cn.ENTROPY] = Hx
        parameter_dct[DENSITY_ARR] = pdf_arr
        parameter_dct[VARIATE_ARR] = variate_arr
        parameter_dct[DX_ARR] =  dx_arr
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
        covariance_arr = pcollection.dct.get(COVARIANCE_ARR, None)
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
            MEAN_ARR: self.gmm.means_, 
            COVARIANCE_ARR: self.gmm.covariances_,
            WEIGHT_ARR: self.gmm.weights_}
        self.pcollection = PCollectionMixture(dct)
        return self.pcollection