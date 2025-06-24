'''Collection classes used by RandomMixture of Gaussian distribution.'''
import iplane.constants as cn  # type: ignore
from iplane.random import PCollection, DCollection

import pandas as pd  # type: ignore
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, List, Tuple, Optional, Dict, cast


################################################
class PCollectionMixture(PCollection):
    # Parameter collection for mixture of Gaussian distributions.
    # Instance variables:
    #   collection_names: list of all names of parameters
    #   dct: dictionary of subset name-value pairs

    def __init__(self, **kwargs)->None:
        """
        Args:
            parameter_dct (Optional[Dict[str, Any]], optional): parameter name-value pairs.
        """
        super().__init__(cn.PC_MIXTURE_NAMES, kwargs)

    def getAll(self) ->Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns all parameters as a tuple of numpy arrays.
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple containing mean_arr, covariance_arr, and weight_arr.
        """
        if not self.isAllValid():
            raise ValueError("Parameter dictionary must contain mean_arr, covariance_arr, and weight_arr.")
        mean_arr = cast(np.ndarray, self.get(cn.PC_MEAN_ARR))
        covariance_arr = cast(np.ndarray, self.get(cn.PC_COVARIANCE_ARR))
        weight_arr = cast(np.ndarray, self.get(cn.PC_WEIGHT_ARR))
        return mean_arr, covariance_arr, weight_arr
    
    def getComponentAndDimension(self) -> Tuple[int, int]:
        """
        Finds the number of components and dimensions for the gaussian distribution.

        Returns:
            Tuple[int, int]: Number of components and number of dimensions for the multivariate distribution.
        """
        if self.isAllValid():
            raise ValueError("Parameter dictionary must contain mean_arr, covariance_arr, and weight_arr.")
        mean_arr, covariance_arr, weight_arr = self.getAll()
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
        collection_dct = dict(self.collection_dct)
        collection_dct[cn.PC_MEAN_ARR] = collection_dct[cn.PC_MEAN_ARR][:, indices] if collection_dct[cn.PC_MEAN_ARR].ndim == 2 else collection_dct[cn.PC_MEAN_ARR][indices]
        collection_dct[cn.PC_COVARIANCE_ARR] = collection_dct[cn.PC_COVARIANCE_ARR][indices, :] if collection_dct[cn.PC_COVARIANCE_ARR].ndim == 2 else collection_dct[cn.PC_COVARIANCE_ARR][indices]
        if len(dimensions) == 1:
            collection_dct[cn.PC_MEAN_ARR] = collection_dct[cn.PC_MEAN_ARR].reshape(num_component)
            collection_dct[cn.PC_COVARIANCE_ARR] = collection_dct[cn.PC_COVARIANCE_ARR].reshape(num_component)
        return PCollectionMixture(**collection_dct)


################################################
class DCollectionMixture(DCollection):
    # Distribution collection for mixture of Gaussian distributions.

    def __init__(self, **kwargs)->None:
        super().__init__(cn.DC_MIXTURE_NAMES, kwargs)
        self.actual_collection_dct = dict(kwargs)
    
    def getAll(self) ->Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Returns all parameters as a tuple of numpy arrays.
                    variate_arr: np.ndarray, density_arr: np.ndarray, dx_arr: np.ndarray, entropy: float
        """
        if not self.isAllValid():
            raise ValueError("Parameter dictionary must contain mean_arr, covariance_arr, and weight_arr.")
        variate_arr, density_arr, dx_arr, entropy = cast(np.ndarray, self.get(cn.DC_VARIATE_ARR)),   \
                cast(np.ndarray, self.get(cn.DC_DENSITY_ARR)), cast(np.ndarray, self.get(cn.DC_DX_ARR)),  \
                cast(float, self.get(cn.DC_ENTROPY))
        return variate_arr, density_arr, dx_arr, entropy

    def getComponentAndDimension(self) -> Tuple[int, int]:
        if super().isAllValid():
            raise ValueError("Parameter dictionary must contain mean_arr, covariance_arr, and weight_arr.")
        variate_arr, _, dx_arr, _ = self.getAll()
        num_component = len(dx_arr)
        if variate_arr.ndim == 1:
            num_dimension = 1
        else:
            num_dimension = variate_arr.shape[1]
        return num_component, num_dimension