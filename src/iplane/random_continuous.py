'''Abstract class for random continuous distributions.'''

"""
All continuous distributions produce a DCollectionContinuous object (or a subclass of it).
This class provides for constructing a collection of continuous distributions.
    variate_arr: numpy array of variates, shape (N, D) where N is the number of samples and D is the number of dimensions.
    density_arr: numpy array of densities, shape (N,)
    dx_arr: numpy array of differential elements, shape (D,)
    entropy: float
Subclasses must implement the following:
    predict(single_variate_arr: np.ndarray, collection: Optional[PCollection] = None) -> np.ndarray
"""

import iplane.constants as cn
from iplane.random import Random, PCollection, DCollection

import collections
import itertools
import pandas as pd  # type: ignore
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, cast


VariateResult = collections.namedtuple("VariateResult", ["variate_arr", "dx_arr"])


################################################
class PCollectionContinuous(PCollection):

    def __init__(self, *args, **kwargs) -> None:
        # Parameter collection for continuous random variables.
        #   C: number of components
        #   N: number of samples
        #   D: number of dimensions
        #   mean_arr: C X D
        #   covariance_arr: C X D X D
        #   weight_arr: C
        super().__init__(*args, **kwargs)
    pass


################################################
class DCollectionContinuous(DCollection):

    def __init__(self, 
                variate_arr:Optional[np.ndarray]=None,
                density_arr:Optional[np.ndarray]=None,
                dx_arr:Optional[np.ndarray]=None,
                entropy:Optional[float]=None)->None:
        # Distribution collection for continuous distributions.
        #   C: number of components
        #   N: number of samples
        #   D: number of dimensions
        #   variate_arr: N X D
        #   density_arr: N
        #   dx_arr: D
        #   entropy: float
        dct = dict(
            variate_arr=variate_arr,
            density_arr=density_arr,
            dx_arr=dx_arr,
            entropy=entropy,
        )
        super().__init__(cn.DC_MIXTURE_NAMES, dct)
        self.actual_collection_dct = dct
        self.isAllValid()

    @property
    def num_dimension(self) -> int:
        """
        Returns the number of dimensions in the variate array.
        """
        if self.get(cn.DC_VARIATE_ARR) is None:
            raise ValueError("Variate array is not set.")
        variate_arr = self.get(cn.DC_VARIATE_ARR)
        if variate_arr.ndim == 1:
            raise ValueError("Variate array must be 2D.")
        return variate_arr.shape[1]

    def __str__(self) -> str:
        """
        Returns a string representation of the DCollectionMixture object.
        """
        variate_arr, density_arr, dx_arr, entropy = self.getAll()
        return (f"DCollectionContinuous(variate_arr={variate_arr}"
                f"\ndensity_arr={density_arr}"
                f"\ndx_arr={dx_arr}"
                f"\nentropy={entropy})")
    
    def getAll(self) ->Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Returns all parameters as a tuple of numpy arrays.
                    variate_arr: np.ndarray, density_arr: np.ndarray, dx_arr: np.ndarray, entropy: float
        """
        variate_arr, density_arr, dx_arr, entropy = cast(np.ndarray, self.get(cn.DC_VARIATE_ARR)),   \
                cast(np.ndarray, self.get(cn.DC_DENSITY_ARR)), cast(np.ndarray, self.get(cn.DC_DX_ARR)),  \
                cast(float, self.get(cn.DC_ENTROPY))
        return variate_arr, density_arr, dx_arr, entropy


################################################
class RandomContinuous(Random):
    """Abstract class for random continuous distributions."""

    def __init__(self, pcollection: Optional[PCollection] = None,
            dcollection: Optional[DCollection] = None,
            total_num_sample:int = cn.TOTAL_NUM_SAMPLE,
            width_std:float = 4.0,
            **kwargs) -> None:
        """ Initializes the RandomContinuous object.
        Args:
            pcollection (Optional[PCollection]): The collection of parameters for the distribution.
            dcollection (Optional[DCollection]): The collection of distributions.
            total_num_sample (int): Total number of samples to generate.
            width_std (float): Two sided width in standard deviations for the variate range.
        """
        super().__init__(pcollection=pcollection, dcollection=dcollection, **kwargs)
        self.width_std = width_std
        self.total_num_sample = total_num_sample

    def makeVariate(self, min_point:np.ndarray, max_point:np.ndarray, num_sample:int) -> VariateResult:
        """
        Generates the variate array for a continuous distribution.

        Args:
            min_point (np.ndarray): Minimum point for each dimension.
            max_point (np.ndarray): Maximum point for each dimension.
            num_sample (int): Number of samples to generate.

        Returns:
            np.ndarray: The generated variate array.
        """
        #
        num_dimension = np.shape(min_point)[0]
        # Construct the variate array
        total_num_sample = self.total_num_sample
        # Calculate the number of samples for each dimension
        num_dim_sample = int(total_num_sample**(1/num_dimension))
        if num_dim_sample < 8:
            msg = "Number of samples per dimension must be at least 8."
            msg += f"\n  Increase max_num_sample so that 8**num_dimesion <= max_num_sample,"
            msg += f"\n  Currently:"
            msg += f"\n    num_sample={num_sample}"
            msg += f"\n    max_num_sample={self.total_num_sample}"
            msg += f"\n    num_dimension={num_dimension})."
            raise ValueError(msg)
        # Create linspace for each dimension
        linspaces:list = []
        for i_dim in range(num_dimension):
            linspace_arr = np.linspace(min_point[i_dim], max_point[i_dim], num=num_dim_sample).flatten()
            linspaces.append(linspace_arr)
        variate_arr = np.array(list(itertools.product(*linspaces)))  # Create a grid of variates
        variate_arr = variate_arr.reshape(-1, num_dimension)  # Reshape to (num_sample, num_dimension)
        dx_arr = np.array([np.mean(np.diff(variate_arr[:, i])) for i in range(num_dimension)])
        #
        return VariateResult(variate_arr=variate_arr, dx_arr=dx_arr)
    
    def predict(self, *args, **kwargs)->np.ndarray:
        """
        Predicts the probability density function (PDF) for a given array of variates using the Gaussian mixture model.

        Returns:
            np.ndarray: Array of predicted density values for each variate.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def makeEntropy(self, density_arr:np.ndarray, dx_arr:np.ndarray) -> float:
        """
        Numerical calculation of differential entropy for a continuous distribution. Using previously
        construte

        Args:
            density (np.ndarray): density array of the distribution.
            dx_arr (np.ndarray): differential elements array of the distribution.   

        Returns:
            float: differential entropy
        """
        entropy = np.sum(density_arr * np.log2(density_arr + 1e-10) * dx_arr)
        return entropy