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
from scipy.integrate import simpson  # type: ignore
from typing import Tuple, Optional, cast

MIN_NUM_DIMENSION_SAMPLE = 5  # Minimum number of samples per dimension for variate array generation


################################################
class PCollectionContinuous(PCollection):

    def __init__(self, *args, **kwargs) -> None:
        # Parameter collection for continuous random variables.
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
        super().__init__(cn.DC_CONTINUOUS_NAMES, dct)
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

    def __init__(self,
            num_variate_sample:int = cn.NUM_VARIATE_SAMPLE,
            axis_length_std:float = cn.AXIS_LENGTH_STD,
            min_num_dimension_coordinate:Optional[int] = None,
            **kwargs) -> None:
        """ Initializes the RandomContinuous object.
        Args:
            num_variate_sample (int): number of samples to generate for the variate array
            axis_length_std (float) length of each axis of variate_arr in units of std
            min_num_dimension_sample (int): Mininum number of coordinate for each dimension
        """
        super().__init__(**kwargs)
        if min_num_dimension_coordinate is None:
            min_num_dimension_coordinate = MIN_NUM_DIMENSION_SAMPLE
        self.axis_length_std = axis_length_std
        self.num_variate_sample = num_variate_sample
        self.min_num_dimension_coordinate = min_num_dimension_coordinate

    def makeVariate(self, min_point:np.ndarray, max_point:np.ndarray, num_variate_sample:int
                ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates the variate array for a continuous distribution.

        Args:
            min_point (np.ndarray): Minimum point for each dimension.
            max_point (np.ndarray): Maximum point for each dimension.
            num_variate_sample (int): Number of samples to generate for the variate array

        Returns:
            np.ndarray: The generated variate array.
        """
        #
        num_dimension = np.shape(min_point)[0]
        # Calculate the number of samples for each dimension
        num_dim_sample = int(num_variate_sample**(1/num_dimension))
        if num_dim_sample < self.min_num_dimension_coordinate:
            msg = f"Number of samples per dimension must be at least {self.min_num_dimension_coordinate}."
            msg += f"\n  Increase max_num_sample so that {self.min_num_dimension_coordinate}**num_dimesion <= max_num_sample,"
            msg += f"\n  Currently:"
            msg += f"\n    num_sample={num_variate_sample}"
            msg += f"\n    max_num_sample={self.num_variate_sample}"
            msg += f"\n    num_dimension={num_dimension})."
            raise ValueError(msg)
        # Create linspace for each dimension
        linspaces:list = []
        dxs:list = []
        for i_dim in range(num_dimension):
            linspace_arr = np.linspace(min_point[i_dim], max_point[i_dim], num=num_dim_sample).flatten()
            linspaces.append(linspace_arr)
            dxs.append(np.mean(np.diff(linspace_arr)))
        variate_arr = np.array(list(itertools.product(*linspaces)))  # Create a grid of variates
        variate_arr = variate_arr.reshape(-1, num_dimension)  # Reshape to (num_sample, num_dimension)
        dx_arr = np.array(dxs)
        #
        return variate_arr, dx_arr

    def makeEntropy(self, density_arr:np.ndarray, dx_arr:np.ndarray) -> float:
        """
        Numerical calculation of differential entropy for a continuous distribution.

        Args:
            density (np.ndarray): density array of the distribution.
            dx_arr (np.ndarray): differential elements array of the distribution.   

        Returns:
            float: differential entropy
        """
        # Do the integration
        dx = float(np.prod(dx_arr))
        integrand = -density_arr*np.log2(density_arr + 1e-30)
        entropy = np.sum(integrand) * dx
        return entropy