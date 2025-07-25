'''Describes a continuous distribution based on empirical data.'''
import iplane.constants as cn
from iplane.random_continuous import RandomContinuous, PCollectionContinuous, DCollectionContinuous  # type: ignore

from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression # type: ignore
from typing import Tuple, Any, Optional, Dict, List, cast


CDF = namedtuple('CDF', ['variate_arr', 'cdf_arr'])
"""To Do
1. Interpolate for multivariate CDF
2. dcollection is CDF
3. construction distribtion for variate_arr
4. variate_arr is a 1D point. point_arr is a 2D array of points.
"""


################################################
class PCollectionEmpirical(PCollectionContinuous):
    # Parameter collection for mixture of Gaussian distributions.

    def __init__(self, training_arr:np.ndarray)->None:
        collection_dct = {
            cn.PC_TRAINING_ARR: training_arr,
        }
        super().__init__(cn.PC_EMPIRICAL_NAMES, collection_dct)


################################################
class DCollectionEmpirical(DCollectionContinuous):
    # Distribution collection for mixture of Gaussian distributions.
    pass


################################################
class RandomEmpirical(RandomContinuous):

    def __init__(self, pcollection:Optional[PCollectionEmpirical]=None,
            dcollection:Optional[DCollectionEmpirical]=None,
            total_num_sample:int=cn.NUM_VARIATE_SAMPLE,
            window_size:int=1)->None:
        super().__init__(pcollection=pcollection, dcollection=dcollection)
        self.total_num_sample = total_num_sample
        self.window_size = window_size
        self._initialize()

    def _initialize(self) -> None:
        self._min_point: Optional[np.ndarray] = None
        self._max_point: Optional[np.ndarray] = None

    @property
    def min_point(self) -> np.ndarray:
        """Minimum point in the variate space."""
        if self._min_point is None:
            if self.pcollection is None:
                raise RuntimeError("PCollection has not been set.")
            sample_arr = cast(np.ndarray, self.pcollection.get(cn.PC_TRAINING_ARR))
            self._min_point = cast(np.ndarray, np.min(sample_arr, axis=0))
        return self._min_point
    
    @property
    def max_point(self) -> np.ndarray:
        """Maximum point in the variate space."""
        if self._max_point is None:
            if self.pcollection is None:
                raise RuntimeError("PCollection has not been set.")
            sample_arr = cast(np.ndarray, self.pcollection.get(cn.PC_TRAINING_ARR))
            self._max_point = cast(np.ndarray, np.max(sample_arr, axis=0))
        return self._max_point

    def makePCollection(self, sample_arr:np.ndarray)-> PCollectionEmpirical:
        """Estimates the PCollectionEmpirical values from a categorical array.

        Args:
            categorical_arr (np.ndarray): _description_
        """
        self.pcollection = PCollectionEmpirical(training_arr=sample_arr)
        self._initialize()
        return self.pcollection

    def _calculateEmpiricalCDF(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates the empirical distribution.

        Returns:
            sorted_sample_arr: Sorted sample array
            cdf_arr: Cumulative distribution function array corresponding to sample_arr
        """
        sample_arr = cast(np.ndarray, self.pcollection.get(cn.PC_TRAINING_ARR))  # type: ignore
        sorted_sample_arr = np.sort(sample_arr)
        # Construct the CDF
        pdf_arr = np.repeat(1.0 / len(sample_arr), len(sample_arr))
        cdf_arr = np.cumsum(pdf_arr)
        return sorted_sample_arr, cdf_arr
    
    def plot(self, dcollection:Optional[DCollectionEmpirical]=None)->None:
        """Plots the distribution"""
        dcollection = cast(DCollectionEmpirical, self.setDCollection(dcollection))
        variate_arr, density_arr, dx_arr, entropy = dcollection.getAll()
        #
        sorted_sample_arr, empirical_cdf_arr = self._calculateEmpiricalCDF()
        variate_arr = dcollection.get(cn.DC_VARIATE_ARR)
        fitted_cdf_arr = np.cumsum(density_arr) * dx_arr
        # Plot the empirical distribution
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        ax.plot(sorted_sample_arr, empirical_cdf_arr, color='red', label='Empirical CDF')
        ax.plot(variate_arr, fitted_cdf_arr, color='blue', label='Fitted CDF')
        ax.set_xlabel('Variate')
        ax.set_ylabel('Density')
        ax.set_title(f'Empirical Distribution (Entropy: {entropy:.2f})')
        ax.legend()
        #plt.plot(variate_arr[0:-(window_size-1)], density_arr); plt.show()
        #plt.plot(variate_arr[0:-(window_size-1)], np.cumsum(density_arr)); plt.show()

    def makeCDF(self, sample_arr:np.ndarray) -> CDF:
        """Constructs a CDF from a two dimensional array of variates. Rows are instances; columns are variables.

        Args:
            sample_arr (np.ndarray) (N X D): An array of points, each of which is a D-dimensional array.

        Returns:
            CDF
                variate_arr (N X D)
                cdf_arr (N): CDF corresponding to the variate_arr
        """
        if sample_arr.ndim != 2:
            raise ValueError(f"Expected 2D array, got {sample_arr.ndim}D array.")
        #
        cdfs:list = []
        num_sample = sample_arr.shape[0]
        num_variable = sample_arr.shape[1]
        #
        for point in sample_arr:
            less_than_arr = sample_arr <= point
            less_satisfies_arr = np.sum(less_than_arr, axis=1) == num_variable
            count_less = np.sum(less_satisfies_arr) - 1
            cdf_val = count_less/num_sample
            cdfs.append(cdf_val)
        # Add a new point if there is none that is greater than all
        import pdb; pdb.set_trace()
        if all([np.all(p != self.min_point) for p in sample_arr]):
            cdfs.append(0)
            full_variate_arr = np.vstack([sample_arr, np.array([self.min_point])])
        else:
            full_variate_arr = sample_arr
        # Add a new point if the max point does't exist
        if all([np.all(p != self.max_point) for p in sample_arr]):
            cdfs.append(1)
            full_variate_arr = np.vstack([sample_arr, np.array([self.max_point])])
        else:
            full_variate_arr = sample_arr
        # Complete the cdf calculations
        cdf_arr = np.array(cdfs, dtype=float)
        #distance_arr = np.sqrt(np.sum(full_variate_arr**2, axis=1))
        #
        return CDF(variate_arr=full_variate_arr, cdf_arr=cdf_arr)