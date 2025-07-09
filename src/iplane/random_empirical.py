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
            total_num_sample:int=cn.TOTAL_NUM_SAMPLE,
            window_size:int=1)->None:
        super().__init__(pcollection, dcollection)
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

    def estimatePCollection(self, sample_arr:np.ndarray)-> PCollectionEmpirical:
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
    
    def makeDCollection(self,
            pcollection:PCollectionEmpirical, variate_arr:Optional[np.ndarray]=None,
            dx_arr:Optional[np.ndarray]=None) -> DCollectionEmpirical:
        """Calculates entropy for the discrete random variable.
        Extends variate_arr by self.window_size so that the result is the same as the original variate_arr.

        Args:
            pcollection (PCollectionDiscrete)
                PCollectionDiscrete with the parameters of the distribution.
            variate_arr (Optional[np.ndarray], optional): Not used for discrete random variables. Defaults to None.            

        Returns:
            DCollectionDiscrete:
        """
        sample_arr = cast(np.ndarray, pcollection.get(cn.PC_TRAINING_ARR))
        if sample_arr.ndim != 2:
            raise ValueError(f"Expected 2D array, got {sample_arr.ndim}D array.")
        # FIXME: Consider multiple dimensions and calculate dx_arr
        if variate_arr is None
            center_point = np.mean(sample_arr, axis=0)
            std_arr = np.std(sample_arr, axis=0)
            min_point = center_point - self.width_std*std_arr
            max_point = center_point + self.width_std*std_arr
            variate_result = self.makeVariate(
                min_point=min_point,
                max_point=max_point,
                num_sample=self.total_num_sample
            )
            variate_arr, dx_arr = variate_result.variate_arr, variate_result.dx_arr
        else:
            if dx_arr is None:
                raise ValueError("dx_arr must be provided if variate_arr is provided.")
        # Use interpolate to construct the density from the empirical distribution
        initial_dcollection = DCollectionEmpirical(
            variate_arr=sorted_sample_arr[:len(density_arr)],
            density_arr=density_arr,
            dx_arr=np.array([dx_arr])
        )
        full_density_arr = self.predict(variate_arr, dcollection=initial_dcollection)
        #
        #
        dcollection = DCollectionEmpirical(
            variate_arr=variate_arr,
            density_arr=full_density_arr,
            dx_arr=np.array([dx_arr])
        )
        entropy = self.calculateEntropy(dcollection)
        dcollection.add(entropy=entropy)
        self.dcollection = dcollection
        return self.dcollection

    # FIXME: Calculate density from interpolation    
    def predict(self, single_variate_arr:np.ndarray,
            pcollection:Optional[PCollectionEmpirical]=None,
            dcollection:Optional[DCollectionEmpirical]=None) -> np.ndarray:
        """Predicts the probability of a variate based on the empirical distribution.

        Args:
            single_variate_arr (np.ndarray): Array of multiple 1D variates to predict.
            pcollection (Optional[PCollectionEmpirical], optional): PCollectionEmpirical with the parameters of the distribution. Defaults to None.

        Returns:
            float: Probability of the variate.
        """
        #
        # FIXME: Do top N closest variates?
        dcollection = cast(DCollectionEmpirical, self.setDCollection(dcollection))
        density_arr = dcollection.get(cn.DC_DENSITY_ARR)
        variate_arr = dcollection.get(cn.DC_VARIATE_ARR)
        # Find the two closest values in the variate_arr to the single_variate_arr and obtain their densities.
        estimates:list = []
        for variate in single_variate_arr:
            squared_difference_arr = (variate - variate_arr)**2
            idx1 = np.argmin(squared_difference_arr)
            value1 = density_arr[idx1]
            squared_difference_arr[idx1] = np.inf  # Ignore the closest point
            idx2 = np.argmin(squared_difference_arr)
            value2 = density_arr[idx2]
            # Interpolate between the two closest points
            distance1 = np.sqrt(np.sum(np.abs(variate_arr[idx1] - single_variate_arr))**2)
            distance2 = np.sqrt(np.sum(np.abs(variate_arr[idx2] - single_variate_arr))**2)
            estimate = (value1 * distance2 + value2 * distance1) / (distance1 + distance2)
            estimates.append(estimate)
        #
        return np.array(estimates)

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