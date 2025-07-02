'''Describes a continuous distribution based on empirical data.'''
import iplane.constants as cn
from iplane.random import Random, PCollection, DCollection  # type: ignore

from collections import namedtuple
import numpy as np
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt
from scipy import interpolate  # type: ignore
from sklearn.isotonic import IsotonicRegression # type: ignore
from typing import Tuple, Any, Optional, Dict, List, cast


CDF = namedtuple('CDF', ['variate_arr', 'cdf_arr'])
"""To Do
1. Interpolate for multivariate CDF
2. dcollection is CDF
3. construction distribtion for variate_arr
"""


################################################
class PCollectionEmpirical(PCollection):
    # Parameter collection for mixture of Gaussian distributions.

    def __init__(self, training_arr:np.ndarray)->None:
        collection_dct = {
            cn.PC_TRAINING_ARR: training_arr,
        }
        super().__init__(cn.PC_EMPIRICAL_NAMES, collection_dct)


################################################
class DCollectionEmpirical(DCollection):
    # Distribution collection for mixture of Gaussian distributions.

    def __init__(self, 
            fitter:Optional[Any]=None,
            variate_arr:Optional[np.ndarray]=None,
            density_arr:Optional[np.ndarray]=None,
            dx_arr:Optional[np.ndarray]=None,
            entropy:Optional[float]=None)->None:
        collection_dct = {
            cn.DC_VARIATE_ARR: variate_arr,
            cn.DC_DENSITY_ARR: density_arr,
            cn.DC_DX_ARR: dx_arr,
            cn.DC_ENTROPY: entropy
        }
        super().__init__(cn.DC_EMPIRICAL_NAMES, collection_dct)

    def getAll(self) ->Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Returns all parameters as a tuple of numpy arrays.
                    variate_arr: np.ndarray, density_arr: np.ndarray, dx_arr: np.ndarray, entropy: float
        """
        variate_arr, density_arr, dx_arr, entropy =  (
                cast(np.ndarray, self.get(cn.DC_VARIATE_ARR)),
                cast(np.ndarray, self.get(cn.DC_DENSITY_ARR)), cast(np.ndarray, self.get(cn.DC_DX_ARR)),
                cast(float, self.get(cn.DC_ENTROPY))
        )
        return variate_arr, density_arr, dx_arr, entropy


################################################
class RandomEmpirical(Random):

    def __init__(self, pcollection:Optional[PCollectionEmpirical]=None,
            dcollection:Optional[DCollectionEmpirical]=None,
            total_num_sample:int=cn.TOTAL_NUM_SAMPLE,
            window_size:int=1)->None:
        super().__init__(pcollection, dcollection)
        self.total_num_sample = total_num_sample
        self.window_size = window_size

    def estimatePCollection(self, sample_arr:np.ndarray)-> PCollectionEmpirical:
        """Estimates the PCollectionEmpirical values from a categorical array.

        Args:
            categorical_arr (np.ndarray): _description_
        """
        sample_arr = np.array(sample_arr, dtype=float)
        sample_arr.sort()
        self.pcollection = PCollectionEmpirical(training_arr=sample_arr)
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
            variate_arr:Optional[np.ndarray]=None,
            pcollection:Optional[PCollection]=None) -> DCollectionEmpirical:
        """Calculates entropy for the discrete random variable.
        Extends variate_arr by self.window_size so that the result is the same as the original variate_arr.

        Args:
            pcollection (PCollectionDiscrete)
                PCollectionDiscrete with the parameters of the distribution.
            variate_arr (Optional[np.ndarray], optional): Not used for discrete random variables. Defaults to None.            

        Returns:
            DCollectionDiscrete:
        """
        pcollection = self.setPCollection(pcollection)
        sample_arr = cast(np.ndarray, pcollection.get(cn.PC_TRAINING_ARR))
        if sample_arr.ndim != 1:
            raise ValueError(f"Expected 1D array, got {sample_arr.ndim}D array.")
        if variate_arr is None:
            variate_arr = cast(np.ndarray, np.linspace(min(sample_arr), max(sample_arr), self.total_num_sample))
        dx_arr = np.array(np.mean(np.diff(variate_arr)))
        # Extend the variate_arr by self.window_size so that the result is the same as the original variate_arr.
        if False:
            tail_arr = variate_arr[-1] + np.cumsum(np.repeat(dx_arr, self.window_size-1))
            new_variate_arr = np.hstack([variate_arr, tail_arr])
        else:
            new_variate_arr = variate_arr
        # Approximate the CDF
        sorted_sample_arr, cdf_arr = self._calculateEmpiricalCDF()
        # Calculate the density function for the sample_arr
        density_arr = np.diff(cdf_arr)/dx_arr
        density_arr = np.hstack([density_arr[0], density_arr])  # Compensate for the differencing
        #   Smooth over a window
        if False:
            weights = np.ones(self.window_size) / self.window_size
            density_arr = np.convolve(density_arr, weights, mode='valid')
        # Calculate density for the variate_arr by interpolating the empirical CDF
        initial_dcollection = DCollectionEmpirical(
            variate_arr=sorted_sample_arr[:len(density_arr)],
            density_arr=density_arr,
            dx_arr=np.array([dx_arr])
        )
        full_density_arr = self.predict(new_variate_arr, initial_dcollection)
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
    
    def calculateEntropy(self, collection:DCollectionEmpirical) -> float:
        """
        Calculates the entropy of a univariate empirical distribution.

        Args:
            covariance_arr (np.ndarray): Covariance matrix of the Gaussian distribution.

        Returns:
            float: Entropy of the Gaussian distribution.
        """
        dcollection = cast(DCollectionEmpirical, collection)
        _, density_arr, dx_arr, entropy = dcollection.getAll()
        entropy = - np.sum(density_arr * np.log2(density_arr + 1e-10) * dx_arr)
        return entropy
    
    def predict(self, single_variate_arr:np.ndarray, collection:Optional[DCollectionEmpirical]=None) -> np.ndarray:
        """Predicts the probability of a variate based on the empirical distribution.

        Args:
            single_variate_arr (np.ndarray): Array of multiple 1D variates to predict.
            pcollection (Optional[PCollectionEmpirical], optional): PCollectionEmpirical with the parameters of the distribution. Defaults to None.

        Returns:
            float: Probability of the variate.
        """
        #
        # FIXME: Do top N closest variates?
        dcollection = cast(DCollectionEmpirical, self.setDCollection(collection))
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

    def makeCDF(self, variate_arr:np.ndarray) -> CDF:
        """Constructs a CDF from a two dimensional array of variates. Rows are instances; columns are variables.

        Args:
            variate_arr (np.ndarray):

        Returns:
            CDF
                variate_arr (N X D)
                cdf_arr (N): CDF corresponding to the variate_arr
        """
        if variate_arr.ndim != 2:
            raise ValueError(f"Expected 2D array, got {variate_arr.ndim}D array.")
        #
        cdfs:list = []
        num_sample = variate_arr.shape[0]
        num_variable = variate_arr.shape[1]
        #
        for variate in variate_arr:
            less_than_arr = variate_arr <= variate
            satisfies_arr = np.sum(less_than_arr, axis=1) == num_variable
            cdfs.append(np.sum(satisfies_arr))
        cdf_arr = np.array(cdfs, dtype=float)
        if np.sum(cdf_arr == num_sample) == 0:
            variate = np.array([max(variate_arr[:, n]) for n in range(num_variable)])
            cdf_arr = np.hstack([cdf_arr, np.array([num_sample])])
            full_variate_arr = np.vstack([variate_arr, variate])
        else:
            full_variate_arr = variate_arr
        idx_arr = np.argsort(cdf_arr)
        full_variate_arr = full_variate_arr[idx_arr]
        cdf_arr = cdf_arr[idx_arr]
        #
        return CDF(variate_arr=full_variate_arr, cdf_arr=cdf_arr)