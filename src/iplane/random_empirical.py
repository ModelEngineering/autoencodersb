'''Describes a continuous distribution based on empirical data.'''
import iplane.constants as cn
from iplane.random import Random, PCollection, DCollection  # type: ignore

import numpy as np
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt
from scipy.interpolate import make_smoothing_spline  # type: ignore
from sklearn.isotonic import IsotonicRegression # type: ignore
from typing import Tuple, Any, Optional, Dict, List, cast


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
            cn.DC_FITTER: fitter,
            cn.DC_VARIATE_ARR: variate_arr,
            cn.DC_DENSITY_ARR: density_arr,
            cn.DC_DX_ARR: dx_arr,
            cn.DC_ENTROPY: entropy
        }
        super().__init__(cn.DC_EMPIRICAL_NAMES, collection_dct)

    def getAll(self) ->Tuple[Any, np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Returns all parameters as a tuple of numpy arrays.
                    variate_arr: np.ndarray, density_arr: np.ndarray, dx_arr: np.ndarray, entropy: float
        """
        fitter, variate_arr, density_arr, dx_arr, entropy =  (
                self.get(cn.DC_FITTER),
                cast(np.ndarray, self.get(cn.DC_VARIATE_ARR)),
                cast(np.ndarray, self.get(cn.DC_DENSITY_ARR)), cast(np.ndarray, self.get(cn.DC_DX_ARR)),
                cast(float, self.get(cn.DC_ENTROPY))
        )
        return fitter, variate_arr, density_arr, dx_arr, entropy


################################################
class RandomEmpirical(Random):

    def __init__(self, pcollection:Optional[PCollectionEmpirical]=None,
            dcollection:Optional[DCollectionEmpirical]=None)->None:
        super().__init__(pcollection, dcollection)

    def estimatePCollection(self, sample_arr:np.ndarray)-> PCollectionEmpirical:
        """Estimates the PCollectionEmpirical values from a categorical array.

        Args:
            categorical_arr (np.ndarray): _description_
        """
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
            variate_arr = cast(np.ndarray, np.linspace(min(sample_arr), max(sample_arr), cn.TOTAL_NUM_SAMPLE))
        dx_arr = np.array(np.mean(np.diff(variate_arr)))
        sorted_sample_arr, cdf_arr = self._calculateEmpiricalCDF()
        # Approximate the CDF
        ir = IsotonicRegression()
        ir.fit(sorted_sample_arr, cdf_arr)
        cdf_arr = ir.predict(variate_arr)
        density_arr = np.diff(cdf_arr)/dx_arr
        density_arr = np.hstack([density_arr[0], density_arr])
        #
        dcollection = DCollectionEmpirical(
            fitter=ir,
            variate_arr=variate_arr,
            density_arr=density_arr,
            dx_arr=np.array([dx_arr])
        )
        entropy = self.calculateEntropy(dcollection)
        dcollection.add(entropy=entropy)
        self.dcollection = dcollection
        plt.plot(variate_arr, density_arr); plt.show()
        import pdb; pdb.set_trace()
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
        _, _, density_arr, dx_arr, entropy = dcollection.getAll()
        entropy = - np.sum(density_arr * np.log2(density_arr + 1e-10) * dx_arr)
        return entropy
    
    def predict(self, variate_arr:np.ndarray, collection:Optional[DCollectionEmpirical]=None) -> float:
        """Predicts the probability of a variate based on the empirical distribution.

        Args:
            variate_arr (np.ndarray): Array of variates to predict.
            pcollection (Optional[PCollectionEmpirical], optional): PCollectionEmpirical with the parameters of the distribution. Defaults to None.

        Returns:
            float: Probability of the variate.
        """
        dcollection = cast(DCollectionEmpirical, self.setDCollection(collection))
        fitter = dcollection.get(cn.DC_FITTER)
        # Interpolate the density function
        density_value = fitter.predict(variate_arr)
        return density_value

    def plot(self, dcollection:Optional[DCollectionEmpirical]=None)->None:
        """Plots the distribution"""
        dcollection = cast(DCollectionEmpirical, self.setDCollection(dcollection))
        _, variate_arr, density_arr, dx_arr, entropy = dcollection.getAll()
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