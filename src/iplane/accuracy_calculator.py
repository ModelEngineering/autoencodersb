'''Calculates accuracy statistics for fractional errors.'''

from collections import namedtuple
import torch
import pandas as pd  # type: ignore
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import List, Optional, cast, Union

PERCENTILE_STATISTICS = [0.9, 0.99]
COL_CDF = 'cdf'
COL_ERROR = 'error'


AccuracyResult = namedtuple('AccuracyResult', ['accuracy', 'mean_absolute_error', 'cdf_df'])


class AccuracyCalculator(object):
    """Calculates accuracy statistics for fractional errors."""

    def __init__(self, error_arr: Union[torch.Tensor, np.ndarray]):
        """
        Args:
            error_arr (np.ndarray): Array of fractional errors.
        """
        if isinstance(error_arr, torch.Tensor):
            error_arr = error_arr.numpy()
        self.error_arr = error_arr
        self.accuracy_result : Union[AccuracyResult, None] = None

    def calculateCDF(self) -> AccuracyResult:
        """
        Calculate the cumulative distribution function (CDF) of the errors.
        
        Returns:
            AccuracyResult: Named tuple containing accuracy, mean absolute error, and CDF DataFrame.
        """
        # Calculate accuracy as the percentage of errors within a threshold
        accuracy = np.mean(np.abs(self.error_arr) < 0.1)
        # Calculate mean absolute error
        mean_absolute_error = np.mean(np.abs(self.error_arr))
        # Create a DataFrame for CDF
        cdf_df = pd.DataFrame({COL_ERROR: np.sort(self.error_arr)})
        cdf_df[COL_CDF] = np.arange(1, len(cdf_df) + 1) / len(cdf_df)
        #
        self.accuracy_result = cast(AccuracyResult, self.accuracy_result)
        self.accuracy_result = AccuracyResult(accuracy=accuracy,
                mean_absolute_error=mean_absolute_error, cdf_df=cdf_df)
        return self.accuracy_result
    
    @classmethod
    def getStatistics(cls, error_arr: Union[torch.Tensor, np.ndarray],
            percentiles: List[float]=PERCENTILE_STATISTICS) -> List[float]:
        """
        Get accuracy statistics for the given error array.

        Args:
            error_arr (np.ndarray): Array of fractional errors.
            percentiles (List[float]): List of percentiles to calculate.

        Returns:
            List[float]: List of accuracy statistics for the specified percentiles.
        """
        if isinstance(error_arr, torch.Tensor):
            error_arr = error_arr.numpy()
        ser = pd.Series(error_arr)
        results = ser.quantile(percentiles).values.tolist() # type: ignore
        return results
    
    def plotCDF(self, ax: Optional[Axes] = None,
            is_plot: bool=True) -> None:
        """
        Plot the CDF of the errors.
        
        Args:
            ax (plt.Axes, optional): Matplotlib axes object. If None, creates a new figure.
            is_plot (bool, optional): Whether to plot the CDF. Defaults to True.
        """
        if ax is None:
            plt.figure(figsize=(10, 6))
            ax = plt.gca()
        # Create data
        cdf_df = self.calculateCDF().cdf_df
        # Plot
        ax.plot(cdf_df[COL_ERROR], cdf_df[COL_CDF], marker='o')
        ax.set_title('Cumulative Distribution Function (CDF) of Errors')
        ax.set_xlabel('Error')
        ax.set_ylabel('CDF')
        ax.grid()
        #
        if is_plot:
            plt.show()

    def plotCDFComparison(self, calculator: 'AccuracyCalculator',
            names: Optional[List[str]] = None,
            ax: Optional[Axes] = None, is_plot: bool=True) -> None:
        """
        Plot the CDF of the errors from this calculator and another calculator.
        
        Args:
            calculator (AccuracyCalculator): Another AccuracyCalculator instance.
            names (Optional[List[str]]): List of names for the plots.
            ax (plt.Axes, optional): Matplotlib axes object. If None, creates a new figure.
            is_plot (bool, optional): Whether to plot the CDF. Defaults to True.
        """
        if ax is None:
            plt.figure(figsize=(10, 6))
            ax = plt.gca()
        if names is None:
            names = ['Current', 'Other']
        # Create data
        cdf_df = self.calculateCDF().cdf_df
        other_cdf_df = calculator.calculateCDF().cdf_df
        # Plot
        ax.plot(cdf_df[COL_ERROR], cdf_df[COL_CDF], marker='o',
                label=names[0])
        ax.plot(other_cdf_df[COL_ERROR], other_cdf_df[COL_CDF], marker='x',
                label=names[1])
        ax.set_title('Cumulative Distribution Function (CDF) Comparison')
        ax.set_xlabel('Error')
        ax.set_ylabel('CDF')
        ax.legend()
        if is_plot:
            plt.show()