'''Interpolation for scalar functions of multiple variables.'''
import iplane.constants as cn
from iplane.random import Random, PCollection, DCollection  # type: ignore

from collections import namedtuple
import numpy as np
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt
from typing import Tuple, Any, Optional, Dict, List, cast


"""
Notation:
    N - number of samples (first dimension of the variate_arr, sample_arr)
    M - number of points
    D - number of variate dimensions (second dimension of the variate_arr)
"""
class Interpolator(object):

    def __init__(self, variate_arr:np.ndarray, sample_arr:np.ndarray,
            is_normalize:bool=True, max_distance:float=1.0,
            size_interpolation_set:int=5,
    ):
        """
        Multivariate interpolator for empirical distributions. A point is a vector in the variate space.
            N - number of samples
            D - number of variate (point) dimensions
        Args:
            variate_arr (np.ndarray N X D)
            sample_arr (np.ndarray N X 1)
            is_normalize (bool, optional): normalize values of the variate_arr by dividing by standard deviation.
            max_distance (float, optional): maximum distance between a variate_arr entry and points to be estimated
            max_size_interpolation_set (int, optional): _description_. Defaults to 5.
        """
        self._checkArray(variate_arr)
        self.variate_arr = np.array(variate_arr, dtype=float)
        self.sample_arr = np.array(sample_arr, dtype=float).reshape(-1)
        self.num_sample = self.variate_arr.shape[0]
        self.num_variate_dimension = self.variate_arr.shape[1]
        self.is_normalize = is_normalize
        self.max_distance = max_distance
        self.size_interpolation_set = size_interpolation_set
        #
        self.std_arr = np.std(self.variate_arr, axis=0)
        self.normalized_variate_arr = self._normalize(self.variate_arr)
        self.min_value = np.array(np.min(self.variate_arr, axis=0))
        self.max_value = np.array(np.max(self.variate_arr, axis=0))

    def isWithinRange(self, point_arr:np.ndarray) -> bool:
        """
        Determines if point is within the range for interpolation

        Args:
            point_arr (np.ndarray): _description_

        Returns:
            bool: _description_
        """
        is_less_than = np.all(point_arr < self.variate_arr)
        is_greater_than = np.all(point_arr > self.variate_arr)
        if is_less_than or is_greater_than:
            return False
        else:
            return True

    def _normalize(self, point_arr:np.ndarray) -> np.ndarray:
        """Normalizes the point array by dividing each variate by its standard deviation.

        Args:
            point_arr (np.ndarray): shape: (N,) or (N, D) where N is the number of samples and D is the number of dimensions.

        Returns:
            np.ndarray: _description_
        """
        return point_arr / self.std_arr
    
    def _getIndexNearestVariate(self, point_arr:np.ndarray, exclude_idxs:List[int]) -> Tuple[int, float]:
        """Finds the closest variates to the given point in the variate space.

        Args:
            point_arr (np.ndarray): Point in the variate space.
            exclude_idxs (List[int]): List of indices to exclude from the search.

        Returns:
            index of closest variate (int). If distance > max_distance, returns -1.
        """
        if self.is_normalize:
            point_arr = np.array(self._normalize(point_arr))
            variate_arr = np.array(self.normalized_variate_arr)
        else:
            variate_arr = np.array(self.variate_arr)
        # Exclude specified indices
        exclude_arr = np.array(exclude_idxs, dtype=int)
        variate_arr[exclude_arr] = np.inf  # Set excluded variates to infinity
        # Calculate squared differences
        distance_arr = np.sqrt(np.sum((variate_arr - point_arr)**2, axis=1))
        idx = cast(int, np.argmin(distance_arr))
        distance = distance_arr[idx]
        if distance > self.max_distance:
            idx = -1
        return idx, distance
    

    def predict(self, point_arr:np.ndarray) -> np.ndarray:
        """Uses nearest variates to estimate the value in the sample_arr space.

        Args:
            point_arr (np.ndarray M X D): Variate to estimate the probability for.

        Returns:
            np.ndarray: Values corresponding to each point in point_arr
        """
        self._checkArray(point_arr)
        #
        results = [self.predictOne(p) for p in point_arr]
        return np.array(results, dtype=float).reshape(-1)
    
    def predictOne(self, point:np.ndarray) -> float:
        """Uses nearest variates to estimate the value in the sample_arr space.

        Args:
            point_arr (np.ndarray D): Variate to estimate the probability for.

        Returns:
            np.ndarray: Values corresponding to each point in point_arr
        """
        # Initializations
        exclude_idxs:list = []
        distances:list = []
        predictions:list = []
        # Iteratively find estimates
        for _ in range(self.size_interpolation_set):
            idx, distance = self._getIndexNearestVariate(point, exclude_idxs)
            if idx < 0:
                break
            predictions.append(self.sample_arr[idx])
            distances.append(distance)
            exclude_idxs.append(idx)
        if len(predictions) == 0:
            result = np.nan
        else:
            weight_arr = np.array(distances, dtype=float)
            weight_arr = 1 / (weight_arr + 1e-8)  # Avoid division by zero
            prediction_arr = np.array(predictions, dtype=float)
            result = np.sum(weight_arr * prediction_arr, axis=0) / np.sum(weight_arr)
        return result
    
    def _checkArray(self, arr):
        """Checks if the array is a numpy array and has the correct dimensions."""
        if not isinstance(arr, np.ndarray):
            raise ValueError("Expected a numpy array.")
        if arr.ndim != 2:
            raise ValueError(f"Expected a 2D array, got {arr.ndim}D array.")