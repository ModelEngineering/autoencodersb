'''Generates synthetic data for training and testing autoencoders.'''

import autoencodersb.constants as cn

import itertools
import numpy as np  # type: ignore
from torch.utils.data import DataLoader
import pandas as pd    # type: ignore
from typing import cast, Optional, Tuple


class DataGenerator(object):

    def __init__(self, 
            num_sample: int = 1000,
            num_independent_feature: int = 2,
            num_feature: int = 10,
            num_data_value: int = 10,
            data_density: float = 1.0,  # Number of values per integer interval
            noise_std: float = 0.0
            ):
        self.num_sample = num_sample
        self.num_independent_feature = num_independent_feature
        self.num_feature = num_feature
        self.num_data_value = num_data_value
        self.data_density = data_density
        self.noise_std = noise_std

    def generateFullData(self) -> DataLoader:
        """Generates the full synthetic dataset."""
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def generateIndependentFeatures(self) -> np.ndarray:
        """
        Generates an array of independent features.

        Returns:
            np.ndarray (N X I): An array of size independent features.
                N is self.num_sample 
                I self.num_independent_feature
        """
        raise NotImplementedError("This method should be overridden by subclasses.") 