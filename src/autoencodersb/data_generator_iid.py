'''Generates synthetic data where the independent variables are iid.'''

import autoencodersb.constants as cn
from autoencodersb.data_generator import DataGenerator  # type: ignore

import itertools
import numpy as np  # type: ignore
from torch.utils.data import DataLoader
import pandas as pd    # type: ignore
from typing import cast, Optional, Tuple


class DataGeneratorIID(DataGenerator):

    def __init__(self, 
            num_sample: int = 1000,
            num_independent_feature: int = 2,
            num_feature: int = 10,
            num_data_value: int = 10,
            data_density: float = 1.0,  # Number of values per integer interval
            noise_std: float = 0.0
            ):
        super().__init__(
            num_sample=num_sample,
            num_independent_feature=num_independent_feature,
            num_feature=num_feature,
            num_data_value=num_data_value,
            data_density=data_density,
            noise_std=noise_std
        )
    
    def generateIndependentFeatures(self) -> np.ndarray:
        """
        Generates an array of independent features.

        Returns:
            np.ndarray (N X I): An array of size independent features.
                N is self.num_sample 
                I self.num_independent_feature
        """
        return np.random.randint(1, self.num_data_value + 1, 
                (self.num_sample, self.num_independent_feature)).astype(np.float32) / self.data_density