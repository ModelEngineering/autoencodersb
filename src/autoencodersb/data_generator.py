'''Generates synthetic data for training and testing autoencoders.'''

from autoencodersb.dataset_csv import DatasetCSV
from autoencodersb.polynomial_collection import PolynomialCollection  # type: ignore
from autoencodersb.dataset_csv import DatasetCSV # type: ignore

import numpy as np  # type: ignore
from torch.utils.data import DataLoader
import pandas as pd    # type: ignore
from typing import Optional, Callable


BATCH_SIZE_FRACTION = 0.1  # Fraction of the number of samples to use as batch size

# To Do
#   1. Implement noise
#   2. Implement sequence data generation


class DataGenerator(object):

    def __init__(self, 
            polynomial_collection: PolynomialCollection,
            num_sample: int = 1000,
            noise_std: float = 0.0,
            is_shuffle: bool = False
        ):
        """Data generator for creating synthetic datasets.

        Args:
            polynomial_collection (PolynomialCollection): Collection of polynomial features.
            num_sample (int, optional): Number of samples to generate. Defaults to 1000.
            noise_std (float, optional): Standard deviation of the noise to add to the features. Defaults to 0.0.
            is_shuffle (bool, optional): Whether to shuffle the dataset. Defaults to False.

        Usage:
            data_generator = DataGenerator(polynomial_collection)
            data_generator.specifyIID(min_value=1.0, max_value=10.0, data_density=0)
            dataloader = data_generator.generate()
            
            data_generator.specifySequence(min_value=1.0, max_value=10.0, data_density=0)
        """
        self.polynomial_collection = polynomial_collection
        self.num_sample = num_sample
        self.noise_std = noise_std
        self.is_shuffle = is_shuffle
        self.independent_feature_func: Callable[[int], np.ndarray] = lambda x: np.array(range(x))

    def generate(self) -> DataLoader:
        """Generates the full synthetic dataset."""
        independent_arr = self.independent_feature_func(self.num_sample)
        # Make a Dataframe
        arr = self.polynomial_collection.generate(independent_arr)
        noise = np.random.normal(0, self.noise_std, arr.shape)
        new_arr = arr + noise
        columns = [str(t) for t in self.polynomial_collection.terms]
        df = pd.DataFrame(new_arr, columns=columns)
        # Construct the DataLoader
        batch_size = int(np.ceil(BATCH_SIZE_FRACTION*self.num_sample))
        dataloader = DataLoader(DatasetCSV(csv_input=df, target_column=None),
                shuffle=self.is_shuffle, batch_size=batch_size)
        return dataloader

    def specifyIID(self, min_value: float = 1.0, max_value: float = 10.0,
            precision: int = 0) -> None:
        """
        Generates an N X I array of i.i.d. samples, where I is the number of independent variables

        Args:
            min_value (float): Minimum value for the uniform distribution.
            max_value (float): Maximum value for the uniform distribution.
            precision (int): Number of decimal places to round the generated values.

        Returns:
            np.ndarray (N X I): An array of floats of size independent features.
                N is self.num_sample
                I is self.num_independent_feature
        """
        def generate(num_sample) -> np.ndarray:
            shape = (num_sample, self.polynomial_collection.num_variable)
            result_arr = np.random.uniform(min_value, max_value, shape)
            final_arr = np.round(result_arr, precision)
            return final_arr
        self.independent_feature_func = generate

    def specifySequence(self, min_value: float = 1, max_value: float = 10.0,
            precision: int = 0, rate: float = -1) -> None:
        """Generates a sequence of features for time series data using a random walk of
            size 10**self.data_density

        Args:
            min_value (float): Minimum value for the uniform distribution.
            max_value (float): Maximum value for the uniform distribution.
            precision (int): Number of decimal places to round the generated values.

        Returns:
            np.ndarray (N X 3):
                X_0: [min_value, max_value]
                X_1: e**-rate*[min_value, max_value]
                X_2: 1 - e**-rate*[min_value, max_value]
        """
        if rate < 0:
            raise ValueError("Rate must be non-negative.")
        #
        def generate(num_sample: int) -> np.ndarray:
            # Generate a sequence of values
            sequence_arr = np.round(np.linspace(min_value, max_value, num=num_sample), precision).reshape(-1, 1)
            exponential_decrease_arr = np.exp(-rate*sequence_arr).reshape(-1, 1)
            exponential_increase_arr = 1 - np.exp(-rate*sequence_arr).reshape(-1, 1)
            #
            result_arr = np.hstack([sequence_arr, exponential_decrease_arr, exponential_increase_arr])
            return result_arr
        #
        self.independent_feature_func = generate