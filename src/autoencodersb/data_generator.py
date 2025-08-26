'''Generates synthetic data for training and testing autoencoders.'''

from autoencodersb import constants as cn  # type: ignore
from autoencodersb.dataset_csv import DatasetCSV
from autoencodersb.polynomial_collection import PolynomialCollection  # type: ignore
from autoencodersb.dataset_csv import DatasetCSV # type: ignore
from autoencodersb.sequence import Sequence# type: ignore

import matplotlib.pyplot as plt
import numpy as np  # type: ignore
from torch.utils.data import DataLoader
import pandas as pd    # type: ignore
from typing import List, Callable, Optional


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
        self.variable_func: Callable[[int], pd.DataFrame] = lambda x: pd.DataFrame(np.array([]))
        # Calculated
        self.data_df = pd.DataFrame()  # Updated by generate
        self.data_dl = DataLoader(DatasetCSV(csv_input=self.data_df, target_column=None),
                shuffle=self.is_shuffle, batch_size=10)

    def generate(self) -> DataLoader:
        """Generates the full synthetic dataset."""
        independent_df = self.variable_func(self.num_sample)
        if len(independent_df) == 0:
            raise ValueError("Independent variable function has not been specified.")
        # Make the DataFrame
        dependent_df = self.polynomial_collection.generate(independent_df.values)
        noise = np.random.normal(0, self.noise_std, dependent_df.shape)
        dependent_df += noise
        self.data_df = pd.concat([independent_df, dependent_df], axis=1)
        # Construct the DataLoader
        batch_size = int(np.ceil(BATCH_SIZE_FRACTION*self.num_sample))
        self.data_dl = DataLoader(DatasetCSV(csv_input=self.data_df, target_column=None),
                shuffle=self.is_shuffle, batch_size=batch_size)
        return self.data_dl

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
        def generate(num_sample) -> pd.DataFrame:
            shape = (num_sample, self.polynomial_collection.num_variable)
            result_arr = np.random.uniform(min_value, max_value, shape)
            final_arr = np.round(result_arr, precision)
            columns = [f"X_{n}" for n in range(self.polynomial_collection.num_variable)]
            return pd.DataFrame(final_arr, columns=columns)
        self.variable_func = generate

    def specifySequences(self, sequences: Optional[List[Sequence]] = None, **kwargs) -> None:
        """Generates sequenceof features

        Args:
            sequences (List[Sequence]): List of Sequence objects defining each independent variable.
            kwargs: Options for sequence constructor

        Returns: List[Sequence]
        """
        #
        def generate(num_sample: int) -> pd.DataFrame:
            # Generate a sequence of values
            if sequences is None:
                new_sequences = [Sequence(num_sample=num_sample, **kwargs)]*self.polynomial_collection.num_variable
            else:
                new_sequences = sequences
            arr = np.hstack([s.generate() for s in new_sequences]).reshape(num_sample, len(new_sequences))
            columns = [f"X_{n}" for n in range(self.polynomial_collection.num_variable)]
            df = pd.DataFrame(arr, columns=columns)
            return df
        #
        self.variable_func = generate

    def plotGeneratedData(self, is_plot: bool = True, x_column: Optional[str] = None) -> None:
        """
        Plots the generated sequences using matplotlib.

        Args:
            is_plot (bool): Whether to display the plot.
            x_column (Optional[str]): The name of the column to use as the x-axis. Defaults to observation index
        """
        if is_plot:
            _, ax = plt.subplots()
            self.data_df.plot(ax=ax, x=x_column)
            ax.set_xlabel("time")
            ax.set_title("Generated Sequences")
            ax.grid()
            plt.show()

    def plotErrorDifference(self, other_df: pd.DataFrame, x_column: Optional[str] = None,
            is_plot: bool = True) -> pd.DataFrame:
        """
        Plots the error difference between the generated data and another DataGenerator's data.
            (other_df - self.data_df)/self.data_df
            x_column (Optional[str]): The name of the column to use as the x-axis. Defaults to observation index
        """
        if x_column is None:
            xv = self.data_df.index.values
        else:
            xv = self.data_df[x_column].values
        error_df = (other_df - self.data_df)/self.data_df
        error_df[x_column] = xv
        if is_plot:
            _, ax = plt.subplots()
            error_df.plot(ax=ax, x=x_column)
            if x_column is None:
                ax.set_xlabel("observation")
            else:
                ax.set_xlabel(x_column)
            ax.set_title("relative error")
            ax.grid()
            plt.show()
        return error_df