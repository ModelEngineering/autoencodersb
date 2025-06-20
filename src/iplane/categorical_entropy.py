'''Calculates entropy for categorical variables.'''

import numpy as np
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt

class CategoricalEntropy(object):

    def __init__(self):
        self.entropy = np.nan
        self.categories = []
        self.probability_ser = pd.Series(dtype=float)

    def calculate(self, categorical_arr:np.ndarray):
        """Calculates data for categorical entropy.

        Args:
            categorical_arr (np.ndarray): _description_
        """
        self.categories, counts = np.unique(categorical_arr, return_counts=True)
        self.probability_ser = pd.Series(counts / len(categorical_arr), index=self.categories)
        self.entropy = -np.sum(self.probability_ser * np.log2(self.probability_ser + 1e-10))

    def plotEntropy(self):
        """Plots the entropy of the categorical variable."""
        if self.entropy is np.nan:
            raise ValueError("Entropy has not been calculated yet.")
        
        plt.figure(figsize=(8, 6))
        plt.bar(self.categories, self.probability_ser, color='skyblue')
        plt.title(f'Categorical Entropy: {self.entropy:.4f}')
        plt.xlabel('Categories')
        plt.ylabel('Probability')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    @classmethod
    def calculateEntropy(cls, categorical_arr:np.ndarray)->float:
        """Calculates entropy for a categorical array.

        Args:
            categorical_arr (np.ndarray): Array of categorical data.

        Returns:
            CategoricalEntropy: An instance with calculated entropy and categories.
        """
        categorical_entropy = cls()
        categorical_entropy.calculate(categorical_arr)
        return categorical_entropy.entropy