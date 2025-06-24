'''Discrete (finite) random variables.'''
import iplane.constants as cn
from iplane.random import Random, PCollection, DCollection  # type: ignore

import numpy as np
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt
from typing import Tuple, Any, Optional, Dict, List, cast


################################################
class PCollectionDiscrete(PCollection):
    # Parameter collection for mixture of Gaussian distributions.

    def __init__(self, parameter_dct:Optional[Dict[str, Any]]=None)->None:
        super().__init__(cn.PC_DISCRETE_NAMES, parameter_dct)

    def __eq__(self, other:Any) -> bool:
        """Check if two ParameterMGaussian objects are equal."""
        if not isinstance(other, PCollectionDiscrete):
            return False
        # Check if all expected parameters are present and equal
        if not self.isAllValid() or not other.isAllValid():
            return False
        for key in cn.PC_DISCRETE_NAMES:
            this_value = self.get(key)
            other_value = other.get(key)
            if isinstance(this_value, np.ndarray):
                if not np.allclose(this_value.flatten(), other_value.flatten()):
                    return False
            else:
                if not np.isclose(this_value, other_value):
                    return False
        return True


################################################
class DCollectionDiscrete(DCollection):
    # Distribution collection for mixture of Gaussian distributions.


    def __init__(self, parameter_dct:Optional[Dict[str, Any]]=None)->None:
        super().__init__(cn.DC_DISCRETE_NAMES, parameter_dct)


################################################
class RandomDiscrete(Random):

    def __init__(self, pcollection:Optional[PCollectionDiscrete]=None,
            dcollection:Optional[DCollectionDiscrete]=None)->None:
        super().__init__(pcollection, dcollection)

    def estimatePCollection(self, sample_arr:np.ndarray)-> PCollectionDiscrete:
        """Estimates the PCollectionDiscrete values from a categorical array.

        Args:
            categorical_arr (np.ndarray): _description_
        """
        category_arr, count_arr = np.unique(sample_arr, return_counts=True)
        probability_arr = count_arr / len(sample_arr)
        dct = {cn.PC_CATEGORY_ARR: category_arr, cn.PC_PROBABILITY_ARR: probability_arr }
        return PCollectionDiscrete(dct)
    
    def makeDCollection(self, pcollection:PCollectionDiscrete) -> DCollectionDiscrete:
        """Calculates entropy for the discrete random variable.

        Args:
            pcollection (PCollectionDiscrete)
            max_num_sample (int): Maximum number of samples to consider.

        Returns:
            DCollectionDiscrete:
        """
        entropy = self.calculateEntropy(pcollection)
        self.dcollection = DCollectionDiscrete({cn.DC_ENTROPY: entropy})
        return self.dcollection

    def plot(self, dcollection:Optional[DCollectionDiscrete]=None)->None:
        """Plots the distribution"""
        if dcollection is None:
            if self.dcollection is None:
                if self.pcollection is None:
                    raise ValueError("PCollection has not been estimated yet.")
                self.dcollection = self.makeDCollection(self.pcollection)
            else:
                dcollection = self.dcollection
        #
        dcollection = cast(DCollectionDiscrete, dcollection)
        plt.figure(figsize=(8, 6))
        category_arr = dcollection.get(cn.PC_CATEGORY_ARR)
        probability_arr = dcollection.get(cn.PC_PROBABILITY_ARR)
        plt.bar(category_arr, probability_arr, color='skyblue')
        plt.title(f'Categorical Entropy: {dcollection.get(cn.DC_ENTROPY)}')
        plt.xlabel('Categories')
        plt.ylabel('Probability')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def calculateEntropy(self, pcollection:PCollectionDiscrete)->float:
        """Analytical calculation of entropy for a categorical array.

        Args:
            categorical_arr (np.ndarray): Array of categorical data.

        Returns:
            CategoricalEntropy: An instance with calculated entropy and categories.
        """
        probability_arr = pcollection.get(cn.PC_PROBABILITY_ARR)
        return -np.sum(probability_arr * np.log2(probability_arr + 1e-10))