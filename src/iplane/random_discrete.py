'''Discrete (finite) random variables.'''
import iplane.constants as cn
from iplane.collection import PCollection, DCollection  # type: ignore
from iplane.random import Random  # type: ignore

import numpy as np
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt
from typing import Tuple, Any, Optional, Dict, List, cast

CATEGORY_ARR = 'category_arr'
PROBABILITY_ARR = 'probability_arr'
PARAMETER_NAMES = [CATEGORY_ARR, PROBABILITY_ARR]
ENTROPY = 'entropy'
DISTRIBUTION_NAMES = [cn.ENTROPY]


################################################
class PCollectionDiscrete(PCollection):
    # Parameter collection for mixture of Gaussian distributions.

    def __init__(self, parameter_dct:Optional[Dict[str, Any]]=None)->None:
        super().__init__(PARAMETER_NAMES, parameter_dct)
    
    def isValid(self) -> bool:
        """Check if the parameter collection is valid."""
        if self.isAnyNull():
            return False
        # Check if all parameters are of the correct type
        if len(self.dct) != len(PARAMETER_NAMES):
            return False
        if np.array(self.dct[CATEGORY_ARR]).ndim != 1:
            return False
        return True

    def __eq__(self, other:Any) -> bool:
        """Check if two ParameterMGaussian objects are equal."""
        if not isinstance(other, PCollectionDiscrete):
            return False
        # Check if all expected parameters are present and equal
        for key in PARAMETER_NAMES:
            if key not in self.dct or key not in other.dct:
                return False
            if np.all(self.dct[key]  != other.dct[key]):
                return False
            if not np.allclose(self.dct[key].flattern(), other.dct[key].flatten()):
                return False
        return True


################################################
class DCollectionDiscrete(DCollection):
    # Distribution collection for mixture of Gaussian distributions.

    def __init__(self, parameter_dct:Optional[Dict[str, Any]]=None)->None:
        super().__init__(parameter_dct)
        if parameter_dct is not None:
            self.isValidDct(parameter_dct, PARAMETER_NAMES)
        super().__init__(parameter_dct)
        # Initialize the properties
        self.entropy = self.dct.get(ENTROPY, None)


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
        dct = {CATEGORY_ARR: category_arr, PROBABILITY_ARR: probability_arr }
        return PCollectionDiscrete(dct)
    
    def makeDCollection(self, pcollection:PCollection) -> DCollectionDiscrete:
        """Calculates entropy for the discrete random variable.

        Args:
            pcollection (PCollectionDiscrete)
            max_num_sample (int): Maximum number of samples to consider.

        Returns:
            DCollectionDiscrete:
        """
        entropy = self.calculateEntropy(pcollection)
        self.dcollection = DCollectionDiscrete({cn.ENTROPY: entropy})
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
        category_arr = dcollection.get(CATEGORY_ARR)
        probability_arr = dcollection.get(PROBABILITY_ARR)
        plt.bar(category_arr, probability_arr, color='skyblue')
        plt.title(f'Categorical Entropy: {dcollection.get(ENTROPY)}')
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
        return -np.sum(pcollection.probability_arr * np.log2(pcollection.probability_ser + 1e-10))