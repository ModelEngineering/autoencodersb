'''Discrete (finite) random variables.'''
import iplane.constants as cn
from iplane.random import Random, PCollection, DCollection  # type: ignore

import numpy as np
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt
from typing import Tuple, Any, Optional, Dict, List, cast


################################################
class PCollectionDiscrete(PCollection):
    # Parameter collection for discrete random variables.

    def __init__(self, collection_dct:Optional[Dict[str, Any]]=None)->None:
        super().__init__(cn.PC_DISCRETE_NAMES, collection_dct)


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
    
    def makeDCollection(self, variate_arr:Optional[np.ndarray]=None,
                pcollection:Optional[PCollectionDiscrete]=None) -> DCollectionDiscrete:
        """Calculates entropy for the discrete random variable.

        Args:
            pcollection (PCollectionDiscrete)
                PCollectionDiscrete with the parameters of the distribution.
            variate_arr (Optional[np.ndarray], optional): Not used for discrete random variables. Defaults to None.            

        Returns:
            DCollectionDiscrete:
        """
        pcollection = cast(PCollectionDiscrete, self.setPCollection(pcollection))
        variate_arr = pcollection.get(cn.PC_CATEGORY_ARR)
        probability_arr = pcollection.get(cn.PC_PROBABILITY_ARR)
        entropy = self.calculateEntropy(pcollection)
        self.dcollection = DCollectionDiscrete({
            cn.DC_ENTROPY: entropy,
            cn.DC_VARIATE_ARR: variate_arr,
            cn.DC_PROBABILITY_ARR: probability_arr}
            )
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

    def calculateEntropy(self, collection:PCollectionDiscrete)->float:
        """Analytical calculation of entropy for a discrete random variable.

        Args:
            categorical_arr (np.ndarray): Array of categorical data.

        Returns:
            CategoricalEntropy: An instance with calculated entropy and categories.
        """
        probability_arr = collection.get(cn.PC_PROBABILITY_ARR)
        return -np.sum(probability_arr * np.log2(probability_arr + 1e-10))