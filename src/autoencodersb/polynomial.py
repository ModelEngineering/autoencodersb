'''Represents a sum of Terms'''
from autoencodersb.term import Term # type: ignore

import numpy as np
import pandas as pd  # type: ignore
from typing import List, Optional, Dict


class Polynomial(object):

    def __init__(self, terms: List[Term]):
        self.terms = terms

    def __repr__(self):
        strs = [str(t) for t in self.terms]
        return " + ".join(strs)
    
    def evaluate(self, independent_variable_arr: np.ndarray) -> np.ndarray:
        """Evaluate the polynomial term.

        Args:
            independent_variable_arr (np.ndarray): N X I array of independent variables

        Returns:
            np.ndarray: Result of the evaluation
        """
        # Evaluate the term by multiplying the coefficient with the independent variables raised to the appropriate powers
        return np.sum([term.generate(independent_variable_arr) for term in self.terms], axis=0)