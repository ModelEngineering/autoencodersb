'''Represents a sum of Terms'''
from autoencodersb.term import Term # type: ignore

import numpy as np
import pandas as pd  # type: ignore
from typing import List, Optional, Dict


class Polynomial(object):

    def __init__(self, terms: List[Term]):
        self.terms = terms
        self.variables: list = []
        _ = [self.variables.extend(t.variables) for t in self.terms] # type: ignore
        self.variables = list(set(self.variables))

    def __repr__(self):
        strs = [str(t) for t in self.terms]
        return "  +  ".join(strs)
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Polynomial):
            return False
        if len(self.terms) != len(other.terms):
            return False
        return all([t1 == t2 for t1, t2 in zip(self.terms, other.terms)])

    def generate(self, independent_variable_arr: np.ndarray) -> np.ndarray:
        """Evaluate the polynomial term.

        Args:
            independent_variable_arr (np.ndarray): N X I array of independent variables

        Returns:
            np.ndarray: Result of the evaluation
        """
        # Evaluate each term
        return np.sum([term.generate(independent_variable_arr) for term in self.terms], axis=0).reshape(-1).astype(np.float32)