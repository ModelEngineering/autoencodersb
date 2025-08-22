'''Constructs a polynomial of arrays.'''

from autoencodersb.term import Term # type: ignore
from autoencodersb.polynomial_ratio import PolynomialRatio  # type: ignore

import numpy as np
from typing import List, Union

"""
Creates synthetic data that are polynomials in an independent variable or ratios of polynomials.

A Term is the product of independent variables
A Polynomial is a sum of Terms.
A PolynomialRatio is the ratio of two Polynomials.
A PolynomialCollection is a collection of Polynomials.
"""

class PolynomialCollection(object):

    def __init__(self, terms: List[Union[Term, PolynomialRatio]], is_random_path: bool = False):
        """
        Args:
            terms (List[Union[Term, TermRatio]]): _description_
            is_random_path (bool, optional): Independent variables are constructed as a random path
        """
        self.terms = terms
        self.num_term = len(self.terms)
        self.num_independent_variable = self._getNumberOfIndependentVariables()

    def _getNumberOfIndependentVariables(self) -> int:
        variable_idxs: list = []
        for term in self.terms:
            variable_idxs.extend(term.exponent_dct.keys())
        return len(set(variable_idxs))

    def __repr__(self):
        return " + ".join([str(term) for term in self.terms])
    
    def generate(self, independent_variable_arr: np.ndarray) -> np.ndarray:
        """Evaluate the polynomial at the given independent variable values."""
        return np.array([term.generate(independent_variable_arr) for term in self.terms])