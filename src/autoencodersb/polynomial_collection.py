'''Constructs a polynomial of arrays.'''

from autoencodersb.term import Term # type: ignore
from autoencodersb.polynomial_ratio import PolynomialRatio  # type: ignore
from autoencodersb.polynomial import Polynomial  # type: ignore

import numpy as np
from typing import List, Union

"""
Creates synthetic data that are polynomials in an independent variable or ratios of polynomials.

A variable is an independent variable denoted by X_n. Its value is provided to the generate method.
A Term is the product of independent variables.
A Polynomial is a sum of Terms.
A PolynomialRatio is the ratio of two Polynomials.
A PolynomialCollection is a List of Polynomial, Term, and/or PolynomialRatio..
"""

class PolynomialCollection(object):

    def __init__(self, terms: Union[List[Term], List[Polynomial], List[PolynomialRatio]], is_random_path: bool = False):
        """
        Args:
            terms (List[Union[Term, TermRatio]]): _description_
            is_random_path (bool, optional): Independent variables are constructed as a random path
        """
        self.terms = terms
        self.num_term = len(self.terms)
        self.variables: list = []
        [self.variables.extend(t.variables) for t in self.terms]  # type: ignore
        self.variables = list(set(self.variables))
        self.num_variable = np.max(self.variables) + 1  # X_0 is a variable.
        self.term_strs = [str(t) for t in self.terms]

    def __repr__(self):
        return ",  ".join([str(t) for t in self.terms])
    
    def generate(self, variable_arr: np.ndarray) -> np.ndarray:
        """Evaluate the polynomial at the given independent variable values."""
        arrs = [term.generate(variable_arr) for term in self.terms]
        result_arr = np.hstack(arrs)
        return result_arr