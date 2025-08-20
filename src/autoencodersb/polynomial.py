'''Constructs a polynomial of arrays.'''

from autoencodersb.polynomial_rational import PolynomialRational  # type: ignore

import numpy as np
from typing import List

"""
A PolynomialTerm of D variables has a coefficient and specifies exponents for 0 or more
    independent variables. The term represents the product of the independent variables raised
    to the power of the exponents times the coefficient.
A PolynomialDivision is the ratio of two PolynomialTerm.
An expression is a collection of PolynomialTerm and PolynomialDivision.
"""

class Polynomial(object):

    def __init__(self, polynomial_rationals: List[PolynomialRational] ):
        self.num_variable = np.sum([t.num_variable for t in polynomial_rationals])
        self.num_term = len(polynomial_rationals)
        self.terms = polynomial_rationals

    def __repr__(self):
        return " + ".join([str(term) for term in self.terms])
    
    def evaluate(self, independent_variable_arr: np.ndarray) -> np.ndarray:
        """Evaluate the polynomial at the given independent variable values."""
        return np.sum([term.evaluate(independent_variable_arr) for term in self.terms], axis=0)

    def make_polynomial(self, degree: int) -> np.ndarray:
        """Construct a polynomial of the given degree."""
        # Create a grid of independent variable values
        x = np.linspace(-1, 1, 100)
        X = np.array(np.meshgrid(*[x]*self.num_variable)).T.reshape(-1, self.num_variable)
        # Compute the polynomial features
        poly = np.hstack([X**d for d in range(1, degree + 1)])
        return poly