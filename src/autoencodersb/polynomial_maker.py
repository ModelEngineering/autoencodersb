'''Constructs a polynomial of arrays.'''and

import numpy as np
from typing import List


class PolynomialRepresentation(object):

    def __init__(self):
        """Initialize the polynomial representation."""
        self.polynomial_terms = []

    def add_term(self, coefficients: List[float], exponents: List[int]):


class PolynomialMaker(object):

    def __init__(self, num_independent_features: int, num_dependent_features: int):
        self.num_independent_features = num_independent_features
        self.num_dependent_features = num_dependent_features

    def make_polynomial(self, degree: int) -> np.ndarray:
        """Construct a polynomial of the given degree."""
        # Create a grid of independent variable values
        x = np.linspace(-1, 1, 100)
        X = np.array(np.meshgrid(*[x]*self.num_independent_features)).T.reshape(-1, self.num_independent_features)
        # Compute the polynomial features
        poly = np.hstack([X**d for d in range(1, degree + 1)])
        return poly