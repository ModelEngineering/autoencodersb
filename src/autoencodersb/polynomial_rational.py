'''Represents a product of independent variables raised to an exponent.'''

from autoencodersb.polynomial_term import PolynomialTerm  # type: ignore

import numpy as np
import pandas as pd  # type: ignore
from typing import List


class PolynomialRational(object):

    def __init__(self, numerator: PolynomialTerm, denominator: PolynomialTerm):
        self.numerator = numerator
        self.denominator = denominator
        exponents = numerator.exponent_arr + denominator.exponent_arr
        self.num_variable = len([e for e in exponents if e != 0])

    def __repr__(self):
        return f"({self.numerator}) / ({self.denominator})"

    def evaluate(self, independent_variable_arr: np.ndarray) -> np.ndarray:
        """Divide this polynomial term by another polynomial term.

        Args:
            other (PolynomialTerm): The polynomial term to divide by.

        Returns:
            PolynomialTerm: The resulting polynomial term after division.
        """
        numerator_value = self.numerator.evaluate(independent_variable_arr)
        denominator_value = self.denominator.evaluate(independent_variable_arr)
        return numerator_value / denominator_value