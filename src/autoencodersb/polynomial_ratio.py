'''Represents a product of independent variables raised to an exponent.'''

from autoencodersb.polynomial import Polynomial# type: ignore

import numpy as np
import pandas as pd  # type: ignore
from typing import List


class PolynomialRatio(object):

    def __init__(self, numerator: Polynomial, denominator: Polynomial):
        self.numerator = numerator
        self.denominator = denominator
        self.variables = list(self.numerator.variables)
        self.variables.extend(self.denominator.variables)
        self.variables = list(set(self.variables))

    def __repr__(self):
        return f"({str(self.numerator)}) / ({str(self.denominator)})"

    def generate(self, independent_variable_arr: np.ndarray) -> np.ndarray:
        """Divide this polynomial term by another polynomial term.

        Args:
            other (PolynomialTerm): The polynomial term to divide by.

        Returns:
            PolynomialTerm: The resulting polynomial term after division.
        """
        numerator_value = self.numerator.generate(independent_variable_arr)
        denominator_value = self.denominator.generate(independent_variable_arr)
        return numerator_value / denominator_value