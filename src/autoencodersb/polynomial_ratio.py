'''Represents a product of independent variables raised to an exponent.'''

from autoencodersb.term import Term  # type: ignore

import numpy as np
import pandas as pd  # type: ignore
from typing import List


class PolynomialRatio(object):

    def __init__(self, numerator: Term, denominator: Term):
        self.numerator = numerator
        self.denominator = denominator
        self.exponent_dct = dict(numerator.exponent_dct)
        self.exponent_dct.update(denominator.exponent_dct)
        self.num_variable = len(self.exponent_dct.keys())

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