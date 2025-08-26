'''Represents a product of independent variables raised to an exponent.'''

from autoencodersb.polynomial import Polynomial# type: ignore
from autoencodersb.term import Term  # type: ignore

import numpy as np
import pandas as pd  # type: ignore
from typing import Optional, Union


class PolynomialRatio(object):

    def __init__(self, numerator: Union[Polynomial, Term], denominator: Polynomial):
        self.numerator = numerator
        self.denominator = denominator
        self.variables: list = list(self.numerator.variables)
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
        numerator_arr = self.numerator.generate(independent_variable_arr).reshape(-1)
        denominator_arr = self.denominator.generate(independent_variable_arr).reshape(-1)
        result = numerator_arr / denominator_arr
        return result.reshape(-1, 1)

    @classmethod 
    def makeHillPolynomialRatio(cls, variable: str, k: Optional[float] = None, n: float = 1) -> 'PolynomialRatio':
        """Creates a Hill equation ratio: X^n / (k + X^n).
        Note that by not specifying the argument n, we obtain a Michaelis-Menten expression.

        Args:
            variable (str): The independent variable, e.g., "X_0".
            k (float): The coefficient for the numerator.
            n (float): The exponent for the variable.

        Returns:
            PolynomialRatio: The constructed Hill equation ratio.
        """
        variable_idx = variable.split("_")[1]
        dct = {"k": 1, f"e{variable_idx}": n}
        numerator_term = Term.make(**dct)  # type: ignore
        denominator_term = Term.make(**dct)  # type: ignore
        constant_term = Term.make(k=k)
        denominator_polynomial = Polynomial(terms=[constant_term, denominator_term])
        return cls(numerator=numerator_term, denominator=denominator_polynomial)