'''Represents a product of independent variables raised to an exponent.'''

import numpy as np
import pandas as pd  # type: ignore
from typing import List


class PolynomialTerm(object):

    def __init__(self, coefficient: float, exponents: List[float]):
        self.coefficient = coefficient
        self.exponent_arr = np.array(exponents)
        self.num_variable = len([e for e in exponents if e != 0])

    def __repr__(self):
        term_strs = [f"X_{n}**{p}" for n, p in enumerate(self.exponent_arr) if p != 0]
        term_str = f"{self.coefficient} * {' * '.join(term_strs)}"
        return term_str
    
    def evaluate(self, independent_variable_arr: np.ndarray) -> np.ndarray:
        """Evaluate the polynomial term.

        Args:
            independent_variable_arr (np.ndarray): N X I array of independent variables

        Returns:
            np.ndarray: Result of the evaluation
        """
        # Evaluate the term by multiplying the coefficient with the independent variables raised to the appropriate powers
        return self.coefficient * np.prod(independent_variable_arr ** self.exponent_arr, axis=1)