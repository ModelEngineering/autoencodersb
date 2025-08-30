'''Constructs a polynomial of arrays.'''

from autoencodersb.term import Term # type: ignore
from autoencodersb.polynomial_ratio import PolynomialRatio  # type: ignore
from autoencodersb.polynomial import Polynomial  # type: ignore

import numpy as np
import pandas as pd # type: ignore
from typing import List, Union, Tuple

"""
Creates synthetic data that are polynomials in an independent variable or ratios of polynomials.

A variable is an independent variable denoted by X_n. Its value is provided to the generate method.
A Term is the product of independent variables.
A Polynomial is a sum of Terms.
A PolynomialRatio is the ratio of two Polynomials.
A PolynomialCollection is a List of Polynomial, Term, and/or PolynomialRatio..
"""

class PolynomialCollection(object):

    def __init__(self, terms: Union[List[Term], List[Polynomial], List[PolynomialRatio]]):
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
        # Number of columns in the output
        self.num_output = self.num_variable + len(self.term_strs)

    def __repr__(self):
        return ",  ".join([str(t) for t in self.terms])
    
    def generate(self, variable_arr: np.ndarray) -> pd.DataFrame:
        """Evaluate the polynomial at the given independent variable values."""
        arrs = [term.generate(variable_arr) for term in self.terms]
        result_arr = np.hstack(arrs)
        return pd.DataFrame(result_arr, columns=self.term_strs)

    @classmethod
    def make(cls,
            is_mm_term: Union[bool, float]=True,
            is_first_order_term: Union[bool, float] = True,
            is_second_order_term: Union[bool, float] = True,
            is_third_order_term: Union[bool, float] = True,
    ) -> 'PolynomialCollection':
        """
        Creates a PolynomialCollection from the specified parameters.

        Args:
            is_mm_term (bool, optional): Whether to include mixed monomials.
            is_first_order_term (bool, optional): Whether to include first-order terms.
            is_second_order_term (bool, optional): Whether to include second-order terms.
            is_third_order_term (bool, optional): Whether to include third-order terms.

        Returns:
            PolynomialCollection: The constructed PolynomialCollection.
        """
        ##
        def getK(k_arg: Union[bool, float]) -> Tuple[bool, Union[float, None]]:
            if isinstance(k_arg, bool):
                return k_arg, None
            return True, k_arg
        ##
        terms:list = []
        is_mm, k_arg = getK(is_mm_term)
        if is_mm:
            # k*X_0/(1 + k'X_0)
            denominator = Polynomial([Term.make(k=1), Term.make(e0=1)])
            numerator = Polynomial([Term.make(k=k_arg, e0=1)])
            polynomial_ratio = PolynomialRatio(numerator, denominator)
            terms.append(polynomial_ratio)
        is_first_order, k_arg = getK(is_first_order_term)
        if is_first_order:
            terms.append(Term.make(k=k_arg, e0=1))
        is_second_order, k_arg = getK(is_second_order_term)
        if is_second_order:
            terms.append(Term.make(k=k_arg, e0=1, e1=1))
        is_third_order, k_arg = getK(is_third_order_term)
        if is_third_order:
            terms.append(Term.make(k=k_arg, e0=1, e1=1, e2=1))
        return cls(terms)