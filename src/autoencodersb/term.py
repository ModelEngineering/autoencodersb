'''Represents a product of independent variables raised to an exponent.'''

import numpy as np
from typing import Optional, Dict


class Term(object):

    def __init__(self, coefficient: float, exponent_dct: Dict[int, float]):
        """
        Args:
            coefficient (float): Constant multiplied by the expression
            exponent_dct (Dict[int, float]): for int variable the float exponet
        """
        self.coefficient = coefficient
        self.exponent_dct = dict(exponent_dct)
        self.variables = [k for k, v in exponent_dct.items() if not np.isclose(v, 0)]

    def __repr__(self):
        term_strs = [f"X_{n}^{p}" if not np.isclose(p, 1) else f"X_{n}"
                for n, p in self.exponent_dct.items() if p != 0]
        product_str = ' * '.join(term_strs)
        #
        if len(term_strs) == 0:
            result = str(self.coefficient)
        elif np.isclose(self.coefficient, 1.0):
            result = product_str
        else:
            result = f"{self.coefficient} * {product_str}"
        return result
    
    def generate(self, variable_arr: np.ndarray) -> np.ndarray:
        """Evaluate the polynomial term.

        Args:
            variable_arr (np.ndarray): N X I array of independent variables

        Returns:
            np.ndarray: Result of the evaluation
        """
        # Error check
        if variable_arr.ndim != 2:
            raise ValueError("Input must be a 2D array")
        # Evaluate the term by multiplying the coefficient with the independent variables raised to the appropriate powers
        exponent_arr = np.zeros(variable_arr.shape[1])
        [exponent_arr.__setitem__(n, p) for n, p in self.exponent_dct.items()]
        arr = self.coefficient * np.prod(variable_arr ** exponent_arr, axis=1)
        arr = arr.reshape(-1, 1).astype(np.float32)
        return arr

    @classmethod
    def make(cls, k: Optional[float] = None, k_min: float = 1.0,
            k_max: float = 10.0, is_k_int: bool = True, **kwargs) -> 'Term':
        """
        Convenince method to make a term.

        Args:
            k (Optional[float]): Coefficient of the term. If None, a random value is generated.
            The following are used if k is None:
                k_min (float): Minimum value for the coefficient.
                k_max (float): Maximum value for the coefficient.
                is_k_int (bool): If True, the coefficient will be an integer.
            kwargs: Keyword arguments of the form "e"<int>=float, the exponent for the indexed
                independent variable.

        """
        # Error check the exponents
        try:
            _ = [float(v[1:]) for v in kwargs.keys()]
        except ValueError:
            import pdb; pdb.set_trace()
            raise ValueError("Invalid exponent format. Use e<int>.")
        # Calculate the coefficient
        if k is None:
            k = np.random.uniform(k_min, k_max)
            if is_k_int:
                k = int(k)
        # Construct the exponent dict
        exponent_dct = {}
        for key, value in kwargs.items():
            if not key.startswith('e'):
                raise ValueError("Invalid exponent format. Use e<int>=float.")
            index = int(key[1:])
            exponent_dct[index] = float(value)
        #
        return cls(k, exponent_dct)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Term):
            raise TypeError("Incompatible types")
        result = (np.isclose(self.coefficient, other.coefficient) and
                self.exponent_dct == other.exponent_dct )
        return bool(result)