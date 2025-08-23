from autoencodersb.term import Term # type: ignore
from autoencodersb.polynomial import Polynomial  # type: ignore
from autoencodersb.polynomial_ratio import PolynomialRatio # type: ignore
import autoencodersb.constants as cn  # type: ignore

import numpy as np
from typing import List, Any, Tuple
import unittest

IGNORE_TESTS = True
IS_PLOT = False
DENOMINATOR = Polynomial([Term.make(k=5), Term.make(k=1, e0=1)])
NUMERATOR = Polynomial([Term.make(2, e0=1)])


########################################
class TestTerm(unittest.TestCase):

    def setUp(self):
        self.polynomial_ratio = PolynomialRatio(NUMERATOR, DENOMINATOR)

    def testConstructor(self):
        if IGNORE_TESTS:
            return
        self.assertEqual(str(self.polynomial_ratio).count("X_0"), 2)
        #
        denominator = Polynomial([Term.make(k=5, e2=4, e5=3), Term.make(k=1, e0=1)])
        numerator = Polynomial([Term.make(2, e0=1)])
        polynomial_ratio = PolynomialRatio(numerator, denominator)
        self.assertEqual(len(polynomial_ratio.variables), 3)

    def testRepr(self):
        #if IGNORE_TESTS:
        #    return
        ratio_str = str(self.polynomial_ratio)
        slash_pos = ratio_str.find("/")
        self.assertNotEqual(slash_pos, -1)
        self.assertEqual(ratio_str[1:slash_pos-2], str(NUMERATOR))
        self.assertEqual(ratio_str[slash_pos + 3:-1], str(DENOMINATOR))

    def testGenerate(self):
        if IGNORE_TESTS:
            return
        independent_variable_arr = np.array(range(2000), dtype=np.float32).reshape(-1, 1)
        result_arr = self.polynomial_ratio.generate(independent_variable_arr)
        self.assertAlmostEqual(result_arr[-1], NUMERATOR.terms[0].coefficient, places=2)
        #
        denominator = Polynomial([Term.make(k=5, e2=4, e5=3), Term.make(k=1, e0=1)])
        numerator = Polynomial([Term.make(2, e0=1)])
        polynomial_ratio = PolynomialRatio(numerator, denominator)
        arr = np.array(range(2000), dtype=np.float32).reshape(-1, 1)
        independent_variable_arr = np.concatenate([arr] * 6, axis=1)
        result_arr = polynomial_ratio.generate(independent_variable_arr)
        import pdb; pdb.set_trace()
        self.assertAlmostEqual(result_arr[-1], 0, places=5)

if __name__ == '__main__':
    unittest.main()