from autoencodersb.term import Term # type: ignore
from autoencodersb.polynomial import Polynomial  # type: ignore
from autoencodersb.polynomial_ratio import PolynomialRatio # type: ignore
import autoencodersb.constants as cn  # type: ignore

import matplotlib.pylab as plt
import numpy as np
import pandas as pd  # type: ignore
from typing import List, Any, Tuple
import unittest

IGNORE_TESTS = False
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
        if IGNORE_TESTS:
            return
        ratio_str = str(self.polynomial_ratio)
        slash_pos = ratio_str.find("/")
        self.assertNotEqual(slash_pos, -1)
        self.assertEqual(ratio_str[1:slash_pos-2], str(NUMERATOR))
        self.assertEqual(ratio_str[slash_pos + 3:-1], str(DENOMINATOR))

    def testGenerate(self):
        if IGNORE_TESTS:
            return
        variable_arr = np.array(range(2000), dtype=np.float32).reshape(-1, 1)
        result_arr = self.polynomial_ratio.generate(variable_arr).reshape(-1)
        self.assertAlmostEqual(result_arr[-1], NUMERATOR.terms[0].coefficient, places=2)
        #
        denominator = Polynomial([Term.make(k=5, e2=4, e5=3), Term.make(k=1, e0=1)])
        numerator = Polynomial([Term.make(2, e0=1)])
        polynomial_ratio = PolynomialRatio(numerator, denominator)
        arr = np.array(range(2000), dtype=np.float32).reshape(-1, 1)
        variable_arr = np.concatenate([arr] * 6, axis=1)
        result_arr = polynomial_ratio.generate(variable_arr).reshape(-1)
        self.assertAlmostEqual(result_arr[-1], 0, places=5)

    def testMakeHillPolynomialRatio(self):
        if IGNORE_TESTS:
            return
        polynomial_ratio = PolynomialRatio.makeHillPolynomialRatio("X_0", k=5, n=2)
        self.assertEqual(len(polynomial_ratio.variables), 1)

    def testMakeHillPolynomialRatioScale(self):
        if IGNORE_TESTS:
            return
        yvs = []
        sequence_arr = np.array(range(1, 101), dtype=np.float32).reshape(-1, 1)
        n_vals = list(range(1, 5))
        for n in n_vals:
            polynomial_ratio = PolynomialRatio.makeHillPolynomialRatio("X_0", k=5, n=n)
            yvs.append(polynomial_ratio.generate(sequence_arr).reshape(-1, 1))
        arr = np.hstack(yvs)
        df = pd.DataFrame(arr, columns=[f"n={n}" for n in n_vals])
        df.plot(kind="line")
        if IS_PLOT:
            plt.show()


if __name__ == '__main__':
    unittest.main()