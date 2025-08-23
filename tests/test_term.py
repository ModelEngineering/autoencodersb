from autoencodersb.term import Term  # type: ignore
import autoencodersb.constants as cn  # type: ignore

import matplotlib.pyplot as plt
import pandas as pd  # type: ignore
import numpy as np
from typing import List, Any, Tuple
import unittest

IGNORE_TESTS = False
IS_PLOT = False
COEFFICIENT = 3
NUM_VARIABLE = 3
EXPONENT_DCT = {n: float(n+1) for n in range(NUM_VARIABLE)}


########################################
class TestTerm(unittest.TestCase):

    def setUp(self):
        self.term = Term(COEFFICIENT, EXPONENT_DCT)

    def testConstructor(self):
        if IGNORE_TESTS:
            return
        self.assertEqual(self.term.coefficient, COEFFICIENT)
        self.assertEqual(self.term.exponent_dct, EXPONENT_DCT)
        self.assertEqual(len(self.term.variables), NUM_VARIABLE)

    def testStr(self):
        if IGNORE_TESTS:
            return
        self.assertEqual(str(self.term), '3 * X_0 * X_1^2.0 * X_2^3.0')

    def testStrScale(self):
        if IGNORE_TESTS:
            return
        ##
        def test(exponent_dct):
            term = Term(2, exponent_dct)
            trues = [f"X_{n}^{p}" in str(term)  for n, p in exponent_dct.items() if p > 1]
            self.assertTrue(all(trues))
        ##
        test(EXPONENT_DCT)
        test({n: 2*n for n in range(10)})

    def testMake(self):
        #if IGNORE_TESTS:
        #    return
        term = Term.make(k=2, e1=1, e2=2)
        self.assertEqual(term, Term(2, {1: 1.0, 2: 2.0}))
        term = Term.make(k=1, e1=3, e2=2, e10=4)
        self.assertEqual(term, Term(1, {1: 3.0, 2: 2.0, 10: 4}))
        term = Term.make(k=1, e1=3, e2=2, e10=1)
        self.assertEqual(term, Term(1, {1: 3.0, 2: 2.0, 10: 1}))

    def testMakeRandomCoefficient(self):
        if IGNORE_TESTS:
            return
        term = Term.make(e1=1, e2=2)
        self.assertEqual(term, Term(term.coefficient, {1: 1.0, 2: 2.0}))

    def testGenerate(self):
        if IGNORE_TESTS:
            return
        term = Term.make(k=1, e1=1, e2=2)
        independent_variable_arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        result = term.generate(independent_variable_arr)
        expected = term.coefficient * np.prod(independent_variable_arr ** np.array([0, 1, 2], dtype=np.float32), axis=1)
        expected = expected.reshape(-1, 1).astype(np.float32)
        np.testing.assert_array_equal(result, expected) 

    def testMakeConstant(self):
        if IGNORE_TESTS:
            return
        term = Term.make(k=5)
        self.assertEqual(term, Term(5, {}))

if __name__ == '__main__':
    unittest.main()