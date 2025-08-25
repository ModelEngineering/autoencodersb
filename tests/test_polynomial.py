from autoencodersb.term import Term # type: ignore
from autoencodersb.polynomial import Polynomial# type: ignore
import autoencodersb.constants as cn  # type: ignore

import matplotlib.pyplot as plt
import pandas as pd  # type: ignore
import numpy as np
from typing import List, Any, Tuple
import unittest

IGNORE_TESTS = False
IS_PLOT = False
COEFFICIENT1 = 3
EXPONENT1_DCT = {n: float(n + 1) for n in range(3)}
COEFFICIENT2 =2.5 
NUM_VARIABLE2 = 2
NUM_VARIABLE = 3  # X0, X1, X2
EXPONENT2_DCT = {n: float(n + 1) for n in range(NUM_VARIABLE2)}
TERM1 = Term(COEFFICIENT1, EXPONENT1_DCT)
TERM2 = Term(COEFFICIENT2, EXPONENT2_DCT)
TERMS = [TERM1, TERM2]


########################################
class TestPolynomial(unittest.TestCase):

    def setUp(self):
        self.polynomial = Polynomial(TERMS)

    def testConstructor(self):
        if IGNORE_TESTS:
            return
        self.assertEqual(len(self.polynomial.terms),  len(TERMS))
        self.assertEqual(self.polynomial.terms, TERMS)
        self.assertEqual(len(self.polynomial.variables), NUM_VARIABLE)

    def testStr(self):
        if IGNORE_TESTS:
            return
        for term in TERMS:
            self.assertIn(str(term), str(self.polynomial))

    def testGenerate(self):
        if IGNORE_TESTS:
            return
        term = Term.make(k=1, e1=1, e3=2)
        terms = [TERM1, term]
        polynomial = Polynomial(terms)
        variable_arr = np.array([[1, 1, 2, 3], [1, 4, 5, 6]], dtype=np.float32)
        result_arr = polynomial.generate(variable_arr)
        self.assertTrue(np.all(result_arr == np.array([33, 6144])))

    def testSum(self):
        if IGNORE_TESTS:
            return
        term = Term.make(k=13)
        terms = [term, Term.make(k=1, e1=1)]
        polynomial = Polynomial(terms)
        self.assertTrue("13" in str(polynomial))

if __name__ == '__main__':
    unittest.main()