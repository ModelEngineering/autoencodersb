from autoencodersb.term import Term # type: ignore
from autoencodersb.polynomial_collection import PolynomialCollection  # type: ignore
from autoencodersb.polynomial import Polynomial  # type: ignore
from autoencodersb.polynomial_ratio import PolynomialRatio # type: ignore
import autoencodersb.constants as cn  # type: ignore

import numpy as np
import unittest

IGNORE_TESTS = False
IS_PLOT = False
TERMS = [Term.make(k=2*n, e0=n) for n in range(3)]
TERMS.append(Term.make(k=1, e4=3))


########################################
class TestPolynomialCollection(unittest.TestCase):

    def setUp(self):
        self.collection = PolynomialCollection(TERMS)

    def testConstructor(self):
        if IGNORE_TESTS:
            return
        self.assertEqual(len(self.collection.terms), len(TERMS))
        self.assertEqual(self.collection.terms, TERMS)
        self.assertEqual(len(self.collection.variables), 2)
        self.assertEqual(np.max(self.collection.variables), 4)
        self.assertEqual(len(self.collection.term_strs), len(TERMS))

    def testRepr(self):
        if IGNORE_TESTS:
            return
        self.assertEqual(len(self.collection.terms), str(self.collection).count(",") + 1)
        for term in TERMS:
            self.assertIn(str(term), repr(self.collection))

    def testGenerate(self):
        if IGNORE_TESTS:
            return
        num_sample = 10
        arrs = [np.array(range(num_sample)).reshape(-1, 1)] * self.collection.num_variable
        variable_arr = np.hstack(arrs)
        result = self.collection.generate(variable_arr)
        self.assertEqual(result.shape, (num_sample, self.collection.num_term))


if __name__ == '__main__':
    unittest.main()