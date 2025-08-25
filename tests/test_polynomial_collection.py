from autoencodersb.term import Term # type: ignore
from autoencodersb.polynomial_collection import PolynomialCollection  # type: ignore
from autoencodersb.polynomial import Polynomial  # type: ignore
from autoencodersb.polynomial_ratio import PolynomialRatio # type: ignore
import autoencodersb.constants as cn  # type: ignore

import numpy as np
import unittest

IGNORE_TESTS = True
IS_PLOT = True
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
        result_df = self.collection.generate(variable_arr)
        self.assertEqual(result_df.shape, (num_sample, self.collection.num_term))

    def testMake(self):
        if IGNORE_TESTS:
            return
        collection = PolynomialCollection.make(is_mm_term=True,
                is_first_order_term=True,
                is_second_order_term=True,
                is_third_order_term=True)
        self.assertIsInstance(collection, PolynomialCollection)
        self.assertEqual(str(collection).count("X_0"), 5)
        self.assertEqual(str(collection).count("X_1"), 2)
        self.assertEqual(str(collection).count("X_2"), 1)


if __name__ == '__main__':
    unittest.main()