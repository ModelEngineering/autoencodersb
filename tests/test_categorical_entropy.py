from iplane.categorical_entropy import CategoricalEntropy  # type: ignore

import numpy as np
from typing import List, Any
import unittest

IGNORE_TESTS = False
IS_PLOT = False
NUM_SAMPLE = 1000
CATEGORICAL_ARR = np.random.randint(0, 5, NUM_SAMPLE)  # Random categorical data with 5 categories


class TestCategoricalEntropy(unittest.TestCase):

    def setUp(self):
        """Set up the test case."""
        self.categorical_entropy = CategoricalEntropy()

    def testConstructor(self):
        self.assertTrue(hasattr(self.categorical_entropy, 'entropy'))

    def testCalculate(self):
        ##
        def test(val:int):
            """Helper function to test entropy calculation."""
            arr = np.random.randint(0, val, NUM_SAMPLE)
            self.categorical_entropy.calculate(arr)
            self.assertEqual(len(self.categorical_entropy.categories), val)
            self.assertAlmostEqual(self.categorical_entropy.entropy, np.log2(val), delta=0.1)
        ##
        test(2)
        test(10)
        test(50)
    
    def testCalculateEntropy(self):
        ##
        def test(val:int):
            """Helper function to test entropy calculation."""
            arr = np.random.randint(0, val, NUM_SAMPLE)
            Hx = self.categorical_entropy.calculateEntropy(arr)
            self.assertAlmostEqual(Hx, np.log2(val), delta=0.1)
        ##
        test(2)
        test(10)
        test(50)


if __name__ == '__main__':
    unittest.main()