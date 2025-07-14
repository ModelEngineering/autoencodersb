from iplane.random_discrete import RandomDiscrete  # type: ignore

import numpy as np
from typing import List, Any
import unittest

IGNORE_TESTS = False
IS_PLOT = False
NUM_SAMPLE = 1000
CATEGORICAL_ARR = np.random.randint(0, 5, NUM_SAMPLE)  # Random categorical data with 5 categories


class TestRandomDiscrete(unittest.TestCase):

    def setUp(self):
        """Set up the test case."""
        self.discrete = RandomDiscrete()

    def testEstimatePCollection(self):
        ##
        def test(val:int):
            """Helper function to test entropy calculation."""
            arr = np.random.randint(0, val, NUM_SAMPLE)
            pcollection = self.discrete.makePCollection(arr)
            self.assertEqual(len(pcollection.get('category_arr')), val)
        ##
        test(2)
        test(10)
        test(50)
    
    def testCalculateEntropy(self):
        ##
        def test(val:int):
            """Helper function to test entropy calculation."""
            arr = np.random.randint(0, val, NUM_SAMPLE)
            pcollection = self.discrete.makePCollection(arr)
            dcollection = self.discrete.makeDCollection(pcollection=pcollection)
            entropy = dcollection.get('entropy')
            self.assertAlmostEqual(entropy, np.log2(val), delta=0.1)
        ##
        test(2)
        test(10)
        test(50)


if __name__ == '__main__':
    unittest.main()