from iplane.random_empirical import RandomEmpirical, PCollectionEmpirical, DCollectionEmpirical  # type: ignore
import iplane.constants as cn  # type: ignore

from matplotlib import pyplot as plt  # type: ignore
import numpy as np
from typing import List, Any, Tuple
import unittest

IGNORE_TESTS = True
IS_PLOT = False
NUM_SAMPLE = 1000
SAMPLE_ARR = np.random.normal(loc=0, scale=1, size=NUM_SAMPLE)  # Sample for testing


class TestRandomEmpirical(unittest.TestCase):

    def setUp(self):
        self.pcollection = PCollectionEmpirical(training_arr=SAMPLE_ARR)
        self.random = RandomEmpirical()

    def testEstimatePCollection(self):
        """Test the estimation of PCollectionEmpirical."""
        if IGNORE_TESTS:
            return
        pcollection = self.random.estimatePCollection(SAMPLE_ARR)
        self.assertIsInstance(pcollection, PCollectionEmpirical)
        self.assertTrue(np.array_equal(pcollection.get(cn.PC_TRAINING_ARR), SAMPLE_ARR))

    def testMakeDCollection(self):
        """Test the creation of DCollectionEmpirical."""
        #if IGNORE_TESTS:
        #    return
        _ = self.random.estimatePCollection(SAMPLE_ARR)
        dcollection = self.random.makeDCollection(pcollection=self.pcollection)
        self.assertIsInstance(dcollection, DCollectionEmpirical)

    def testCalculateEntropy(self):
        """Test the calculation of entropy."""
        if IGNORE_TESTS:
            return
        return
        self.random.estimatePCollection(SAMPLE_ARR)
        dcollection = self.random.makeDCollection()
        entropy = self.random.calculateEntropy(dcollection)
        self.assertIsInstance(entropy, float)

if __name__ == '__main__':
    unittest.main()