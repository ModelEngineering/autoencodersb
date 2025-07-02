from iplane.interpolator import Interpolator  # type: ignore
import iplane.constants as cn  # type: ignore

from matplotlib import pyplot as plt  # type: ignore
import numpy as np
from typing import List, Any, Tuple
import unittest

IGNORE_TESTS = True
IS_PLOT = True
NUM_SAMPLE = 50
STD = 4
VARIATE_ARR = np.array(range(2*NUM_SAMPLE)).reshape((NUM_SAMPLE, 2))
SAMPLE_ARR = np.array(VARIATE_ARR)
SAMPLE_ARR[:, 1] = 5 + VARIATE_ARR[:, 1]


class TestInterpolator(unittest.TestCase):

    def setUp(self):
        self.interpolator = Interpolator( variate_arr=VARIATE_ARR, sample_arr=SAMPLE_ARR, is_normalize=True)

    def testNormalize(self):
        if IGNORE_TESTS:
            return
        """Test the normalization of variate array."""
        normalized = self.interpolator._normalize(VARIATE_ARR)
        self.assertTrue(np.allclose(normalized, VARIATE_ARR / np.std(VARIATE_ARR, axis=0)))

    def testConstructor(self):
        if IGNORE_TESTS:
            return
        """Test the constructor initializes correctly."""
        self.assertIsInstance(self.interpolator, Interpolator)
        self.assertEqual(self.interpolator.variate_arr.shape, (NUM_SAMPLE, 2))
        self.assertEqual(self.interpolator.sample_arr.shape, (NUM_SAMPLE, 2)) 

    def testGetIndexNearestVariateBasic(self):
        """Test finding the nearest variate."""
        if IGNORE_TESTS:
            return
        exclude_idxs = []
        for is_normalized  in [True, False]:
            interpolator = Interpolator( variate_arr=VARIATE_ARR, sample_arr=SAMPLE_ARR,
                    is_normalize=is_normalized)
            for point in VARIATE_ARR:
                idx, distance = interpolator._getIndexNearestVariate(point, exclude_idxs)
                self.assertIsInstance(idx, (np.int64, int))
                self.assertIsInstance(distance, float)
                self.assertTrue(0 <= idx < NUM_SAMPLE)
                self.assertTrue(np.all(VARIATE_ARR[idx] == point))

    def testGetIndexNearestVariateExcludes(self):
        """Test finding the nearest variate."""
        if IGNORE_TESTS:
            return
        for is_normalized  in [True, False]:
            interpolator = Interpolator( variate_arr=VARIATE_ARR, sample_arr=SAMPLE_ARR,
                    is_normalize=is_normalized)
            for idx, point in enumerate(VARIATE_ARR):
                exclude_idxs = [idx]
                point[0] = point[0] + 1  # Modify the point to ensure it is not equal to the variate
                idx, _ = interpolator._getIndexNearestVariate(point, exclude_idxs)
                self.assertIsInstance(idx, (np.int64, int))
                self.assertTrue(-1 <= idx < NUM_SAMPLE)
                if idx >= 0:
                    self.assertFalse(np.all(VARIATE_ARR[idx, :] == point))

    def testPredict(self):
        """Test the predict method."""
        #if IGNORE_TESTS:
        #    return
        for is_normalized  in [True, False]:
            interpolator = Interpolator( variate_arr=VARIATE_ARR, sample_arr=SAMPLE_ARR,
                    is_normalize=is_normalized)
            for idx, point in enumerate(VARIATE_ARR):
                result = interpolator.predict(point)
                self.assertIsInstance(result, np.ndarray)
                self.assertEqual(result.shape, (2,))
                self.assertTrue(np.allclose(result, SAMPLE_ARR[idx, :], atol=1e-5))

if __name__ == '__main__':
    unittest.main()