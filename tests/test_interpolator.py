from iplane.interpolator import Interpolator  # type: ignore
from iplane.random_mixture_collection import PCollectionMixture, DCollectionMixture  # type: ignore
from iplane.random_mixture import RandomMixture  # type: ignore
from iplane.random_empirical import RandomEmpirical  # type: ignore
import iplane.constants as cn  # type: ignore

from matplotlib import pyplot as plt  # type: ignore
import numpy as np
from scipy.stats import norm, multivariate_normal # type: ignore
import unittest

IGNORE_TESTS = False
IS_PLOT = False
NUM_SAMPLE = 10
STD = 4
VARIATE_ARR = np.array(range(20)).reshape(-1, 2)  # Sample for testing
SAMPLE_ARR = 100*VARIATE_ARR[:, 0] + VARIATE_ARR[:, 1]


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
        self.assertEqual(self.interpolator.sample_arr.shape, (NUM_SAMPLE,)) 

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
        if IGNORE_TESTS:
            return
        for is_normalized  in [True, False]:
            interpolator = Interpolator( variate_arr=VARIATE_ARR, sample_arr=SAMPLE_ARR,
                    is_normalize=is_normalized)
            result_arr = interpolator.predict(VARIATE_ARR)
            self.assertIsInstance(result_arr, np.ndarray)
            self.assertEqual(len(result_arr), len(SAMPLE_ARR))
            self.assertLessEqual(np.std(result_arr-SAMPLE_ARR), 1e-3)  # Check if the result is close to zero

    def testPredictScaled(self):
        """Scale Test the predict method."""
        if IGNORE_TESTS:
            return
        def test(num_sample, noise=0.1):
            variate_arr = np.array(range(num_sample)).reshape(-1, 2)  # Sample for testing
            sample_arr = 100*variate_arr[:, 0] + variate_arr[:, 1] + np.random.normal(0, noise, num_sample//2)
            for is_normalized  in [True, False]:
                interpolator = Interpolator( variate_arr=variate_arr, sample_arr=sample_arr,
                        is_normalize=is_normalized)
                result_arr = interpolator.predict(variate_arr)
                #shift_arr = np.random.normal(0, 1, variate_arr.shape)
                #result_arr = interpolator.predict(variate_arr + shift_arr)
                self.assertIsInstance(result_arr, np.ndarray)
                self.assertEqual(len(result_arr), len(sample_arr))
                self.assertLessEqual(np.std(result_arr-sample_arr), 1e-3)  # Check if the result is close to zero
                if IS_PLOT:
                    plt.scatter(variate_arr[:, 0], sample_arr, label='Sample')
                    plt.scatter(variate_arr[:, 0], result_arr, label='Prediction', alpha=0.5)
                    plt.title('Prediction vs Sample')
                    plt.xlabel('Variate Dimension 1')
                    plt.ylabel('Sample Value')
                    plt.legend()
                    plt.grid()
                    plt.show()
        #
        test(20, noise=10)
        test(100)
        test(10000)



if __name__ == '__main__':
    unittest.main()