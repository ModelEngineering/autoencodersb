from iplane.interpolator import Interpolator  # type: ignore
from iplane.random_mixture_collection import PCollectionMixture, DCollectionMixture  # type: ignore
from iplane.random_mixture import RandomMixture  # type: ignore
from iplane.random_empirical import RandomEmpirical  # type: ignore
import iplane.constants as cn  # type: ignore

from matplotlib import pyplot as plt  # type: ignore
import numpy as np
from scipy.stats import norm, multivariate_normal # type: ignore
import unittest

IGNORE_TESTS = True
IS_PLOT = True
NUM_SAMPLE = 10
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
        if IGNORE_TESTS:
            return
        for is_normalized  in [True, False]:
            interpolator = Interpolator( variate_arr=VARIATE_ARR, sample_arr=SAMPLE_ARR,
                    is_normalize=is_normalized)
            for idx, point in enumerate(VARIATE_ARR):
                result = interpolator.predict(point)
                self.assertIsInstance(result, np.ndarray)
                self.assertEqual(result.shape, (2,))
                self.assertTrue(np.allclose(result, SAMPLE_ARR[idx, :], atol=1e-5))

    def testPredictUnivariateDistribution(self)->None:
        """Test the predict method."""
        if IGNORE_TESTS:
            return
        num_sample = 50
        variate_arr = np.random.normal(0, 1, num_sample)
        sorted_variate_arr = np.array(variate_arr)
        sorted_variate_arr.sort()
        sorted_sample_arr = np.cumsum(np.repeat(1/num_sample, num_sample))
        permutation = np.random.permutation(range(num_sample))
        final_variate_arr = np.reshape(sorted_variate_arr[permutation], (-1, 1))
        final_sample_arr = sorted_sample_arr[permutation]
        #
        interpolator = Interpolator( variate_arr=final_variate_arr, sample_arr=final_sample_arr,
                is_normalize=False)
        errors:list = []
        for _ in range(1000):
            point = np.array(np.random.uniform(-4, 4))
            if not interpolator.isWithinRange(point):
                continue
            probability = norm.cdf(point, 0, 1)
            result = interpolator.predict(point)
            errors.append(abs(probability - result[0]))
        print(np.mean(errors), np.std(errors))

    def testPredictMultivariateDistribution(self)->None:
        """Test the predict method for multivariate distribution."""
        #if IGNORE_TESTS:
        #    return
        NUM_DIM = 4 
        NUM_SAMPLE = int(1e2)
        NUM_ITERATION = 100
        VARIANCE = 10
        MEAN_ARR = np.repeat(0, NUM_DIM)
        COVARIANCE_ARR = np.eye(NUM_DIM, NUM_DIM) * VARIANCE
        WEIGHT_ARR = np.array([1.0])
        pcollection = PCollectionMixture(
            mean_arr=np.array([MEAN_ARR]),
            covariance_arr=np.array([COVARIANCE_ARR]),
            weight_arr=WEIGHT_ARR,
        )
        random_mixture = RandomMixture()
        random_empirical = RandomEmpirical()
        sample_arr = random_mixture.generateSample(pcollection, num_sample=NUM_SAMPLE)
        _ = random_empirical.estimatePCollection(sample_arr)
        cdf = random_empirical.makeCDF(sample_arr)
        variate_arr = cdf.variate_arr
        cdf_arr = cdf.cdf_arr
        #
        interpolator = Interpolator( variate_arr=variate_arr, sample_arr=cdf_arr,
                is_normalize=True, max_distance=1, size_interpolation_set=5)
        errors:list = []
        avg_errors:list = []
        results:list = []
        probabilities:list = []
        for _ in range(NUM_ITERATION):
            point = np.random.uniform(-3, 3, (NUM_DIM,))
            if not interpolator.isWithinRange(point):
                continue
            if NUM_DIM == 1:
                probability = norm.cdf(point, 0, scale=VARIANCE**0.5)
            else:
                probability = multivariate_normal.cdf(point, mean=MEAN_ARR, cov=COVARIANCE_ARR)  # type: ignore
            result = interpolator.predict(point)
            if np.isnan(result[0]):
                continue
            results.append(result[0])
            probabilities.append(probability)
            errors.append(abs(probability - result[0]))
            avg_errors.append(probability - result[0])
        if np.mean(errors) > 1:
            import pdb; pdb.set_trace()
        print(f"\nmean error in prob: {np.mean(errors)}")
        print(f"avg mean error in prob: {np.mean(avg_errors)}")
        print(f"std: {np.std(errors)}")
        print(f"Frac succ: {len(errors)/NUM_ITERATION}")



if __name__ == '__main__':
    unittest.main()