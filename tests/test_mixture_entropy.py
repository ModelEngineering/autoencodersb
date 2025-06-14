from iplane.mixture_entropy import MixtureEntropy  # type: ignore

from matplotlib import pyplot as plt  # type: ignore
import numpy as np
import unittest

IGNORE_TESTS = True
IS_PLOT = True


class TestMixtureEntropy(unittest.TestCase):
    """Test class for MixtureEntropy."""

    def setUp(self):
        """Set up the test case."""
        pass

    def testGenerateMixture(self):
        """Test the generateMixture method."""
        if IGNORE_TESTS:
            return
        num_samples = [100, 200]
        means = np.array([0, 10])
        convariances = MixtureEntropy.makeConvariance([0.5, 0.5])
        mixture_entropy = MixtureEntropy(
            means=means,
            covariances=convariances)
        arr = mixture_entropy.generateMixture(num_samples=num_samples)
        self.assertEqual(arr.shape[0], np.sum(num_samples))

    def testGenerateMixture2D(self):
        """Test the generateMixture method."""
        #if IGNORE_TESTS:
        #    return
        NUM_DIM = 3
        num_samples = [100, 200]
        means = np.array([0, 10])
        convariances = MixtureEntropy.makeConvariance([0.5, 0.5])
        mixture_entropy = MixtureEntropy(
            means=means,
            covariances=convariances,
            num_dim=NUM_DIM)
        arr = mixture_entropy.generateMixture(num_samples=num_samples)
        if IS_PLOT:
            plt.hist(arr, bins=30)
            plt.title(f"Mixture of Gaussian Distributions with {NUM_DIM} Dimensions")
            plt.show()
        self.assertEqual(arr.shape[0], np.sum(num_samples))
        self.assertEqual


if __name__ == '__main__':
    unittest.main()