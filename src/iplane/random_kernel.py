'''Random for a kernel density estimation.'''
import iplane.constants as cn  # type: ignore
from iplane.random_continuous import RandomContinuous, PCollectionContinuous, DCollectionContinuous # type: ignore

from scipy.stats import gaussian_kde  # type: ignore
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, cast

###############################################
class PCollectionKernel(PCollectionContinuous):

    def __init__(self, training_arr:np.ndarray, kde:gaussian_kde):
        """
        Args:
            training_arr (np.ndarray): Array of training data points.
            kde (gaussian_kde): Kernel density estimator object.
        """
        super().__init__(cn.PC_KERNEL_NAMES, dict(training_arr=training_arr, kde=kde))
        self.isValid()

    @property
    def num_dimension(self) -> int:
        """
        Returns:
            int: Number of dimensions in the training data.
        """
        training_arr = self.get(cn.PC_TRAINING_ARR)
        if training_arr is None:
            raise ValueError("Training array is not set.")
        return training_arr.shape[1]


###############################################
class DCollectionKernel(DCollectionContinuous):
    pass


###############################################
class RandomKernel(RandomContinuous):
    """Handles Gaussian Kernel Models."""

    def __init__(self,
            num_variate_sample:int = cn.NUM_VARIATE_SAMPLE,
            **kwargs
            ):
        """
        Initializes the KernelEntropy object.
        Args:
            num_variate_sample (int): total number of samples to generate.
            num_component (int): number of components in the mixture model.
            random_state (int): random state for reproducibility.
        """
        super().__init__(**kwargs)
        self.num_variate_sample = num_variate_sample
        # Use k-means clustering to initialize the Gaussian Kernel Model

    def generateSample(self, pcollection:PCollectionKernel, num_sample:int) -> np.ndarray:
        """
        Generates synthetic data for a multidimensional Gaussian Kernel Model.
            Each Gaussian component is defined by its mean and covariance marix.
            Components are indexed by the first array index
            Dimensions are indexed by the second array index for mean.

        Args: N is number of components, D is number of dimensions, K is number of categories.
            sample_arr (np.ndarray N): Number of samples to generate for each component.
            mean_arr (np.ndarray N X D): Mean of each Gaussian component.
            covariance_arr (np.ndarray N X D X D): Covariance matrix for each Gaussian component.
                    if D = 1, then covariance_arr is N X 1

        Returns:
            np.array (num_sample, 1), int. total count is = sum(num_samples)
        """
        raise NotImplementedError("This method is not implemented for RandomKernel.")
    
    def makeDCollection(self, variate_arr:Optional[np.ndarray]=None,
            pcollection:Optional[PCollectionKernel]=None) -> DCollectionKernel:
        """
        Calculates the probability density function (PDF) for a multi-dimensional Gaussian mixture model
        and calculates its differential entropy.

        Args:
            pcollection (PCollectionKernel): The collection of parameters for the Gaussian mixture model.
            variate_arr (Optional[np.ndarray]): Optional array of variates to evaluate the PDF.

        Returns:
            DistributionCollectionMGaussian: The distribution object containing the variate array, PDF array, dx array, and entropy.
        """
        # Initializations
        pcollection = cast(PCollectionKernel, self.setPCollection(pcollection))
        training_arr = cast(np.ndarray, pcollection.get(cn.PC_TRAINING_ARR))
        # Construct the the variate and dx
        std_arr = np.std(training_arr, axis=0)
        mean_arr = np.mean(training_arr)
        min_point = mean_arr - 0.5*self.axis_length_std*std_arr
        max_point = mean_arr + 0.5*self.axis_length_std*std_arr
        variate_arr, dx_arr = self.makeVariate(min_point, max_point, self.num_variate_sample)
        # Estimate the density and entropy
        density_arr = self.predict(variate_arr, pcollection)
        entropy = self.makeEntropy(density_arr=density_arr, dx_arr=dx_arr)
        # Make the DCollection
        self.dcollection = DCollectionKernel(
            variate_arr=variate_arr,
            density_arr=density_arr,
            dx_arr=dx_arr,
            entropy=entropy)
        return self.dcollection
    
    def predict(self, small_variate_arr:np.ndarray,
                pcollection:Optional[PCollectionKernel]=None) -> np.ndarray:
        """
        Predicts the probability density function (PDF) for a given array of variates using the Gaussian Kernel Model.

        Args:
            small_variate_arr (np.ndarray): Single variate array of shape (1, num_dimension).

        Returns:
            np.ndarray: Array of predicted PDF values for each variate.
        """
        # Error checking
        if pcollection is None:
            if self.pcollection is None:
                raise ValueError("PCollection has not been estimated yet.")
            else:
                pcollection = self.pcollection
            pcollection = cast(PCollectionKernel, self.pcollection)
        # Initializations
        kde = pcollection.get(cn.PC_KDE)
        density_arr = kde(small_variate_arr.T)
        return density_arr

    def makePCollection(self, sample_arr:np.ndarray)->PCollectionKernel:
        """
        Estimates the parameters of a Gaussian Kernel Model from the sample array.

        Args:
            sample_arr: array of one dimensional variates

        Returns:
            PCollectionKernel
        """
        # Check the data
        if not isinstance(sample_arr, np.ndarray):
            raise ValueError("sample_arr must be an array.")
        if sample_arr.ndim != 2:
            raise ValueError("sample_arr must be 2d.")
        #
        kde = gaussian_kde(sample_arr.T, bw_method='silverman')
        #
        self.pcollection = PCollectionKernel(
            training_arr=sample_arr,
            kde=kde
        )
        return self.pcollection