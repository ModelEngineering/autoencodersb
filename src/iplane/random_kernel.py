'''Random for a kernel density estimation.'''
import iplane.constants as cn  # type: ignore
from iplane.random_continuous import RandomContinuous, PCollectionContinuous, DCollectionContinuous # type: ignore

import numpy as np
import matplotlib.pyplot as plt # type: ignore
from scipy.stats import gaussian_kde  # type: ignore
from scipy.optimize import minimize # type: ignore
from typing import Optional, cast, List


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

    def deprecated_findVariate(self, cdf:cn.CDF, cdf_val:float)-> np.ndarray:
        """
        Finds a variate whose CDF is close to the given CDF value.

        Args:
            cdf (CDF): The cumulative distribution function.
            cdv_val (float): The CDF value to find the corresponding variate for.

        Returns:
            np.ndarray: The variate (point) or np.array([np.nan])
        """
        ##
        def evaluatePoint(point:np.ndarray, cdf_val:float) -> float:
            """
            Evaluates the difference between the CDF value and the given CDF value.

            Args:
                point (np.ndarray): The point to evaluate.
                cdf_val (float): The CDF value to compare against.

            Returns:
                float: The absolute difference.
            """
            #estimated_cdf_val = interpolator.predictOne(point)
            trues = np.all(cdf.variate_arr <= point, axis=1)
            estimated_cdf_val = np.mean(trues)
            #print(point, estimated_cdf_val, cdf_val)
            return estimated_cdf_val - cdf_val
        ##
        num_iter = 1000
        std_incr = 0.1 # Amount by which points are changed in std increments
        # Characteristics of the variate
        std_arr = np.std(cdf.variate_arr, axis=0)
        # Search for the best fitting variate
        random_idx = np.random.randint(0, cdf.variate_arr.shape[0])
        point_estimate = np.array([np.nan])
        point = cdf.variate_arr[random_idx]
        #result = minimize(fun=evaluatePoint, x0=x0, args=(cdf_val,), method='Nelder-Mead',
        #            options=dict(maxiter=1000), tol=1e-2)
        point_estimate = np.array([np.nan])
        is_last_increase = None
        is_flip = False
        last_estimation_error = 0
        for _ in range(num_iter):
            estimation_error = evaluatePoint(point, cdf_val)
            if np.abs(estimation_error) < 1e-2:
                point_estimate = point
                import pdb; pdb.set_trace()  # Debugging breakpoint
                break
            print(estimation_error, point)
            if estimation_error < 0:
                # If the estimation error is negative, we need to increase the point
                point += std_incr * std_arr
                if is_last_increase == False:
                    if is_flip:
                        point_estimate = point
                        break
                    is_flip = True
                else:
                    is_flip = False
                is_last_increase = True
            else:
                # If the estimation error is positive, we need to decrease the point
                point -= std_incr * std_arr
                if is_last_increase == True:
                    if is_flip:
                        point_estimate = point
                        break
                    is_flip = True
                else:
                    is_flip = False
                is_last_increase = False
        #
        return point_estimate

    @staticmethod
    def calculateCDFValue(point, sample_arr:np.ndarray) -> float:
        """
        Calculates the cumulative distribution function (CDF) for a given point and sample array.

        Args:
            point (np.ndarray): The point at which to evaluate the CDF.
            sample_arr (np.ndarray): The array of samples used to estimate the CDF.

        Returns:
            float:
        """
        # Compute the CDF using the empirical distribution
        trues = sample_arr <= point
        estimate = np.mean(np.sum(trues, axis=1) == sample_arr.shape[1])
        return estimate

    def _findVariate(self, cdf:cn.CDF, cdf_val:float)-> np.ndarray:
        """
        Finds a variate whose CDF is close to the given CDF value.

        Args:
            cdf (CDF): The cumulative distribution function.
            cdv_val (float): The CDF value to find the corresponding variate for.

        Returns:
            np.ndarray: The variate (point) or np.array([np.nan])
        """
        ##
        def evaluatePoint(point:np.ndarray) -> float:
            """
            Evaluates the difference between the CDF value and the given CDF value.

            Args:
                point (np.ndarray): The point to evaluate.
                cdf_val (float): The CDF value to compare against.

            Returns:
                float: The absolute difference.
            """
            estimate_error = (self.calculateCDFValue(point, cdf.variate_arr) - cdf_val)**2
            return estimate_error
        ##
        results:list = []
        for _ in range(100):
            random_idx = np.random.randint(0, cdf.variate_arr.shape[0])
            point = cdf.variate_arr[random_idx]
            result = minimize(fun=evaluatePoint, x0=point, method='BFGS')
            if evaluatePoint(result.x) < 1e-2:
                return result.x
            results.append(result)
        # Find the best result
        results.sort(key=lambda r: evaluatePoint(r.x))
        return results[0].x

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
        cdf = self.makeCDF(pcollection.get(cn.PC_TRAINING_ARR))
        uniform_arr = np.random.uniform(0, 1, num_sample)
        sample_arr = np.array([self._findVariate(cdf, v) for v in uniform_arr])
        return sample_arr
    
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
        if self.pcollection is None:
            if pcollection is None:
                raise ValueError("PCollection has not been estimated yet.")
            else:
                self.pcollection = pcollection
        pcollection = cast(PCollectionKernel, self.setPCollection(pcollection))
        #
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
        Predicts the probability density for an a collection of variates (points) using the empirical distribution.

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
    
    def makeCDF(self, sample_arr:np.ndarray) -> cn.CDF:
        """Constructs a CDF from a two dimensional array of variates. Rows are instances; columns are variables.
        The CDF is constructed from the sample directed acyclic graph (DAG) of variates. The verticies
        of the DAG are elements (points) of sample_arr. An arc is drawn from a point A to a point B if A is less than or equal to B,
        and there is no point C such that A < C < B. The CDF is constructed by counting the number of points
        that are less than or equal to each point in sample_arr. The sample DAG is not explicitly constructed,
        but the CDF is constructed by counting the number of points that are less than or equal to each point in sample_arr.
        

        Args:
            sample_arr (np.ndarray) (N X D): An array of points, each of which is a D-dimensional array.

        Returns:
            CDF
                variate_arr (N X D)
                cdf_arr (N): CDF corresponding to the variate_arr
        """
        if sample_arr.ndim != 2:
            raise ValueError(f"Expected 2D array, got {sample_arr.ndim}D array.")
        #
        min_point = np.min(sample_arr, axis=0)
        max_point = np.max(sample_arr, axis=0)
        cdfs:list = []
        num_sample = sample_arr.shape[0]
        num_variable = sample_arr.shape[1]
        #
        for point in sample_arr:
            less_than_arr = sample_arr <= point
            less_satisfies_arr = np.sum(less_than_arr, axis=1) == num_variable
            count_less = np.sum(less_satisfies_arr) - 1
            cdf_val = count_less/num_sample
            cdfs.append(cdf_val)
        # Ensure that there is a minimum point
        if 1 in cdfs:
            idx = cdfs.index(1)
            min_point = sample_arr[idx]
            full_variate_arr = sample_arr
        else:
            cdfs.append(0)
            full_variate_arr = np.vstack([sample_arr, np.array([min_point])])
        # Ensure that there is a maximum point
        num_sample = full_variate_arr.shape[0]
        if num_sample in cdfs:
            idx = cdfs.index(num_sample)
            max_point = sample_arr[idx]
            full_variate_arr = sample_arr
        else:
            cdfs.append(1)
            full_variate_arr = np.vstack([full_variate_arr, np.array([max_point])])
        # Complete the cdf calculations
        cdf_arr = np.array(cdfs, dtype=float)
        #distance_arr = np.sqrt(np.sum(full_variate_arr**2, axis=1))
        #
        return cn.CDF(variate_arr=full_variate_arr, cdf_arr=cdf_arr, variate_min=min_point, variate_max=max_point)
    
    @classmethod
    def estimateEntropy(cls, sample_arr:np.ndarray, num_variate_sample:int) -> float:
        """
        Estimates the differential entropy for sample data using the kernel density estimation method.

        Args:
            sample_arr (np.ndarray): Array of samples to estimate the entropy from.
            num_variate_sample (int): Number of variates to sample from the distribution.

        Returns:
            float: Estimated differential entropy.
        """
        random_kernel = cls(num_variate_sample=num_variate_sample)
        pcollection = random_kernel.makePCollection(sample_arr)
        dcollection = random_kernel.makeDCollection(pcollection=pcollection)
        return dcollection.get(cn.DC_ENTROPY)
    
    def makeMarginal(self, dimensions:List[int]) -> 'RandomKernel':
        """
        Create a marginal distribution by selecting specific dimensions from the PCollection.

        Args:
            dimensions (List[int]): List of dimensions to include in the marginal distribution.

        Returns:
            RandomKernel: A new instance of RandomKernel with the specified dimensions.
        """
        if self.pcollection is None:
            raise ValueError("PCollection has not been estimated yet.")
        training_arr = self.pcollection.get(cn.PC_TRAINING_ARR)
        marginal_training_arr = training_arr[:, np.array(dimensions)]
        marginal_pcollection = self.makePCollection(marginal_training_arr)
        marginal_dcollection = self.makeDCollection(pcollection=marginal_pcollection)
        return RandomKernel(num_variate_sample=self.num_variate_sample, pcollection=marginal_pcollection,
                dcollection=marginal_dcollection)

    @classmethod 
    def makeNormalizedMutualInformation(cls, sample_arr1:np.ndarray, sample_arr2:np.ndarray,
            min_num_dimension_coordinate:Optional[int]=None) -> float:
        """
        Calculate the mutual information between two sets of samples. Normalizes by the sum of entropies of the two samples.

        Args:
            sample_arr1 (np.ndarray): First set of samples.
            sample_arr2 (np.ndarray): Second set of samples.

        Returns:
            float: Normalized mutual information between the two sets of samples.
        """
        ##
        def makeEntropy(sample_arr:np.ndarray) -> float:
            """Calculate the entropy of a sample array."""
            if sample_arr.ndim != 2:
                raise ValueError("Sample array must be 2D.")
            random = RandomKernel(num_variate_sample=sample_arr.shape[0],
                    min_num_dimension_coordinate=min_num_dimension_coordinate)
            pcollection = random.makePCollection(sample_arr)
            dcollection = random.makeDCollection(pcollection=pcollection)
            return dcollection.get(cn.DC_ENTROPY)
        ##
        #
        if sample_arr1.shape[0] != sample_arr2.shape[0]:
            raise ValueError("Both sample arrays must have the same number of rows.")
        # Create the joint sample
        joint_sample_arr = np.concatenate([sample_arr1, sample_arr2], axis=1)
        entropy_joint = makeEntropy(joint_sample_arr)
        entropy1 = makeEntropy(sample_arr1)
        entropy2 = makeEntropy(sample_arr2)
        # Calculate mutual information
        mutual_info = (entropy1 + entropy2 - entropy_joint)/(entropy1 + entropy2)
        return mutual_info