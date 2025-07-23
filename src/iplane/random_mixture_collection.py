'''Collection classes used by RandomMixture of Gaussian distribution.'''

import iplane.constants as cn  # type: ignore
from iplane.random_continuous import PCollectionContinuous, DCollectionContinuous # type: ignore

import collections
import pandas as pd  # type: ignore
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, List, Tuple, Optional, Dict, cast


PCollectionShape = collections.namedtuple('PCollectionShape', ['num_component', 'num_dimension'])
DCollectionShape = collections.namedtuple('DCollectionShape', ['num_sample', 'num_dimension'])


################################################
class PCollectionMixture(PCollectionContinuous):
    # Parameter collection for mixture of Gaussian distributions.
    #   C: number of components
    #   D: number of dimensions
    # Instance variables:
    #   collection_names: list of all names of parameters
    #   collection_dct: dictionary of subset name-value pairs
    #   mean_arr: C X D, mean of each Gaussian component
    #   covariance_arr: C X D X D, covariance matrix for each Gaussian component
    #   weight_arr: C, weight of each Gaussian component, sum(weight_arr) =

    def __init__(self,
                mean_arr:np.ndarray=np.array([[100]]),
                covariance_arr:np.ndarray=np.array([[[1]]]),
                weight_arr:np.ndarray=np.array([1.0]))->None:
        """
        Args:
            parameter_dct (Optional[Dict[str, Any]], optional): parameter name-value pairs.
        """
        dct = dict(
            mean_arr=mean_arr,
            covariance_arr=covariance_arr,
            weight_arr=weight_arr,
        )
        super().__init__(cn.PC_MIXTURE_NAMES, dct)
        self.isValid()
        # Calculated
        self._std_point = np.array([])
        self._center_point = np.array([])

    @property
    def std_point(self) -> np.ndarray:
        """
        Returns a point representing one standard deviation in each dimension.
        """
        if self._std_point.size == 0:
            self._std_point = np.sqrt(np.diagonal(self.get(cn.PC_COVARIANCE_ARR), axis1=1, axis2=2))
        return self._std_point
    
    @property
    def center_point(self) -> np.ndarray:
        """
        Returns a center point for the distribution, typically the mean.
        """
        if self._center_point.size == 0:
            mean_arr, covariance_arr, _ = self.getAll()
            self._center_point = np.mean(self.get(cn.PC_MEAN_ARR), axis=0)
        return self._center_point
        
    
    @property
    def num_dimension(self) -> int:
        """
        Returns the number of dimensions for the distribution.
        """
        return len(self.center_point)

    def __str__(self) -> str:
        """
        Returns a string representation of the PCollectionMixture object.
        """
        mean_arr, covariance_arr, weight_arr = self.getAll()
        return ("mean_arr, covariance_arr, weight_arr"
                f"\nPCollectionMixture(mean_arr={mean_arr}"
                f"\ncovariance_arr={covariance_arr}, "
                f"\nweight_arr={weight_arr})")

    def getAll(self) ->Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns all parameters as a tuple of numpy arrays.
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple containing mean_arr, covariance_arr, and weight_arr.
        """
        mean_arr = cast(np.ndarray, self.get(cn.PC_MEAN_ARR))
        covariance_arr = cast(np.ndarray, self.get(cn.PC_COVARIANCE_ARR))
        weight_arr = cast(np.ndarray, self.get(cn.PC_WEIGHT_ARR))
        return mean_arr, covariance_arr, weight_arr
    
    def getShape(self) -> PCollectionShape:
        """
        Finds the number of components and dimensions for the gaussian distribution.

        Returns:
            Number of components
            Number of dimensions
        """
        mean_arr, covariance_arr, weight_arr = self.getAll()
        # Consistent number of components and dimensions
        if weight_arr.ndim != 1:
            raise ValueError("Weight must have 1 dimension.")
        is_correct_shape = False
        if mean_arr.ndim == 2 and covariance_arr.ndim == 3:
            is_correct_shape = True
        elif mean_arr.ndim == 1 and covariance_arr.ndim == 1:
            is_correct_shape = True
        if not is_correct_shape:
            import pdb; pdb.set_trace()
            raise ValueError("mean_arr must be 2D and covariance_arr must be 3D.")
        num_component = mean_arr.shape[0]
        num_dimension = mean_arr.shape[1] if mean_arr.ndim == 2 else 1
        if not np.all([mean_arr.shape[0], covariance_arr.shape[0], weight_arr.shape[0]] == [num_component] * 3):
            raise ValueError("Mean, covariance, and weight arrays must have the same number of components.")
        #
        return PCollectionShape(num_component=num_component, num_dimension= num_dimension)
    
    def select(self, dimensions:List[int]) -> 'PCollectionMixture':
        """
        Selects a subset of the parameters based on the provided indices.

        Args:
            indices (List[int]): List of indices to select from the parameter collection.

        Returns:
            PCollectionMixture: A new PCollectionMixture object with the selected parameters.
        """
        # Check if the dimensions are valid
        shape = self.getShape()
        num_component, num_dimension = shape.num_component, shape.num_dimension
        if num_dimension == 1:
            raise ValueError("Cannot select dimensions from a 1D Gaussian mixture.")
        if max(dimensions) >= num_dimension:
            raise ValueError(f"Dimensions must be less than {num_dimension}. Provided dimensions: {dimensions}")
        # Create the new PCollectionMixture with selected dimensions
        indices = np.array(dimensions, dtype=int)
        collection_dct = dict(self.collection_dct)
        collection_dct[cn.PC_MEAN_ARR] = collection_dct[cn.PC_MEAN_ARR][:, indices]
        collection_dct[cn.PC_COVARIANCE_ARR] = collection_dct[cn.PC_COVARIANCE_ARR][:, indices, indices]
        if len(dimensions) == 1:
            collection_dct[cn.PC_MEAN_ARR] = collection_dct[cn.PC_MEAN_ARR].reshape(num_component, 1)
            collection_dct[cn.PC_COVARIANCE_ARR] = collection_dct[cn.PC_COVARIANCE_ARR].reshape(num_component, 1, 1)
        return PCollectionMixture(**collection_dct)
    
    def isAllValid(self):
        """
        Checks if the elements of the collection have the correct dimensions.

        Returns:
            bool: True if the dictionary is valid, False otherwise.
        """
        super().isAllValid()
        # Check if all required keys are present
        mean_arr, covariance_arr, weight_arr = self.getAll()
        shape = self.getShape()
        num_component, num_dimension = shape.num_component, shape.num_dimension
        if (mean_arr is not None) and (mean_arr.shape != (num_component, num_dimension)):
            raise ValueError(f"Mean array must have shape ({num_component}, {num_dimension}).")
        if (covariance_arr is not None) and (covariance_arr.shape != (num_component, num_dimension, num_dimension)):
            raise ValueError(f"Covariance array must have shape ({num_component}, {num_dimension}, {num_dimension}).")
        if (weight_arr is not None) and (weight_arr.shape != (num_component,)):
            raise ValueError(f"Weight array must have shape ({num_component},).")
        
    def reshape(self, name:str, value:Union[int, float, np.ndarray],
            has_component:bool=True, has_dimension:bool=False) -> np.ndarray:
        """
        Reshapes the element for the collection based on the provided name and dimensions.

        Args:
            name (str): Name of the parameter to reshape.
            value (Union[int, float, np.ndarray]): Value to reshape.
            has_component (bool): Whether the parameter has a multivariate component.
            has_dimension (bool): Whether the parameter has a multivariable dimension.

        Returns:
            np.ndarray: Reshaped parameter array.
        """
        result = cn.NULL_ARR
        if name not in self.collection_names:
            raise ValueError(f"Parameter '{name}' is not in the collection names.")
        if name == cn.PC_MEAN_ARR:
            # Mean array is C X D
            if isinstance(value, (float, int)):
                result = np.array([[value]])
            elif value.ndim == 2:
                result = value
            elif value.ndim == 1:
                if has_component:
                    result = value.reshape(-1, 1)
                elif has_dimension:
                    result = value.reshape(1, -1)
                else:
                    result = np.reshape(value, (1,1))
            else:
                raise ValueError(f"Mean array has invalid shape {value.shape}.")
        if name == cn.PC_COVARIANCE_ARR:
            # Covariance array is C X D X D
            if isinstance(value, (float, int)):
                result = np.array([[[value]]])
            elif value.ndim == 1:
                if has_component:
                    result = value.reshape(-1, 1, 1)
                elif has_dimension:
                    mat = np.zeros((1, value.shape[0], value.shape[0]), dtype=value.dtype)
                    result = mat
                else:
                    result = np.reshape(value, (1,1))
            else:
                raise ValueError(f"Covariance array has invalid shape {value.shape}.")
        if name == cn.PC_WEIGHT_ARR:
            # Weight array is C
            if isinstance(value, (float, int)):
                result = np.array([value])
            elif value.ndim == 1:
                result = value
            else:
                raise ValueError(f"Weight array has invalid shape {value.shape}.")
        #
        if result is cn.NULL_ARR:
            raise ValueError(f"Parameter '{name}' has no valid shape.")
        return result
    
    @classmethod
    def make(cls, 
            num_component:int=2,
            num_dim:int=1,
            variance:float=0.5,
            covariance:float=0.0,  # Between non-identical dimen
            weight_arr:Optional[np.ndarray]=None,
            ) -> 'PCollectionMixture':
        """
        Factory method to create a PCollectionMixture instance.

        Args:
            num_component (int): Number of components in the mixture.
            num_dim (int): Number of dimensions for each component.
            variance (float): Variance for the Gaussian components.
            covariance (float): Covariance between different dimensions.
            weight_arr (Optional[np.ndarray]): Optional array of weights for each component.
                    If None, weights are uniform.

        Returns:
            PCollectionMixture: Instance of PCollectionMixture. Weights are proportional to
            the sample size of each component.
        """
        means:list = []
        covariances:list = []
        if weight_arr is None:
            weight_arr = np.repeat(1/num_component, num_component)
        # Construct the means and covariances for each component
        for n_component in range(num_component):
                means.append([5*n_component + 0.2*n_dim for n_dim in range(1, num_dim + 1)])
                # Covariances
                matrix = np.repeat(covariance, num_dim*num_dim).reshape((num_dim, num_dim))
                np.fill_diagonal(matrix, variance)
                covariances.append(matrix)
        covariance_arr = np.array(covariances)
        mean_arr = np.reshape(np.array(means), (num_component, num_dim))
        pcollection = PCollectionMixture(mean_arr= mean_arr, covariance_arr=covariance_arr,
                weight_arr= weight_arr)
        return pcollection



################################################
class DCollectionMixture(DCollectionContinuous):
    # Distribution collection for mixture of Gaussian distributions.
    #   C: number of components
    #   N: number of samples
    #   D: number of dimensions
    #   variate_arr: N X D
    #   density_arr: N
    #   dx_arr: D
    #   entropy: float

    def __init__(self, 
                variate_arr:Optional[np.ndarray]=None,
                density_arr:Optional[np.ndarray]=None,
                dx_arr:Optional[np.ndarray]=None,
                entropy:Optional[float]=None)->None:
        dct = dict(
            variate_arr=variate_arr,
            density_arr=density_arr,
            dx_arr=dx_arr,
            entropy=entropy,
        )
        super().__init__(variate_arr=variate_arr, density_arr=density_arr, dx_arr=dx_arr, entropy=entropy)
        self.actual_collection_dct = dct
        self.isAllValid()

    def __str__(self) -> str:
        """
        Returns a string representation of the DCollectionMixture object.
        """
        variate_arr, density_arr, dx_arr, entropy = self.getAll()
        return (f"DCollectionMixture(variate_arr={variate_arr}"
                f"\ndensity_arr={density_arr}"
                f"\ndx_arr={dx_arr}"
                f"\nentropy={entropy})")
    
    def getAll(self) ->Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Returns all parameters as a tuple of numpy arrays.
                    variate_arr: np.ndarray, density_arr: np.ndarray, dx_arr: np.ndarray, entropy: float
        """
        variate_arr, density_arr, dx_arr, entropy = cast(np.ndarray, self.get(cn.DC_VARIATE_ARR)),   \
                cast(np.ndarray, self.get(cn.DC_DENSITY_ARR)), cast(np.ndarray, self.get(cn.DC_DX_ARR)),  \
                cast(float, self.get(cn.DC_ENTROPY))
        return variate_arr, density_arr, dx_arr, entropy

    def getShape(self) -> DCollectionShape:
        """
        Returns:
            num_sample (int): Number of samples in the variate array.
            num_dimension (int): Number of dimensions in the variate array.
        """
        variate_arr = self.get(cn.DC_VARIATE_ARR)
        if variate_arr is None:
            raise ValueError("Variate array must not be None.")
        return DCollectionShape(num_sample=variate_arr.shape[0], num_dimension=variate_arr.shape[1])
    
    def isAllValid(self):
        """
        Checks if the elements of the collection have the correct dimensions.

        Returns:
            bool: True if the dictionary is valid, False otherwise.
        """
        super().isAllValid()
        # Check if all required keys are present
        variate_arr, density_arr, dx_arr, entropy = self.getAll()
        collection_shape = self.getShape()
        num_sample = collection_shape.num_sample
        num_dimension = collection_shape.num_dimension
        if (variate_arr is not None) and (variate_arr.shape != (num_sample, num_dimension)):
            raise ValueError(f"Variate array must have shape ({num_sample}, {num_dimension}).")
        if (density_arr is not None) and (density_arr.shape != (num_sample,)):
            raise ValueError(f"Density array must have shape ({num_sample},).")
        if (dx_arr is not None) and (dx_arr.shape != (num_dimension,)):
            raise ValueError(f"dx array must have shape ({num_dimension},).")
        if (entropy is not None) and (not isinstance(entropy, (int, float))):
            raise ValueError("Entropy must be a float value.")
