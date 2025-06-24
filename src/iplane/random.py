'''Random estimates parameters and generates random samples from parameters.'''

""" 
This is an abstract class for estimating parameters of a distribution.
A subclass implements the estimate and generate methods. It is responsible for constructing
a Parameter object appropriate for these methods. Generated distributions are stored in a Distribution object.
"""
import iplane.constants as cn

import numpy as np
from typing import Any, Optional, Dict, List

ENTROPY = 'entropy'


############################################
class Collection(object):
    # Base class for collections of parameter and distribution information

    def __init__(self, collection_names:List[str], collection_dct:Optional[Dict[str, Any]]=None)->None:
        """
        Args:
            collection_names (List[str]): Names of all parameters
            collection_dct (Optional[Dict[str, Any]], optional): parameter name-value pairs.
        """
        self.collection_names = list(collection_names)
        if collection_dct is None:
            self.collection_dct = {}
        else:
            self.collection_dct = dict(collection_dct)
        if not self.isValid():
            raise ValueError(f"Collection dictionary must contain all expected keys: {self.collection_names}")

    def get(self, name:str) -> Any:
        """Get the value of a parameter by its name."""
        return self.collection_dct.get(name, None)

    def isValid(self) -> bool:
        """Check if the dictionary contains only valid keys."""
        trues =  [key in self.collection_names for key in self.collection_dct.keys()]
        return all(trues)

    def isAllValid(self) -> bool:
        """Check if the parameter dictionary contains all expected keys and values are non-null."""
        for key in self.collection_names:
            if not key in self.collection_dct:
                return False
            if self.collection_dct[key] is None:
                return False
            value = self.collection_dct[key]
            if isinstance(value, np.ndarray):   
                if value.size == 0 or np.isnan(value).any():
                    return False
            elif isinstance(value, (int, float)):
                if np.isnan(value):
                    return False
            else:
                raise ValueError(f"Unsupported type for key '{key}': {type(value)}")
        return True

    def add(self, collection_dct:Dict[str, Any]) -> None:
        """Add a key-value pair to the parameter."""
        diff = set(self.collection_names)
        diff = set(collection_dct.keys()) - set(self.collection_names)
        if len(diff) > 0:
            raise ValueError(f"Collection dictionary has unexpected keys: {diff}")
        self.collection_dct.update(collection_dct)

    def __eq__(self, other:Any) -> bool:
        """Check if two Collection objects are equal."""
        if not str(type(self)) == str(type(other)):
            raise RuntimeError(f"Cannot compare {str(type(self))} with {str(type(other))}.")
        # Check if all expected parameters are present and equal
        if not self.isAllValid() or not other.isAllValid():
            return False
        # Compare the values
        for key in self.collection_names:
            this_value = self.get(key)
            other_value = other.get(key)
            if isinstance(this_value, np.ndarray):
                if not np.allclose(this_value.flatten(), other_value.flatten()):
                    return False
            else:
                if not np.isclose(this_value, other_value):
                    return False
        return True


############################################
class PCollection(Collection):
    """Collection of Parameters of a distribution."""
    pass
    

############################################
class DCollection(Collection):
    """Container for the representation and properties of a distribution"""
    pass


############################################
class Random(object):

    def __init__(self, pcollection:Optional[Any]=None,
            dcollection:Optional[Any]=None, **kwargs) -> None:
        self.pcollection = pcollection
        self.dcollection = dcollection

    def estimatePCollection(self, sample_arr: np.ndarray) -> PCollection:
        """Estimates the Parameter instance for from the data array."""
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def makeDCollection(self, pcollection:Any) -> DCollection:
        """Create a Distribution object from the ParameterCollection."""
        raise NotImplementedError("This method should be overridden by subclasses.")

    def generateSample(self, pcollection:Any, num_sample:int) -> np.ndarray:
        """Generate a random sample from the self.actual_parameter."""
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def calculateEntropy(self, pcollection:Any) -> float:
        """Analytic calculation of the entropy of the distribution based on the parameters of the distribution."""
        raise NotImplementedError("This method should be overridden by subclasses.")

    def evaluate(self, pcollection:Any, num_samples:int) -> bool:
        """Evaluate the calculation using a round trip of estimation and generation."""
        sample_arr = self.generateSample(pcollection, num_samples)
        estimated_pcollection = self.estimatePCollection(sample_arr)
        dcollection = self.makeDCollection(estimated_pcollection)
        return dcollection == self.makeDCollection(pcollection)