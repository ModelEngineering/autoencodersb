'''Random estimates parameters and generates random samples from parameters.'''

""" 
This is an abstract class for estimating parameters of a distribution.
A subclass implements the estimate and generate methods. It is responsible for constructing
a Parameter object appropriate for these methods. Generated distributions are stored in a Distribution object.
"""
import iplane.constants as cn

import numpy as np
from typing import Tuple, Any, Optional, Dict, List

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
        self.collection_names = collection_names
        if collection_dct is None:
            collection_dct = {}
        else:
            self.isValidDct(self.collection_names, collection_dct)
        self.dct: Dict[str, Any] = dict(collection_dct)

    def get(self, name:str) -> Any:
        """Get the value of a parameter by its name."""
        return self.dct.get(name, None)

    @staticmethod
    def isValidDct(parameter_names:List[str], dct:Dict[str, Any]) -> bool:
        """Check if the parameter dictionary contains all expected keys."""
        return all(key in parameter_names for key in dct)


############################################
class PCollection(Collection):
    """Collection of Parameters of a distribution."""

    def add(self, collection_dct:Dict[str, Any]) -> None:
        """Add a key-value pair to the parameter."""
        self.dct.update(collection_dct)
    
    def isValid(self) -> bool:
        raise NotImplementedError("isValid must be implemented in subclasses.")
    
    def __eq__(self, other:Any) -> bool:
        """Check if two Parameter objects are equal."""
        raise NotImplementedError("Equality check must be implemented in subclasses.")
    

############################################
class DCollection(Collection):
    """Container for the representation and properties of a distribution"""
        
    def __eq__(self, other:Any) -> bool:
        """Check if two Parameter objects are equal."""
        if not isinstance(other, DCollection):
            return False
        if (not ENTROPY in self.dct) or (not ENTROPY not in other.dct):
            return False
        return bool(np.isclose(self.dct[ENTROPY], other.dct[ENTROPY], atol=1e-8))
    
    def isValid(self) -> bool:
        raise NotImplementedError("isValid must be implemented in subclasses.")


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