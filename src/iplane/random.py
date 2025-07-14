'''Abstract class for estimating parameters and generating random samples from parameters.'''

""" 
This is an abstract class for estimating parameters of a distribution.
A subclass implements the estimate and generate methods. It is responsible for constructing
a Parameter object appropriate for these methods. Generated distributions are stored in a Distribution object.
"""

import numpy as np
from typing import Any, Optional, Dict, List, Union, cast

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
        self.isValid()

    def get(self, name:str) -> Any:
        """Get the value of a parameter by its name."""
        return self.collection_dct.get(name, None)

    def isValid(self):
        """Check if the dictionary contains only valid keys."""
        trues =  [key in self.collection_names for key in self.collection_dct.keys()]
        if not all(trues):
            diff = set(self.collection_dct.keys()) - set(self.collection_names)
            raise ValueError(f"Collection dictionary has unexpected keys: {diff}")

    def isAllValid(self)->None:
        """Check if the parameter dictionary contains all expected keys and values are non-null."""
        for key in self.collection_names:
            if not key in self.collection_dct:
                raise ValueError(f"Collection dictionary is missing key: {key}")
            if self.collection_dct[key] is None:
                raise ValueError(f"Collection dictionary has None value for key: {key}")
            value = self.collection_dct[key]
            if isinstance(value, np.ndarray):   
                if value.size == 0 or np.isnan(value).any():
                    raise ValueError(f"Collection dictionary has empty or NaN value for key: {key}")
            elif isinstance(value, (int, float)):
                if np.isnan(value):
                    raise ValueError(f"Collection dictionary has NaN value for key: {key}")
            else:
                raise ValueError(f"Collection dictionary has unexpected type for key: {key}, value: {value}")

    def add(self, **kwargs) -> None:
        """Add a key-value pair to the parameter."""
        diff = set(self.collection_names)
        diff = set(kwargs.keys()) - set(self.collection_names)
        if len(diff) > 0:
            raise ValueError(f"Collection dictionary has unexpected keys: {diff}")
        self.collection_dct.update(kwargs)

    def __eq__(self, other:Any) -> bool:
        """Check if two Collection objects are equal."""
        if not str(type(self)) == str(type(other)):
            raise RuntimeError(f"Cannot compare {str(type(self))} with {str(type(other))}.")
        # Check if all expected parameters are present and equal
        try:
            self.isAllValid()
            other.isAllValid()
        except ValueError:
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
    
    def __str__(self) -> str:
        """
        Returns a string representation of the Collections object.
        """
        lines:list = [f"\n{self.__class__.__name__}:"]
        for key in self.collection_names:
            value = self.get(key)
            lines.append(f"{key}:\n{value}")
        msg = "\n".join(lines)
        return(msg)


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

    def makePCollection(self, sample_arr: np.ndarray) -> PCollection:
        """Estimates the Parameter instance for from the data array."""
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def predict(self, *args, **kwargs) -> np.ndarray:
        """Predict the probability of a single variate_arr (multiple dimensions)."""
        """Uses saved PCollection if this is not specified."""
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def makeDCollection(self, *args, **kwargs)  -> DCollection:
        """Create a Distribution object from the ParameterCollection."""
        raise NotImplementedError("This method should be overridden by subclasses.")

    def generateSample(self, pcollection:Any, num_sample:int) -> np.ndarray:
        """Generate a random sample from the self.actual_parameter."""
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def calculateEntropy(self, collection:Any) -> float:
        """Analytic calculation of the entropy of the distribution based on the parameters of the distribution."""
        raise NotImplementedError("This method should be overridden by subclasses.")

    def evaluate(self, pcollection:Any, num_samples:int) -> bool:
        """Evaluate the calculation using a round trip of estimation and generation."""
        sample_arr = self.generateSample(pcollection, num_samples)
        estimated_pcollection = self.makePCollection(sample_arr)
        dcollection = self.makeDCollection(pcollection=estimated_pcollection)
        return dcollection == self.makeDCollection(pcollection)
    
    def setPCollection(self, pcollection:Union[PCollection, None]) -> PCollection:
        """Set the PCollection for this Random instance."""
        if pcollection is None:
            if self.pcollection is None:
                raise RuntimeError("PCollection has not been set.")
            pcollection = self.pcollection
        return cast(PCollection, pcollection)
    
    def setDCollection(self, dcollection:Union[DCollection, None]) -> DCollection:
        """Set the DCollection for this Random instance."""
        if dcollection is None:
            if self.dcollection is None:
                raise RuntimeError("DCollection has not been set.")
            dcollection = self.dcollection
        return cast(DCollection, dcollection)