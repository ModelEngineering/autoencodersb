from iplane.random import Collection, PCollection, DCollection  # type: ignore
import iplane.constants as cn  # type: ignore

import numpy as np
from typing import List, Any, Tuple
import unittest

IGNORE_TESTS = True
IS_PLOT = False
DATE = "date"
ELDERBERRY = "elderberry"
APPLE = "apple"
APPLE_ARR = np.array([1, 2, 3])
BANANA = "banana"
MISSING_NAMES = [DATE, ELDERBERRY]  # Names that are not in the dictionary
COLLECTION_NAMES = [APPLE, BANANA, "cherry", DATE, ELDERBERRY]
COLLECTION_DCT = {
    "apple": APPLE_ARR,
    "banana": np.array([4, 5, 6]),
    "cherry": 3
}


########################################
class TestCollection(unittest.TestCase):

    def setUp(self):
        self.collection = Collection(COLLECTION_NAMES, COLLECTION_DCT)

    def testConstructor(self):
        """Test the constructor of Collection."""
        self.assertEqual(self.collection.collection_names, COLLECTION_NAMES)
        self.assertEqual(self.collection.collection_dct, COLLECTION_DCT)

    def getGet(self):
        """Test the get method of Collection."""
        # Test valid keys
        self.assertTrue(np.array_equal(self.collection.get(APPLE), APPLE_ARR))
        self.assertEqual(self.collection.get("cherry"), 3)

    def testIsValid(self):
        """Test the isValid method of Collection."""
        pass


########################################
class TestPCollection(unittest.TestCase):

    def setUp(self):
        pass


########################################
class TestPDollection(unittest.TestCase):

    def setUp(self):
        pass


if __name__ == '__main__':
    unittest.main()