from iplane.random import Collection, PCollection, DCollection  # type: ignore
import iplane.constants as cn  # type: ignore

import numpy as np
from typing import List, Any, Tuple
import unittest

IGNORE_TESTS = False
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
        if IGNORE_TESTS:
            return
        self.assertEqual(self.collection.collection_names, COLLECTION_NAMES)
        self.assertEqual(self.collection.collection_dct, COLLECTION_DCT)

    def getGet(self):
        """Test the get method of Collection."""
        # Test valid keys
        if IGNORE_TESTS:
            return
        self.assertTrue(np.array_equal(self.collection.get(APPLE), APPLE_ARR))
        self.assertEqual(self.collection.get("cherry"), 3)

    def testIsValid(self):
        """Test the isValid method of Collection."""
        if IGNORE_TESTS:
            return
        collection_dct = dict(COLLECTION_DCT)
        collection_dct["missing"] = np.array([1, 2, 3])
        with self.assertRaises(ValueError):
            self.collection = Collection(COLLECTION_NAMES, collection_dct)
    
    def testIsAllValid(self):
        """Test the isAllValid method of Collection."""
        if IGNORE_TESTS:
            return
        self.assertFalse(self.collection.isAllValid())
        # All keys but None values
        collection_dct = dict(COLLECTION_DCT)
        collection_dct[DATE] = None 
        collection_dct[ELDERBERRY] = np.nan
        self.collection = Collection(COLLECTION_NAMES, collection_dct)
        self.assertFalse(self.collection.isAllValid())
        # All keys and values are valid
        collection_dct[DATE] = 3
        collection_dct[ELDERBERRY] = np.array([7, 8, 9])
        self.collection = Collection(COLLECTION_NAMES, collection_dct)
        self.assertTrue(self.collection.isAllValid())

    def testEq(self):
        """Test the __eq__ method of Collection."""
        if IGNORE_TESTS:
            return
        # Same collection
        collection_dct = dict(COLLECTION_DCT)
        collection_dct[DATE] = np.array([[1, 2.0, 3], [4, 5, 6]])
        collection_dct[ELDERBERRY] = np.array([4, 5, 6])
        collection1 = Collection(COLLECTION_NAMES, collection_dct)
        collection2 = Collection(COLLECTION_NAMES, collection_dct)
        self.assertTrue(collection1 == collection2)
        # Different collections
        collection_dct2 = dict(COLLECTION_DCT)
        collection_dct2["banana"] = np.array([7, 8, 9])
        collection3 = Collection(COLLECTION_NAMES, collection_dct2)
        self.assertFalse(collection1 == collection3)

    def testAdd(self):
        """Test the add method of Collection."""
        if IGNORE_TESTS:
            return
        # Add a new key-value pair
        new_dct = {DATE: np.array([1, 2, 3]), ELDERBERRY: np.array([4, 5, 6])}
        self.collection.add(new_dct)
        self.assertTrue(self.collection.get(DATE) is not None)
        self.assertTrue(np.array_equal(self.collection.get(DATE), np.array([1, 2, 3])))
        # Add an existing key-value pair
        with self.assertRaises(ValueError):
            self.collection.add({"missing": APPLE_ARR})


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