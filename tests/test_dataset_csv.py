from autoencodersb.dataset_csv import DatasetCSV # type: ignore
import autoencodersb.constants as cn  # type: ignore

import pandas as pd    # type: ignore
import unittest

IGNORE_TESTS = False
IS_PLOT = False:while
NUM_EPOCH = 3

DATASET_CSV_PATH = "tests/test_dataset_csv.csv"
TARGET_COLUMN = "target"  # Assuming the target column is named 'target'
DF = pd.read_csv(DATASET_CSV_PATH)


class TestDatasetCSV(unittest.TestCase):

    def setUp(self):
        self.dataset = DatasetCSV(csv_input=DATASET_CSV_PATH, target_column=TARGET_COLUMN)
        self.datasetdf = DatasetCSV(csv_input=DF, target_column=TARGET_COLUMN)

    def checkLen(self, dataset):
        self.assertEqual(len(dataset), 4)

    def checkGetitem(self, dataset):
        features, target = dataset[0]
        self.assertEqual(features.shape, (3,))
        self.assertEqual(target, 100)

    def testLen(self):
        if IGNORE_TESTS:
            return
        self.checkLen(self.dataset)
        self.checkLen(self.dataset)

    def testGetitemCSV(self):
        if IGNORE_TESTS:
            return
        self.checkGetitem(self.dataset)
        self.checkGetitem(self.datasetdf)


if __name__ == '__main__':
    unittest.main()