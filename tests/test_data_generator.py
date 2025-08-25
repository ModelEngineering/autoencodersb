from autoencodersb.data_generator import DataGenerator # type: ignore
from autoencodersb.polynomial_collection import PolynomialCollection  # type: ignore
from autoencodersb.sequence import Sequence # type: ignore
import autoencodersb.constants as cn  # type: ignore

from torch.utils.data import DataLoader
import numpy as np
import unittest

IGNORE_TESTS = False
IS_PLOT = True


########################################
class TestDataGenerator(unittest.TestCase):

    def setUp(self):
        self.polynomial_collection = PolynomialCollection.make(
            is_mm_term=True,
            is_first_order_term=True,
            is_second_order_term=True,
            is_third_order_term=True)
        self.num_sample = 100
        self.generator = DataGenerator(polynomial_collection=self.polynomial_collection,
                num_sample=self.num_sample)

    def testConstructor(self):
        if IGNORE_TESTS:
            return
        self.assertIsInstance(self.generator, DataGenerator)
        self.assertEqual(self.generator.polynomial_collection, self.polynomial_collection)

    def testGenerate(self):
        if IGNORE_TESTS:
            return
        self.generator.specifyIID()
        dataloader = self.generator.generate()
        self.assertTrue(isinstance(dataloader, DataLoader))
        #
        self.generator.specifySequences()
        dataloader = self.generator.generate()
        self.assertTrue(isinstance(dataloader, DataLoader))
        self.assertEqual(len(self.generator.data_df), self.num_sample)

    def testPlot(self):
        if IGNORE_TESTS:
            return
        for seq_type in cn.SEQ_TYPES:
            sequences = [Sequence(seq_type=seq_type)]*self.polynomial_collection.num_variable
            self.generator.specifySequences(sequences=sequences)
            self.generator.generate()
            df = self.generator.plotGeneratedData(is_plot=IS_PLOT)


if __name__ == '__main__':
    unittest.main()