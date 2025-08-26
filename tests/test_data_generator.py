from autoencodersb.data_generator import DataGenerator # type: ignore
from autoencodersb.polynomial_collection import PolynomialCollection  # type: ignore
from autoencodersb.sequence import Sequence # type: ignore
import autoencodersb.constants as cn  # type: ignore

import pandas as pd  # type: ignore
import numpy as np
from torch.utils.data import DataLoader
from typing import cast
import unittest

IGNORE_TESTS = True
IS_PLOT = True


########################################
class TestDataGenerator(unittest.TestCase):

    def setUp(self):
        self.polynomial_collection = PolynomialCollection.make(
            is_mm_term=True,
            is_first_order_term=True,
            is_second_order_term=True,
            is_third_order_term=False)
        self.num_sample = 20
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
        self.generator.specifySequences(seq_type=cn.SEQ_EXPONENTIAL, density=10)
        dataloader = self.generator.generate()
        self.assertTrue(isinstance(dataloader, DataLoader))
        self.assertEqual(len(self.generator.data_df), self.num_sample)

    def testPlotGeneratedData(self):
        if IGNORE_TESTS:
            return
        for seq_type in cn.SEQ_TYPES:
            sequences = [Sequence(num_sample=self.num_sample, seq_type=seq_type)]*self.polynomial_collection.num_variable
            self.generator.specifySequences(sequences=sequences)
            self.generator.generate()
            self.generator.plotGeneratedData(x_column="X_0", is_plot=IS_PLOT)

    def testPlotErrorDifference(self):
        if IGNORE_TESTS:
            return
        for seq_type in cn.SEQ_TYPES:
            sequences = [Sequence(num_sample=self.num_sample, seq_type=seq_type)]*self.polynomial_collection.num_variable
            self.generator.specifySequences(sequences=sequences)
            self.generator.generate()
            df = self.generator.plotErrorDifference(other_df=self.generator.data_df, is_plot=IS_PLOT)
            mss = df.std().sum()
            self.assertTrue(mss < 1e-3)

    def testNoNoise(self):
        if IGNORE_TESTS:
            return
        polynomial_collection = PolynomialCollection.make(
            is_mm_term=False,
            is_first_order_term=True,
            is_second_order_term=False,
            is_third_order_term=False)
        num_sample = 200
        generator = DataGenerator(polynomial_collection=polynomial_collection, num_sample=num_sample)
        sequences = [Sequence(num_sample=num_sample, seq_type=cn.SEQ_EXPONENTIAL, rate=0.001)]*polynomial_collection.num_variable
        generator.specifySequences(sequences=sequences)
        _ = generator.generate()
        noise_generator = DataGenerator(polynomial_collection=polynomial_collection,
                num_sample=num_sample, noise_std=0.00)
        noise_generator.specifySequences(sequences=sequences)
        _ = noise_generator.generate()
        x_column = "X_0"
        error_df = noise_generator.plotErrorDifference(generator.data_df, x_column=x_column, is_plot=IS_PLOT)
        test_df = error_df.drop(columns=[x_column])
        self.assertTrue((error_df*test_df).sum().sum() < 1e-3)

    def testNoise(self):
        #if IGNORE_TESTS:
        #    return
        polynomial_collection = PolynomialCollection.make(
            is_mm_term=False,
            is_first_order_term=True,
            is_second_order_term=False,
            is_third_order_term=False)
        num_sample = 2000
        noise_var = 1
        sequences = [Sequence(num_sample=num_sample, seq_type=cn.SEQ_EXPONENTIAL, rate=0.001)]*polynomial_collection.num_variable
        noise_generator = DataGenerator(polynomial_collection=polynomial_collection,
                num_sample=num_sample, noise_std=np.sqrt(noise_var))
        noise_generator.specifySequences(sequences=sequences)
        _ = noise_generator.generate()
        x_column = "X_0"
        data_df = noise_generator.data_df.copy()
        sequence_var = data_df[x_column].var()
        data_column = list(data_df.columns)[1]
        column_var = data_df[data_column].var()
        k_val = polynomial_collection.terms[0].coefficient # type: ignore
        k_val = cast(float, k_val)
        expected_var = (k_val**2) * sequence_var + noise_var  # type: ignore
        self.assertAlmostEqual(column_var, expected_var, delta=1e-1)

if __name__ == '__main__':
    unittest.main()